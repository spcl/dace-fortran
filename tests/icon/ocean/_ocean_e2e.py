"""Shared harness for the ICON-O ocean kernel end-to-end numerical tests.

Builds two shared libs with the SAME flat C ABI and drives both from one set of
random ctypes buffers: DUT is the kernel lowered to a DaCe SDFG through the
auto-generated ``bind(c)`` shim; REF is the original Fortran kernel through the
SAME shim retargeted to call it directly (:func:`_retarget_shim`) -- identical
ABI, no hand-authored per-kernel caller.

Each kernel runs in its own subprocess (:func:`run_kernel_e2e`): the SDFG
``.so`` is dlopen'd RTLD_GLOBAL, which would otherwise leak DaCe runtime
symbols into the next kernel's in-process flang/MLIR build.
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# Fortran C-ABI type -> (numpy dtype, ctypes value type).
_NP = {"real(c_double)": np.float64, "integer(c_int)": np.int32, "logical(c_bool)": np.int8}
_CT = {"real(c_double)": ctypes.c_double, "integer(c_int)": ctypes.c_int, "logical(c_bool)": ctypes.c_bool}

# ICON refin-ctrl index arrays (verts/cells/edges%{start,end}_{block,index}) are
# ALLOCATABLE with a NEGATIVE lower bound (min_rl:max_rl) and read at negative
# levels (end_block(-10)).  The bind_c shim carries each array's per-dim lower
# bound (<arr>_lb<i>), so these are sized to [-16, 16] (covers ICON's widest,
# edges' [-10,10]) instead of mesh size n; under int_fill every slot is the one
# valid block/index so any level read is in-bounds.
_REFIN_CTRL_RE = re.compile(r"_(?:start|end)_(?:block|index)$")
_REFIN_CTRL_LBOUND = -16
_REFIN_CTRL_EXTENT = 33


def _refin_ctrl_flat(sym: str, suffix_re: str) -> str:
    """Flat array name behind a per-dim extent (``_d<i>``) / lower-bound (``_lb<i>``) C-ABI arg; unchanged if no such suffix."""
    return re.sub(suffix_re, "", sym)


def _module_symbol_map(binding: str) -> dict:
    """``sym -> module`` map from the DUT binding's ``use <module>, only: <sym>__mod => <sym>`` imports."""
    sym_module = {}
    for module, renames in re.findall(r"use\s+(\w+),\s*only:\s*([^\n]+)", binding):
        for _alias, sym in re.findall(r"(\w+)__mod\s*=>\s*(\w+)", renames):
            sym_module[sym] = module
    return sym_module


def _size_derived_module_dims(binding: str):
    """ICON grid-DIMENSION module globals the DUT binding derives from an array
    extent (``n_zlev = int(size(psi_c, dim=2), c_int)``, ``nproma``, etc).

    An isolated kernel reads these as BSS=0, sizing automatic locals to zero ->
    OOB; the SDFG path derives them as free symbols instead, so the reference
    must seed them too (every extent here is mesh size ``n``).  Returns
    ``[(sym, module), ...]`` for syms with a ``size()`` derivation; a directly-read
    global goes through :func:`_resolve_module_seeds` instead.
    """
    sym_module = _module_symbol_map(binding)
    seen, out = set(), []
    for sym in re.findall(r"^\s*(\w+)\s*=\s*int\(size\(\w+,\s*dim=\d+\),\s*c_int\)", binding, re.M):
        if sym in sym_module and sym not in seen:
            seen.add(sym)
            out.append((sym, sym_module[sym]))
    return out


def _resolve_module_seeds(binding: str, seeds: dict):
    """Resolve ``{fortran_sym: int_value}`` config-global seeds to
    ``[(mangled_symbol, length, value), ...]`` for ctypes ``in_dll`` seeding on
    BOTH the DUT and reference ``.so``.

    Some ICON config globals (``nproma``, ``nflatlev``) are read straight from
    the module on both sides, so an isolated run sees BSS=0 -> OOB with no array
    extent to derive them from; the test supplies namelist values and the
    harness seeds both ``.so``s identically.  ``length`` comes from the
    binding's ``allocate(<sym>(N))`` (else 1 for a scalar).
    """
    sym_module = _module_symbol_map(binding)
    out = []
    for sym, value in seeds.items():
        module = sym_module.get(sym)
        if module is None:
            raise KeyError(f"module-seed '{sym}' is not imported as '{sym}__mod => {sym}' in the binding")
        m = re.search(rf"allocate\({sym}\((\d+)\)\)", binding)
        length = int(m.group(1)) if m else 1
        out.append((f"__{module}_MOD_{sym}", length, int(value)))
    return out


# Kept alive for the child's lifetime -- descriptor's base_addr points into these, and the child os._exit()s before any Fortran teardown.
_ALLOC_SEED_KEEPALIVE: list = []


def _resolve_module_array_seeds(binding: str, seeds: dict):
    """Resolve ``{fortran_sym: length}`` allocatable-module-ARRAY seeds to
    ``[(mangled_symbol, length), ...]``.

    Some ICON module globals are ``REAL(8), ALLOCATABLE`` (e.g. ``vct_a``) that
    the kernel indexes directly; an isolated run leaves them UNALLOCATED (REF
    SEGVs, SDFG reads a garbage size-1 fallback).  Like
    :func:`_resolve_module_seeds`, the test supplies the length and the harness
    allocates + fills identical values on BOTH ``.so``s.
    """
    sym_module = _module_symbol_map(binding)
    out = []
    for sym, length in seeds.items():
        module = sym_module.get(sym)
        if module is None:
            raise KeyError(f"module-array-seed '{sym}' is not imported as '{sym}__mod => {sym}' in the binding")
        out.append((f"__{module}_MOD_{sym}", int(length)))
    return out


def _seed_alloc_array(cdll, mangled: str, length: int):
    """Allocate + fill a ``REAL(8), ALLOCATABLE :: <sym>(:)`` module global in a
    loaded ``.so`` by writing its gfortran array descriptor in place.

    Layout (gfortran 15, rank-1 allocatable, 64-byte descriptor): base_addr@0,
    offset@8 (= -lbound), elem_len@16, dtype@24 {version:i32, rank:i8, type:i8,
    attribute:i16}, span@32, dim0 {stride@40, lbound@48, ubound@56}.  The fill is
    a deterministic ramp -- only IDENTITY across the two ``.so``s matters for
    bit-exactness."""
    buf = (ctypes.c_double * length)(*[float(i + 1) for i in range(length)])
    _ALLOC_SEED_KEEPALIVE.append(buf)
    addr = ctypes.addressof((ctypes.c_byte * 64).in_dll(cdll, mangled))
    ctypes.c_void_p.from_address(addr + 0).value = ctypes.addressof(buf)
    ctypes.c_ssize_t.from_address(addr + 8).value = -1  # offset = -lbound
    ctypes.c_size_t.from_address(addr + 16).value = 8  # elem_len (real*8)
    ctypes.c_int32.from_address(addr + 24).value = 0  # dtype.version
    ctypes.c_int8.from_address(addr + 28).value = 1  # dtype.rank
    ctypes.c_int8.from_address(addr + 29).value = 3  # dtype.type (BT_REAL)
    ctypes.c_int16.from_address(addr + 30).value = 0  # dtype.attribute
    ctypes.c_ssize_t.from_address(addr + 32).value = 8  # span
    ctypes.c_ssize_t.from_address(addr + 40).value = 1  # dim0.stride
    ctypes.c_ssize_t.from_address(addr + 48).value = 1  # dim0.lbound
    ctypes.c_ssize_t.from_address(addr + 56).value = length  # dim0.ubound


def _global_bind_lines(indent: str, global_binds, deferred_members=None, pointer_members=None):
    """Fortran that copies a module-global derived type's components from the
    shim's reconstructed dummy struct, emitted just before the call on BOTH shims.

    A single-TU extraction drops the module global's initialiser (ICON's
    ``init_ho_params``), so a kernel that reads ``v_params%<comp>`` instead of the
    ``p_phys_param`` dummy it was handed sees an unassociated pointer / a zero
    scalar.  The DUT gets the same global marshalled by the binding, so the copy
    runs on both sides and both read byte-identical values.

    Deferred-shape components (the binding guards them with
    ``associated()``/``allocated()``) are allocated from the source's own bounds
    and then value-copied -- NOT pointer-associated: the SDFG marshals the global
    and the dummy into two independent arrays, so aliasing them on the reference
    would propagate a write the DUT does not see.
    """
    deferred_members = deferred_members or set()
    pointer_members = pointer_members or set()
    out = []
    for _module, obj, comp, src in global_binds:
        key = (obj.lower(), comp.lower())
        if key in deferred_members:
            guard = "associated" if key in pointer_members else "allocated"
            out.append(f"{indent}if (.not. {guard}({obj} % {comp})) allocate({obj} % {comp}, source = {src})")
        out.append(f"{indent}{obj} % {comp} = {src}")
    return out


def _global_bind_imports(global_binds) -> list:
    """``use <module>, only: <object>`` lines for the module globals bound by
    :func:`_global_bind_lines` (dedup, order-preserving)."""
    seen, out = set(), []
    for module, obj, _comp, _src in global_binds:
        if (module, obj) not in seen:
            seen.add((module, obj))
            out.append(f"  use {module}, only: {obj}")
    return out


def _retarget_shim(shim: str,
                   dace_name: str,
                   entry: str,
                   module_dims=None,
                   n_val=None,
                   solver_allocs=None,
                   pointer_members=None,
                   extra_refmod_imports=None,
                   global_binds=None,
                   deferred_members=None) -> str:
    """Rewrite the auto ``<dace_name>_c`` shim into ``<dace_name>_ref_c``: same
    flat->struct reconstruction, but ``use``/``call``/name retarget to the
    ORIGINAL kernel ``entry`` instead of ``<dace_name>_dace``.

    ``logical(c_bool)`` args are coerced to the kernel's LOGICAL kind at the
    call site (C ABI keeps them ``c_bool``), same as the SDFG path.

    ``module_dims`` (grid-DIMENSION globals from :func:`_size_derived_module_dims`)
    are seeded to ``n_val`` before the call, mirroring ICON's namelist init --
    else the isolated reference reads them as 0.

    ``solver_allocs`` (``[(object, component, dims), ...]``) builds the
    allocatable work members of module-level solver objects that a single-TU
    extraction's stubbed ``ocean_solve_construct`` leaves unallocated (the stock
    write would SEGV; the DUT gets this scratch from its own marshalling layer,
    so only REF needs it).  ``dims`` may reference reconstructed dummy structs
    and the seeded grid-dim globals (``nproma__refmod``), both in scope before
    the call.

    ``global_binds`` (``[(module, object, component, source), ...]``) copies a
    module global's components from the shim's reconstructed dummy struct; see
    :func:`_global_bind_lines`."""
    mod, proc = entry.split("::")
    proc = proc.lower()
    module_dims = module_dims or []
    solver_allocs = solver_allocs or []
    global_binds = global_binds or []
    # Distinct solver objects to ``use`` from ``mod`` (dedup, order-preserving).
    solver_objs = []
    for obj, _comp, _dims in solver_allocs:
        if obj not in solver_objs:
            solver_objs.append(obj)
    bool_args = set(re.findall(r"logical\(c_bool\),\s*value\s*::\s*(\w+)", shim))
    out = []
    for ln in shim.splitlines():
        if ln.strip().startswith(f"use {dace_name}_dace_bindings"):
            extra = "".join(f", {o}" for o in solver_objs)
            out.append(f"  use {mod}, only: {proc}{extra}")
            # ``<sym>__refmod => <module>::<sym>`` rename avoids clashing with any
            # same-named shim local; seeded just before the call below.
            for sym, module in module_dims:
                out.append(f"  use {module}, only: {sym}__refmod => {sym}")
            # Seeded grid-dim globals a solver-alloc extent references (nproma__refmod):
            # imported (renamed) so the extent resolves; value comes from the ctypes module-seed.
            for sym, module in (extra_refmod_imports or []):
                out.append(f"  use {module}, only: {sym}__refmod => {sym}")
            out.extend(_global_bind_imports(global_binds))
            continue
        if f"call {dace_name}_dace_finalize()" in ln:
            continue
        m = re.match(rf"(\s*)call {dace_name}_dace\((.*)\)\s*$", ln)
        if m:
            indent, arglist = m.group(1), m.group(2)
            for sym, _module in module_dims:
                out.append(f"{indent}{sym}__refmod = {n_val}")
            # Stubbed solver objects' allocatable work arrays (skipped by the empty
            # ocean_solve_construct); zero-init so a read-before-write of an unwritten
            # slot is deterministic across DUT and reference.
            for obj, comp, dims in solver_allocs:
                guard = "associated" if (obj.lower(), comp.lower()) in (pointer_members or set()) else "allocated"
                out.append(f"{indent}if (.not. {guard}({obj} % {comp})) allocate({obj} % {comp}({dims}))")
                out.append(f"{indent}{obj} % {comp} = 0.0d0")
            out.extend(_global_bind_lines(indent, global_binds, deferred_members, pointer_members))
            args = [a.strip() for a in arglist.split(",")]
            args = [f"logical({a})" if a in bool_args else a for a in args]
            out.append(f"{indent}call {proc}({', '.join(args)})")
            continue
        ln = ln.replace(f"subroutine {dace_name}_c(", f"subroutine {dace_name}_ref_c(")
        ln = ln.replace(f"end subroutine {dace_name}_c", f"end subroutine {dace_name}_ref_c")
        ln = ln.replace(f"name='{dace_name}_c'", f"name='{dace_name}_ref_c'")
        out.append(ln)
    return "\n".join(out) + "\n"


def _inject_dut_shim_prologue(shim: str,
                              dace_name: str,
                              entry: str,
                              module_dims=None,
                              n_val=None,
                              solver_allocs=None,
                              pointer_members=None,
                              extra_refmod_imports=None,
                              global_binds=None,
                              deferred_members=None) -> str:
    """Set up the stubbed module globals in the DUT shim -- symmetric to
    :func:`_retarget_shim`'s REF-side injection, but KEEPING the
    ``<dace_name>_dace`` (SDFG) call.

    ``solver_allocs``: ``ocean_solve_construct`` is stubbed empty by single-TU
    extraction, so these members stay unallocated; the binding's live-member
    marshalling sizes each SoA companion from ``size(host_member)``, so without
    this the marshalling takes the degenerate ``(1,1)`` fallback and the kernel's
    mesh-bounded writes smash the heap.  Same ``(object, component, dims)`` list
    the REF shim uses.

    ``global_binds``: same list the REF shim uses -- the binding marshals the
    module global into its own SDFG argument, so the DUT needs the identical
    host-side values or the two sides start from different inputs.
    """
    mod, _proc = entry.split("::")
    module_dims = module_dims or []
    solver_allocs = solver_allocs or []
    global_binds = global_binds or []
    if not solver_allocs and not global_binds:
        return shim
    solver_objs = []
    for obj, _comp, _dims in solver_allocs:
        if obj not in solver_objs:
            solver_objs.append(obj)
    out = []
    for ln in shim.splitlines():
        if ln.strip().startswith(f"use {dace_name}_dace_bindings"):
            out.append(ln)
            if solver_objs:
                out.append(f"  use {mod}, only: {', '.join(solver_objs)}")
            for sym, module in module_dims:
                out.append(f"  use {module}, only: {sym}__refmod => {sym}")
            # Seeded grid-dim globals a solver-alloc extent references (nproma__refmod):
            # imported (renamed) so the extent resolves; value comes from the ctypes module-seed.
            for sym, module in (extra_refmod_imports or []):
                out.append(f"  use {module}, only: {sym}__refmod => {sym}")
            out.extend(_global_bind_imports(global_binds))
            continue
        m = re.match(rf"(\s*)call {dace_name}_dace\((.*)\)\s*$", ln)
        if m:
            indent = m.group(1)
            for sym, _module in module_dims:
                out.append(f"{indent}{sym}__refmod = {n_val}")
            for obj, comp, dims in solver_allocs:
                # t_ocean_solve mixes ALLOCATABLE (x_loc_wp/res_loc_wp) and POINTER
                # (b_loc_wp) members -- guard picked per the binding's own associated()/allocated() use.
                guard = "associated" if (obj.lower(), comp.lower()) in (pointer_members or set()) else "allocated"
                out.append(f"{indent}if (.not. {guard}({obj} % {comp})) allocate({obj} % {comp}({dims}))")
                out.append(f"{indent}{obj} % {comp} = 0.0d0")
            out.extend(_global_bind_lines(indent, global_binds, deferred_members, pointer_members))
            out.append(ln)
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _parse_abi(shim: str):
    """Recover the shim's flat C ABI: ``(header_args, value_ftype, ptr_ftype,
    ptr_shape_expr, dim_symbols, ptr_local)``.  ``ptr_local`` maps each pointer
    header arg (``<x>_p``) to its ``c_f_pointer`` local name -- the key
    :func:`run_kernel_e2e`'s ``array_overrides`` uses to pin a buffer.
    """
    m = re.search(r"subroutine\s+\w+\(([^)]*)\)\s*bind", shim, re.S)
    header = [a.strip() for a in m.group(1).replace("&", " ").split(",") if a.strip()]
    value_ftype = {
        name: ftype
        for ftype, name in re.findall(r"(integer\(c_int\)|real\(c_double\)|logical\(c_bool\)),\s*value\s*::\s*(\w+)",
                                      shim)
    }
    ptr_ftype, ptr_shape, ptr_local = {}, {}, {}
    for p, local, shp in re.findall(r"call c_f_pointer\((\w+),\s*(\w+),\s*\[([^\]]*)\]\)", shim):
        ptr_shape[p] = shp
        ptr_local[p] = local
        dt = re.search(rf"(real\(c_double\)|integer\(c_int\)|logical\(c_bool\)),\s*pointer\s*::\s*{local}\b", shim)
        ptr_ftype[p] = dt.group(1)
    dim_symbols = set()
    for shp in ptr_shape.values():
        dim_symbols |= set(re.findall(r"[A-Za-z_]\w*", shp))
    return header, value_ftype, ptr_ftype, ptr_shape, dim_symbols, ptr_local


def synth_call_inputs(shim,
                      *,
                      n,
                      seed,
                      float_range=(-1.0, 1.0),
                      int_fill=None,
                      scalar_overrides=None,
                      array_overrides=None,
                      mesh_buffers=None):
    """Parse the shim's flat C ABI and synthesize ``(call_plan, inputs, ptr_args,
    ptr_local)`` -- ctypes call plan plus random/pinned input buffers.

    Shared by the fork-based single-rank differential (:func:`_build_and_compare`)
    and the in-process 2-rank MPI driver; only divergence is the driver
    overriding ``comm`` with a live ``mpi4py`` handle via ``scalar_overrides``.
    """
    scalar_overrides = scalar_overrides or {}
    array_overrides = array_overrides or {}
    mesh_buffers = mesh_buffers or {}
    header, value_ftype, ptr_ftype, ptr_shape, dim_symbols, ptr_local = _parse_abi(shim)
    # Refin-ctrl arrays span the full level range, not mesh size n -- size their
    # extent symbols to _REFIN_CTRL_EXTENT so the negative lower bound stays in bounds.
    dimvals = {
        s: (_REFIN_CTRL_EXTENT if _REFIN_CTRL_RE.search(_refin_ctrl_flat(s, r"_d\d+$")) else n)
        for s in dim_symbols
    }
    rng = np.random.default_rng(seed)
    inputs, call_plan = {}, []
    for arg in header:
        if arg in ptr_shape:
            shape = tuple(int(eval(tok, {}, dimvals)) for tok in ptr_shape[arg].split(","))
            npdt = _NP[ptr_ftype[arg]]
            override = array_overrides.get(ptr_local[arg])
            mesh_buf = mesh_buffers.get(ptr_local[arg])
            if mesh_buf is not None:  # explicit per-element buffer (real mesh connectivity)
                if tuple(mesh_buf.shape) != shape:
                    raise ValueError(f"mesh buffer for {ptr_local[arg]!r} has shape "
                                     f"{tuple(mesh_buf.shape)}, harness expects {shape}")
                base = np.asfortranarray(mesh_buf.astype(npdt))
            elif override is not None:  # pinned to a constant (e.g. a valid loop bound)
                base = np.asfortranarray(np.full(shape, override, dtype=npdt))
            elif npdt == np.float64:
                base = np.asfortranarray(rng.uniform(float_range[0], float_range[1], shape).astype(npdt))
            elif int_fill is not None:  # degenerate valid mesh: every count/index/bound == int_fill
                # One in-domain block/edge/vertex per slot, every connectivity index -> int_fill:
                # exactly one in-bounds iteration everywhere, no vacuous range, no OOB composite index.
                base = np.asfortranarray(np.full(shape, int_fill, dtype=npdt))
            else:  # integer count / index array -> in-bounds [1, n]
                base = np.asfortranarray(rng.integers(1, n + 1, shape).astype(npdt))
            inputs[arg] = base
            call_plan.append(("ptr", arg, ctypes.c_void_p))
        else:
            ft = value_ftype[arg]
            lb_flat = _refin_ctrl_flat(arg, r"_lb\d+$")
            if lb_flat != arg:  # an array lower-bound arg ``<flat>_lb<i>``
                v = scalar_overrides.get(arg, _REFIN_CTRL_LBOUND if _REFIN_CTRL_RE.search(lb_flat) else 1)
            else:
                # Extent args ride dimvals (refin-ctrl-aware) to match the c_f_pointer shape.
                int_default = int_fill if int_fill is not None else 0
                v = scalar_overrides.get(
                    arg, dimvals[arg] if arg in dim_symbols else (60.0 if ft == "real(c_double)" else int_default))
            call_plan.append(("val", arg, _CT[ft], v))
    return call_plan, inputs, list(ptr_shape), ptr_local


def _invoke(so_path, call_plan, bufs, sym, sdfg_so=None, module_seeds=None, array_seeds=None):
    # SDFG .so dlopen'd RTLD_GLOBAL first so the DaCe runtime is live and its symbols resolve for the binding .so.
    if sdfg_so is not None:
        ctypes.CDLL(str(sdfg_so), mode=ctypes.RTLD_GLOBAL)
    cdll = ctypes.CDLL(str(so_path))
    # ICON namelist/config globals otherwise read as BSS 0 -- seeded identically
    # on DUT and reference (see _resolve_module_seeds) to stay bit-exact.
    for mangled, length, value in (module_seeds or []):
        cell = (ctypes.c_int * length).in_dll(cdll, mangled)
        for i in range(length):
            cell[i] = value
    # Allocatable module ARRAYS read module-direct (e.g. vct_a) -- unallocated in isolation; allocate + fill identically on both .so's.
    for mangled, length in (array_seeds or []):
        _seed_alloc_array(cdll, mangled, length)
    fn = getattr(cdll, sym)
    argtypes, args = [], []
    for kind, arg, ct, *rest in call_plan:
        argtypes.append(ct)
        args.append(bufs[arg].ctypes.data_as(ct) if kind == "ptr" else ct(rest[0]))
    fn.argtypes = argtypes
    fn.restype = None
    fn(*args)


def _run_in_fork(so_path,
                 call_plan,
                 inputs,
                 ptr_args,
                 sym,
                 save_prefix: Path,
                 sdfg_so=None,
                 module_seeds=None,
                 array_seeds=None):
    """Run ``sym`` from ``so_path`` in a forked child on a copy of ``inputs``,
    saving each output to ``<save_prefix>_<arg>.npy``, then ``os._exit`` so the
    child's (occasionally heap-corrupting) teardown never runs.  Separate
    processes for DUT/REF also avoid a cross-allocator double-free seen when
    both share one process.  Returns the child's wait status (0 == clean).
    """
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            bufs = {k: v.copy() for k, v in inputs.items()}
            _invoke(so_path, call_plan, bufs, sym, sdfg_so=sdfg_so, module_seeds=module_seeds, array_seeds=array_seeds)
            for k in ptr_args:
                np.save(f"{save_prefix}_{k}.npy", bufs[k])
        except BaseException:
            traceback.print_exc()
            code = 1
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    return status


def _build_and_compare(tu_path: Path,
                       entry: str,
                       scalar_overrides: dict,
                       array_overrides: dict,
                       float_range: tuple,
                       n: int,
                       seed: int,
                       out: Path,
                       int_fill=None,
                       module_seeds=None,
                       module_array_seeds=None,
                       do_not_emit=None,
                       prelude_paths=None,
                       inject_use_mpi=False,
                       ref_solver_allocs=None,
                       ref_global_binds=None):
    """Worker body (runs in a subprocess): build DUT + REF, drive both, return
    ``(max_diff, n_changed)``.  Imports deferred so the module loads cheaply in
    the orchestrating parent.

    ``do_not_emit`` drops halo/MPI/sync/timer/logging externals from the DUT
    SDFG so a single-rank dycore carries no MPI.  ``prelude_paths`` (extra .f90
    files, e.g. the mpi stub + no-op point-to-point impls) compile ahead of the
    TU on both sides.  ``inject_use_mpi`` gives inlined ``mo_mpi`` a ``use mpi``
    so its dual-typed real*8/real*4 calls resolve through the stub's assumed-type
    interface (no ``-fallow-argument-mismatch``).
    """
    import dace

    from dace_fortran.build import build_sdfg
    from dace_fortran.bindings import build_fortran_library
    from dace_fortran.external import apply_external_functions, clear_external_registry

    # Drop halo/MPI/sync/timer externals from the DUT SDFG; REF keeps the real
    # bodies but its mpi_* leaves resolve to the no-op prelude impls below.
    clear_external_registry()
    if do_not_emit:
        apply_external_functions(do_not_emit=list(do_not_emit))

    # Compiled ahead of the TU on the reference build (and DUT binding, for use
    # mpi): the mpi stub module + no-op point-to-point impls.
    extra_prelude = [Path(p) for p in (prelude_paths or [])]

    # Reference-TU fixup so the extracted single-TU is gfortran-EXECUTABLE (SDFG
    # build reads the RAW tu_path -- bridge resolves separately).  mo_mpi's
    # inlined wrappers call one mpi_recv/mpi_isend with both real*8 and real*4
    # buffers; use mpi binds them to the stub's type(*) interface (no
    # -fallow-argument-mismatch).  Devirtualised call names (<base>_deconiface_N
    # / _deconproc_N) are NOT undefined -- each renames an in-TU body via USE --
    # only the raw mpi_* leaves need the no-op prelude impls.
    ref_tu = tu_path
    if inject_use_mpi:
        src = tu_path.read_text()
        if "MODULE mo_mpi\n" not in src:
            raise RuntimeError("inject_use_mpi: inlined 'MODULE mo_mpi' anchor not found in TU")
        ref_tu = out / f"{tu_path.stem}_ref.f90"
        ref_tu.write_text(src.replace("MODULE mo_mpi\n", "MODULE mo_mpi\n  use mpi\n", 1))

    # Bit-exact differential needs IEEE-strict FP on the DUT: DaCe's default
    # -ffast-math contracts a*b+c into an FMA (dot-product kernels round ~1 ulp
    # off gfortran) -- drop it and pin -ffp-contract=off; per-subprocess config change.
    cpu_args = dace.Config.get("compiler", "cpu", "args").replace("-ffast-math", "")
    if "-ffp-contract" not in cpu_args:
        cpu_args += " -ffp-contract=off"
    if os.environ.get("OCEAN_E2E_ASAN"):
        # DaCe emits aligned heap transients (``new T DACE_ALIGN(64)[N]`` ->
        # aligned ``operator new[]``) paired with a plain ``delete[]``; glibc's
        # aligned_alloc pointers free() cleanly so it is benign at runtime, but
        # ASAN's stricter new/delete tracking flags every such transient at
        # teardown. Run the ASAN diagnostic with
        # ``ASAN_OPTIONS=...:new_delete_type_mismatch=0`` (env, read by libasan
        # at process start -- can't be set in-file once LD_PRELOAD has loaded it)
        # so the run reaches the bit-exact assert instead of aborting on the nit.
        cpu_args += " -fsanitize=address -fno-omit-frame-pointer -g"
    dace.Config.set("compiler", "cpu", "args", value=cpu_args)

    sdfg = build_sdfg(tu_path.read_text(), entry=entry, name=tu_path.stem, out_dir=str(out / "sdfg"))
    clear_external_registry()  # DUT SDFG built -- drop the drop-list so it can't leak
    dace_name = sdfg.name  # bind(c) symbols + SDFG exports key off this, NOT name=
    lib = build_fortran_library(sdfg,
                                out_dir=str(out / "lib"),
                                prelude_sources=[*extra_prelude, ref_tu],
                                bind_c_shim=True)
    shim = Path(lib.bind_c_shim_f90).read_text()
    # Seed the reference's grid-dim module globals (nproma/n_zlev) the DUT derives
    # from array extents -- else isolated reference reads 0 and OOBs.  Recovered
    # from the binding's ``<sym> = int(size(...))`` lines.
    binding_files = list((out / "lib").glob("*bindings.f90"))
    binding_text = binding_files[0].read_text() if binding_files else ""
    module_dims = _size_derived_module_dims(binding_text) if binding_text else []
    # Config globals read straight from the module (nproma/nflatlev/...) -- BSS 0
    # in isolation, so seeded on both sides to test-supplied namelist values.
    seed_specs = _resolve_module_seeds(binding_text, module_seeds or {}) if binding_text else []
    array_specs = _resolve_module_array_seeds(binding_text, module_array_seeds or {}) if binding_text else []
    # t_ocean_solve mixes ALLOCATABLE and POINTER scratch members; the pre-alloc
    # guard must match each kind (allocated rejects a pointer) -- recovered from
    # the binding's own associated(obj % comp) guards.
    pointer_members = {(o.lower(), c.lower())
                       for o, c in re.findall(r"associated\(\s*(\w+)\s*%\s*(\w+)\s*\)", binding_text)}
    # A global-bind target needs an allocate only when the member is deferred-shape;
    # the binding guards exactly those with associated()/allocated(), so its own
    # guards classify array-vs-scalar members without a second spec field.
    deferred_members = pointer_members | {(o.lower(), c.lower())
                                          for o, c in re.findall(r"allocated\(\s*(\w+)\s*%\s*(\w+)\s*\)", binding_text)}

    # A solver-alloc extent may reference a seeded grid-dim global via its
    # <sym>__refmod rename; those not in module_dims come from module_seeds and
    # are imported (renamed) into both shims -- value comes from the ctypes module-seed, not a shim assign.
    extra_refmod_imports = []
    if ref_solver_allocs:
        sym_module = _module_symbol_map(binding_text)
        size_derived = {s for s, _m in module_dims}
        referenced = set()
        for _obj, _comp, dims in ref_solver_allocs:
            referenced |= set(re.findall(r"(\w+)__refmod", dims))
        for sym in sorted(referenced - size_derived):
            module = sym_module.get(sym)
            if module is None:
                raise KeyError(f"solver-alloc extent references '{sym}__refmod' but '{sym}' is not a "
                               f"module import in the binding")
            extra_refmod_imports.append((sym, module))

    # DUT: pre-allocate the stubbed solver-scratch host members (same list the
    # REF shim builds) and re-link -- live-member marshalling then sizes each SoA
    # companion from the real mesh shape instead of the degenerate (1,1) fallback
    # (heap smash).  Mirrors the REF compile below; SDFG-linked binding untouched.
    dut_so_path = lib.so_path
    if ref_solver_allocs or ref_global_binds:
        dut_shim = out / f"{dace_name}_c_dut.f90"
        dut_shim.write_text(
            _inject_dut_shim_prologue(shim,
                                      dace_name,
                                      entry,
                                      module_dims,
                                      n,
                                      solver_allocs=ref_solver_allocs,
                                      pointer_members=pointer_members,
                                      extra_refmod_imports=extra_refmod_imports,
                                      global_binds=ref_global_binds,
                                      deferred_members=deferred_members))
        dut_so_path = out / f"lib{dace_name}_dut.so"
        sdfg_so = Path(lib.sdfg_so)
        _asan = ["-fsanitize=address", "-fno-omit-frame-pointer"] if os.environ.get("OCEAN_E2E_ASAN") else []
        rl = subprocess.run([
            "gfortran", *_asan, "-shared", "-fPIC", "-ffree-line-length-none", "-O3", "-g", "-fno-fast-math",
            "-ffp-contract=off", "-frounding-math", "-fopenmp", f"-J{out}", *[str(p) for p in extra_prelude],
            str(ref_tu),
            str(lib.bindings_f90),
            str(dut_shim), "-o",
            str(dut_so_path), f"-L{sdfg_so.parent}", f"-Wl,-rpath,{sdfg_so.parent}", f"-l:{sdfg_so.name}"
        ],
                            capture_output=True,
                            text=True,
                            cwd=str(out))
        if rl.returncode != 0:
            raise RuntimeError(f"DUT shim-prologue relink failed:\n{rl.stderr[-3000:]}")

    ref_shim = out / f"{dace_name}_ref_c.f90"
    ref_shim.write_text(
        _retarget_shim(shim,
                       dace_name,
                       entry,
                       module_dims,
                       n,
                       solver_allocs=ref_solver_allocs,
                       pointer_members=pointer_members,
                       extra_refmod_imports=extra_refmod_imports,
                       global_binds=ref_global_binds,
                       deferred_members=deferred_members))
    ref_so = out / f"lib{dace_name}_ref.so"
    # No -fallow-argument-mismatch: it would silence a genuine shim/kernel ABI
    # mismatch instead of failing loudly.  Dual-typed MPI kernels are made sound
    # via TYPE(*) interfaces in the MPI stub, not by suppressing the diagnostic.
    # OCEAN_E2E_FINIT: poison uninitialised locals so a read shows up as a sNaN
    # (and a traceable trap) instead of whatever the heap held.  Diagnostic only
    # -- a reference that needs this to be reproducible is itself the bug.
    _finit = ["-finit-real=snan", "-finit-integer=-99999999", "-fbacktrace", "-g"
              ] if os.environ.get("OCEAN_E2E_FINIT") else []
    # OCEAN_E2E_ASAN: instrument the REF too so a real heap-buffer-overflow (not a
    # benign 8-vs-n_zlev array-conformance, which -fcheck=bounds would false-flag)
    # reports a precise line.  Run pytest with LD_PRELOAD=$(gcc -print-file-name=libasan.so).
    _asan_ref = ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"] if os.environ.get("OCEAN_E2E_ASAN") else []
    r = subprocess.run(
        [
            "gfortran",
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            # IEEE-strict FP to match the DUT: no FMA contraction, no fast-math reassociation.
            "-ffp-contract=off",
            "-fno-fast-math",
            *_finit,
            *_asan_ref,
            "-o",
            str(ref_so),
            # Prelude (mpi stub + no-op impls) compiles ahead of the TU so use mpi
            # resolves and single-rank halo calls are no-ops.
            *[str(p) for p in extra_prelude],
            str(ref_tu),
            str(ref_shim)
        ],
        capture_output=True,
        text=True,
        cwd=str(out))
    if r.returncode != 0:
        raise RuntimeError(f"reference .so compile failed:\n{r.stderr[-3000:]}")

    # Structured per-array input buffers can't ride argv as JSON, so
    # run_kernel_e2e drops them as a sidecar .npz; feed the same bytes to both forks.
    mesh_buffers = {}
    mesh_npz = out / "mesh_buffers.npz"
    if mesh_npz.exists():
        with np.load(mesh_npz) as zf:
            mesh_buffers = {k: zf[k] for k in zf.files}

    call_plan, inputs, ptr_args, _ptr_local = synth_call_inputs(shim,
                                                                n=n,
                                                                seed=seed,
                                                                float_range=float_range,
                                                                int_fill=int_fill,
                                                                scalar_overrides=scalar_overrides,
                                                                array_overrides=array_overrides,
                                                                mesh_buffers=mesh_buffers)
    sd = _run_in_fork(dut_so_path,
                      call_plan,
                      inputs,
                      ptr_args,
                      f"{dace_name}_c",
                      out / "dut",
                      sdfg_so=lib.sdfg_so,
                      module_seeds=seed_specs,
                      array_seeds=array_specs)
    sr = _run_in_fork(ref_so,
                      call_plan,
                      inputs,
                      ptr_args,
                      f"{dace_name}_ref_c",
                      out / "ref",
                      module_seeds=seed_specs,
                      array_seeds=array_specs)

    dut = {k: np.load(out / f"dut_{k}.npy") for k in ptr_args if (out / f"dut_{k}.npy").exists()}
    ref = {k: np.load(out / f"ref_{k}.npy") for k in ptr_args if (out / f"ref_{k}.npy").exists()}
    missing = [k for k in ptr_args if k not in dut or k not in ref]
    if missing:
        raise RuntimeError(f"run did not produce all outputs (dut status={sd}, ref status={sr}); "
                           f"missing {missing[:5]}")

    max_diff, n_changed = 0.0, 0
    for arg in ptr_args:
        d = dut[arg].astype(np.float64)
        rf = ref[arg].astype(np.float64)
        # Bit-exact comparison treats IDENTICAL non-finite results as equal (same
        # NaN/Inf bits on both sides is not a divergence); a genuine mismatch
        # (finite vs non-finite, opposite-signed Inf/NaN) sets +Inf so max_diff fails the == 0.0 gate.
        equal = (d == rf) | (np.isnan(d) & np.isnan(rf))
        if not equal.all():
            with np.errstate(invalid="ignore"):
                block = np.abs(d[~equal] - rf[~equal])
            block[~np.isfinite(block)] = np.inf
            diff = float(block.max())
            if not np.isfinite(diff) or diff > max_diff:
                max_diff = diff
        if not np.array_equal(dut[arg], inputs[arg]):
            n_changed += 1
    return max_diff, n_changed


def run_kernel_e2e(tu_path: Path,
                   entry: str,
                   *,
                   scalar_overrides=None,
                   array_overrides=None,
                   float_range=(-1.0, 1.0),
                   n: int = 8,
                   seed: int = 0,
                   int_fill=None,
                   module_seeds=None,
                   module_array_seeds=None,
                   do_not_emit=None,
                   prelude_paths=None,
                   inject_use_mpi=False,
                   ref_solver_allocs=None,
                   ref_global_binds=None,
                   mesh_buffers=None) -> dict:
    """Run one kernel's e2e build+compare in an isolated subprocess.  Returns
    ``{passed, max_diff, n_changed, output}``; ``passed`` is False (with captured
    output) on any build/lowering/compile/run failure rather than crashing pytest.

    ``module_seeds`` seeds ICON config globals the DUT binding reads straight
    from the module (``nproma``, ``nflatlev``/``nrdmax``) on both ``.so``s --
    isolated reads are BSS 0 (OOB) with no extent to derive them from.

    ``ref_global_binds`` (``[[module, object, component, source], ...]``) copies a
    module-global derived type's components from a reconstructed dummy struct in
    both shims, for kernels that read the global instead of the dummy they were
    handed (see :func:`_global_bind_lines`).
    """
    # _session_scratch is gitignored (absent on a fresh CI checkout) -- create it before carving a per-run tempdir.
    scratch_root = _HERE.parent.parent.parent / "_session_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="ocean_e2e_", dir=str(scratch_root)))
    # numpy arrays can't ride the child's argv as JSON -- sidecar .npz in the
    # per-run out dir instead; _build_and_compare loads it and applies to both forks.
    if mesh_buffers:
        np.savez(out / "mesh_buffers.npz", **mesh_buffers)
    env = dict(os.environ)
    # tests/ (for icon.ocean) + repo root (for dace_fortran) on the child path.
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE.parents[1]), str(_HERE.parents[2]), env.get("PYTHONPATH", "")])
    env["TMPDIR"] = str(out)
    env.setdefault("UCX_VFS_ENABLE", "n")
    proc = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve()),
        str(tu_path.resolve()), entry,
        json.dumps(scalar_overrides or {}),
        json.dumps(array_overrides or {}),
        json.dumps(list(float_range)),
        str(n),
        str(seed),
        str(out),
        json.dumps(int_fill),
        json.dumps(module_seeds or {}),
        json.dumps(list(do_not_emit or [])),
        json.dumps([str(p) for p in (prelude_paths or [])]),
        json.dumps(bool(inject_use_mpi)),
        json.dumps(module_array_seeds or {}),
        json.dumps(ref_solver_allocs or []),
        json.dumps(ref_global_binds or [])
    ],
                          capture_output=True,
                          text=True,
                          env=env)
    output = proc.stdout + "\n" + proc.stderr
    passed = any(ln.startswith("RESULT: PASS") for ln in proc.stdout.splitlines())
    max_diff = next((float(ln.split(":", 1)[1]) for ln in proc.stdout.splitlines() if ln.startswith("MAXDIFF:")), None)
    n_changed = next((int(ln.split(":", 1)[1]) for ln in proc.stdout.splitlines() if ln.startswith("CHANGED:")), 0)
    return {"passed": passed, "max_diff": max_diff, "n_changed": n_changed, "output": output}


def _main(argv):
    (tu_path, entry, overrides_json, array_overrides_json, float_range_json, n, seed, out, int_fill_json,
     module_seeds_json, do_not_emit_json, prelude_paths_json, inject_use_mpi_json, module_array_seeds_json,
     ref_solver_allocs_json, ref_global_binds_json) = argv[1:17]
    try:
        max_diff, n_changed = _build_and_compare(Path(tu_path),
                                                 entry,
                                                 json.loads(overrides_json),
                                                 json.loads(array_overrides_json),
                                                 tuple(json.loads(float_range_json)),
                                                 int(n),
                                                 int(seed),
                                                 Path(out),
                                                 json.loads(int_fill_json),
                                                 module_seeds=json.loads(module_seeds_json),
                                                 module_array_seeds=json.loads(module_array_seeds_json),
                                                 do_not_emit=json.loads(do_not_emit_json),
                                                 prelude_paths=json.loads(prelude_paths_json),
                                                 inject_use_mpi=json.loads(inject_use_mpi_json),
                                                 ref_solver_allocs=json.loads(ref_solver_allocs_json),
                                                 ref_global_binds=json.loads(ref_global_binds_json))
        print(f"MAXDIFF: {max_diff}", flush=True)
        print(f"CHANGED: {n_changed}", flush=True)
        print("RESULT: PASS", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
