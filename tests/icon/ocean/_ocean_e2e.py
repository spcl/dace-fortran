"""Shared harness for the ICON-O ocean kernel end-to-end numerical tests.

For one ocean single-TU kernel it builds two shared libraries that share the
SAME flat C ABI and drives both from one set of random ctypes buffers:

  DUT  -- the kernel lowered to a DaCe SDFG, compiled, and reached through the
          AUTO-GENERATED ``bind(c)`` shim (``build_fortran_library(
          bind_c_shim=True)``).
  REF  -- the ORIGINAL Fortran kernel, reached through the SAME shim
          *retargeted* to ``call <kernel>`` instead of ``call <entry>_dace``
          (:func:`_retarget_shim`).  Identical flat->struct reconstruction, so
          identical C ABI -- an apples-to-apples comparison with no
          hand-authored per-kernel caller.

Each kernel runs in its OWN subprocess (:func:`run_kernel_e2e`), mirroring the
``extract_single_tu`` pattern: the SDFG ``.so`` is dlopen'd ``RTLD_GLOBAL`` so
the DaCe runtime initialises, which would otherwise leak symbols into a later
in-process flang/MLIR build of the next kernel and corrupt it.  Process
isolation makes both the load mode and the MLIR build per-kernel-clean.
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

# ICON refinement-control index arrays -- ``verts/cells/edges%{start,end}_{block,index}``
# -- are ``ALLOCATABLE(:)`` but allocated ``(min_rl : max_rl)`` (a NEGATIVE lower bound)
# and read at refinement-control levels that run negative (``end_block(-10)``).  The
# bind_c shim now carries each array's per-dim lower bound (``<arr>_lb<i>``), so the
# harness gives these arrays that negative lower bound and sizes them to span the whole
# refin-ctrl level range instead of the mesh size ``n``.  ``[-16, 16]`` comfortably
# covers ICON's levels (velocity's widest is the edges' ``[-10, 10]``); under
# ``int_fill`` every slot is the single valid block / index, so the read is in-bounds
# at any level and the binding's ``offset = lbound`` matches the SDFG's indexing.
_REFIN_CTRL_RE = re.compile(r"_(?:start|end)_(?:block|index)$")
_REFIN_CTRL_LBOUND = -16
_REFIN_CTRL_EXTENT = 33


def _refin_ctrl_flat(sym: str, suffix_re: str) -> str:
    """The array flat name behind a per-dim extent (``_d<i>``) or lower-bound
    (``_lb<i>``) C-ABI arg, for the refin-ctrl test.  ``re.sub`` of the given
    suffix; returns the arg unchanged when it carries no such suffix."""
    return re.sub(suffix_re, "", sym)


def _module_symbol_map(binding: str) -> dict:
    """``sym -> module`` for every ICON namelist/config global the DUT binding
    imports as ``use <module>, only: <sym>__mod => <sym>``.  Shared by the
    size-derived grid-dim seeding (:func:`_size_derived_module_dims`) and the
    config-global seeding (:func:`_resolve_module_seeds`)."""
    sym_module = {}
    for module, renames in re.findall(r"use\s+(\w+),\s*only:\s*([^\n]+)", binding):
        for _alias, sym in re.findall(r"(\w+)__mod\s*=>\s*(\w+)", renames):
            sym_module[sym] = module
    return sym_module


def _size_derived_module_dims(binding: str):
    """The ICON grid-DIMENSION module globals the DUT binding derives from an
    array extent -- ``n_zlev = int(size(psi_c, dim=2), c_int)`` for the
    ``mo_ocean_nml::n_zlev`` global, ``nproma`` from ``mo_parallel_config``, etc.

    ICON sets these from its namelist at model init; an extracted kernel called
    in isolation reads them UNINITIALISED (BSS = 0), which sizes the kernel's
    automatic local arrays (``this_vort_flux(n_zlev, 2)``) to zero -> OOB.  The
    SDFG path is immune (the bridge derives them as free symbols from the array
    extents -- the ``size(...)`` lines this parses), so a faithful reference
    caller must seed them the same way.  Every array extent in this harness is
    the mesh size ``n``, so the reference sets each to ``n``.

    Returns ``[(sym, module), ...]``.  Only module-origin syms (those the binding
    imports as ``<sym>__mod => <sym>``) that ALSO carry a ``size()`` derivation
    qualify -- a module-SOURCED global the DUT reads *directly* (``nproma =
    int(nproma__mod, c_int)``) is instead seeded on BOTH sides via
    :func:`_resolve_module_seeds`."""
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

    Unlike the size-derived grid dims, some ICON config globals are read by the
    DUT binding straight from the module (``nproma = int(nproma__mod, c_int)``;
    ``nflatlev = nflatlev__mod``), so an isolated run reads them as BSS 0 on BOTH
    sides -- ``nproma = 0`` sizes automatic locals to zero, ``nflatlev(jg) = 0``
    makes ``DO jk = nflatlev, nlev`` start at 0, both out of bounds.  There is no
    array extent to derive them from, so the test supplies the namelist values
    (the policy) and the harness seeds both ``.so``s identically (the mechanism),
    keeping the differential bit-exact.  ``length`` is the array extent from the
    binding's ``allocate(<sym>(N))`` (a per-domain global like ``nflatlev(10)``),
    else 1 for a scalar; every element is set to ``value``."""
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


# Seeded allocatable-array buffers are kept alive for the child's lifetime -- the
# descriptor's ``base_addr`` points into them, and the child ``os._exit``s before
# any Fortran teardown, so they are never freed.
_ALLOC_SEED_KEEPALIVE: list = []


def _resolve_module_array_seeds(binding: str, seeds: dict):
    """Resolve ``{fortran_sym: length}`` allocatable-module-ARRAY seeds to
    ``[(mangled_symbol, length), ...]``.

    Some ICON module globals are ``REAL(8), ALLOCATABLE`` (``mo_vertical_coord_
    table::vct_a`` -- the vertical coordinate table) that the kernel indexes
    directly (``vct_a(jk)``).  An isolated run leaves them UNALLOCATED (a null
    descriptor), so the real reference OOB-reads / SEGVs and the SDFG reads the
    binding's size-1 defensive fallback (garbage).  Neither crosses the bind(c)
    ABI (module-direct reads), so -- like the scalar :func:`_resolve_module_seeds`
    -- the test supplies the length and the harness allocates + fills the SAME
    values on BOTH ``.so``s, keeping the differential bit-exact."""
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


def _retarget_shim(shim: str,
                   dace_name: str,
                   entry: str,
                   module_dims=None,
                   n_val=None,
                   solver_allocs=None,
                   pointer_members=None) -> str:
    """Rewrite the auto ``<dace_name>_c`` shim into a ``<dace_name>_ref_c`` that
    calls the ORIGINAL kernel ``entry`` (``module::proc``) instead of the
    ``<dace_name>_dace`` binding.  The flat->struct reconstruction (header,
    ``c_f_pointer`` aliases, struct alloc + copy-in) is reused verbatim; only
    the ``use`` line, the final ``call`` and the subroutine name change.

    ``logical(c_bool)`` value args are coerced to the kernel's default
    ``LOGICAL`` (kind 4) at the call site -- the C ABI keeps logicals as
    ``c_bool``, and the conversion to the callee's logical kind happens here, the
    same way the binding wrapper handles it for the SDFG path.

    Grid DIMENSION module globals (``module_dims`` = ``[(sym, module), ...]`` from
    :func:`_size_derived_module_dims`) the kernel reads are seeded to ``n_val``
    (the mesh size -- every extent in this harness) before the call, mirroring
    ICON's namelist init -- otherwise the isolated reference reads them as 0.

    ``solver_allocs`` (``[(object, component, dims), ...]``) builds the allocatable
    work components of module-level DERIVED-TYPE solver objects the stock routine
    reads.  A single-TU extraction stubs the solver's allocator
    (``ocean_solve_construct``) to an empty body, so the real Fortran variable
    (``free_sfc_solver%x_loc_wp``) stays unallocated and the stock write SEGVs; the
    DUT SDFG supplies this scratch from its marshalling layer, so only the REF
    needs it.  ``object`` lives in the entry's own module ``mod``; ``dims`` is a
    Fortran extent expression that may reference the reconstructed dummy structs
    (``patch_3d % p_patch_2d(1) % alloc_cell_blocks``) and the seeded grid-dim
    globals (``nproma__refmod``), both in scope + populated before the call."""
    mod, proc = entry.split("::")
    proc = proc.lower()
    module_dims = module_dims or []
    solver_allocs = solver_allocs or []
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
            continue
        if f"call {dace_name}_dace_finalize()" in ln:
            continue
        m = re.match(rf"(\s*)call {dace_name}_dace\((.*)\)\s*$", ln)
        if m:
            indent, arglist = m.group(1), m.group(2)
            for sym, _module in module_dims:
                out.append(f"{indent}{sym}__refmod = {n_val}")
            # Build the stubbed solver objects' allocatable work arrays (the
            # module-global reconstruction the empty ``ocean_solve_construct``
            # skips); zero-init so a read-before-write of an unwritten slot
            # (``res_loc_wp(1)`` when the solve stub leaves it untouched) is
            # deterministic across the DUT and the reference.
            for obj, comp, dims in solver_allocs:
                guard = "associated" if (obj.lower(), comp.lower()) in (pointer_members or set()) else "allocated"
                out.append(f"{indent}if (.not. {guard}({obj} % {comp})) allocate({obj} % {comp}({dims}))")
                out.append(f"{indent}{obj} % {comp} = 0.0d0")
            args = [a.strip() for a in arglist.split(",")]
            args = [f"logical({a})" if a in bool_args else a for a in args]
            out.append(f"{indent}call {proc}({', '.join(args)})")
            continue
        ln = ln.replace(f"subroutine {dace_name}_c(", f"subroutine {dace_name}_ref_c(")
        ln = ln.replace(f"end subroutine {dace_name}_c", f"end subroutine {dace_name}_ref_c")
        ln = ln.replace(f"name='{dace_name}_c'", f"name='{dace_name}_ref_c'")
        out.append(ln)
    return "\n".join(out) + "\n"


def _inject_dut_solver_allocs(shim: str,
                              dace_name: str,
                              entry: str,
                              module_dims=None,
                              n_val=None,
                              solver_allocs=None,
                              pointer_members=None) -> str:
    """Pre-allocate the stubbed module-global solver-scratch host members in the
    DUT ``<dace_name>_c`` shim -- symmetric to :func:`_retarget_shim`'s REF-side
    injection, but KEEPING the ``<dace_name>_dace`` (SDFG) call.

    The single-TU extraction stubs ``ocean_solve_construct`` to an empty body, so
    ``free_sfc_solver%x_loc_wp`` (+ twins) stay unallocated on entry.  The binding's
    live-member marshalling sizes each SoA companion from ``size(host_member)``, so
    the host member must be allocated at its real (mesh) shape here first; otherwise
    the marshalling takes the degenerate ``(1,1)`` fallback and the kernel's
    mesh-bounded writes overrun it and smash the heap.  Same ``(object, component,
    dims)`` list the REF shim uses -- ``dims`` may reference the reconstructed dummy
    structs (``patch_3d % p_patch_2d(1) % alloc_cell_blocks``) and the seeded
    grid-dim globals (``nproma__refmod``), both in scope + populated before the
    call."""
    mod, _proc = entry.split("::")
    module_dims = module_dims or []
    solver_allocs = solver_allocs or []
    if not solver_allocs:
        return shim
    solver_objs = []
    for obj, _comp, _dims in solver_allocs:
        if obj not in solver_objs:
            solver_objs.append(obj)
    out = []
    for ln in shim.splitlines():
        if ln.strip().startswith(f"use {dace_name}_dace_bindings"):
            out.append(ln)
            out.append(f"  use {mod}, only: {', '.join(solver_objs)}")
            for sym, module in module_dims:
                out.append(f"  use {module}, only: {sym}__refmod => {sym}")
            continue
        m = re.match(rf"(\s*)call {dace_name}_dace\((.*)\)\s*$", ln)
        if m:
            indent = m.group(1)
            for sym, _module in module_dims:
                out.append(f"{indent}{sym}__refmod = {n_val}")
            for obj, comp, dims in solver_allocs:
                # ``x_loc_wp`` / ``res_loc_wp`` are ALLOCATABLE, ``b_loc_wp`` is a
                # POINTER (mixed kinds in ``t_ocean_solve``) -- pick the guard the
                # member's kind accepts (``allocated`` rejects a pointer and vice
                # versa), taken from the binding's own associated()/allocated() use.
                guard = "associated" if (obj.lower(), comp.lower()) in (pointer_members or set()) else "allocated"
                out.append(f"{indent}if (.not. {guard}({obj} % {comp})) allocate({obj} % {comp}({dims}))")
                out.append(f"{indent}{obj} % {comp} = 0.0d0")
            out.append(ln)
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _parse_abi(shim: str):
    """Recover the shim's flat C ABI: ``(header_args, value_ftype, ptr_ftype,
    ptr_shape_expr, dim_symbols, ptr_local)``.  ``dim_symbols`` are the
    identifiers any ``c_f_pointer`` shape references -- the array extents the
    caller supplies.  ``ptr_local`` maps each pointer header arg (``<x>_p``)
    to its readable ``c_f_pointer`` local name (``<x>``), the key
    :func:`run_kernel_e2e`'s ``array_overrides`` uses to pin a buffer's
    contents."""
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


def _invoke(so_path, call_plan, bufs, sym, sdfg_so=None, module_seeds=None, array_seeds=None):
    # The SDFG .so is dlopen'd RTLD_GLOBAL first so the DaCe runtime (OpenMP
    # offload init, etc.) is live and its symbols resolve for the binding .so.
    if sdfg_so is not None:
        ctypes.CDLL(str(sdfg_so), mode=ctypes.RTLD_GLOBAL)
    cdll = ctypes.CDLL(str(so_path))
    # ICON namelist/config globals the isolated kernel would otherwise read as
    # BSS 0 (nproma block size, per-domain nflatlev/nrdmax) -- seeded here, after
    # the .so loads and before the call, identically on the DUT and reference .so
    # (see _resolve_module_seeds) so the differential stays bit-exact.
    for mangled, length, value in (module_seeds or []):
        cell = (ctypes.c_int * length).in_dll(cdll, mangled)
        for i in range(length):
            cell[i] = value
    # Allocatable module ARRAYS the kernel reads module-direct (``vct_a``) --
    # unallocated in isolation; allocate + fill identically on both .so's.
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
    """Load ``so_path`` and run ``sym`` in a forked child on a fresh copy of
    ``inputs``, saving each output buffer to ``<save_prefix>_<arg>.npy``, then
    ``os._exit`` so the child's (occasionally heap-corrupting) ctypes/DaCe-
    runtime teardown never runs.  Loading the DUT (DaCe runtime) and the REF
    (plain gfortran) ``.so``s in SEPARATE processes also avoids the
    nondeterministic cross-allocator double-free seen when both share one
    process.  Returns the child's wait status (0 == clean)."""
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
                       ref_solver_allocs=None):
    """The worker body (runs in a subprocess): build DUT + REF, drive both,
    return ``(max_diff, n_changed)``.  Imports are deferred so the module loads
    cheaply in the parent (which only orchestrates the subprocess).

    ``do_not_emit`` (list of routine names) drops halo / MPI / sync / timer /
    logging externals from the DUT SDFG (``apply_external_functions``) so a
    single-rank dycore like ``solve_nh`` carries no MPI.  ``prelude_paths``
    (extra ``.f90`` files) are compiled ahead of the TU on BOTH the DUT binding
    and the REF -- the ``mpi`` module stub + its no-op point-to-point impls that
    let the reference link and run single-rank without real MPI.
    ``inject_use_mpi`` gives the inlined ``mo_mpi`` a ``use mpi`` so its
    dual-typed real*8/real*4 calls resolve through the stub's one assumed-type
    interface (no ``-fallow-argument-mismatch``)."""
    import dace

    from dace_fortran.build import build_sdfg
    from dace_fortran.bindings import build_fortran_library
    from dace_fortran.external import apply_external_functions, clear_external_registry

    # Drop the halo / MPI / sync / timer externals from the DUT SDFG so no MPI
    # survives in a single-rank run.  The REF keeps the (real, single-TU) bodies
    # but their ``mpi_*`` leaves resolve to the no-op prelude impls below.
    clear_external_registry()
    if do_not_emit:
        apply_external_functions(do_not_emit=list(do_not_emit))

    # The reference build (and, for ``use mpi``, the DUT binding) compile these
    # ahead of the TU: the ``mpi`` stub module + no-op point-to-point impls.
    extra_prelude = [Path(p) for p in (prelude_paths or [])]

    # Reference-TU fixup so the extracted single-TU is gfortran-EXECUTABLE (the
    # SDFG build always reads the RAW ``tu_path`` -- the bridge resolves things its
    # own way).  ``mo_mpi``'s inlined wrappers call one ``mpi_recv`` / ``mpi_isend``
    # with both real*8 and real*4 buffers; a ``use mpi`` binds them to the stub's
    # single ``type(*)`` assumed-type interface (no -fallow-argument-mismatch).
    # The bridge's devirtualised call names (``<base>_deconiface_<N>`` /
    # ``<base>_deconproc_<N>``) are NOT undefined -- each rides a ``USE <mod>,
    # ONLY: <name> => <base>`` rename of an in-TU body -- so the TU compiles as-is;
    # only the raw ``mpi_*`` leaves are genuinely undefined (no-op prelude impls).
    ref_tu = tu_path
    if inject_use_mpi:
        src = tu_path.read_text()
        if "MODULE mo_mpi\n" not in src:
            raise RuntimeError("inject_use_mpi: inlined 'MODULE mo_mpi' anchor not found in TU")
        ref_tu = out / f"{tu_path.stem}_ref.f90"
        ref_tu.write_text(src.replace("MODULE mo_mpi\n", "MODULE mo_mpi\n  use mpi\n", 1))

    # A bit-exact differential needs IEEE-strict FP on the DUT.  DaCe's default CPU
    # args carry ``-ffast-math``, which contracts ``a*b+c`` into an FMA (and lets
    # the compiler reassociate) -- so a dot-product-heavy kernel (ICON's rbf /
    # cells2verts interpolation) rounds ~1 ulp away from the plain-gfortran
    # reference.  Drop ``-ffast-math`` and pin ``-ffp-contract=off`` so the SDFG
    # rounds bit-for-bit like the reference; this is a per-subprocess config change.
    cpu_args = dace.Config.get("compiler", "cpu", "args").replace("-ffast-math", "")
    if "-ffp-contract" not in cpu_args:
        cpu_args += " -ffp-contract=off"
    dace.Config.set("compiler", "cpu", "args", value=cpu_args)

    sdfg = build_sdfg(tu_path.read_text(), entry=entry, name=tu_path.stem, out_dir=str(out / "sdfg"))
    clear_external_registry()  # DUT SDFG built -- drop the drop-list so it can't leak
    dace_name = sdfg.name  # bind(c) symbols + SDFG exports key off this, NOT name=
    lib = build_fortran_library(sdfg,
                                out_dir=str(out / "lib"),
                                prelude_sources=[*extra_prelude, ref_tu],
                                bind_c_shim=True)
    shim = Path(lib.bind_c_shim_f90).read_text()
    # Seed the reference's grid-dimension module globals (nproma / n_zlev) the DUT
    # binding derives from array extents -- else the isolated reference reads them
    # as 0 and OOBs.  Recovered from the binding's ``<sym> = int(size(...))`` lines.
    binding_files = list((out / "lib").glob("*bindings.f90"))
    binding_text = binding_files[0].read_text() if binding_files else ""
    module_dims = _size_derived_module_dims(binding_text) if binding_text else []
    # Config globals the DUT binding reads straight from the module (nproma /
    # nflatlev / ...) -- BSS 0 in isolation, so seeded on BOTH sides to the
    # test-supplied namelist values.
    seed_specs = _resolve_module_seeds(binding_text, module_seeds or {}) if binding_text else []
    array_specs = _resolve_module_array_seeds(binding_text, module_array_seeds or {}) if binding_text else []
    # ``t_ocean_solve`` mixes ALLOCATABLE (``x_loc_wp`` / ``res_loc_wp``) and
    # POINTER (``b_loc_wp``) scratch members; the shim's pre-alloc guard must match
    # each member's kind (``allocated`` rejects a pointer and vice versa).  Recover
    # the pointer members from the binding's own ``associated(obj % comp)`` guards.
    pointer_members = {(o.lower(), c.lower())
                       for o, c in re.findall(r"associated\(\s*(\w+)\s*%\s*(\w+)\s*\)", binding_text)}

    # DUT: pre-allocate the stubbed solver-scratch host members in the shim (the
    # same list the REF shim builds) and re-link the DUT .so.  The binding's
    # live-member marshalling then sizes each SoA companion from the real (mesh)
    # member shape instead of the degenerate (1,1) fallback that the kernel's
    # mesh-bounded writes would overrun (heap smash).  Mirrors the REF compile
    # below; keeps the SDFG-linked ``<dace_name>_dace`` binding untouched.
    dut_so_path = lib.so_path
    if ref_solver_allocs:
        dut_shim = out / f"{dace_name}_c_dut.f90"
        dut_shim.write_text(
            _inject_dut_solver_allocs(shim,
                                      dace_name,
                                      entry,
                                      module_dims,
                                      n,
                                      solver_allocs=ref_solver_allocs,
                                      pointer_members=pointer_members))
        dut_so_path = out / f"lib{dace_name}_dut.so"
        sdfg_so = Path(lib.sdfg_so)
        rl = subprocess.run([
            "gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-O3", "-g", "-fno-fast-math",
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
            raise RuntimeError(f"DUT solver-alloc relink failed:\n{rl.stderr[-3000:]}")

    ref_shim = out / f"{dace_name}_ref_c.f90"
    ref_shim.write_text(
        _retarget_shim(shim,
                       dace_name,
                       entry,
                       module_dims,
                       n,
                       solver_allocs=ref_solver_allocs,
                       pointer_members=pointer_members))
    ref_so = out / f"lib{dace_name}_ref.so"
    # No ``-fallow-argument-mismatch``: it silences REAL argument-type errors, so a
    # genuine ABI mismatch between the shim and the kernel would compile to a wrong
    # call instead of failing loudly.  The pure-compute ocean kernels have no such
    # mismatch; a kernel with dual-typed MPI buffers (``solve_nh``'s real*8/real*4
    # ``mpi_recv``) is made sound with ``TYPE(*)`` assumed-type interfaces in the MPI
    # stub, not by suppressing the diagnostic.
    r = subprocess.run(
        [
            "gfortran",
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            # IEEE-strict FP to match the DUT (``-ffp-contract=off`` above): no FMA
            # contraction, no fast-math reassociation, so both sides round identically.
            "-ffp-contract=off",
            "-fno-fast-math",
            "-o",
            str(ref_so),
            # Prelude (mpi stub module + no-op point-to-point impls) compiles ahead of
            # the TU so its ``use mpi`` resolves and single-rank halo calls are no-ops.
            *[str(p) for p in extra_prelude],
            str(ref_tu),
            str(ref_shim)
        ],
        capture_output=True,
        text=True,
        cwd=str(out))
    if r.returncode != 0:
        raise RuntimeError(f"reference .so compile failed:\n{r.stderr[-3000:]}")

    header, value_ftype, ptr_ftype, ptr_shape, dim_symbols, ptr_local = _parse_abi(shim)
    # Refin-ctrl arrays span the full refin-ctrl level range, not the mesh size ``n`` --
    # size their per-dim extent symbols (``<arr>_d<i>``) to ``_REFIN_CTRL_EXTENT`` so the
    # negative lower bound (supplied below) leaves every read in bounds.
    dimvals = {
        s: (_REFIN_CTRL_EXTENT if _REFIN_CTRL_RE.search(_refin_ctrl_flat(s, r"_d\d+$")) else n)
        for s in dim_symbols
    }

    # Structured per-array input buffers (a physically-consistent mesh's
    # connectivity / subset-range arrays) can't ride argv as JSON, so
    # ``run_kernel_e2e`` drops them into ``out`` as a sidecar .npz; load once and
    # feed the same bytes to both the DUT and REF forks so the differential stays
    # bit-exact.
    mesh_buffers = {}
    mesh_npz = out / "mesh_buffers.npz"
    if mesh_npz.exists():
        with np.load(mesh_npz) as zf:
            mesh_buffers = {k: zf[k] for k in zf.files}

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
                # A single in-domain block, one edge/vertex per connectivity slot, and every
                # connectivity index pointing at element ``int_fill`` -> exactly one in-bounds
                # iteration everywhere (no vacuous empty range, no out-of-bounds composite index).
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
                # Extent args ride ``dimvals`` (refin-ctrl-aware) so the passed extent
                # matches the buffer the ``c_f_pointer`` shape reconstructs.
                int_default = int_fill if int_fill is not None else 0
                v = scalar_overrides.get(
                    arg, dimvals[arg] if arg in dim_symbols else (60.0 if ft == "real(c_double)" else int_default))
            call_plan.append(("val", arg, _CT[ft], v))

    ptr_args = list(ptr_shape)
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
        # Bit-exact comparison treats IDENTICAL non-finite results as equal: two NaNs
        # (or the same signed Inf) at a position are the same bits, so a degenerate
        # input that deterministically drives BOTH sides to the same Inf/NaN is not a
        # divergence.  Only a genuine mismatch -- finite vs non-finite, +Inf vs -Inf,
        # or a lone NaN -- diverges: those positions carry +Inf so max_diff fails the
        # ``== 0.0`` gate instead of a real DUT overflow masquerading as a match.
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
                   mesh_buffers=None) -> dict:
    """Run one kernel's e2e build+compare in an isolated subprocess.  Returns
    ``{passed, max_diff, n_changed, output}``.  ``passed`` is False (with the
    captured output) on any build / lowering / compile / run failure -- a
    kernel that does not yet lower surfaces here rather than crashing pytest.

    ``module_seeds`` (``{fortran_sym: int_value}``) seeds ICON config globals the
    DUT binding reads straight from the module (``nproma``, per-domain
    ``nflatlev`` / ``nrdmax``) on BOTH the DUT and reference ``.so`` -- an
    isolated kernel reads them as BSS 0 (OOB), and there is no array extent to
    derive them from, so the test supplies the namelist values."""
    # ``_session_scratch`` is gitignored, so it is absent on a fresh checkout
    # (CI) -- create it before carving a per-run tempdir out of it.
    scratch_root = _HERE.parent.parent.parent / "_session_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="ocean_e2e_", dir=str(scratch_root)))
    # numpy arrays can't be JSON-encoded onto the child's argv, so hand any structured
    # input buffers to the worker as a sidecar .npz in the per-run ``out`` dir (already
    # threaded to the child); ``_build_and_compare`` loads it and applies it to both forks.
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
        json.dumps(ref_solver_allocs or [])
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
     ref_solver_allocs_json) = argv[1:16]
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
                                                 ref_solver_allocs=json.loads(ref_solver_allocs_json))
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
