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
    qualify -- a module-SOURCED global (``no_dual_edges = int(no_dual_edges__mod,
    c_int)``) reads the same module default on both sides, so it needs no seed."""
    sym_module = {}
    for module, renames in re.findall(r"use\s+(\w+),\s*only:\s*([^\n]+)", binding):
        for _alias, sym in re.findall(r"(\w+)__mod\s*=>\s*(\w+)", renames):
            sym_module[sym] = module
    seen, out = set(), []
    for sym in re.findall(r"^\s*(\w+)\s*=\s*int\(size\(\w+,\s*dim=\d+\),\s*c_int\)", binding, re.M):
        if sym in sym_module and sym not in seen:
            seen.add(sym)
            out.append((sym, sym_module[sym]))
    return out


def _retarget_shim(shim: str, dace_name: str, entry: str, module_dims=None, n_val=None) -> str:
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
    ICON's namelist init -- otherwise the isolated reference reads them as 0."""
    mod, proc = entry.split("::")
    proc = proc.lower()
    module_dims = module_dims or []
    bool_args = set(re.findall(r"logical\(c_bool\),\s*value\s*::\s*(\w+)", shim))
    out = []
    for ln in shim.splitlines():
        if ln.strip().startswith(f"use {dace_name}_dace_bindings"):
            out.append(f"  use {mod}, only: {proc}")
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
            args = [a.strip() for a in arglist.split(",")]
            args = [f"logical({a})" if a in bool_args else a for a in args]
            out.append(f"{indent}call {proc}({', '.join(args)})")
            continue
        ln = ln.replace(f"subroutine {dace_name}_c(", f"subroutine {dace_name}_ref_c(")
        ln = ln.replace(f"end subroutine {dace_name}_c", f"end subroutine {dace_name}_ref_c")
        ln = ln.replace(f"name='{dace_name}_c'", f"name='{dace_name}_ref_c'")
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


def _invoke(so_path, call_plan, bufs, sym, sdfg_so=None):
    # The SDFG .so is dlopen'd RTLD_GLOBAL first so the DaCe runtime (OpenMP
    # offload init, etc.) is live and its symbols resolve for the binding .so.
    if sdfg_so is not None:
        ctypes.CDLL(str(sdfg_so), mode=ctypes.RTLD_GLOBAL)
    cdll = ctypes.CDLL(str(so_path))
    fn = getattr(cdll, sym)
    argtypes, args = [], []
    for kind, arg, ct, *rest in call_plan:
        argtypes.append(ct)
        args.append(bufs[arg].ctypes.data_as(ct) if kind == "ptr" else ct(rest[0]))
    fn.argtypes = argtypes
    fn.restype = None
    fn(*args)


def _run_in_fork(so_path, call_plan, inputs, ptr_args, sym, save_prefix: Path, sdfg_so=None):
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
            _invoke(so_path, call_plan, bufs, sym, sdfg_so=sdfg_so)
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
                       int_fill=None):
    """The worker body (runs in a subprocess): build DUT + REF, drive both,
    return ``(max_diff, n_changed)``.  Imports are deferred so the module loads
    cheaply in the parent (which only orchestrates the subprocess)."""
    from dace_fortran.build import build_sdfg
    from dace_fortran.bindings import build_fortran_library

    sdfg = build_sdfg(tu_path.read_text(), entry=entry, name=tu_path.stem, out_dir=str(out / "sdfg"))
    dace_name = sdfg.name  # bind(c) symbols + SDFG exports key off this, NOT name=
    lib = build_fortran_library(sdfg, out_dir=str(out / "lib"), prelude_sources=[tu_path], bind_c_shim=True)
    shim = Path(lib.bind_c_shim_f90).read_text()
    # Seed the reference's grid-dimension module globals (nproma / n_zlev) the DUT
    # binding derives from array extents -- else the isolated reference reads them
    # as 0 and OOBs.  Recovered from the binding's ``<sym> = int(size(...))`` lines.
    binding_files = list((out / "lib").glob("*bindings.f90"))
    module_dims = _size_derived_module_dims(binding_files[0].read_text()) if binding_files else []

    ref_shim = out / f"{dace_name}_ref_c.f90"
    ref_shim.write_text(_retarget_shim(shim, dace_name, entry, module_dims, n))
    ref_so = out / f"lib{dace_name}_ref.so"
    r = subprocess.run([
        "gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-fallow-argument-mismatch", "-o",
        str(ref_so),
        str(tu_path),
        str(ref_shim)
    ],
                       capture_output=True,
                       text=True,
                       cwd=str(out))
    if r.returncode != 0:
        raise RuntimeError(f"reference .so compile failed:\n{r.stderr[-3000:]}")

    header, value_ftype, ptr_ftype, ptr_shape, dim_symbols, ptr_local = _parse_abi(shim)
    dimvals = {s: n for s in dim_symbols}

    rng = np.random.default_rng(seed)
    inputs, call_plan = {}, []
    for arg in header:
        if arg in ptr_shape:
            shape = tuple(int(eval(tok, {}, dimvals)) for tok in ptr_shape[arg].split(","))
            npdt = _NP[ptr_ftype[arg]]
            override = array_overrides.get(ptr_local[arg])
            if override is not None:  # pinned to a constant (e.g. a valid loop bound)
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
            int_default = int_fill if int_fill is not None else 0
            v = scalar_overrides.get(arg, n if arg in dim_symbols else
                                     (60.0 if ft == "real(c_double)" else int_default))
            call_plan.append(("val", arg, _CT[ft], v))

    ptr_args = list(ptr_shape)
    sd = _run_in_fork(lib.so_path, call_plan, inputs, ptr_args, f"{dace_name}_c", out / "dut", sdfg_so=lib.sdfg_so)
    sr = _run_in_fork(ref_so, call_plan, inputs, ptr_args, f"{dace_name}_ref_c", out / "ref")

    dut = {k: np.load(out / f"dut_{k}.npy") for k in ptr_args if (out / f"dut_{k}.npy").exists()}
    ref = {k: np.load(out / f"ref_{k}.npy") for k in ptr_args if (out / f"ref_{k}.npy").exists()}
    missing = [k for k in ptr_args if k not in dut or k not in ref]
    if missing:
        raise RuntimeError(f"run did not produce all outputs (dut status={sd}, ref status={sr}); "
                           f"missing {missing[:5]}")

    max_diff, n_changed = 0.0, 0
    for arg in ptr_args:
        d, rf = dut[arg], ref[arg]
        diff = float(np.abs(d.astype(np.float64) - rf.astype(np.float64)).max())
        # Python ``max(0.0, nan) == 0.0`` silently swallows a NaN/Inf DUT, so a
        # diverged or overflowed output would masquerade as a perfect match.
        # Carry a non-finite diff forward so it fails the tolerance check.
        if not np.isfinite(diff) or diff > max_diff:
            max_diff = diff
        if not np.array_equal(d, inputs[arg]):
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
                   int_fill=None) -> dict:
    """Run one kernel's e2e build+compare in an isolated subprocess.  Returns
    ``{passed, max_diff, n_changed, output}``.  ``passed`` is False (with the
    captured output) on any build / lowering / compile / run failure -- a
    kernel that does not yet lower surfaces here rather than crashing pytest."""
    # ``_session_scratch`` is gitignored, so it is absent on a fresh checkout
    # (CI) -- create it before carving a per-run tempdir out of it.
    scratch_root = _HERE.parent.parent.parent / "_session_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="ocean_e2e_", dir=str(scratch_root)))
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
        json.dumps(int_fill)
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
    tu_path, entry, overrides_json, array_overrides_json, float_range_json, n, seed, out, int_fill_json = argv[1:10]
    try:
        max_diff, n_changed = _build_and_compare(Path(tu_path), entry, json.loads(overrides_json),
                                                 json.loads(array_overrides_json), tuple(json.loads(float_range_json)),
                                                 int(n), int(seed), Path(out), json.loads(int_fill_json))
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
