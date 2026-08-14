#!/usr/bin/env python3
"""Run the STOCK ``dace_fortran.pipelines.optimize`` on the shipped ORIGINAL
SDFG FILE (``addusxx_g_original.sdfg``) of this sample, emit bindings +
.so, and re-apply the manual artifact fixes of
dace-fortran-manual-fixes-6c99810.patch.

Contrast with each sample's ``build_opt_lib.py``:

  * The SDFG fed to the pipeline is loaded FROM THE FILE -- the raw builder
    output, INCLUDING its known bugs (writer-less scal_* gate scalars, bug 3
    of dace-fortran-fixes-needed-6c99810.md).  No len1-staging monkeypatch,
    no SDFG-level copy-in repair, no stage-by-stage instrumentation: strictly
    ``optimize(sdfg)``.
  * Bug 3 is therefore fixed AFTER codegen, exactly like the reference patch:
    ``bool scal_okvan;`` -> ``bool scal_okvan; scal_okvan = okvan[0];``.
  * ``build_fortran_library`` needs the flatten plan / Fortran interface that
    only ``SDFGBuilder`` computes and that do NOT survive an .sdfg round-trip
    (the frozen signature DOES -- it lives in ``sdfg.frontend_metadata``).
    They are recovered from ``outputs/<k>.hlfir`` via the builder's METADATA
    ONLY -- no SDFG is built from it; the compiled SDFG is the file-loaded one.

Produces
    outputs/<k>_original_opt.sdfg
    outputs/lib_orig_opt/<k>_bindings.f90, lib<k>.so (guarded),
                         libdace_kernel_<k>.so (+dacestub copies)
NOTE: this sample's .dacecache/addusxx_g is WIPED for fresh codegen each run; it
is shared with the run_regex_default/build_opt_lib flows (all regenerable;
the *.handpatched.bak backups are never touched).

    cd /workspace/dace-fortran/samples/addusxx_g && PYTHONHASHSEED=0 python3 run_optimize_original.py
"""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")

import dace  # noqa: E402
from dace.sdfg import nodes as dace_nodes  # noqa: E402

from dace_fortran import keep_external  # noqa: E402
import dace_fortran.bindings.block_builders as bb  # noqa: E402
from dace_fortran.bindings import build_fortran_library  # noqa: E402
from dace_fortran.bindings.frozen_signature import get_frozen_signature, refreeze  # noqa: E402
from dace_fortran.pipelines import optimize  # noqa: E402

GFC_FLAGS = ["-shared", "-fPIC", "-ffree-line-length-none", "-O3", "-g", "-fno-fast-math",
             "-ffp-contract=off", "-frounding-math", "-fopenmp",
             "-fallow-argument-mismatch", "-std=legacy"]

STUBS = [
    "mp_sum", "mp_bcast", "mp_barrier", "mp_gather", "mp_size", "mp_rank",
    "mp_circular_shift_left", "result_sum", "reduce_base_real", "poolcollect",
    "zgemm", "zgemv", "zaxpy", "zcopy", "zdotc", "zdscal",
    "get_buffer", "save_buffer", "open_buffer", "close_buffer",
    "start_clock", "stop_clock", "errore", "infomsg",
]

_GUARD_ANCHOR = "    ! ----- Symbol population (input-derived; before allocates) -----"

_GUARD_ADDUSXX = """    ! ----- STRICT SCOPE GUARD (injected by run_optimize_original.py) -----
    ! hlfir-strip-character-runtime folds every flag=='c'/'r'/'i' compare to
    ! TRUE, so the SDFG hard-bakes the FIRST arm of the IF/ELSEIF chain: the
    ! flag='c' complex (k-point) path.  The gamma-trick 'r'/'i' paths and the
    ! real bec arrays are silently ignored -- refuse them.
    if (gamma_only__mod) then
      write (*, '(a)') 'FATAL(addusxx_g_dace): gamma_only = .true. -- only the &
        &k-point (flag=''c'') path is baked into this SDFG build; aborting.'
      error stop 8
    end if
    if (.not. (present(becphi_c) .and. present(becpsi_c))) then
      write (*, '(a)') 'FATAL(addusxx_g_dace): becphi_c/becpsi_c missing -- the &
        &flag=''c'' path needs both complex bec arrays; aborting.'
      error stop 8
    end if
    ! (guard on real bec arrays removed: the flattened input checks demand
    !  present _r dummies; the gamma compute arms are not in this SDFG)
"""

KERNELS = {
    "addusxx_g": {"entry": "us_exx::addusxx_g", "guard": _GUARD_ADDUSXX},
}

_PRESENT_RE = re.compile(r"(?m)^(\s*)(\w+_present) = 0  ! optional absent \(not forwarded by wrapper\)$")
_SCAL_DECL_RE = re.compile(r"(?m)^(\s*)(bool|int|long long|int64_t|float|double)\s+(scal_\w+);\s*$")

_ORIG_INIT_SYM_NAMES = bb._init_sym_names


def patch_bindings(bindings: Path, guard: str) -> None:
    """Fixes 1+2 of the reference patch: *_present -> 1, guard without _r abort."""
    txt = bindings.read_text()
    txt, n = _PRESENT_RE.subn(
        r"\1\2 = 1  ! FIXED: data IS forwarded (c_loc below); guard/driver ensure presence", txt)
    if n:
        print(f"  bindings fix 1: {n} *_present symbols forced to 1", flush=True)
    elif "_present" not in txt:
        print("  bindings fix 1: *_present plumbing eliminated by the pipeline -- nothing to patch", flush=True)
    elif "_present = 1  ! FIXED" not in txt:
        raise RuntimeError("bindings carry *_present symbols but fix-1 pattern matched nothing")
    if _GUARD_ANCHOR not in txt:
        raise RuntimeError("guard anchor not found in generated bindings")
    if "STRICT SCOPE GUARD" not in txt:
        txt = txt.replace(_GUARD_ANCHOR, guard + _GUARD_ANCHOR, 1)
    bindings.write_text(txt)
    print("  bindings fix 2: strict scope guard injected", flush=True)


def _enclosing_function(src: str, pos: int):
    """(name, params_span) of the function definition enclosing byte offset ``pos``."""
    best = None
    for fm in re.finditer(r"(?m)^(?:inline\s+)?(?:DACE_EXPORTED\s+)?void\s+(\w+)\(([^)]*)\)", src):
        if fm.start() > pos:
            break
        best = fm
    if best is None:
        raise RuntimeError("no enclosing function found for scal_* declaration")
    return best.group(1), (best.start(2), best.end(2))


def _thread_param(src: str, func: str, source: str) -> str:
    """Append ``source`` to ``func``'s parameter list and to every call of it.

    Needed when a writer-less scal_* lives in a nested-SDFG loop_body function
    that never received the source array; the callers all have it in scope
    (it is an ABI signature array)."""
    tm = re.search(rf"(\w+(?:::\w+)?)\s*\*\s*__restrict__\s+{source}\b", src)
    if tm is None:
        raise RuntimeError(f"cannot infer pointer type of {source} for parameter threading")
    ptype = tm.group(1)
    defn = re.search(rf"(?m)^((?:inline\s+)?void\s+{func}\()([^)]*)\)", src)
    src = src[:defn.end(2)] + f", {ptype}* __restrict__ {source}" + src[defn.end(2):]
    n_calls, pos = 0, 0
    while True:
        i = src.find(f"{func}(", pos)
        if i < 0:
            break
        if re.search(r"void\s+$", src[max(0, i - 16):i]):
            pos = i + 1          # the definition itself
            continue
        close = src.index(");", i)
        src = src[:close] + f", {source}" + src[close:]
        pos = close + len(source) + 4
        n_calls += 1
    if n_calls == 0:
        raise RuntimeError(f"threaded {source} into {func} but found no call sites")
    print(f"  cpp fix 3: threaded {source} into {func} (+{n_calls} call sites)", flush=True)
    return src


def patch_cpp_scal_copyins(cpp: Path) -> int:
    """Fix 3, artifact form (the reference patch's copy-in hunks): give every
    declared-but-never-assigned ``scal_<x>[_N]`` local its ``= <x>[0]`` copy-in.
    A dead local (declared, never read) is left alone.  When the enclosing
    function lacks the source array, it is threaded in as a new parameter."""
    src = cpp.read_text()
    patched = 0
    for m in list(_SCAL_DECL_RE.finditer(src)):
        indent, ctype, name = m.groups()
        rest = src.replace(m.group(0), "", 1)
        if re.search(rf"(?m)^[^/\n]*\b{name}\s*=[^=]", rest) or f"&{name}" in rest:
            continue  # already assigned (or filled through a pointer)
        if not re.search(rf"\b{name}\b", rest):
            print(f"  cpp fix 3: {name} declared but never read -- left alone", flush=True)
            continue
        base = name[len("scal_"):]
        cand = [base, re.sub(r"_\d+$", "", base)]
        source = next((c for c in cand if re.search(rf"\b{c}\b", rest)), None)
        if source is None:
            raise RuntimeError(f"writer-less {name} in {cpp.name} has no source array "
                               f"(tried {cand}) -- cannot wire copy-in")
        decl = re.search(rf"(?m)^{re.escape(indent)}{ctype}\s+{name};\s*$", src)
        func, (p0, p1) = _enclosing_function(src, decl.start())
        if not re.search(rf"\b{source}\b", src[p0:p1]) and not func.startswith("__program"):
            src = _thread_param(src, func, source)
            decl = re.search(rf"(?m)^{re.escape(indent)}{ctype}\s+{name};\s*$", src)
        src = src[:decl.start()] + (f"{indent}{ctype} {name}; {name} = {source}[0];"
                                    f"  /* FIXED: missing copy-in */") + src[decl.end():]
        patched += 1
        print(f"  cpp fix 3: {name} = {source}[0] wired", flush=True)
    if patched == 0:
        print("  cpp fix 3: no writer-less scal_* locals -- nothing to patch", flush=True)
    cpp.write_text(src)
    return patched


def patch_cpp_eigts(cpp: Path) -> None:
    """Fix 4 of the reference patch: rebase eigts1/2/3 to (-nr:nr) bounds."""
    src = cpp.read_text()
    if "QE_EIGTS_REBASED" in src:
        print("  cpp fix 4: already applied", flush=True)
        return
    total = 0
    for k in ("eigts1", "eigts2", "eigts3"):
        if f"{k}[" not in src:
            print(f"  cpp fix 4: {k} not referenced (0 sites)", flush=True)
            continue
        pat = re.compile(rf"({k}\[\(\(\({k}_d0 \* \([^]]+?\)\) \+ [A-Za-z0-9_]+\) - 1)\)\]")
        src, nk = pat.subn(rf"\1 + (({k}_d0 + 1) / 2))]  /* FIXED: lbound -nr offset */", src)
        if nk == 0:
            offending = [ln.strip() for ln in src.splitlines() if f"{k}[" in ln]
            raise RuntimeError(f"cpp fix 4: {k} referenced but regex matched 0 sites:\n  "
                               + "\n  ".join(offending[:5]))
        total += nk
    src = "// QE_EIGTS_REBASED (run_optimize_original.py)\n" + src
    cpp.write_text(src)
    print(f"  cpp fix 4: {total} eigts sites rebased", flush=True)


def rebuild_kernel(build_dir: Path, name: str) -> None:
    """compile_commands.json replay as ONE compile+link (no generator files survive)."""
    import json
    import shlex
    cc = json.loads((build_dir / "compile_commands.json").read_text())
    entry = next(e for e in cc if e["file"].endswith(f"{name}.cpp"))
    args, skip = [], False
    for a in shlex.split(entry["command"]):
        if skip:
            skip = False
            continue
        if a in ("-MD", "-c") or a == entry["file"]:
            continue
        if a in ("-MT", "-MF", "-o"):
            skip = True
            continue
        args.append(a)
    out = build_dir / f"lib{name}.so"
    subprocess.check_call(args + ["-shared", "-o", str(out), entry["file"]],
                          cwd=entry["directory"])
    print(f"  kernel relinked via compile_commands replay -> {out.name}", flush=True)


def run_kernel(name: str, cfg: dict) -> None:
    sdir = HERE
    out = sdir / "outputs"
    original = out / f"{name}_original.sdfg"
    hlfir = out / f"{name}.hlfir"
    src_tu = out / f"{name}.F90"
    lib_src = out / "lib"
    lib_out = out / "lib_orig_opt"
    for f in (original, hlfir, src_tu, lib_src / "mpi_stub_gnu.f90"):
        if not f.is_file():
            sys.exit(f"missing {f}")
    lib_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name}: load original SDFG " + "=" * 40, flush=True)
    sdfg = dace.SDFG.from_file(str(original))
    if get_frozen_signature(sdfg) is None:
        sys.exit(f"{original.name} carries no frozen signature in frontend_metadata")
    print(f"  loaded {original.name}: {sum(1 for _ in sdfg.all_nodes_recursive())} nodes", flush=True)

    # ---- interface/plan recovery (metadata only; the builder's SDFG is never built)
    t0 = time.time()
    from dace_fortran.hlfir_to_sdfg import DEFAULT_PIPELINE, SDFGBuilder
    builder = SDFGBuilder(str(hlfir), pipeline=DEFAULT_PIPELINE, entry=cfg["entry"])
    sdfg._fortran_interface_raw = builder._fortran_interface_raw
    sdfg._flatten_plan_raw = builder.module.get_flatten_plan()
    print(f"  iface+plan recovered from {hlfir.name} in {time.time() - t0:.0f}s", flush=True)

    # ---- STRICTLY the stock pipeline ----------------------------------------
    t0 = time.time()
    optimize(sdfg)
    print(f"  pipelines.optimize() done in {time.time() - t0:.0f}s "
          f"({sum(1 for _ in sdfg.all_nodes_recursive())} nodes)", flush=True)
    refreeze(sdfg)   # REAL drift check vs the original builder freeze from the file
    print("  refreeze ok (array ABI identical to the original snapshot)", flush=True)
    sdfg.save(str(out / f"{name}_original_opt.sdfg"))

    # init-signature fix (emitter monkeypatch, per kernel; see build_opt_lib.py)
    used_syms = {str(s) for s in sdfg.used_symbols(all_symbols=False)}
    bb._init_sym_names = lambda frozen: [s for s in _ORIG_INIT_SYM_NAMES(frozen) if s in used_syms]
    dropped = [s for s in _ORIG_INIT_SYM_NAMES(sdfg._frozen_signature) if s not in used_syms]
    print(f"  init-signature fix: dropped {len(dropped)} codegen-eliminated symbols", flush=True)

    # ---- bindings + .so ------------------------------------------------------
    # fresh codegen every run: a stale (possibly half-patched) cache would be
    # reused silently on an SDFG-hash match
    shutil.rmtree(HERE / ".dacecache" / name, ignore_errors=True)
    t0 = time.time()
    gnu_stub = lib_out / "mpi_stub_gnu.f90"
    shutil.copy2(lib_src / "mpi_stub_gnu.f90", gnu_stub)
    try:
        build_fortran_library(sdfg, out_dir=str(lib_out), name=name,
                              prelude_sources=[gnu_stub, src_tu],
                              extra_flags=["-fallow-argument-mismatch", "-std=legacy"])
    finally:
        bb._init_sym_names = _ORIG_INIT_SYM_NAMES
    print(f"  lib{name}.so built in {time.time() - t0:.0f}s", flush=True)

    # ---- manual artifact fixes ----------------------------------------------
    bindings = lib_out / f"{name}_bindings.f90"
    patch_bindings(bindings, cfg["guard"])
    cpp = HERE / ".dacecache" / name / "src" / "cpu" / f"{name}.cpp"
    patch_cpp_scal_copyins(cpp)
    patch_cpp_eigts(cpp)
    kernel_build = HERE / ".dacecache" / name / "build"
    rebuild_kernel(kernel_build, name)

    # ---- guarded relink (stage-6 recipe) ------------------------------------
    kernel_copy = lib_out / f"libdace_kernel_{name}.so"
    shutil.copy2(kernel_build / f"lib{name}.so", kernel_copy)
    subprocess.check_call(["patchelf", "--set-soname", kernel_copy.name, str(kernel_copy)])
    for extra in kernel_build.glob("libdacestub_*.so"):
        shutil.copy2(extra, lib_out / extra.name)
    cmd = ["gfortran"] + GFC_FLAGS + [f"-J{lib_out}", str(gnu_stub), str(src_tu), str(bindings),
                                      "-o", str(lib_out / f"lib{name}.so"),
                                      f"-L{lib_out}", "-Wl,-rpath,$ORIGIN",
                                      f"-Wl,-rpath,{kernel_build}",
                                      f"-l:libdace_kernel_{name}.so"]
    subprocess.check_call(cmd, cwd=str(lib_out))
    need = subprocess.run(["readelf", "-d", str(lib_out / f"lib{name}.so")],
                          capture_output=True, text=True).stdout
    if f"libdace_kernel_{name}.so" not in need:
        raise RuntimeError("kernel NEEDED entry missing after relink")
    print(f"  guarded lib{name}.so relinked into {lib_out.name}", flush=True)

    # sanity: __dace_init arg counts must match C vs Fortran interface
    c_init = re.search(rf"__dace_init_{name}\(([^)]*)\)", cpp.read_text()).group(1)
    f_init = re.search(rf"function dace_init_{name}\(([^)]*)\)", bindings.read_text()).group(1)
    n_c = len([a for a in c_init.split(",") if a.strip()])
    n_f = len([a for a in f_init.split(",") if a.strip()])
    if n_c != n_f:
        raise RuntimeError(f"__dace_init arg-count mismatch: C {n_c} vs Fortran {n_f}")
    print(f"  init-signature check OK: {n_c} args on both sides", flush=True)


def main() -> int:
    picked = sys.argv[1:] or list(KERNELS)
    unknown = [k for k in picked if k not in KERNELS]
    if unknown:
        sys.exit(f"unknown kernel(s) {unknown}; choose from {list(KERNELS)}")
    for nm in STUBS:
        keep_external(nm, stub=True)
    for name in picked:
        run_kernel(name, KERNELS[name])
    print("\nall kernels done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
