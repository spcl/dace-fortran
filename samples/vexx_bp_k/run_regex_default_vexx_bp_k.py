#!/usr/bin/env python3
"""Stage-2-onward copy of f2dace-qe-source/run_regex_default_vexx_bp_k.py:
ingest the READY single TU ``outputs/vexx_bp_k.F90`` (no merge stage -- edit
that file to change what flang sees) and produce

  outputs/vexx_bp_k.hlfir             flang -fc1 -emit-hlfir
  outputs/vexx_bp_k_normalized.mlir   after DEFAULT_PIPELINE
  outputs/vexx_bp_k.sdfg              built SDFG
  outputs/lib/                        every compiled module (.mod) + mpi stub,
                                      then the bindings + guarded .so chain:
                                      vexx_bp_k_bindings.f90, libvexx_bp_k.so,
                                      libdace_kernel_vexx_bp_k.so (+dacestub)

    cd /workspace/dace-fortran/samples/vexx_bp_k && python3 run_regex_default_vexx_bp_k.py
"""
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LIB = OUT / "lib"
SRC_TU = OUT / "vexx_bp_k.F90"
NAME = "vexx_bp_k"
BIND_NAME = "vexx_bp_k"
ENTRY = "exx_bp::vexx_bp_k"
GFC_FLAGS = ["-shared", "-fPIC", "-ffree-line-length-none", "-O3", "-g", "-fno-fast-math",
             "-ffp-contract=off", "-frounding-math", "-fopenmp",
             "-fallow-argument-mismatch", "-std=legacy"]
_OPENMPI_EXTRA = ["/opt/hpcx/ompi4/include", "/opt/hpcx/ompi5/include"]

if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")

from dace_fortran import keep_external  # noqa: E402
from dace_fortran.build import _find_flang, _flang_intrinsic_modules_path  # noqa: E402
from dace_fortran.flang_codebase import find_openmpi_include, mpi_stub_source  # noqa: E402

#: strict scope guard injected into the generated wrapper (see NEGRP_FINDINGS.md)
_GUARD_ANCHOR = "    ! ----- Symbol population (input-derived; before allocates) -----"
_GUARD = """    ! ----- STRICT SCOPE GUARD (injected by run_regex_default_vexx_bp_k.py) -----
    ! The SDFG drops the inter-rank collectives (exxbuff circular shift,
    ! deexx mp_sum, result_sum); running with negrp > 1 would return silently
    ! corrupted hpsi.
    if (negrp__mod /= 1) then
      write (*, '(a,i0,a)') 'FATAL(vexx_bp_k_dace): negrp = ', negrp__mod, &
        ' -- only negrp == 1 is supported by this SDFG build; aborting.'
      error stop 8
    end if
"""

#: do_not_emit stub families (same as the TU-producing run: their bodies are
#: already stubbed inside the TU; the registry drives hlfir-drop-stub-calls)
STUBS = [
    "mp_sum", "mp_bcast", "mp_barrier", "mp_gather", "mp_size", "mp_rank",
    "mp_circular_shift_left", "result_sum", "reduce_base_real", "poolcollect",
    "zgemm", "zgemv", "zaxpy", "zcopy", "zdotc", "zdscal",
    "get_buffer", "save_buffer", "open_buffer", "close_buffer",
    "start_clock", "stop_clock", "errore", "infomsg",
]

results = {}


def stage(n, label):
    print(f"\n=== stage {n}: {label} " + "=" * max(0, 50 - len(label)))
    results[n] = ["FAILED", label]


def done(n, msg):
    results[n] = ["ok", results[n][1] + f" -- {msg}"]
    print(f"    ok: {msg}")


def _bisect_default(hlfir: Path, pipeline: str, entry: str):
    """On pipeline failure: fresh module, entry set, one pass at a time."""
    print("\n--- pass-by-pass bisect " + "-" * 40)
    from dace_fortran.build_bridge import hb
    from dace_fortran.builder import _resolve_entry_symbol
    m = hb.HLFIRModule()
    if not m.parse_file(str(hlfir)):
        print("bisect: cannot re-parse the .hlfir")
        return
    m.set_entry_symbol(_resolve_entry_symbol(m, entry))
    for i, p in enumerate([p for p in pipeline.split(",") if p.strip()], 1):
        try:
            m.run_passes(p)
            print(f"  {i:2d} {p:45s} ok")
        except Exception as e:
            print(f"  {i:2d} {p:45s} FAILED: {str(e)[:400]}")
            if not p.startswith("hlfir-verify"):
                break
    else:
        (OUT / f"{NAME}_normalized_partial.mlir").write_text(m.dump())
        print(f"  wrote {NAME}_normalized_partial.mlir")


def main() -> int:
    if not SRC_TU.is_file():
        sys.exit(f"input TU not found: {SRC_TU}")
    LIB.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    for nm in STUBS:
        keep_external(nm, stub=True)

    # ---------------------------------------------------------------- stage 1b
    stage("1b", "MODULE mpi stub -> mpi.mod (flang, into outputs/lib)")
    ompi = find_openmpi_include() or next((d for d in _OPENMPI_EXTRA if (Path(d) / "mpif-config.h").is_file()), None)
    if ompi is None:
        raise RuntimeError("no OpenMPI mpif-*.h headers found")
    stub = LIB / "mpi_stub.f90"
    stub.write_text(mpi_stub_source())
    subprocess.check_call([_find_flang(), "-fc1", "-fsyntax-only", "-I", ompi, str(stub)], cwd=str(LIB))
    done("1b", f"mpi.mod from {ompi}")

    # ---------------------------------------------------------------- stage 2
    stage(2, "flang -fc1 -emit-hlfir (modules into outputs/lib)")
    t0 = time.time()
    hlfir = OUT / f"{NAME}.hlfir"
    # replicate build._emit_hlfir's invocation, but with cwd = outputs/lib so
    # every emitted .mod lands there instead of beside the artifacts
    cmd = [_find_flang(), "-fc1", "-cpp", "-U_OPENMP", "-U_OPENACC", "-I", str(LIB)]
    intrinsic = _flang_intrinsic_modules_path(cmd[0])
    if intrinsic is not None:
        cmd += ["-fintrinsic-modules-path", str(intrinsic)]
    cmd += ["-emit-hlfir", str(SRC_TU), "-o", str(hlfir)]
    subprocess.check_call(cmd, cwd=str(LIB))
    done(2, f"{hlfir.name}: {hlfir.stat().st_size // 1024} KiB in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- stage 3
    stage(3, "DEFAULT_PIPELINE -> normalised IR")
    from dace_fortran.hlfir_to_sdfg import DEFAULT_PIPELINE, SDFGBuilder
    t0 = time.time()
    try:
        builder = SDFGBuilder(str(hlfir), pipeline=DEFAULT_PIPELINE, entry=ENTRY)
    except Exception:
        traceback.print_exc()
        _bisect_default(hlfir, DEFAULT_PIPELINE, ENTRY)
        raise
    norm = OUT / f"{NAME}_normalized.mlir"
    norm.write_text(builder.module.dump())
    done(3, f"{norm.name}: {norm.stat().st_size // 1024} KiB in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- stage 4
    stage(4, "SDFG build")
    t0 = time.time()
    sdfg = builder.build()
    if sdfg.name != BIND_NAME:
        # the bindings emitter derives the C symbols (__dace_init_<name>...)
        # from BIND_NAME, so the kernel must be compiled under that name or
        # ld's --as-needed silently drops it at the stage-6 relink
        from dace_fortran.bindings.frozen_signature import refreeze
        sdfg.name = BIND_NAME
        refreeze(sdfg)
    sdfg.save(str(OUT / f"{NAME}.sdfg"))
    done(4, f"{NAME}.sdfg: {sum(1 for _ in sdfg.all_nodes_recursive())} nodes in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- stage 5
    stage(5, f"bindings + .so (name={BIND_NAME}, into outputs/lib)")
    import re
    import shutil
    from dace_fortran.bindings import build_fortran_library
    t0 = time.time()
    # gfortran cannot read the flang-format .mod files stage 2 left in LIB
    # (and -J LIB makes it look there): purge them; the compile regenerates
    # GNU .mods for every module in its place
    purged = sum(1 for m2 in LIB.glob("*.mod") if (m2.unlink() or True))
    print(f"    purged {purged} flang .mod files from outputs/lib")
    stub_txt = mpi_stub_source()
    inc_re = re.compile(r"(?im)^\s*INCLUDE\s+'([\w.-]+)'\s*$")
    for _ in range(4):
        new = inc_re.sub(lambda mm: (Path(ompi) / mm.group(1)).read_text(errors="replace"), stub_txt)
        if new == stub_txt:
            break
        stub_txt = new
    gnu_stub = LIB / "mpi_stub_gnu.f90"
    gnu_stub.write_text(stub_txt)
    build_fortran_library(sdfg, out_dir=str(LIB), name=BIND_NAME,
                          prelude_sources=[gnu_stub, SRC_TU],
                          extra_flags=["-fallow-argument-mismatch", "-std=legacy"])
    done(5, f"lib{BIND_NAME}.so in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- stage 6
    stage(6, "inject strict negrp==1 guard + relink")
    t0 = time.time()
    bindings = LIB / f"{BIND_NAME}_bindings.f90"
    txt = bindings.read_text()
    if _GUARD_ANCHOR not in txt:
        raise RuntimeError("guard anchor not found in generated bindings")
    if "STRICT SCOPE GUARD" not in txt:
        txt = txt.replace(_GUARD_ANCHOR, _GUARD + _GUARD_ANCHOR, 1)
        bindings.write_text(txt)
    # link a RENAMED, SONAME-retagged COPY of the dace kernel so the binding
    # (same basename) can never satisfy its own kernel NEEDED entry
    kernel_build = Path.cwd() / ".dacecache" / sdfg.name / "build"
    kernel_copy = LIB / f"libdace_kernel_{BIND_NAME}.so"
    shutil.copy2(kernel_build / f"lib{sdfg.name}.so", kernel_copy)
    subprocess.check_call(["patchelf", "--set-soname", kernel_copy.name, str(kernel_copy)])
    for extra in kernel_build.glob("libdacestub_*.so"):
        shutil.copy2(extra, LIB / extra.name)
    cmd = ["gfortran"] + GFC_FLAGS + [f"-J{LIB}", str(gnu_stub), str(SRC_TU), str(bindings),
                                      "-o", str(LIB / f"lib{BIND_NAME}.so"),
                                      f"-L{LIB}", "-Wl,-rpath,$ORIGIN", f"-Wl,-rpath,{kernel_build}",
                                      f"-l:libdace_kernel_{BIND_NAME}.so"]
    subprocess.check_call(cmd, cwd=str(LIB))
    need = subprocess.run(["readelf", "-d", str(LIB / f"lib{BIND_NAME}.so")],
                          capture_output=True, text=True).stdout
    if f"libdace_kernel_{BIND_NAME}.so" not in need:
        raise RuntimeError("kernel NEEDED entry missing after relink")
    done(6, f"guarded lib{BIND_NAME}.so relinked in {time.time() - t0:.0f}s")

    print(f"\nall stages complete in {time.time() - t00:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 1
    print("\nsummary:")
    for n, (st, label) in results.items():
        print(f"  stage {n}: [{st:6s}] {label}")
    sys.exit(rc)
