"""Extract one ICON-O kernel into a single, self-contained, COMPILING ``.f90``.

This is the ``input -> single TU`` stage:

  merge_used_modules (regex closure, NO mpi/netcdf library stubs)
  -> inline_to_single_tu(expand_cpp=True, tolerate_external_uses=True)
     (cpp pre-pass, CONTIGUOUS strip, external-USE tolerance for
      netcdf/mpi/cdi, consistent namelist pruning, prune to the kernel)
  -> gfortran -fsyntax-only  (the TU must be valid, compiling Fortran)

Lowering the TU to an SDFG is a SEPARATE concern handled elsewhere.

Run as a subprocess (with a virtual-memory cap) so the fparser parse of
the ~137k-line merged closure cannot OOM the host.  Prints
``RESULT: PASS``, ``TU_PATH: <path>`` and ``TU_LINES: <n>`` on success.

Usage:
    _extract_single_tu.py <source_relpath> <module::entry> <out_dir> [mem_gb]
"""
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main(argv):
    source_relpath, entry, out_dir = argv[1], argv[2], Path(argv[3])
    mem_gb = float(argv[4]) if len(argv) > 4 else 10.0
    halo_mode = argv[5] if len(argv) > 5 else "external"
    # Cap the SOFT address-space limit only, leaving the inherited HARD limit
    # untouched.  Raising the hard limit raises ValueError on a host that already
    # constrains it (e.g. a CI cgroup) -- and that would crash before any RESULT
    # marker is printed, which the harness would surface only as an opaque fail.
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    cap = int(mem_gb * 1024**3)
    if hard != resource.RLIM_INFINITY:
        cap = min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    os.environ.setdefault("UCX_VFS_ENABLE", "n")
    out_dir.mkdir(parents=True, exist_ok=True)

    from icon.ocean._ocean_harness import ocean_config, SRC, ocean_search_dirs
    from dace_fortran import inline_to_single_tu
    from dace_fortran.preprocess import merge_used_modules

    cfg = ocean_config(halo_mode)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    t0 = time.time()
    try:
        log(f"merge_used_modules ({source_relpath}) [halo={halo_mode}]")
        merged = merge_used_modules((SRC / source_relpath).read_text(), search_dirs=ocean_search_dirs())
        mp = out_dir / "merged.F90"
        mp.write_text(merged)
        log(f"  {len(merged.splitlines())} lines merged")

        # In the "inlined" halo mode the concrete comm-pattern arm is force-included
        # (it is reached only via the externalised factory) and kept alive past the
        # early USE-reachability prune via a USE injection into the entry module --
        # the monomorphisation pass then has it to retype to.  No-op for "external".
        sources = {str(mp): merged}
        entry_mod = entry.split("::")[0]
        use_lines = []
        for rel in cfg["force_include"]:
            content = (SRC / rel).read_text()
            sources[str(SRC / rel)] = content
            m = re.search(r"(?im)^\s*MODULE\s+(\w+)\s*$", content)
            if m:
                use_lines.append(f"  USE {m.group(1)}")
            log(f"  force-included {rel} (module {m.group(1) if m else '?'})")
        if use_lines:
            merged, nsub = re.subn(rf"(?im)^(\s*MODULE\s+{re.escape(entry_mod)}\s*$)",
                                   lambda mm: mm.group(1) + "\n" + "\n".join(use_lines),
                                   merged,
                                   count=1)
            sources[str(mp)] = merged
            log(f"  injected {len(use_lines)} force-include USE(s) into module {entry_mod} (matched {nsub})")

        log(f"inline_to_single_tu(expand_cpp, tolerate_external_uses, monomorphize) [halo={halo_mode}]")
        tu = inline_to_single_tu(sources,
                                 entry=entry,
                                 out_dir=out_dir,
                                 name="kernel_tu",
                                 expand_cpp=True,
                                 defines=cfg["defines"],
                                 include_dirs=[SRC / "include"],
                                 external_functions=cfg["external_functions"],
                                 do_not_emit=cfg["do_not_emit"],
                                 make_return_false=cfg["make_return_false"],
                                 rename_specifics=cfg["rename_specifics"],
                                 tolerate_external_uses=True)
        n = len(Path(tu).read_text().splitlines())
        log(f"  single TU: {n} lines in {time.time()-t0:.0f}s")
        print(f"TU_PATH: {tu}", flush=True)
        print(f"TU_LINES: {n}", flush=True)

        # Compile-check in a clean dir (the repo root carries stale flang
        # ``.mod`` files that gfortran refuses to read).
        log("gfortran -fsyntax-only")
        cdir = out_dir / "cc"
        cdir.mkdir(exist_ok=True)
        cf = cdir / Path(tu).name
        shutil.copy(tu, cf)
        r = subprocess.run(["gfortran", "-fsyntax-only", "-ffree-line-length-none", cf.name],
                           cwd=str(cdir),
                           capture_output=True,
                           text=True)
        if r.returncode != 0:
            print(r.stderr[-4000:], flush=True)
            print(f"RESULT: FAIL compile after {time.time()-t0:.0f}s", flush=True)
            return 1
        log(f"  COMPILES in {time.time()-t0:.0f}s total")
        print("RESULT: PASS", flush=True)
        return 0
    except MemoryError:
        print(f"RESULT: FAIL MemoryError (hit {mem_gb}GB cap) after {time.time()-t0:.0f}s", flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        print(f"RESULT: FAIL after {time.time()-t0:.0f}s", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
