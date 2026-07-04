"""Extract one ICON atmosphere kernel into a single, self-contained,
COMPILING ``.f90`` -- the ``input -> single TU`` stage for the dynamical core.

Mirrors :mod:`icon.ocean._extract_single_tu` but for the atmosphere solver and
WITHOUT externalising the halo exchange: ``sync_patch_array`` / ``exchange_data``
are inlined, and the inliner's default monomorphisation pass devirtualises the
(single-arm, post-cpp) ``t_comm_pattern`` dispatch into static calls.

  merge_used_modules (regex closure)
  -> inline_to_single_tu(expand_cpp=True, tolerate_external_uses=True,
                         monomorphize=True [default])
  -> gfortran -fsyntax-only

Run as a memory-capped subprocess so the fparser parse of the ~140k-line merged
closure cannot OOM the host.  Prints ``RESULT: PASS``, ``TU_PATH:`` and
``TU_LINES:`` on success.

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
    mem_gb = float(argv[4]) if len(argv) > 4 else 12.0
    halo_mode = argv[5] if len(argv) > 5 else "inlined"
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    cap = int(mem_gb * 1024**3)
    if hard != resource.RLIM_INFINITY:
        cap = min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    os.environ.setdefault("UCX_VFS_ENABLE", "n")
    out_dir.mkdir(parents=True, exist_ok=True)

    from icon.atmosphere._atmo_harness import atmo_config, SRC, atmo_search_dirs
    from dace_fortran import inline_to_single_tu
    from dace_fortran.preprocess import merge_used_modules

    cfg = atmo_config(halo_mode, entry)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    t0 = time.time()
    try:
        log(f"merge_used_modules ({source_relpath}) [halo={halo_mode}]")
        merged = merge_used_modules((SRC / source_relpath).read_text(), search_dirs=atmo_search_dirs())
        mp = out_dir / "merged.F90"
        mp.write_text(merged)
        log(f"  {len(merged.splitlines())} lines merged")

        # Force-include the concrete comm-pattern arm module(s): they are reached
        # only via the externalised factory, so the USE-graph merge never pulls
        # them in -- but the monomorphisation pass needs the arm to retype to.
        # Adding the source is not enough: the inliner's first step prunes every
        # module not USE-reachable from the entry (``keep_sorted_used_modules``),
        # which would drop the arm BEFORE monomorphisation runs.  So we also
        # inject a ``USE <arm-module>`` into the entry module, keeping it alive
        # until the retype makes the arm genuinely reachable (its exchange bodies
        # inlined), after which normal pruning keeps exactly what is needed.
        sources = {str(mp): merged}
        for name, content in cfg["extra_sources"].items():
            sources[str(out_dir / name)] = content
            log(f"  spliced extra source {name} ({len(content.splitlines())} lines)")
        entry_mod = entry.split("::")[0]
        use_lines = []
        for rel in cfg["force_include"]:
            content = (SRC / rel).read_text()
            sources[str(SRC / rel)] = content
            m = re.search(r"(?im)^\s*MODULE\s+(\w+)\s*$", content)
            if m:
                use_lines.append(f"  USE {m.group(1)}")
            log(f"  force-included {rel} (module {m.group(1) if m else '?'}, {len(content.splitlines())} lines)")
        if use_lines:
            merged, nsub = re.subn(rf"(?im)^(\s*MODULE\s+{re.escape(entry_mod)}\s*$)",
                                   lambda mm: mm.group(1) + "\n" + "\n".join(use_lines),
                                   merged,
                                   count=1)
            sources[str(mp)] = merged
            log(f"  injected {len(use_lines)} force-include USE(s) into module {entry_mod} (matched {nsub})")

        log("inline_to_single_tu(expand_cpp, tolerate_external_uses, monomorphize)")
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
                                 specialize_at_source=cfg["specialize_at_source"],
                                 keep_type_components=cfg.get("keep_type_components"),
                                 checkpoint_dir=(os.environ.get("ATMO_CHECKPOINT_DIR") or None),
                                 tolerate_external_uses=True)
        n = len(Path(tu).read_text().splitlines())
        log(f"  single TU: {n} lines in {time.time()-t0:.0f}s")
        print(f"TU_PATH: {tu}", flush=True)
        print(f"TU_LINES: {n}", flush=True)

        log("gfortran -fsyntax-only")
        cdir = out_dir / "cc"
        cdir.mkdir(exist_ok=True)
        cf = cdir / Path(tu).name
        shutil.copy(tu, cf)
        # ``-fallow-argument-mismatch``: the inlined halo leaves raw, interface-less
        # ``mpi_*`` calls whose buffer argument is REAL(8) / REAL(4) / INTEGER / a
        # scalar or array across the type-specific wrappers -- exactly the
        # type-generic external every real MPI build compiles with this flag.
        r = subprocess.run(
            ["gfortran", "-fsyntax-only", "-ffree-line-length-none", "-fallow-argument-mismatch", cf.name],
            cwd=str(cdir),
            capture_output=True,
            text=True)
        if r.returncode != 0:
            print(r.stderr[-6000:], flush=True)
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
