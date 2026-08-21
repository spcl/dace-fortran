#!/usr/bin/env python3
"""End-to-end integration: DaCe-backed ``solve_nh`` inside a real ICON atmosphere run.

Workflow (mirrors ``run_icon_e2e.sh`` for velocity, adapted to ``solve_nh``):

  1. Build a stock ICON atmosphere binary (unpatched) for reference.
  2. Lower ICON's real ``mo_solve_nonhydro.f90`` to an SDFG and emit
     ``libsolve_nh.so`` + the ``solve_nh_dace_icon`` wrapper.
  3. Patch ``mo_solve_nonhydro.f90`` to dispatch to ``solve_nh_dace_icon``
     (differential driver: runs the SDFG on the real state, the renamed
     original body on a deep clone, and compares bit-for-bit).
  4. Configure + build the patched ICON, linking ``libsolve_nh.so``.
  5. Run both binaries on the EXCLAIM aquaplanet R02B04 experiment with
     ``pinit_seed=0`` and 2-3 MPI ranks.
  6. Assert every ``[diff] solve_nh TOTAL: 0`` line in the patched run
     shows bit-exact agreement.

All work happens under a configurable scratch root; the ICON source tree is
restored to pristine on exit so the live submodule is not left patched.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_DACE_FORTRAN = _HERE.parent
_TESTS_ICON_FULL = _DACE_FORTRAN / "tests" / "icon" / "full"

sys.path.insert(0, str(_DACE_FORTRAN / "tests"))
from icon.full._icon_solve_nh_patch import (
    SOLVE_NH_WRAPPER_NAME,
    write_patched_solve_nh,
)
from icon.full._icon_build import ensure_icon_built

#: Default experiment from the ICON source tree.
_EXP = "exclaim_ape_R02B04"


def _run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None, check: bool = True):
    """Run a command, printing it first."""
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=check)


def _stage_runscript_helpers(icon_src: Path, build_dir: Path):
    """Symlink source/run helpers into the build's run/ directory."""
    run_src = icon_src / "run"
    run_dst = build_dir / "run"
    run_dst.mkdir(parents=True, exist_ok=True)
    setup = build_dir / "run" / "set-up.info"
    if not setup.is_symlink() and not setup.is_file():
        setup.symlink_to(build_dir / "set-up.info")
    for entry in run_src.iterdir():
        dst = run_dst / entry.name
        if not dst.exists():
            dst.symlink_to(entry)


def _build_icon(label: str,
                icon_src: Path,
                build_dir: Path,
                dace_libs_dir: Optional[Path] = None,
                patched: bool = False,
                fresh: bool = False,
                jobs: int = 1):
    """Configure + build ICON; ``patched`` applies the solve_nh patch first."""
    solve_nh_f90 = icon_src / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
    solve_nh_bak = solve_nh_f90.with_suffix(".f90.bak")

    if patched:
        if not solve_nh_bak.is_file():
            shutil.copy2(solve_nh_f90, solve_nh_bak)
        else:
            shutil.copy2(solve_nh_bak, solve_nh_f90)
        write_patched_solve_nh(solve_nh_bak if solve_nh_bak.is_file() else solve_nh_f90, solve_nh_f90)
        print(f"[build_icon:{label}] patched {solve_nh_f90}", flush=True)

    if fresh and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if dace_libs_dir is not None:
        env["DACE_LIBS_DIR"] = str(dace_libs_dir)
    else:
        env.pop("DACE_LIBS_DIR", None)
    env["BUILD_DIR"] = str(build_dir)
    env["ICON_SRC"] = str(icon_src)

    _run(["bash", str(_HERE / "configure_icon_solve_nh_cpu.sh")], cwd=build_dir, env=env)
    _run(["make", f"-j{jobs}"], cwd=build_dir, env=env)
    shutil.copy2(build_dir / "bin" / "icon", build_dir / "bin" / f"icon.{label}")


def _make_experiment(icon_src: Path, build_dir: Path, exp_name: str, grid_dir: Path) -> Path:
    """Create the lane-specific experiment file from the source template."""
    run_src = icon_src / "run"
    exp_src = run_src / f"exp.{exp_name}"
    exp_dst = build_dir / "run" / f"exp.{exp_name}"
    shutil.copy2(exp_src, exp_dst)

    text = exp_dst.read_text()
    # Point the experiment at the local grid cache.
    text = re.sub(r'^input_folder=.*', f'input_folder="{grid_dir}"', text, flags=re.M)
    text = re.sub(r'^grid_id=.*', 'grid_id=0012', text, flags=re.M)
    text = re.sub(r'^atmo_dyn_grid=.*', f'atmo_dyn_grid="{grid_dir}/icon_grid_0012_R02B04_G.nc"', text, flags=re.M)
    # Short run + frequent output so the test finishes quickly and has dumps.
    text = re.sub(r'^end_date=.*', 'end_date="2000-01-01T00:00:10Z"', text, flags=re.M)
    text = re.sub(r'^start_output=.*', 'start_output="2000-01-01T00:00:02Z"', text, flags=re.M)
    text = re.sub(r'^atm_file_interval=.*', 'atm_file_interval="PT2S"', text, flags=re.M)
    text = re.sub(r'^atm_output_interval=.*', 'atm_output_interval="PT2S"', text, flags=re.M)
    # pinit_seed guard: force 0 to avoid the unallocated soil-temp perturbation segfault.
    text = re.sub(r'pinit_seed\s*=\s*[-]?[0-9]+', 'pinit_seed = 0', text, flags=re.M)
    text = re.sub(r'init_seed\s*=\s*[-]?[0-9]+', 'init_seed = 0', text, flags=re.M)
    text = re.sub(r'seed\s*=\s*[-]?[0-9]+', 'seed = 0', text, flags=re.M)
    # Pure compute run so the halo exchange is exercised at low rank count.
    text = re.sub(r'num_io_procs\s*=\s*[0-9]+', 'num_io_procs = 0', text, flags=re.M)
    exp_dst.write_text(text)
    return exp_dst


def _run_icon(label: str, build_dir: Path, icon_src: Path, exp_name: str, nranks: int) -> Path:
    """Run the labelled ICON binary on the experiment; returns the experiments dir."""
    icon_bin = build_dir / "bin" / f"icon.{label}"
    icon = build_dir / "bin" / "icon"
    shutil.copy2(icon_bin, icon)

    exp_dir = build_dir / "experiments" / exp_name
    exp_label_dir = build_dir / "experiments" / f"{exp_name}.{label}"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    if exp_label_dir.exists():
        shutil.rmtree(exp_label_dir)

    _stage_runscript_helpers(icon_src=icon_src, build_dir=build_dir)
    _run(["bash", str(icon_src / "make_runscripts"), exp_name], cwd=icon_src)

    run_script = build_dir / "run" / f"exp.{exp_name}.run"
    log = build_dir / f"icon_run.{label}.log"

    env = os.environ.copy()
    env["no_of_nodes"] = "1"
    env["mpi_procs_pernode"] = str(nranks)
    env["mpi_total_procs"] = str(nranks)

    with open(log, "w") as lf:
        rc = subprocess.call(
            ["mpirun", "--oversubscribe", "-n", str(nranks), "bash",
             str(run_script)],
            cwd=build_dir / "run",
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    print(f"[run_icon:{label}] rc={rc}", flush=True)

    if exp_dir.exists():
        shutil.move(exp_dir, exp_label_dir)
    return log


def _check_diff_log(log: Path) -> int:
    """Parse the patched run log for ``[diff] solve_nh TOTAL: N`` lines.

    Returns the number of solve_nh calls observed and asserts every diff is 0.
    """
    pattern = re.compile(r"\[diff\] solve_nh TOTAL:\s*(\d+)")
    calls = 0
    bad = 0
    with open(log) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                calls += 1
                total = int(m.group(1))
                if total != 0:
                    bad += 1
                    print(f"  BIT-EXACT FAILURE: {line.strip()}", flush=True)
    print(f"[check_diff_log] observed {calls} solve_nh differential call(s), {bad} non-zero", flush=True)
    return calls, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icon-src",
                    type=Path,
                    default=_TESTS_ICON_FULL / "icon-model",
                    help="ICON source tree (checked-out submodule).")
    ap.add_argument("--work-dir", type=Path, required=True, help="Scratch root for builds, grids, and artifacts.")
    ap.add_argument("--grid-dir", type=Path, default=None, help="R02B04 grid cache directory.")
    ap.add_argument("--exp", type=str, default=_EXP, help="Base experiment name under icon-model/run/.")
    ap.add_argument("--nranks", type=int, default=2, help="MPI rank count for both runs.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel make jobs for ICON builds.")
    ap.add_argument("--release",
                    action="store_true",
                    help="Use -O3 for the solve_nh binding (default -O0 for bit-exactness).")
    ap.add_argument("--skip-stock",
                    action="store_true",
                    help="Skip the stock reference run (use existing stock binary).")
    ap.add_argument("--skip-dace-build",
                    action="store_true",
                    help="Skip the solve_nh SDFG + binding build (use existing DACE_LIBS_DIR).")
    args = ap.parse_args()

    icon_src = args.icon_src.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dace_libs_dir = work_dir / "solve_nh_dace_libs"
    stock_build = work_dir / "icon_build_stock"
    dace_build = work_dir / "icon_build_dace"
    grid_dir = (args.grid_dir or work_dir / "icon_grids").resolve()

    solve_nh_f90 = icon_src / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
    solve_nh_bak = solve_nh_f90.with_suffix(".f90.bak")

    try:
        # 1. Stock ICON build (also produces the .mod files the DaCe binding needs).
        if not args.skip_stock:
            _build_icon("stock", icon_src, stock_build, dace_libs_dir=None, patched=False, jobs=args.jobs)
        else:
            print("[main] skipping stock build", flush=True)

        # 2. Build the solve_nh SDFG + binding from the pristine source.
        if not args.skip_dace_build:
            _run([
                sys.executable,
                "-m",
                "scripts.build_icon_solve_nh_libs",
                "--icon-src",
                str(icon_src),
                "--icon-build",
                str(stock_build),
                "--out-dir",
                str(dace_libs_dir),
            ] + (["--release"] if args.release else []),
                 cwd=_DACE_FORTRAN)
        else:
            print("[main] skipping DaCe solve_nh build", flush=True)

        # 3. Patched ICON build linking the DaCe solve_nh library.
        _build_icon("dace", icon_src, dace_build, dace_libs_dir=dace_libs_dir, patched=True, jobs=args.jobs)

        # 4. Fetch / cache the R02B04 grid.
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_file = grid_dir / "icon_grid_0012_R02B04_G.nc"
        if not grid_file.is_file():
            url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0012_R02B04_G.nc"
            print(f"[main] fetching grid from {url}", flush=True)
            _run(["wget", "-q", "-O", str(grid_file), url])
        else:
            print(f"[main] using cached grid {grid_file}", flush=True)

        # 5. Generate the lane experiment file.
        _make_experiment(icon_src, stock_build, args.exp, grid_dir)
        _make_experiment(icon_src, dace_build, args.exp, grid_dir)

        # 6. Run both binaries.
        stock_log = _run_icon("stock", stock_build, icon_src, args.exp, args.nranks)
        dace_log = _run_icon("dace", dace_build, icon_src, args.exp, args.nranks)

        # 7. Verify the patched run is bit-exact at every solve_nh call.
        calls, bad = _check_diff_log(dace_log)
        if calls == 0:
            print("ERROR: no solve_nh differential calls observed -- did the patched run reach solve_nh?",
                  file=sys.stderr)
            return 1
        if bad:
            print(f"ERROR: {bad} solve_nh call(s) diverged from stock", file=sys.stderr)
            return 1

        print("=== solve_nh ICON atmosphere e2e PASSED ===")
        return 0
    finally:
        # Restore the pristine source so the shared tree is not left patched.
        if solve_nh_bak.is_file():
            shutil.copy2(solve_nh_bak, solve_nh_f90)
            print("[main] restored pristine mo_solve_nonhydro.f90", flush=True)


if __name__ == "__main__":
    sys.exit(main())
