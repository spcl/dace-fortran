#!/usr/bin/env python3
"""End-to-end integration: DaCe-backed ``solve_nh`` inside a real ICON atmosphere run.

Workflow (mirrors ``run_icon_e2e.sh`` for velocity, adapted to ``solve_nh``):

  1. Build a stock ICON atmosphere binary (unpatched) for reference.
  2. Lower the ``solve_nh`` translation unit to an SDFG and emit
     ``libsolve_nh.so`` + the ``solve_nh_dace_icon`` wrapper.
  3. Patch ``mo_solve_nonhydro.f90`` to dispatch to ``solve_nh_dace_icon``.
  4. Configure + build the patched ICON, linking ``libsolve_nh.so``.
  5. Run both binaries on the EXCLAIM aquaplanet R02B04 experiment with
     ``pinit_seed=0`` and 2-3 MPI ranks.
  6. Compare selected dycore output fields in the generated NetCDF files.

By default the patched binary runs the SDFG dycore only and the script
compares the resulting model output against the stock run.  With
``--diff-driver`` the patched binary runs both the SDFG and the original
Fortran body on a deep clone and reports bit-exact differences per call
(this is expensive and mainly useful for debugging).

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

import netCDF4
import numpy as np

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
    """Symlink source/run helpers into the build's run/ directory.

    Also repoint ``icon_src/run/set-up.info`` at this build's
    ``run/set-up.info`` so ``make_runscripts`` hardcodes ``basedir`` to
    ``build_dir``.
    """
    run_src = icon_src / "run"
    run_dst = build_dir / "run"
    run_dst.mkdir(parents=True, exist_ok=True)
    src_setup = run_src / "set-up.info"
    src_setup.unlink(missing_ok=True)
    src_setup.symlink_to(run_dst / "set-up.info")
    for entry in run_src.iterdir():
        dst = run_dst / entry.name
        if not dst.exists():
            dst.symlink_to(entry)


def _build_icon(label: str,
                icon_src: Path,
                build_dir: Path,
                dace_libs_dir: Optional[Path] = None,
                patched: bool = False,
                dace_only: bool = True,
                fresh: bool = False,
                jobs: int = 1,
                reuse: bool = False):
    """Configure + build ICON; ``patched`` applies the solve_nh patch first."""
    solve_nh_f90 = icon_src / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
    solve_nh_bak = solve_nh_f90.with_suffix(".f90.bak")

    if patched:
        if not solve_nh_bak.is_file():
            shutil.copy2(solve_nh_f90, solve_nh_bak)
        else:
            shutil.copy2(solve_nh_bak, solve_nh_f90)
        write_patched_solve_nh(solve_nh_bak if solve_nh_bak.is_file() else solve_nh_f90,
                               solve_nh_f90,
                               dace_only=dace_only)
        print(f"[build_icon:{label}] patched {solve_nh_f90}", flush=True)

    if reuse and (build_dir / "bin" / f"icon.{label}").is_file():
        print(f"[build_icon:{label}] reusing existing binary", flush=True)
        return

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
    src_bin = build_dir / "bin" / "icon"
    dst_bin = build_dir / "bin" / f"icon.{label}"
    if dst_bin.exists() or dst_bin.is_symlink():
        dst_bin.unlink()
    shutil.copy2(src_bin, dst_bin)


def _make_experiment(icon_src: Path, build_dir: Path, exp_name: str, grid_dir: Path) -> Path:
    """Create the lane-specific experiment file from the source template."""
    run_src = icon_src / "run"
    exp_src = run_src / f"exp.{exp_name}"
    exp_dst = build_dir / "run" / f"exp.{exp_name}"
    shutil.copy2(exp_src, exp_dst)

    text = exp_dst.read_text()
    # Point the experiment at the local grid cache.  ``atmo_dyn_grid`` is
    # expressed in terms of ``input_folder`` in the EXCLAIM template, so updating
    # ``input_folder`` is sufficient.
    text = re.sub(r'^input_folder=.*', f'input_folder="{grid_dir}"', text, flags=re.M)
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
    # The default EXCLAIM APE experiment selects ecRad (inwp_radiation=4), but the
    # CPU-only ICON build in this lane does not include ECRAD.  Disable radiation so
    # the run reaches the dycore without aborting in mo_nwp_phy_init.
    text = re.sub(r'inwp_radiation\s*=\s*[0-9]+', 'inwp_radiation = 0', text, flags=re.M)
    # EXCLAIM aquaplanet template does not contain pinit_seed / num_io_procs; append
    # them to the experiment section so the namelist still sees them.
    if 'pinit_seed' not in text:
        text += '\npinit_seed = 0\n'
    if 'num_io_procs' not in text:
        text += '\nnum_io_procs = 0\n'
    if 'inwp_radiation' not in text:
        text += '\ninwp_radiation = 0\n'
    exp_dst.write_text(text)
    return exp_dst


def _run_icon(label: str, build_dir: Path, icon_src: Path, exp_name: str, nranks: int) -> Path:
    """Run the labelled ICON binary on the experiment; returns the experiments dir."""
    icon_bin = (build_dir / "bin" / f"icon.{label}").resolve()

    exp_dir = build_dir / "experiments" / exp_name
    exp_label_dir = build_dir / "experiments" / f"{exp_name}.{label}"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    if exp_label_dir.exists():
        shutil.rmtree(exp_label_dir)

    _stage_runscript_helpers(icon_src=icon_src, build_dir=build_dir)

    # make_runscripts reads the experiment file from icon_src/run.  Substitute
    # our locally-modified copy temporarily, then restore the original template.
    src_exp = icon_src / "run" / f"exp.{exp_name}"
    local_exp = build_dir / "run" / f"exp.{exp_name}"
    saved_exp = icon_src / "run" / f"exp.{exp_name}.orig"
    if saved_exp.exists():
        saved_exp.unlink()
    shutil.copy2(src_exp, saved_exp)
    shutil.copy2(local_exp, src_exp)
    try:
        _run(["bash", str(icon_src / "make_runscripts"), exp_name], cwd=icon_src)
    finally:
        shutil.copy2(saved_exp, src_exp)
        saved_exp.unlink()

    # make_runscripts writes into icon_src/run and hardcodes basedir from the
    # current set-up.info.  Copy the generated script into this build's run/
    # directory so stock and dace builds do not share (and overwrite) one script.
    src_run_script = icon_src / "run" / f"exp.{exp_name}.run"
    run_script = build_dir / "run" / f"exp.{exp_name}.run"
    if run_script.exists() or run_script.is_symlink():
        run_script.unlink()
    shutil.copy2(src_run_script, run_script)
    # The generated script names ${basedir}/bin/icon; point it at the labelled
    # binary so we never overwrite a running executable (text file busy).
    run_script.write_text(
        re.sub(r'^export MODEL=.*bin/icon".*$', f'export MODEL="{icon_bin}"', run_script.read_text(), flags=re.M))
    log = build_dir / f"icon_run.{label}.log"

    # The run script itself calls the MPI launcher via $START; do not wrap it
    # in another mpirun.  Export the rank count so the script recomputes
    # mpi_total_procs = no_of_nodes * mpi_procs_pernode.
    env = os.environ.copy()
    env["no_of_nodes"] = "1"
    env["mpi_procs_pernode"] = str(nranks)

    with open(log, "w") as lf:
        rc = subprocess.call(
            ["bash", str(run_script)],
            cwd=build_dir / "run",
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    print(f"[run_icon:{label}] rc={rc}", flush=True)

    if exp_dir.exists():
        shutil.move(exp_dir, exp_label_dir)
    return log


#: Output variables whose values are dominated by the dycore and are compared
#: between the stock and DaCe runs.
_DYCORE_VARS = ("u", "v", "w", "temp", "geopot", "u_10m", "v_10m", "t_2m")


def _check_diff_log(log: Path) -> tuple[int, int]:
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


def _compare_output_files(stock_dir: Path, dace_dir: Path) -> tuple[int, int]:
    """Compare selected dycore variables in matching output NetCDF files.

    Returns ``(compared_variables, mismatches)``.  A mismatch is any element
    that is not bit-exact between the two runs.
    """
    stock_files = {p.name: p for p in stock_dir.glob("*.nc")}
    dace_files = {p.name: p for p in dace_dir.glob("*.nc")}
    common = sorted(set(stock_files) & set(dace_files))
    compared = 0
    mismatches = 0
    for name in common:
        stock_path = stock_files[name]
        dace_path = dace_files[name]
        with netCDF4.Dataset(stock_path, "r") as ds_s, netCDF4.Dataset(dace_path, "r") as ds_d:
            for var in _DYCORE_VARS:
                if var not in ds_s.variables or var not in ds_d.variables:
                    continue
                a = np.asarray(ds_s.variables[var][:])
                b = np.asarray(ds_d.variables[var][:])
                if a.shape != b.shape:
                    print(f"  SHAPE MISMATCH: {name}:{var} {a.shape} vs {b.shape}", flush=True)
                    mismatches += 1
                    continue
                compared += 1
                if not np.array_equal(a, b):
                    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    max_diff = float(np.nanmax(diff))
                    mismatches += 1
                    print(f"  MISMATCH: {name}:{var} max_abs_diff={max_diff:.6e}", flush=True)
    print(
        f"[compare_outputs] compared {compared} variable(s) across {len(common)} file(s), "
        f"{mismatches} mismatched",
        flush=True)
    return compared, mismatches


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
    ap.add_argument("--reuse-build",
                    action="store_true",
                    help="Skip ICON configure/make if bin/icon.<label> already exists.")
    ap.add_argument("--diff-driver",
                    action="store_true",
                    help=("Use the expensive differential driver (runs original + SDFG in one binary and "
                          "compares per-call).  Default compares model output files after separate runs."))
    ap.add_argument(
        "--dace-libs-dir",
        type=Path,
        default=None,
        help="Directory with an existing libsolve_nh.so + .mod files.  Overrides the default work-dir location.")
    args = ap.parse_args()

    icon_src = args.icon_src.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dace_libs_dir = (args.dace_libs_dir or work_dir / "solve_nh_dace_libs").resolve()
    stock_build = work_dir / "icon_build_stock"
    dace_build = work_dir / "icon_build_dace"
    grid_dir = (args.grid_dir or work_dir / "icon_grids").resolve()

    solve_nh_f90 = icon_src / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
    solve_nh_bak = solve_nh_f90.with_suffix(".f90.bak")

    try:
        # 1. Stock ICON build (also produces the .mod files the DaCe binding needs).
        if not args.skip_stock:
            _build_icon("stock",
                        icon_src,
                        stock_build,
                        dace_libs_dir=None,
                        patched=False,
                        jobs=args.jobs,
                        reuse=args.reuse_build)
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
        _build_icon("dace",
                    icon_src,
                    dace_build,
                    dace_libs_dir=dace_libs_dir,
                    patched=True,
                    dace_only=not args.diff_driver,
                    jobs=args.jobs,
                    reuse=args.reuse_build)

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

        # 7. Verify the patched run matches the stock run.
        if args.diff_driver:
            calls, bad = _check_diff_log(dace_log)
            if calls == 0:
                print("ERROR: no solve_nh differential calls observed -- did the patched run reach solve_nh?",
                      file=sys.stderr)
                return 1
            if bad:
                print(f"ERROR: {bad} solve_nh call(s) diverged from stock", file=sys.stderr)
                return 1
        else:
            stock_exp_dir = stock_build / "experiments" / f"{args.exp}.stock"
            dace_exp_dir = dace_build / "experiments" / f"{args.exp}.dace"
            compared, bad = _compare_output_files(stock_exp_dir, dace_exp_dir)
            if compared == 0:
                print("ERROR: no comparable output variables found -- did both runs produce NetCDF output?",
                      file=sys.stderr)
                return 1
            if bad:
                print(f"ERROR: {bad} compared variable(s) diverged from stock", file=sys.stderr)
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
