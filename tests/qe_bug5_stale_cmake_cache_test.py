"""Reproducer for bug 5 (dace-fortran-fixes-needed-6c99810.md):
``.dacecache/<name>/build`` must remain rebuildable after editing the
generated C++.

DaCe's command cache can skip the CMake configure step; in that mode the
build directory keeps ``rebuild.sh`` instead of ``CMakeCache.txt``/``Makefile``.
Either path is acceptable, so the test verifies that ONE documented in-place
rebuild command works after ``sdfg.compile()``.

Companion doc: bug5_stale_cmake_cache.md
"""
import subprocess
from pathlib import Path

import pytest

import dace


def _tiny_sdfg(tmp_path: Path):
    sdfg = dace.SDFG("rebuild_probe")
    sdfg.add_array("x", (1, ), dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    t = st.add_tasklet("t", {}, {"o"}, "o = 1.0")
    w = st.add_write("x")
    st.add_edge(t, "o", w, None, dace.Memlet("x[0]"))
    sdfg.build_folder = str(tmp_path / "cache")
    return sdfg


def test_generated_kernel_is_rebuildable_in_place(tmp_path: Path):
    """After ``sdfg.compile()``, a hand-edited generated .cpp can be rebuilt
    with either ``cmake --build .`` (when CMake configured in the build dir)
    or the shipped ``rebuild.sh`` recipe (when the command cache replayed)."""
    sdfg = _tiny_sdfg(tmp_path)
    sdfg.compile()

    build_dir = Path(sdfg.build_folder) / "build"
    assert build_dir.is_dir(), f"no build dir at {build_dir}"

    cpps = list((Path(sdfg.build_folder) / "src").rglob("*.cpp"))
    assert cpps, "no generated kernel sources found"
    cpps[0].touch()

    so = Path(sdfg.build_folder) / "build" / f"lib{sdfg.name}.so"
    assert so.exists(), f"expected shared library at {so}"
    mtime_before = so.stat().st_mtime

    if (build_dir / "CMakeCache.txt").exists():
        res = subprocess.run(["cmake", "--build", "."], cwd=build_dir, capture_output=True, text=True, timeout=300)
        recipe = "cmake --build ."
    else:
        rebuild_script = build_dir / "rebuild.sh"
        assert rebuild_script.exists(), ("build dir has neither CMakeCache.txt nor rebuild.sh; "
                                         "hand-fixed kernel .cpps cannot be rebuilt")
        res = subprocess.run(["sh", str(rebuild_script)], cwd=build_dir, capture_output=True, text=True, timeout=300)
        recipe = str(rebuild_script)

    assert res.returncode == 0, (f"in-place rebuild broken: {recipe} -> rc={res.returncode}\n"
                                 f"stdout: {res.stdout[-400:]}\nstderr: {res.stderr[-400:]}")
    assert so.stat().st_mtime > mtime_before, "shared library was not re-linked"
