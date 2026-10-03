"""Reproducer for bug 5 (dace-fortran-fixes-needed-6c99810.md):
``.dacecache/<name>/build`` must remain rebuildable after editing the
generated C++.

DaCe's command cache replays a recorded build without configuring CMake, which
leaves nothing to rebuild with in place. A build meant for hand-editing therefore
turns ``compiler.command_cache`` off, and ``cmake --build .`` then rebuilds it.

Companion doc: bug5_stale_cmake_cache.md
"""

import os
import subprocess
import tempfile
from pathlib import Path

import dace


def _tiny_sdfg(tmp_path: Path):
    sdfg = dace.SDFG("rebuild_probe")
    sdfg.add_array("x", (1,), dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    t = st.add_tasklet("t", {}, {"o"}, "o = 1.0")
    w = st.add_write("x")
    st.add_edge(t, "o", w, None, dace.Memlet("x[0]"))
    sdfg.build_folder = str(tmp_path / "cache")
    return sdfg


def test_generated_kernel_is_rebuildable_in_place(tmp_path: Path):
    """With the command cache off, a hand-edited generated .cpp rebuilds with ``cmake --build .``."""
    sdfg = _tiny_sdfg(tmp_path)
    with dace.config.set_temporary("compiler", "command_cache", value=False):
        sdfg.compile()

    build_dir = Path(sdfg.build_folder) / "build"
    assert (build_dir / "CMakeCache.txt").exists(), f"CMake did not configure {build_dir}"

    cpps = list((Path(sdfg.build_folder) / "src").rglob("*.cpp"))
    assert cpps, "no generated kernel sources found"
    so = build_dir / f"lib{sdfg.name}.so"
    assert so.exists(), f"expected shared library at {so}"
    mtime_before = so.stat().st_mtime
    os.utime(cpps[0], (mtime_before + 10, mtime_before + 10))  # a hand-edit, without waiting a clock tick

    res = subprocess.run(["cmake", "--build", "."], cwd=build_dir, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, (
        f"in-place rebuild broken: rc={res.returncode}\nstdout: {res.stdout[-400:]}\nstderr: {res.stderr[-400:]}"
    )
    assert so.stat().st_mtime > mtime_before, "shared library was not re-linked"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_generated_kernel_is_rebuildable_in_place(Path(tmp))
