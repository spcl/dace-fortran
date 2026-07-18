"""End-to-end smoke for ``cmake/DaceFortran.cmake``: a self-contained CMake
project includes the module, configures, builds, and asserts the preprocess
custom commands ran and every source landed.  Pins the build-system-integration
contract:

    include(DaceFortran)
    dace_fortran_preprocess(TARGET mylib SOURCES kernel.f90
                            SEARCH_DIRS utils
                            PASSES all_defaults rewrite_external)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HAVE_CMAKE = shutil.which("cmake") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CMAKE_MODULE_DIR = _REPO_ROOT / "cmake"

pytestmark = pytest.mark.skipif(not _HAVE_CMAKE, reason="cmake not on PATH")


def _write_project(tmp_path: Path) -> Path:
    """Stage a minimal CMake project exercising the module -- no Fortran compiler
    needed; the only target is the preprocess-source aggregator."""
    # The kernel that needs preprocessing.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kernel.f90").write_text("""\
SUBROUTINE run(out_val, x, f)
  IMPLICIT NONE
  REAL(KIND=wp), INTENT(IN) :: x, f
  REAL(KIND=wp), INTENT(OUT) :: out_val
  REAL(KIND=wp) :: dscale
  EXTERNAL :: dscale
  out_val = dscale(x, f)
END SUBROUTINE
""")
    # Sidecar module ``utils_mod`` containing ``dscale``.
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "utils_mod.f90").write_text("""\
MODULE utils_mod
  IMPLICIT NONE
CONTAINS
  REAL(KIND=8) FUNCTION dscale(x, f)
    REAL(KIND=8), INTENT(IN) :: x, f
    dscale = x * f
  END FUNCTION dscale
END MODULE utils_mod
""")
    # CMakeLists.txt -- this is the load-bearing UX we're testing.
    cmakelists = tmp_path / "CMakeLists.txt"
    cmakelists.write_text(f"""
cmake_minimum_required(VERSION 3.18)
project(dace_fortran_integration_smoke NONE)

list(PREPEND CMAKE_MODULE_PATH "{_CMAKE_MODULE_DIR}")
set(DACE_FORTRAN_PYTHON "{sys.executable}" CACHE FILEPATH "")
include(DaceFortran)

dace_fortran_preprocess(
    TARGET mylib
    SOURCES src/kernel.f90
    SEARCH_DIRS utils
    PASSES all_defaults rewrite_external)

# Re-export the preprocessed source list so the test can inspect it.
file(WRITE "${{CMAKE_BINARY_DIR}}/preprocessed_sources.txt"
     "${{mylib_PREPROCESSED_SOURCES}}")
""")
    return tmp_path


def test_cmake_configure_succeeds(tmp_path):
    """``cmake -S . -B build`` returns 0 -- DaceFortran.cmake imports
    cleanly and the ``dace_fortran_preprocess`` call is well-formed."""
    proj = _write_project(tmp_path)
    res = subprocess.run(["cmake", "-S", str(proj), "-B", str(proj / "build")], capture_output=True, text=True)
    assert res.returncode == 0, \
        f"cmake -S failed:\nstdout={res.stdout}\nstderr={res.stderr}"


def test_cmake_build_runs_preprocess_and_emits_sources(tmp_path):
    """``cmake --build`` invokes the preprocess commands; rewritten ``.f90`` lands
    in the build tree reflecting every requested pass."""
    proj = _write_project(tmp_path)
    subprocess.check_call(["cmake", "-S", str(proj), "-B", str(proj / "build")], stdout=subprocess.DEVNULL)
    res = subprocess.run(["cmake", "--build", str(proj / "build")], capture_output=True, text=True)
    assert res.returncode == 0, \
        f"cmake --build failed:\nstdout={res.stdout}\nstderr={res.stderr}"

    out = (proj / "build" / "dace_fortran_preprocessed" / "src" / "kernel.f90")
    assert out.is_file(), \
        f"no preprocessed kernel at {out}\nbuild tree:\n" + \
        "\n".join(str(p.relative_to(proj)) for p in proj.rglob("*"))
    rewritten = out.read_text()
    # ``KIND=wp`` -> ``KIND=8`` (all_defaults includes normalize-kind).
    assert "KIND=8" in rewritten
    assert "KIND=wp" not in rewritten
    # ``EXTERNAL :: dscale`` -> ``USE utils_mod, ONLY: dscale``.
    assert "USE utils_mod, ONLY: dscale" in rewritten
    assert "EXTERNAL :: dscale" not in rewritten


def test_cmake_returns_preprocessed_sources_variable_to_parent_scope(tmp_path):
    """The macro sets ``<TARGET>_PREPROCESSED_SOURCES`` in the parent scope,
    verified via the test project echoing it to ``preprocessed_sources.txt``."""
    proj = _write_project(tmp_path)
    subprocess.check_call(["cmake", "-S", str(proj), "-B", str(proj / "build")], stdout=subprocess.DEVNULL)
    listing = (proj / "build" / "preprocessed_sources.txt").read_text()
    assert "kernel.f90" in listing, \
        f"expected kernel.f90 in mylib_PREPROCESSED_SOURCES, got: {listing!r}"


def test_cmake_rebuilds_when_input_source_changes(tmp_path):
    """Editing ``src/kernel.f90`` makes ``cmake --build`` regenerate the output
    (CMake's DEPENDS on the input source)."""
    proj = _write_project(tmp_path)
    subprocess.check_call(["cmake", "-S", str(proj), "-B", str(proj / "build")], stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(proj / "build")], stdout=subprocess.DEVNULL)
    out = (proj / "build" / "dace_fortran_preprocessed" / "src" / "kernel.f90")
    mtime1 = out.stat().st_mtime

    # content change so the file digest changes (mtime alone isn't enough on some FS)
    src = proj / "src" / "kernel.f90"
    src.write_text(src.read_text() + "\n! cache-buster\n")
    # bump mtime explicitly -- some FS round to seconds
    import os
    import time
    time.sleep(1.05)
    os.utime(src, None)

    subprocess.check_call(["cmake", "--build", str(proj / "build")], stdout=subprocess.DEVNULL)
    mtime2 = out.stat().st_mtime
    assert mtime2 > mtime1, \
        f"expected rebuild after input change; mtime1={mtime1} mtime2={mtime2}"
