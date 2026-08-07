"""Auto-build + import velocity_data (``from velocity_data_build import vd``); pattern
from dace_fortran/build_bridge.py minus its LLVM detection (host C++20 is enough)."""
import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
MODULE = "velocity_data"


def ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def local_so() -> Path:
    return HERE / f"{MODULE}{ext_suffix()}"


def needs_build() -> bool:
    so = local_so()
    if not so.exists():
        return True
    so_mtime = so.stat().st_mtime
    sources = [HERE / "velocity_data.cpp", HERE / "CMakeLists.txt", *sorted((HERE / "third_party").glob("*.hpp"))]
    return any(src.stat().st_mtime > so_mtime for src in sources)


def python_cmake_hints() -> list[str]:
    # From dace_fortran/build_bridge.py: a pyenv/venv prefix often ships no headers of
    # its own, so hand cmake the real sysconfig paths.
    hints = []
    include = sysconfig.get_path("include")
    if include and os.path.isdir(include):
        hints.append(f"-DPython_INCLUDE_DIR={include}")
    return hints


def build() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.check_call(
        ["cmake", str(HERE), f"-DPython_EXECUTABLE={sys.executable}", *python_cmake_hints()],
        cwd=BUILD_DIR,
    )
    subprocess.check_call(["cmake", "--build", ".", f"-j{os.cpu_count() or 4}"], cwd=BUILD_DIR)
    target = BUILD_DIR / f"{MODULE}{ext_suffix()}"
    if not target.exists():
        candidates = list(BUILD_DIR.rglob(f"{MODULE}{ext_suffix()}"))
        if not candidates:
            raise RuntimeError(f"build succeeded but {MODULE}{ext_suffix()} not found under {BUILD_DIR}")
        target = candidates[0]
    link = local_so()
    link.unlink(missing_ok=True)
    link.symlink_to(target)


def import_velocity_data():
    if needs_build():
        build()
    if str(HERE) not in sys.path:  # the .so lives beside this file, not on any package path
        sys.path.insert(0, str(HERE))
    return importlib.import_module(MODULE)


vd = import_velocity_data()
