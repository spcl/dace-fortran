"""Unit tests for the pure helpers in ``dace_fortran.build_bridge``.

Covers only ``_python_cmake_hints`` (sysconfig-derived cmake hints, the
venv/pyenv fix) and ``needs_build`` (source-vs-``.so`` mtime gate); the
cmake/LLVM shell-out paths are exercised by CI's build step, not mocked here.
"""
import os
import subprocess
import sys
import sysconfig

from dace_fortran import build_bridge


def test_python_cmake_hints_point_at_real_files():
    """Every ``-DPython_*=<path>`` hint resolves, so cmake's find_package works from a venv/pyenv prefix."""
    hints = build_bridge._python_cmake_hints()

    inc = next((h for h in hints if h.startswith("-DPython_INCLUDE_DIR=")), None)
    assert inc is not None, f"no include hint in {hints}"
    inc_dir = inc.split("=", 1)[1]
    assert os.path.isdir(inc_dir)
    assert os.path.isfile(os.path.join(inc_dir, "Python.h"))

    # The library hint is conditional (only when a shared libpython
    # actually exists); when present it must resolve to a real file.
    for h in hints:
        if h.startswith("-DPython_LIBRARY="):
            assert os.path.exists(h.split("=", 1)[1])


def test_python_cmake_hints_match_running_interpreter():
    """Include hint matches the active interpreter's sysconfig path exactly (no hard-coded prefix)."""
    hints = build_bridge._python_cmake_hints()
    expected_inc = sysconfig.get_path("include")
    assert f"-DPython_INCLUDE_DIR={expected_inc}" in hints


def test_needs_build_true_when_source_newer(tmp_path, monkeypatch):
    """A ``.cpp`` newer than the linked ``.so`` forces a rebuild;
    an up-to-date ``.so`` does not."""
    so = tmp_path / build_bridge._so_name()
    src = tmp_path / "bridge.cpp"
    src.write_text("// stub\n")
    so.write_bytes(b"")

    monkeypatch.setattr(build_bridge, "_HERE", tmp_path)
    monkeypatch.setattr(build_bridge, "_local_so", lambda: so)

    # .so newer than the source -> no rebuild.
    os.utime(src, (1, 1))
    os.utime(so, (2, 2))
    assert build_bridge.needs_build() is False

    # Source touched after the .so -> rebuild.
    os.utime(so, (1, 1))
    os.utime(src, (2, 2))
    assert build_bridge.needs_build() is True


def test_needs_build_true_when_so_missing(tmp_path, monkeypatch):
    """No linked ``.so`` at all -> must build."""
    monkeypatch.setattr(build_bridge, "_HERE", tmp_path)
    monkeypatch.setattr(build_bridge, "_local_so", lambda: tmp_path / "absent.so")
    assert build_bridge.needs_build() is True


def test_importing_the_module_does_not_load_the_extension():
    """``python -m dace_fortran.build_bridge`` rebuilds the ``.so``; loading it during import would
    leave the process holding the copy it is about to replace, and the README promises the bridge
    compiles on first use rather than at import.  A subprocess keeps an earlier test's ``hb`` out."""
    code = "import dace_fortran.build_bridge, sys; print('dace_fortran.hlfir_bridge' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "False", out


def test_an_incomplete_prefix_is_not_selected(monkeypatch, tmp_path):
    """An LLVM prefix can ship LLVMConfig.cmake and neither flang nor MLIR. Selecting it
    configures a build that cannot emit HLFIR and links a second LLVM's MLIR."""
    prefix = tmp_path / "llvm-x"
    (prefix / "lib" / "cmake" / "llvm").mkdir(parents=True)
    monkeypatch.setattr(build_bridge, "find_flang", lambda v: None)
    assert build_bridge._prefix_builds_the_bridge(str(prefix), "22") is False

    (prefix / "lib" / "cmake" / "mlir").mkdir(parents=True)
    monkeypatch.setattr(build_bridge, "find_flang", lambda v: "/usr/bin/flang-new-22")
    assert build_bridge._prefix_builds_the_bridge(str(prefix), "22") is True


def test_a_differently_configured_build_dir_is_reported(monkeypatch, tmp_path):
    """Reconfiguring in place leaves entries cmake does not overwrite pointing at the old prefix,
    which is how one module ends up linking parts of two LLVM installs."""
    monkeypatch.setattr(build_bridge, "_BUILD_DIR", tmp_path)
    (tmp_path / "CMakeCache.txt").write_text("LLVM_VERSION:STRING=21\n"
                                             "LLVM_DIR:PATH=/opt/llvm-21/lib/cmake/llvm\n")

    assert build_bridge._cache_conflicts({"LLVM_VERSION": "21"}) == []

    conflicts = build_bridge._cache_conflicts({"LLVM_VERSION": "22"})
    assert len(conflicts) == 1 and "LLVM_VERSION" in conflicts[0], conflicts
