"""Unit tests for the pure helpers in ``dace_fortran.build_bridge``.

Covers only ``_python_cmake_hints`` (sysconfig-derived cmake hints, the
venv/pyenv fix) and ``needs_build`` (source-vs-``.so`` mtime gate); the
cmake/LLVM shell-out paths are exercised by CI's build step, not mocked here.
"""
import os
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
