"""Preprocessor (cpp) handling: the pipeline must run flang's preprocessor
before emitting HLFIR and feed it the build's ``-D`` / ``-I`` flags.

Three routes, all exercised here:

* **cmake** -- ``compile_commands.json`` records the compile as a
  ``"command"`` string; ``-D`` / ``-I`` are parsed out of it.
* **automake / bear** -- the same artefact records an ``"arguments"``
  list instead; the parser handles both forms.
* **no build system** -- ``build_sdfg(..., defines=[...])`` passes the
  cpp config explicitly (and the source's ``#ifdef`` branch is selected
  accordingly, proving cpp ran before HLFIR emission).
"""
import json
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang
from dace_fortran.emit_hlfir import _parse_compile_commands

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_parse_compile_commands_cmake_command_string(tmp_path):
    """cmake/ninja form: a single ``command`` string -- ``-I`` / ``-D``
    (both ``-Dx`` and ``-D x`` spellings) are extracted in order."""
    cc = tmp_path / "compile_commands.json"
    src = tmp_path / "k.f90"
    src.write_text("subroutine k()\nend subroutine k\n")
    cc.write_text(json.dumps([{
        "directory": str(tmp_path),
        "command": f"flang-new-21 -cpp -I/opt/inc -DUSE_DBL -D NPROMA=8 -c {src}",
        "file": str(src),
    }]))
    parsed = _parse_compile_commands(cc)
    assert len(parsed) == 1
    _, includes, defines = parsed[0]
    assert "/opt/inc" in includes
    assert "USE_DBL" in defines and "NPROMA=8" in defines


def test_parse_compile_commands_automake_arguments_list(tmp_path):
    """bear/autotools form: an ``arguments`` list -- same extraction."""
    cc = tmp_path / "compile_commands.json"
    src = tmp_path / "k.f90"
    src.write_text("subroutine k()\nend subroutine k\n")
    cc.write_text(json.dumps([{
        "directory": str(tmp_path),
        "arguments": ["mpif90", "-cpp", "-I", "/usr/inc", "-D__ICON__",
                      "-c", str(src)],
        "file": str(src),
    }]))
    parsed = _parse_compile_commands(cc)
    assert len(parsed) == 1
    _, includes, defines = parsed[0]
    assert "/usr/inc" in includes
    assert "__ICON__" in defines


_GATED_SRC = """
module kern_mod
  use iso_c_binding
  implicit none
contains
subroutine kern(n, x)
  implicit none
  integer(c_int), intent(in) :: n
#ifdef USE_DBL
  real(c_double), intent(inout) :: x(n)
#else
  real(c_float), intent(inout) :: x(n)
#endif
  integer :: i
  do i = 1, n
     x(i) = x(i) + 1
  end do
end subroutine kern
end module kern_mod
"""


def test_defines_select_cpp_branch_without_build_system(tmp_path):
    """No build system: ``build_sdfg(defines=[...])`` feeds flang's cpp.
    The ``#ifdef USE_DBL`` branch picks real(8) vs real(4), so the SDFG
    arg dtype proves the define reached the preprocessor before HLFIR."""
    sdfg_dbl = build_sdfg(_GATED_SRC, tmp_path / "dbl", name="kern",
                          entry="kern_mod::kern", defines=["USE_DBL"]).build()
    assert str(sdfg_dbl.arrays["x"].dtype) == "double"

    sdfg_flt = build_sdfg(_GATED_SRC, tmp_path / "flt", name="kern",
                          entry="kern_mod::kern").build()
    assert str(sdfg_flt.arrays["x"].dtype) == "float"
