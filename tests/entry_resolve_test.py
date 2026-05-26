"""Resolve a plain Fortran procedure name to its mangled flang symbol (and
back), so callers pass ``solve_nh`` instead of ``_QMmo_solve_nonhydroPsolve_nh``.
"""
from pathlib import Path

import pytest

from dace_fortran.emit_hlfir import demangle_entry, resolve_entry


def test_demangle_entry():
    assert demangle_entry("_QMmo_solve_nonhydroPsolve_nh") == "mo_solve_nonhydro::solve_nh"
    assert demangle_entry("_QPadd_array") == "add_array"
    assert demangle_entry("_QMmo_xFcompute") == "mo_x::compute"


def test_resolve_passthrough_for_mangled():
    assert resolve_entry("_QMmo_xPfoo", []) == "_QMmo_xPfoo"


def _src(tmp_path, name, text):
    p = tmp_path / f"{name}.f90"
    p.write_text(text)
    return p


def test_resolve_module_subroutine(tmp_path):
    s = _src(tmp_path, "m", """
module mo_solve_nonhydro
  implicit none
contains
  subroutine solve_nh(a)
    real, intent(inout) :: a(:)
  end subroutine solve_nh
end module mo_solve_nonhydro
""")
    assert resolve_entry("solve_nh", [s]) == "_QMmo_solve_nonhydroPsolve_nh"


def test_resolve_free_subroutine(tmp_path):
    s = _src(tmp_path, "f", "subroutine foo(n)\n  integer :: n\nend subroutine foo\n")
    assert resolve_entry("foo", [s]) == "_QPfoo"


def test_resolve_skips_interface_blocks(tmp_path):
    # An ``interface`` declaration of ``bar`` must not count as a definition.
    s = _src(tmp_path, "i", """
module mo_a
  interface
    subroutine bar(x)
      real :: x
    end subroutine bar
  end interface
contains
  subroutine baz()
  end subroutine baz
end module mo_a
""")
    with pytest.raises(ValueError, match="no subroutine"):
        resolve_entry("bar", [s])
    assert resolve_entry("baz", [s]) == "_QMmo_aPbaz"


def test_resolve_ambiguous_needs_qualifier(tmp_path):
    s1 = _src(tmp_path, "a", "module mo_a\ncontains\nsubroutine run()\nend subroutine run\nend module mo_a\n")
    s2 = _src(tmp_path, "b", "module mo_b\ncontains\nsubroutine run()\nend subroutine run\nend module mo_b\n")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_entry("run", [s1, s2])
    # module::proc disambiguates.
    assert resolve_entry("mo_b::run", [s1, s2]) == "_QMmo_bPrun"
