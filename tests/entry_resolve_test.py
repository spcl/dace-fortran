"""Resolve a Fortran procedure name (``solve_nh`` / ``module::proc``) against Fortran sources."""

import pytest

from dace_fortran.entry_names import QualifiedEntry, resolve_entry_name

_SOLVE_NH = """
module mo_solve_nonhydro
  implicit none
contains
  subroutine solve_nh(a)
    real, intent(inout) :: a(:)
  end subroutine solve_nh
end module mo_solve_nonhydro
"""


def test_resolve_module_subroutine():
    assert resolve_entry_name([_SOLVE_NH], "solve_nh") == QualifiedEntry("mo_solve_nonhydro", "solve_nh")


def test_resolve_free_function():
    src = "integer function foo(n) ! the entry\n  integer :: n\n  foo = n\nend function foo\n"
    assert resolve_entry_name([src], "foo") == QualifiedEntry(None, "foo")


def test_resolve_skips_interface_blocks():
    # An ``interface`` declaration of ``bar`` must not count as a definition.
    src = """
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
"""
    with pytest.raises(ValueError, match="no procedure"):
        resolve_entry_name([src], "bar")
    assert resolve_entry_name([src], None) == QualifiedEntry("mo_a", "baz")


def test_resolve_ambiguous_needs_qualifier():
    a = "module mo_a\ncontains\nsubroutine run()\nend subroutine run\nend module mo_a\n"
    b = "module mo_b\ncontains\nsubroutine run()\nend subroutine run\nend module mo_b\n"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_entry_name([a, b], "run")
    with pytest.raises(ValueError, match="multiple procedures"):
        resolve_entry_name([a, b], None)
    # module::proc disambiguates.
    assert resolve_entry_name([a, b], "mo_b::run") == QualifiedEntry("mo_b", "run")


if __name__ == "__main__":
    test_resolve_module_subroutine()
    test_resolve_free_function()
    test_resolve_skips_interface_blocks()
    test_resolve_ambiguous_needs_qualifier()
