"""Regression: module-level allocatables get TARGET; pointers and derived-type components do not."""
from dace_fortran.bindings.build_fortran_library import _ensure_target_on_module_deferred_arrays

_SRC = """
module mo_types
  type :: t_coeffs
    real(kind=8), pointer :: ptr_arr(:, :)
    real(kind=8), allocatable :: alloc_arr(:)
  end type t_coeffs
  real(kind=8), pointer :: mod_ptr(:, :)
  real(kind=8), allocatable :: mod_alloc(:)
end module mo_types
"""


def test_target_added_to_module_allocatables_only():
    patched = _ensure_target_on_module_deferred_arrays(_SRC)
    lines = [l.strip() for l in patched.splitlines()]
    assert "real(kind=8), pointer :: mod_ptr(:, :)" in lines
    assert "real(kind=8), allocatable, target :: mod_alloc(:)" in lines
    assert "real(kind=8), pointer :: ptr_arr(:, :)" in lines
    assert "real(kind=8), allocatable :: alloc_arr(:)" in lines
    assert "target :: ptr_arr" not in patched
    assert "target :: alloc_arr" not in patched
    assert "pointer, target" not in patched
