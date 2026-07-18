"""Regression: a top-level POINTER entry arg must be declared ``pointer`` (not
``target``) in the generated wrapper head so ``associated(<arg>)`` compiles
(commit aff27ae).

A POINTER arg tested with ``ASSOCIATED(...)`` carries a ``<name>_allocated``
presence guard; the wrapper's ``merge(1, 0, associated(<name>))`` fold is
legal only on a POINTER dummy -- pre-fix, outer array dummies were declared
plain ``target`` and gfortran rejected the fold. Builds + gfortran-links a
synthetic kernel via ``build_fortran_library`` and inspects the wrapper decl.
"""
import shutil

import pytest

from _util import have_flang
from dace_fortran.bindings import build_fortran_library
from dace_fortran.build import make_builder

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Minimal shape that mints a v_allocated presence guard: POINTER arg gated on ASSOCIATED(...).
_SRC = """
subroutine kern(v, n)
  integer, intent(in) :: n
  real(8), pointer, intent(inout) :: v(:, :, :)
  integer :: i
  if (associated(v)) then
    do i = 1, n
      v(i, 1, 1) = v(i, 1, 1) * 2.0d0
    end do
  end if
end subroutine kern
"""


def test_pointer_outer_arg_declared_pointer_and_wrapper_compiles(tmp_path):
    """The wrapper declares the POINTER arg ``pointer`` (not ``target``) and the
    ``associated(...)`` presence fold gfortran-compiles."""
    # Short SDFG name via make_builder: xdist suffix would blow Fortran's 63-char identifier limit.
    sdfg = make_builder(_SRC, entry="kern", name="ptrarg", out_dir=str(tmp_path / "sdfg")).build()

    # Precondition: ASSOCIATED lowered to a presence guard free symbol (else no fold to declare pointer for).
    assert any(str(s) == "v_allocated" for s in sdfg.free_symbols), \
        f"expected a v_allocated presence guard; free symbols = {sorted(str(s) for s in sdfg.free_symbols)}"

    # green build = fold gfortran-compiles (a target decl would fail with 'must be a POINTER').
    lib = build_fortran_library(sdfg, out_dir=str(tmp_path / "lib"))
    assert lib.so_path.is_file(), "wrapper library did not link"

    wrapper = lib.bindings_f90.read_text()
    assert "pointer :: v(:,:,:)" in wrapper, \
        f"POINTER arg not declared 'pointer' in wrapper head:\n{wrapper}"
    assert "target :: v(:,:,:)" not in wrapper, \
        "POINTER arg wrongly declared 'target' (associated() would not compile)"
    assert "merge(1, 0, associated(v))" in wrapper, \
        f"expected an associated() presence fold in the wrapper:\n{wrapper}"
