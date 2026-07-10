"""Regression: a top-level POINTER entry arg must be declared ``pointer`` (not
``target``) in the generated wrapper head so ``associated(<arg>)`` compiles
(commit aff27ae).

An arg the original entry declared POINTER -- ICON ocean
``REAL(8), POINTER, INTENT(INOUT) :: vn(:,:,:)`` / ``veloc_adv_horz_e(:,:,:)`` --
that the kernel body tests with ``ASSOCIATED(...)`` carries a presence guard
``<name>_allocated`` in the SDFG free symbols.  ``build_wrapper_head`` emits its
presence fold as ``merge(1, 0, associated(<name>))``, which is legal ONLY on a
POINTER dummy.  The pre-fix ``_outer_decl`` declared every outer array dummy
plain ``target``, so gfortran rejected the fold with ``'pointer' argument of
'associated' intrinsic must be a POINTER`` and the binding failed to compile.

The fix flips the outer decl to ``pointer`` exactly for args that carry a
``<name>_allocated`` guard (``c_loc`` accepts an associated POINTER so the body
is unaffected).  This drives a small synthetic kernel with that shape rather
than the heavy real ``ocean_veloc_adv`` case: build the SDFG, emit + gfortran-
link the wrapper via ``build_fortran_library`` (a green build IS the ``associated``
fold compiling), and string-inspect the wrapper decl.
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

# A top-level ``REAL(8), POINTER, INTENT(INOUT)`` array arg the kernel gates on
# ``ASSOCIATED(...)`` -- the minimal shape that mints a ``v_allocated`` presence
# guard whose wrapper fold is ``merge(1, 0, associated(v))``.
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
    # Short SDFG name via make_builder directly (the test-util xdist suffix
    # would blow Fortran's 63-char identifier limit on ``<name>_dace_finalize``).
    sdfg = make_builder(_SRC, entry="kern", name="ptrarg", out_dir=str(tmp_path / "sdfg")).build()

    # Precondition: the POINTER arg's ``ASSOCIATED`` test lowered to a presence
    # guard free symbol (otherwise there is no fold to declare ``pointer`` for).
    assert any(str(s) == "v_allocated" for s in sdfg.free_symbols), \
        f"expected a v_allocated presence guard; free symbols = {sorted(str(s) for s in sdfg.free_symbols)}"

    # build_fortran_library emits the wrapper AND gfortran-links it -- a
    # successful build is the gfortran-compiles-the-fold assertion (a ``target``
    # decl would fail with the 'must be a POINTER' error).
    lib = build_fortran_library(sdfg, out_dir=str(tmp_path / "lib"))
    assert lib.so_path.is_file(), "wrapper library did not link"

    wrapper = lib.bindings_f90.read_text()
    # The outer dummy is declared POINTER, not TARGET.
    assert "pointer :: v(:,:,:)" in wrapper, \
        f"POINTER arg not declared 'pointer' in wrapper head:\n{wrapper}"
    assert "target :: v(:,:,:)" not in wrapper, \
        "POINTER arg wrongly declared 'target' (associated() would not compile)"
    # And the presence fold is the ``associated`` form (legal only on a POINTER).
    assert "merge(1, 0, associated(v))" in wrapper, \
        f"expected an associated() presence fold in the wrapper:\n{wrapper}"
