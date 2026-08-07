"""QE ``xclib_dft_is`` pattern: ``LEN_TRIM`` (fir.iterate_while) + char
normalisation loop + ``SELECT CASE`` on a string.

``LEN_TRIM(what)`` on an assumed-length CHARACTER dummy lowers to
``fir.iterate_while`` bounded by ``unboxchar#1 - 1`` -- the string-LENGTH half
of ``fir.unboxchar``, which the bridge's expression renderers did not handle,
surfacing as ``NotImplementedError: emit_scalar_assign: unresolved operand
placeholder ``?`` in ``__sc_7 = (? - 1)`` (QE h_psi build, fourth fatal's
residual after the scf.index_switch fix).  Character data is outside the
bridge's numerical-equivalence contract (hlfir-strip-character-runtime folds
the string compares); the length renders as a benign constant so the residual
scaffolding still builds.

Pinned: build + validate (the numerical outcome is the folded no-error path:
``xclib_dft_is('HYBRID')`` collapses to the first-case arm by design of the
strip pass, so only the surrounding integer plumbing is checked).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Mirrors ast_v1_h_psi.f90:88-133 (xclib_dft_is + capital), wrapped so the
# SELECT CASE result lands in an integer output.
_SRC = """
subroutine probe_xclib(what, res)
  implicit none
  character(len=*), intent(in) :: what
  integer, intent(out) :: res
  character(len=15) :: cwhat
  integer :: i, ln
  ln = len_trim(what)
  do i = 1, ln
    cwhat(i:i) = capital(what(i:i))
  end do
  select case (cwhat(1:ln))
  case ('GRADIENT')
    res = 1
  case ('META')
    res = 2
  case ('HYBRID')
    res = 3
  case default
    res = 0
  end select
contains
  function capital(in_char)
    implicit none
    character(len=1), intent(in) :: in_char
    character(len=1) :: capital
    character(len=26), parameter :: lower = 'abcdefghijklmnopqrstuvwxyz'
    character(len=26), parameter :: upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    integer :: k
    do k = 1, 26
      if (in_char == lower(k:k)) then
        capital = upper(k:k)
        return
      end if
    end do
    capital = in_char
  end function capital
end subroutine probe_xclib
"""


def test_len_trim_select_case_builds(tmp_path):
    """The SDFG builds and validates (was: NotImplementedError on ``(? - 1)``)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_xclib", entry="probe_xclib").build()
    sdfg.validate()


def test_len_trim_select_case_runs(tmp_path):
    """The built SDFG calls without error and writes an in-range result."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_xclib", entry="probe_xclib").build()
    what = np.frombuffer(b"HYBRID         "[:15], dtype=np.uint8).copy()
    res = np.zeros(1, dtype=np.int32)
    kwargs = {"res": res}
    # Character dummy: pass bytes if the SDFG exposes it, else just res.
    if "what" in sdfg.arrays:
        kwargs["what"] = what
    sdfg(**kwargs)
    assert 0 <= int(res[0]) <= 3, f"res out of SELECT CASE range: {res[0]}"
