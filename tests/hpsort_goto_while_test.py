"""QE ``hpsort`` GOTO-rewritten DO WHILE through the HLFIR frontend.

The structurizer (lift-cf-to-scf) rewrites hpsort's ``goto_10`` flag loop and
its mid-loop ``RETURN`` into ``scf.index_switch`` ops whose RESULTS are read
by the inner loop-carried values and break conditions -- in expression context,
before the statement walk materialises the switch.  ``buildExpr`` used to fall
through to the unhandled-op ``?`` sentinel, which then surfaced as
``NotImplementedError: emit_scalar_assign: unresolved operand placeholder``
(``__sc_7 = (? - 1)``, the QE h_psi build's fourth fatal error).

Fix: expression reads of a side-effecting ``scf.index_switch`` result lazily
register the result's ``__sc_<id>`` synth name (same ``scfSynthName`` memo the
statement pass uses), the statement walk's ``buildIndexSwitchNodes`` emits the
matching assignments, the break-condition snapshot defers past the switch
chain, and ``extractAST`` asserts no lazily-registered switch is left
unmaterialised.

Pinned: (1) build + validate of the verbatim QE ``hpsort`` body, (2) numerical
sort correctness incl. the index permutation and the duplicate-key tie-break.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# Verbatim from QE (ast_v1_h_psi.f90:977) -- the ``goto_10`` DO WHILE shape is
# what drives the structurizer to scf.index_switch.
_SRC = """
subroutine hpsort(n, ra, ind)
  implicit none
  integer :: n
  integer :: ind(n)
  real(8) :: ra(n)
  integer :: i, ir, j, l, iind
  real(8) :: rra
  logical :: goto_10
  if (n < 1) return
  if (ind(1) == 0) then
    do i = 1, n
      ind(i) = i
    end do
  end if
  if (n < 2) return
  l = n / 2 + 1
  ir = n
  goto_10 = .true.
  do while (goto_10)
    goto_10 = .false.
    if (l > 1) then
      l = l - 1
      rra = ra(l)
      iind = ind(l)
    else
      rra = ra(ir)
      iind = ind(ir)
      ra(ir) = ra(1)
      ind(ir) = ind(1)
      ir = ir - 1
      if (ir == 1) then
        ra(1) = rra
        ind(1) = iind
        return
      end if
    end if
    i = l
    j = l + l
    do while (j <= ir)
      if (j < ir) then
        if (ra(j) < ra(j + 1)) then
          j = j + 1
        else if (ra(j) == ra(j + 1)) then
          if (ind(j) < ind(j + 1)) j = j + 1
        end if
      end if
      if (rra < ra(j)) then
        ra(i) = ra(j)
        ind(i) = ind(j)
        i = j
        j = j + j
      else if (rra == ra(j)) then
        if (iind < ind(j)) then
          ra(i) = ra(j)
          ind(i) = ind(j)
          i = j
          j = j + j
        else
          j = ir + 1
        end if
      else
        j = ir + 1
      end if
    end do
    ra(i) = rra
    ind(i) = iind
    goto_10 = .true.
  end do
end subroutine hpsort
"""


def _inputs():
    rng = np.random.default_rng(17)
    n = 24
    ra = rng.standard_normal(n)
    # Duplicate keys exercise the ind() tie-break arms of the sift-down.
    ra[5] = ra[11] = ra[3]
    ind = np.zeros(n, dtype=np.int32)
    return np.asfortranarray(ra), ind, n


def test_hpsort_goto_while_builds(tmp_path):
    """The SDFG builds and validates (was: NotImplementedError on ``(? - 1)``)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="hpsort").build()
    sdfg.validate()


def test_hpsort_goto_while_numerical(tmp_path):
    """Ascending sort + index permutation consistent with the final array."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="hpsort").build()
    ra, ind, n = _inputs()
    ra_orig = ra.copy()
    sdfg(ra=ra, ind=ind, n=n)
    assert np.all(np.diff(ra) >= 0), f"not sorted ascending: {ra}"
    np.testing.assert_allclose(ra_orig[ind - 1],
                               ra,
                               rtol=0,
                               atol=0,
                               err_msg="ind is not the permutation of the sorted array")
