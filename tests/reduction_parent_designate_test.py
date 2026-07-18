"""Mode-C reductions (COUNT/ANY/ALL over a comparison mask) when the source is a
*section* of a higher-rank array, e.g. ``m(:, pos1)``.

The elemental walker materialises the comparison into a transient mask; the
read-access collector must walk the parent designate chain
(``expandDesignateChain``) so accesses match the underlying array's full rank
-- without it, ``m(:, pos1)`` records rank-1 ``index_exprs=['ei0']`` against
rank-2 ``m``, a malformed memlet. Pins against elementals.inc/control_flow.inc
walker divergence reintroducing the mismatch.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_count_parent_designate_scalar_dim(tmp_path: Path):
    """``COUNT(m(:, pos1) .eq. 5)`` -- elemental walks a section whose parent
    designate has a scalar dim; collectReadAccesses must thread it through."""
    src = """
SUBROUTINE count_parent_dg(m, pos1, res)
  integer, dimension(7,4) :: m
  integer :: pos1
  integer, dimension(2) :: res
  res(1) = COUNT(m(:, pos1) .eq. 5)
END SUBROUTINE count_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='count_parent_dg').build()

    m = np.zeros((7, 4), order='F', dtype=np.int32)
    m[2, 1] = 5
    m[5, 1] = 5
    m[3, 2] = 5
    res = np.zeros(2, order='F', dtype=np.int32)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] == 2

    sdfg(m=m, pos1=3, res=res)
    assert res[0] == 1

    sdfg(m=m, pos1=4, res=res)
    assert res[0] == 0


def test_any_parent_designate_scalar_dim(tmp_path: Path):
    """``ANY(m(:, pos1) .gt. 0)`` -- same parent-section shape as the COUNT case;
    pins buildElementalAnyAllReduce's walker."""
    src = """
SUBROUTINE any_parent_dg(m, pos1, res)
  integer, dimension(5,3) :: m
  integer :: pos1
  logical, dimension(2) :: res
  res(1) = ANY(m(:, pos1) .gt. 0)
END SUBROUTINE any_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='any_parent_dg').build()

    m = np.zeros((5, 3), order='F', dtype=np.int32)
    m[2, 1] = 7
    # res is Fortran LOGICAL -- pass np.bool_ so the C ABI dtype matches the SDFG's bool* declaration
    res = np.zeros(2, order='F', dtype=np.bool_)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] != 0

    sdfg(m=m, pos1=3, res=res)
    assert res[0] == 0


def test_all_parent_designate_scalar_dim(tmp_path: Path):
    """``ALL(m(:, pos1) .gt. 0)`` -- Mode-C ALL counterpart; false if any element
    of the parent-designate slice fails the predicate."""
    src = """
SUBROUTINE all_parent_dg(m, pos1, res)
  integer, dimension(4,3) :: m
  integer :: pos1
  logical, dimension(2) :: res
  res(1) = ALL(m(:, pos1) .gt. 0)
END SUBROUTINE all_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='all_parent_dg').build()

    m = np.ones((4, 3), order='F', dtype=np.int32)
    # res is Fortran LOGICAL -- pass np.bool_
    res = np.zeros(2, order='F', dtype=np.bool_)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] != 0

    m[1, 1] = 0
    sdfg(m=m, pos1=2, res=res)
    assert res[0] == 0
