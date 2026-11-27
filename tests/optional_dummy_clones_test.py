# Copyright 2026 ETH Zurich and the dace-fortran authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inlined copies of one OPTIONAL dummy stay separate variables.

ICON's ``exchange_data_mult_mixprec`` is inlined three times in ``solve_nh``; its OPTIONAL
``recv2_dp`` is bound to ``z_rho_e``, ``z_dwdz_dd`` and the POINTER member ``p_nh%prog(nnew)%exner``.
A POINTER actual reaches an OPTIONAL dummy through a present test, so that copy is a variable of its
own, while the other two alias their actuals under the same uniq_name. It took the name of one
sibling's actual and the source of its own: ``z_dwdz_dd`` became a section of ``exner``, its
memset wrote ``exner``, and its extent stayed a free symbol the binding refused.
"""

import numpy as np
import pytest

from tests._util import build_sdfg

SOURCE = """
MODULE clones
CONTAINS
  SUBROUTINE bump(n, a, b)
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(INOUT) :: a(:)
    REAL(8), INTENT(INOUT), OPTIONAL :: b(:)
    a(1:n) = a(1:n) + 1.0d0
    IF (PRESENT(b)) b(1:n) = b(1:n) + 2.0d0
  END SUBROUTINE
END MODULE
SUBROUTINE optional_clones(n, x, y, z)
  USE clones
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: x(n), y(n)
  REAL(8), INTENT(INOUT), TARGET :: z(n)
  REAL(8), POINTER :: p(:)
  REAL(8) :: first(n), last(n)
  first = 0.0d0
  last = 10.0d0
  p => z
  {calls}
  y = first + last
END SUBROUTINE
"""

ORDERS = {
    "pointer_between": ("CALL bump(n, x, first)", "CALL bump(n, x, p)", "CALL bump(n, x, last)"),
    "pointer_first": ("CALL bump(n, x, p)", "CALL bump(n, x, first)", "CALL bump(n, x, last)"),
    "pointer_last": ("CALL bump(n, x, first)", "CALL bump(n, x, last)", "CALL bump(n, x, p)"),
}


@pytest.mark.parametrize("order", ORDERS)
def test_an_optional_dummy_bound_to_a_pointer_and_to_locals_keeps_each_actual(tmp_path, order: str):
    src = SOURCE.replace("{calls}", "\n  ".join(ORDERS[order]))
    sdfg = build_sdfg(src, tmp_path / "sdfg", name=f"optional_clones_{order}", entry="optional_clones").build()
    n = 5
    x = np.zeros(n, order="F")
    y = np.zeros(n, order="F")
    z = np.arange(n, dtype=np.float64)
    # p_allocated is ASSOCIATED(p), which the binding passes.
    sdfg(n=np.int32(n), x=x, y=y, z=z, p_allocated=1)
    np.testing.assert_array_equal(x, np.full(n, 3.0))
    np.testing.assert_array_equal(z, np.arange(n) + 2.0)
    np.testing.assert_array_equal(y, np.full(n, 2.0 + 12.0))


if __name__ == "__main__":
    for order in ORDERS:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            test_an_optional_dummy_bound_to_a_pointer_and_to_locals_keeps_each_actual(Path(tmp), order)
