"""Section reductions appearing in an ``IF`` condition.

Bridge gap (closed 2026-06-02): ``IF (k >= MINVAL(kmin(iv, :)))`` and its
``MAXVAL`` / ``SUM`` / ``PRODUCT`` siblings used to emerge as
``(k >= ?)`` because ``buildExprWithSubscripts`` had no handler for
``hlfir.minval`` etc., and the ``?`` placeholder propagated into an
interstate-edge assignment that DaCe's symbolic engine then rejected
at specialise time (``SyntaxError: invalid syntax`` while parsing
``(k >= ?)``).

The fix unfolds a constant-extent section reduction inline at the
condition site (``min(min(min(arr[i, 0], arr[i, 1]), arr[i, 2]), arr[i, 3])``
for the Graupel ``MINVAL(kmin(iv, 1:4))`` pattern), keeping the
emitted condition parseable.

These tests anchor the pattern so future regressions in the same area
surface as test failures rather than silent ``?`` placeholders.
"""

import pytest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src_text: str, tmp_path, name: str):
    """Build the SDFG without running the numerical pipeline."""
    sdfg = build_sdfg(src_text, tmp_path / "sdfg", name=name).build()
    sdfg.validate()
    return sdfg


def _conditions_with(sdfg, fragment: str):
    """Return interstate-edge assignment values containing ``fragment``."""
    out = []
    for edge in sdfg.all_interstate_edges():
        if not edge.data.assignments:
            continue
        for _, value in edge.data.assignments.items():
            if fragment in value:
                out.append(value)
    return out


_MINVAL_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE minval_run(n, ke, kmin, out)
    INTEGER, INTENT(IN) :: n, ke
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: kmin(n, 4)
    INTEGER :: iv, k
    kmin(:, :) = ke + 1
    DO iv = 1, n
      DO k = 1, ke
        IF (k >= MINVAL(kmin(iv, :))) THEN
          out(iv) = out(iv) + 1.0_real64
        END IF
      END DO
    END DO
  END SUBROUTINE minval_run
END MODULE m
"""


def test_minval_const_extent_section_in_if_condition(tmp_path):
    """``IF (k >= MINVAL(kmin(iv, :)))`` -- ICON ``mo_aes_graupel:341``."""
    sdfg = _build(_MINVAL_SOURCE, tmp_path, "minval_run")
    hits = _conditions_with(sdfg, "min(")
    assert hits, "no nested min(...) emerged from MINVAL section reduction"
    expected = "min(min(min(kmin"
    assert any(expected in h for h in hits), (
        f"MINVAL did not unfold to nested min over a constant-extent section; got: {hits}")


_MAXVAL_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE maxval_run(n, ke, kmax, out)
    INTEGER, INTENT(IN) :: n, ke
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: kmax(n, 4)
    INTEGER :: iv, k
    kmax(:, :) = 0
    DO iv = 1, n
      DO k = 1, ke
        IF (k <= MAXVAL(kmax(iv, :))) THEN
          out(iv) = out(iv) + 1.0_real64
        END IF
      END DO
    END DO
  END SUBROUTINE maxval_run
END MODULE m
"""


def test_maxval_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_MAXVAL_SOURCE, tmp_path, "maxval_run")
    hits = _conditions_with(sdfg, "max(")
    assert hits, "no nested max(...) emerged from MAXVAL section reduction"
    assert any("max(max(max(kmax" in h for h in hits), (
        f"MAXVAL did not unfold to nested max; got: {hits}")


_SUM_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE sum_run(n, weights, out)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: weights(n, 3)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (SUM(weights(iv, :)) > 5) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE sum_run
END MODULE m
"""


def test_sum_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_SUM_SOURCE, tmp_path, "sum_run")
    hits = _conditions_with(sdfg, "weights[")
    assert hits, "SUM section reduction did not surface in interstate-edge condition"
    assert any("+" in h and "weights[" in h for h in hits)


_PRODUCT_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE product_run(n, factors, out)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: factors(n, 3)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (PRODUCT(factors(iv, :)) > 8) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE product_run
END MODULE m
"""


def test_product_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_PRODUCT_SOURCE, tmp_path, "product_run")
    hits = _conditions_with(sdfg, "factors[")
    assert hits
    assert any("*" in h and "factors[" in h for h in hits)


_RUNTIME_EXTENT_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE runtime_run(n, m, arr, out)
    INTEGER, INTENT(IN) :: n, m
    INTEGER, INTENT(IN) :: arr(n, m)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (1 >= MINVAL(arr(iv, :))) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE runtime_run
END MODULE m
"""


def test_runtime_extent_section_falls_through(tmp_path):
    """Runtime-extent section ``arr(iv, 1:m)`` cannot unfold inline.

    The bridge should refuse cleanly (no nested ``min(...)`` with literals,
    no ``?`` leaking past); the downstream emitter then renders a clear
    diagnostic.  Either outcome is acceptable for this anchor; what
    must NOT happen is a silent ``?`` placeholder reaching an
    interstate-edge assignment.
    """
    try:
        sdfg = _build(_RUNTIME_EXTENT_SOURCE, tmp_path, "runtime_run")
    except Exception:
        return  # bridge refused, that's fine
    for value in _conditions_with(sdfg, "?"):
        if "?" in value:
            pytest.fail(f"runtime-extent reduction left a ?-placeholder: {value!r}")
