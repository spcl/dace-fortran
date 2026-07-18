"""Frozen-signature drift, exercised through the *real* generation path.

Complements ``frozen_signature_test.py`` (hand-built SDFGs): here the
signature comes from ``SDFGBuilder.build()``, drifted by a post-build
mutation, and must be rejected by ``build_fortran_library`` before any
compile/emit. Positive control checks the gate doesn't false-positive.
"""

from pathlib import Path

import dace
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import SignatureDriftError, build_fortran_library

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module axpy_mod
  implicit none
contains
  subroutine axpy(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: a
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = a * x(i) + y(i)
    end do
  end subroutine axpy
end module axpy_mod
"""


def _build(tmp_path: Path):
    """Build the ``axpy`` SDFG through the bridge; returns SDFG with ``_frozen_signature`` auto-attached."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="axpy", entry="axpy_mod::axpy").build()
    sdfg.validate()
    assert getattr(sdfg, "_frozen_signature",
                   None) is not None, ("SDFGBuilder.build() must auto-attach a frozen signature")
    return sdfg


def _build_lib(sdfg, tmp_path: Path):
    return build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path / "lib"))


def test_untouched_sdfg_passes_drift_gate(tmp_path: Path):
    """Positive control: an unmutated build passes the drift gate (a later
    emit step may fail on the stub interface, but never with drift)."""
    sdfg = _build(tmp_path)
    try:
        _build_lib(sdfg, tmp_path)
    except SignatureDriftError:
        pytest.fail("drift gate misfired on an undrifted bridge SDFG")
    except Exception:
        pass  # stub iface/plan: post-gate failure expected & irrelevant


def test_added_arg_after_freeze_raises(tmp_path: Path):
    """Adding a non-transient array after freeze drifts ``sdfg.arglist()``;
    the builder must reject it instead of linking an uncallable library."""
    sdfg = _build(tmp_path)
    sdfg.add_array("z_drift", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    with pytest.raises(SignatureDriftError, match="signature drift"):
        _build_lib(sdfg, tmp_path)


def test_dtype_change_after_freeze_raises(tmp_path: Path):
    """Retyping an existing arg (float64 -> float32) is the most dangerous
    drift (order/count look identical); the per-arg dtype guard must catch it."""
    sdfg = _build(tmp_path)
    sdfg.arrays["y"].dtype = dace.float32
    with pytest.raises(SignatureDriftError, match="dtype"):
        _build_lib(sdfg, tmp_path)


def test_extra_free_symbol_after_freeze_raises(tmp_path: Path):
    """A pass that introduces a new *used* free symbol changes the
    SDFG's callable surface; the free-symbol set guard must fire."""
    sdfg = _build(tmp_path)
    sdfg.add_symbol("drift_sym", dace.int64)
    sink = sdfg.sink_nodes()[0]
    tail = sdfg.add_state("drift_tail")
    sdfg.add_edge(sink, tail, dace.InterstateEdge(condition="drift_sym > 0"))
    with pytest.raises(SignatureDriftError):
        _build_lib(sdfg, tmp_path)
