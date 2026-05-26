"""Frozen-signature drift, exercised through the *real* generation path.

``frozen_signature_test.py`` checks ``FrozenSignature.verify_against``
against hand-built synthetic SDFGs.  This module closes the gap the
contract actually guards: a signature snapshotted by
``SDFGBuilder.build()`` (``sdfg._frozen_signature``), then drifted by a
post-build SDFG mutation, must be rejected by the dace-fortran binding
builder -- not silently linked into a Fortran library that disagrees
with the wrapper.

The drift gate used to be a ``dace/codegen/codegen.py`` hook; it now
lives in ``build_fortran_library``, which runs ``verify_against``
*before* compiling or emitting anything.  So the negative cases raise
on the real bridge SDFG without a hand-built ``OriginalInterface`` /
``FlattenPlan`` (the gate is reached first).  The positive control
asserts the gate does not false-positive: a later emit/link step
legitimately fails on the stub interface, but never with
``SignatureDriftError``.
"""

from pathlib import Path

import dace
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import SignatureDriftError, build_fortran_library

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
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
"""


def _build(tmp_path: Path):
    """Build the ``axpy`` SDFG through the bridge.

    :param tmp_path: pytest scratch dir.
    :returns: built SDFG with ``_frozen_signature`` auto-attached.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="axpy", entry="axpy").build()
    sdfg.validate()
    assert getattr(sdfg, "_frozen_signature",
                   None) is not None, ("SDFGBuilder.build() must auto-attach a frozen signature")
    return sdfg


def _build_lib(sdfg, tmp_path: Path):
    return build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path / "lib"))


def test_untouched_sdfg_passes_drift_gate(tmp_path: Path):
    """Positive control: a build nobody mutated still matches its own
    snapshot, so the drift gate in ``build_fortran_library`` passes
    (a later emit step fails on the stub interface, never with drift)."""
    sdfg = _build(tmp_path)
    try:
        _build_lib(sdfg, tmp_path)
    except SignatureDriftError:
        pytest.fail("drift gate misfired on an undrifted bridge SDFG")
    except Exception:
        pass  # stub iface/plan: post-gate failure expected & irrelevant


def test_added_arg_after_freeze_raises(tmp_path: Path):
    """A transformation that adds a non-transient array after the
    signature was frozen drifts ``sdfg.arglist()``; the builder must
    reject it instead of linking a library the binding can't call."""
    sdfg = _build(tmp_path)
    sdfg.add_array("z_drift", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    with pytest.raises(SignatureDriftError, match="signature drift"):
        _build_lib(sdfg, tmp_path)


def test_dtype_change_after_freeze_raises(tmp_path: Path):
    """Silently retyping an existing arg (float64 -> float32) is the
    most dangerous drift -- order/count look identical.  The per-arg
    dtype guard in ``verify_against`` must catch it."""
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
