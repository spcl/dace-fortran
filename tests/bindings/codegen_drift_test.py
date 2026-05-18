"""Drift gate -- ``build_fortran_library`` must refuse to emit/link a
Fortran binding when the SDFG's live arglist has drifted from its
attached ``FrozenSignature``.

The drift contract used to live in a ``dace/codegen/codegen.py`` hook;
it now lives in the dace-fortran ``build_fortran_library`` entrypoint
(dace-core ``compile`` / ``generate_code`` stay vanilla).  This module
asserts the gate fires at that new layer.

``build_fortran_library`` runs ``frozen.verify_against(sdfg)`` *before*
compiling or emitting anything, so the drift cases raise on the
synthetic SDFGs here without needing a real ``OriginalInterface`` /
``FlattenPlan`` (the gate is reached first).  The happy-path / unpinned
cases assert the gate does not false-positive -- a later build step may
fail on the synthetic kernel, but never with ``SignatureDriftError``.
"""

import dace
import pytest

from dace_fortran.bindings import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
    build_fortran_library,
)


def _demo_sdfg() -> dace.SDFG:
    """Small SDFG: ``a`` and ``b`` as two non-transient float64
    arrays + one free symbol ``n``."""
    sdfg = dace.SDFG("demo")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    return sdfg


def _pin(sdfg: dace.SDFG) -> FrozenSignature:
    fs = FrozenSignature(
        entry="demo",
        mangled="_QPdemo",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="a",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("n", ),
                      intent="in"),
            FrozenArg(fortran_name="b",
                      sdfg_name="b",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("n", ),
                      intent="inout"),
        ),
        free_symbols=("n", ),
    )
    sdfg._frozen_signature = fs
    return fs


def _assert_gate_passes(sdfg, tmp_path):
    """``build_fortran_library`` must not raise drift on ``sdfg``.

    A later emit/link step legitimately fails on these synthetic
    kernels (no real interface/plan); only ``SignatureDriftError``
    indicates the gate misfired.
    """
    try:
        build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path))
    except SignatureDriftError:
        pytest.fail("drift gate misfired on an undrifted SDFG")
    except Exception:
        pass  # synthetic kernel: post-gate failure expected & irrelevant


def test_build_library_honours_frozen_signature_happy_path(tmp_path):
    """When the SDFG hasn't drifted, the drift gate passes."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    sdfg.add_state("s0", is_start_block=True)
    _assert_gate_passes(sdfg, tmp_path)


def test_build_library_raises_on_arg_removal(tmp_path):
    """Transformation dropped array ``b`` -> drift -> raise."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    sdfg.add_state("s0", is_start_block=True)
    del sdfg.arrays["b"]
    with pytest.raises(SignatureDriftError):
        build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path))


def test_build_library_raises_on_dtype_change(tmp_path):
    """Transformation changed ``a`` to float32 -> drift -> raise."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    sdfg.add_state("s0", is_start_block=True)
    sdfg.arrays["a"].dtype = dace.float32
    with pytest.raises(SignatureDriftError):
        build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path))


def test_build_library_requires_a_pinned_sdfg(tmp_path):
    """``build_fortran_library`` is the dace-fortran-only binding path
    and needs a ``build()``-pinned SDFG.  The old "plain SDFGs are
    unaffected" guarantee is now structural: dace-core ``codegen`` is
    vanilla (no drift hook), so a plain SDFG simply has no contract to
    apply.  Calling the binding builder on an unpinned SDFG is a clear
    usage error  --  not a silent pass, not a drift error."""
    sdfg = _demo_sdfg()
    sdfg.add_state("s0", is_start_block=True)
    assert not hasattr(sdfg, "_frozen_signature") or sdfg._frozen_signature is None
    with pytest.raises(ValueError, match="_frozen_signature"):
        build_fortran_library(sdfg, iface=None, plan=None, out_dir=str(tmp_path))
