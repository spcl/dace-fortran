"""A module scalar is read from its module, never off an array that names it as an extent.

QE's ``vexx_bp_k_gpu`` (USPP path) takes ``becpsi % r`` with SDFG shape ``(nkb, nbnd)``; on the k-point path the
caller allocates it ``(1, 1)``. A binding that sourced ``nkb = size(becpsi % r, 1)`` passed ``nkb = 1`` to an SDFG that
sizes every ``nkb`` buffer by it, and the kernel overflowed them.
"""

import pytest

from dace_fortran.bindings.block_builders import _build_symbol_assigns
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, ModuleOrigin


def becpsi_signature(nkb_origin: dict) -> FrozenSignature:
    becpsi_r = FrozenArg(
        fortran_name="becpsi_r",
        sdfg_name="becpsi_r",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=2,
        shape=("nkb", "1"),
    )
    return FrozenSignature(
        entry="k", mangled="_QPk", args=(becpsi_r,), free_symbols=("nkb",), module_symbol_origins=nkb_origin
    )


def assigns(frozen: FrozenSignature) -> list[str]:
    becpsi_r = OriginalArg(name="becpsi_r", fortran_type="real(c_double)", rank=2, shape=(":", ":"), intent="in")
    return _build_symbol_assigns(frozen, FlattenPlan(), set(), OriginalInterface(entry="k", args=(becpsi_r,)))


def test_module_scalar_is_read_from_its_module_not_from_an_array_extent():
    lines = assigns(becpsi_signature({"nkb": ModuleOrigin("uspp", "nkb")}))

    assert "    nkb = int(nkb__mod, c_int)" in lines
    assert not any(line.lstrip().startswith("nkb =") and "size(" in line for line in lines)


def test_symbol_of_no_module_still_takes_the_array_extent():
    lines = assigns(becpsi_signature({}))

    assert "    nkb = int(size(becpsi_r, dim=1), c_int)" in lines


def test_a_free_symbol_with_no_source_is_a_build_error():
    frozen = FrozenSignature(entry="k", mangled="_QPk", args=(), free_symbols=("nij",))

    with pytest.raises(ValueError, match="no source for the free symbol 'nij'"):
        assigns(frozen)


if __name__ == "__main__":
    test_module_scalar_is_read_from_its_module_not_from_an_array_extent()
    test_symbol_of_no_module_still_takes_the_array_extent()
    test_a_free_symbol_with_no_source_is_a_build_error()
