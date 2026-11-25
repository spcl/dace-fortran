# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The caller declares the block dimensions of assumed-shape arrays small, so DaCe's offload sees ICON's jb."""

import dace
import pytest

from dace_fortran.pipelines import declare_small_dimensions


def blocked() -> dace.SDFG:
    """``a(:, :)`` and ``b(:, :, :)`` as dace-fortran builds them: each extent its own ``<array>_d<k>`` symbol."""
    sdfg = dace.SDFG("blocked")
    for name, rank in (("a", 2), ("b", 3)):
        extents = [f"{name}_d{k}" for k in range(rank)]
        for extent in extents:
            sdfg.add_symbol(extent, dace.int64)
        sdfg.add_array(name, extents, dace.float64)
    return sdfg


def test_the_named_block_dimensions_are_small_and_no_other():
    sdfg = blocked()
    assert declare_small_dimensions(sdfg, {"a": [1], "b": [2]}) == {"a_d1", "b_d2"}
    assert {"a_d1", "b_d2"} <= set(sdfg.facts().small)
    assert not {"a_d0", "b_d0", "b_d1"} & set(sdfg.facts().small)


def test_an_unknown_array_is_an_error():
    with pytest.raises(KeyError):
        declare_small_dimensions(blocked(), {"c": [0]})


if __name__ == "__main__":
    test_the_named_block_dimensions_are_small_and_no_other()
    test_an_unknown_array_is_an_error()
