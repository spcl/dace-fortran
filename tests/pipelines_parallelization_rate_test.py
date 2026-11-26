# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The samples' gate: every parallelizable loop maps, and the map count stays under its bound."""

import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

from dace_fortran.pipelines import assert_parallelized, num_loops, parallelization_rate, sequential_loops


def two_loops() -> dace.SDFG:
    """Two sibling loops that a pipeline has not touched."""
    sdfg = dace.SDFG("two_loops")
    first = LoopRegion("first", "i < 4", "i", "i = 0", "i = i + 1")
    second = LoopRegion("second", "j < 4", "j", "j = 0", "j = j + 1")
    first.add_state(is_start_block=True)
    second.add_state(is_start_block=True)
    sdfg.add_node(first, is_start_block=True)
    sdfg.add_node(second)
    sdfg.add_edge(first, second, dace.InterstateEdge())
    return sdfg


def test_a_loop_left_behind_lowers_the_rate():
    sdfg = two_loops()

    assert num_loops(sdfg) == 2
    assert parallelization_rate(sdfg, loops_before=4) == 0.5


def test_loops_that_must_stay_sequential_do_not_count_against_the_rate():
    assert parallelization_rate(two_loops(), loops_before=4, sequential=2) == 1.0


def test_a_program_with_only_sequential_loops_is_fully_parallelized():
    assert parallelization_rate(two_loops(), loops_before=2, sequential=2) == 1.0


def test_the_gate_fails_on_an_unexpected_loop():
    with pytest.raises(AssertionError, match=r"50.0% parallelized \(2 of 4 loops left"):
        assert_parallelized(two_loops(), loops_before=4, max_maps=10)


def test_the_gate_fails_above_the_map_bound():
    sdfg = dace.SDFG("mapped")
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet("t", {"i": "0:4"}, {}, "pass", {})

    with pytest.raises(AssertionError, match="1 maps with a bound of 0"):
        assert_parallelized(sdfg, loops_before=1, max_maps=0)


def test_fusing_below_the_bound_passes():
    assert_parallelized(dace.SDFG("fused"), loops_before=1, max_maps=10)


def versioned(parallel_arm_maps: bool) -> dace.SDFG:
    """``if (c) <map or loop> else <loop>``: run-time versioning of one loop, its fallback in the else arm."""
    sdfg = dace.SDFG("versioned")
    sdfg.add_symbol("c", dace.int32)
    conditional = ConditionalBlock("version")
    sdfg.add_node(conditional, is_start_block=True)
    fast = ControlFlowRegion("fast", sdfg)
    if parallel_arm_maps:
        fast.add_state(is_start_block=True).add_mapped_tasklet("t", {"i": "0:4"}, {}, "pass", {})
    else:
        fast.add_node(LoopRegion("fast_loop", "i < 4", "i", "i = 0", "i = i + 1"), is_start_block=True)
        fast.nodes()[0].add_state(is_start_block=True)
    slow = ControlFlowRegion("slow", sdfg)
    fallback = LoopRegion("fallback", "k < 4", "k", "k = 0", "k = k + 1")
    fallback.add_state(is_start_block=True)
    slow.add_node(fallback, is_start_block=True)
    conditional.add_branch(CodeBlock("c > 0"), fast)
    conditional.add_branch(None, slow)
    return sdfg


# Reproducer: graupel's gate read "95.7% parallelized (3 of 48 loops left, 1 expected)": each loop
# ParallelizeUnderConstraint versioned kept its in-order fallback, which the gate counted as a loop left
def test_a_versioned_loop_counts_as_parallelized():
    assert parallelization_rate(versioned(parallel_arm_maps=True), loops_before=1) == 1.0


def test_a_fallback_whose_other_arm_is_still_a_loop_counts():
    assert sequential_loops(versioned(parallel_arm_maps=False)) == 2


if __name__ == "__main__":
    test_a_loop_left_behind_lowers_the_rate()
    test_loops_that_must_stay_sequential_do_not_count_against_the_rate()
    test_a_program_with_only_sequential_loops_is_fully_parallelized()
    test_the_gate_fails_on_an_unexpected_loop()
    test_the_gate_fails_above_the_map_bound()
    test_fusing_below_the_bound_passes()
    test_a_versioned_loop_counts_as_parallelized()
    test_a_fallback_whose_other_arm_is_still_a_loop_counts()
