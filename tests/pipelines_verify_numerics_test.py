# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``optimize(verify_inputs=...)`` must catch a pipeline that changes values.

``validate()`` at the end of the pipeline is structural: it accepts a graph that is well-formed but
computes something different. This checks the numerical gate both ways -- silent on a
value-preserving run, loud on a corrupted one -- because a gate that never fires proves nothing.
"""
import numpy as np
import pytest

import dace

from dace_fortran.pipelines import optimize, verify_numerics


def scale_sdfg(name: str, factor: float = 2.0) -> dace.SDFG:
    """``B[i] = A[i] * factor`` over a loop, i.e. something LoopToMap will turn into a map."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    before = sdfg.add_state('before', is_start_block=True)
    body = sdfg.add_state('body')
    after = sdfg.add_state('after')
    sdfg.add_loop(before, body, after, 'i', '0', 'i < 16', 'i + 1')
    tasklet = body.add_tasklet('scale', {'inp'}, {'out'}, f'out = inp * {factor}')
    body.add_edge(body.add_access('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', body.add_access('B'), None, dace.Memlet('B[i]'))
    return sdfg


def test_optimize_verifies_numerics():
    """A value-preserving pipeline run must pass the gate, and still parallelize."""
    sdfg = scale_sdfg('verify_ok')
    a = np.random.rand(16)
    inputs = {'A': a, 'B': np.zeros(16)}

    optimize(sdfg, verify_inputs=inputs)

    # The caller's arrays are inputs to the check, never scratch for it.
    assert np.array_equal(inputs['B'], np.zeros(16)), 'verify_numerics wrote through the caller arrays'
    out = np.zeros(16)
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, a * 2.0)


def test_verify_numerics_catches_a_changed_result():
    """The gate must FAIL when the two SDFGs disagree -- otherwise it guards nothing."""
    reference = scale_sdfg('verify_ref', factor=2.0)
    corrupted = scale_sdfg('verify_bad', factor=3.0)  # stands in for a miscompiling pipeline

    with pytest.raises(AssertionError, match='changed the numerics'):
        verify_numerics(reference, corrupted, {'A': np.random.rand(16), 'B': np.zeros(16)})


def test_verify_numerics_is_bit_exact_not_approximate():
    """One ULP is a failure: the pipeline never reassociates arithmetic, so it cannot drift."""
    reference = scale_sdfg('verify_exact_ref', factor=1.0)
    nudged = scale_sdfg('verify_exact_bad', factor=1.0 + 2**-52)

    with pytest.raises(AssertionError, match='changed the numerics'):
        verify_numerics(reference, nudged, {'A': np.full(16, 1.0), 'B': np.zeros(16)})


if __name__ == '__main__':
    test_optimize_verifies_numerics()
    test_verify_numerics_catches_a_changed_result()
    test_verify_numerics_is_bit_exact_not_approximate()
