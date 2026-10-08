"""Tests for ``BindOmpThreadCount``: the symbol a thread-strided ``CPU_Persistent`` map uses for the team
size is bound at SDFG entry from ``omp_get_max_threads()``, not passed in by the caller."""

import ctypes

import dace
import numpy as np
import pytest

from dace_fortran.omp_threads import OMP_NUM_THREADS_SYMBOL, BindOmpThreadCount

N = 64
#: Room for one flag per thread of any machine.
M = 4096


def max_threads() -> int:
    """The thread count the generated library sees: the OpenMP runtime reads ``OMP_NUM_THREADS`` once."""
    libgomp = ctypes.CDLL("libgomp.so.1")
    libgomp.omp_get_max_threads.restype = ctypes.c_int
    return libgomp.omp_get_max_threads()


def thread_strided_sdfg(name: str) -> dace.SDFG:
    """``out[t] = 1`` for every ``t in 0:__omp_num_threads`` outside a parallel region, and a persistent map
    striding ``A`` over the team inside one."""
    nthreads = dace.symbol(OMP_NUM_THREADS_SYMBOL)

    @dace.program
    def tester(A: dace.float64[N], out: dace.int32[M]):
        for t in dace.map[0:nthreads]:
            out[t] = 1
        for tid in dace.map[0:nthreads] @ dace.ScheduleType.CPU_Persistent:
            for i in dace.map[tid:N:nthreads] @ dace.ScheduleType.Sequential:
                A[i] += 1

    sdfg = tester.to_sdfg()
    sdfg.name = name
    return sdfg


def test_unbound_symbol_is_a_call_argument():
    """Without the pass the team size is part of the call signature."""
    sdfg = thread_strided_sdfg("omp_threads_unbound")
    assert OMP_NUM_THREADS_SYMBOL in {str(s) for s in sdfg.free_symbols}
    with pytest.raises(KeyError, match=OMP_NUM_THREADS_SYMBOL):
        sdfg(A=np.zeros(N), out=np.zeros(M, dtype=np.int32))


def test_symbol_is_bound_at_entry():
    sdfg = thread_strided_sdfg("omp_threads_structure")
    scalar = BindOmpThreadCount().apply_pass(sdfg, {})
    sdfg.validate()

    assert scalar in sdfg.arrays and sdfg.arrays[scalar].transient
    assert OMP_NUM_THREADS_SYMBOL not in {str(s) for s in sdfg.free_symbols}
    assert OMP_NUM_THREADS_SYMBOL not in sdfg.arglist()

    first = sdfg.start_state
    (tasklet,) = [n for n in first.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert tasklet.side_effects
    assert tasklet.language == dace.Language.CPP
    assert "omp_get_max_threads" in tasklet.code.as_string
    (edge,) = sdfg.out_edges(first)
    assert edge.data.assignments == {OMP_NUM_THREADS_SYMBOL: scalar}


def test_kernel_runs_without_the_thread_count_argument():
    """The generated code takes the team size from the OpenMP runtime: every thread strides its share of ``A``
    and the outside map covers exactly ``omp_get_max_threads()`` flags."""
    sdfg = thread_strided_sdfg("omp_threads_run")
    BindOmpThreadCount().apply_pass(sdfg, {})
    code = sdfg.generate_code()[0].clean_code
    assert "omp_get_max_threads()" in code

    A = np.zeros(N)
    out = np.zeros(M, dtype=np.int32)
    sdfg(A=A, out=out)

    assert np.all(A == 1)
    assert out.sum() == max_threads()


def test_optimize_binds_the_symbol():
    """The optimization recipe leaves no team-size argument behind."""
    from dace_fortran.pipelines import optimize

    sdfg = optimize(thread_strided_sdfg("omp_threads_optimize"))
    assert OMP_NUM_THREADS_SYMBOL not in {str(s) for s in sdfg.free_symbols}

    A = np.zeros(N)
    out = np.zeros(M, dtype=np.int32)
    sdfg(A=A, out=out)
    assert np.all(A == 1)
    assert out.sum() == max_threads()


def test_pass_without_the_symbol_is_a_no_op():
    @dace.program
    def plain(A: dace.float64[N]):
        for i in dace.map[0:N]:
            A[i] += 1

    sdfg = plain.to_sdfg()
    states = sdfg.number_of_nodes()
    assert BindOmpThreadCount().apply_pass(sdfg, {}) is None
    assert sdfg.number_of_nodes() == states


def test_pass_is_idempotent():
    sdfg = thread_strided_sdfg("omp_threads_twice")
    assert BindOmpThreadCount().apply_pass(sdfg, {}) is not None
    states = sdfg.number_of_nodes()
    assert BindOmpThreadCount().apply_pass(sdfg, {}) is None
    assert sdfg.number_of_nodes() == states


if __name__ == "__main__":
    test_unbound_symbol_is_a_call_argument()
    test_symbol_is_bound_at_entry()
    test_kernel_runs_without_the_thread_count_argument()
    test_optimize_binds_the_symbol()
    test_pass_without_the_symbol_is_a_no_op()
    test_pass_is_idempotent()
