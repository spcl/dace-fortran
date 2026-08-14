#!/usr/bin/env python3
"""Symbol-gated phases for the QE GPU SDFGs, so a host timer brackets COMPUTE only.

Both offloaded SDFGs (``addusxx_g_gpu.sdfg``, ``newdxx_g_gpu.sdfg``) carry their own
``_cpu_to_gpu_copy_in`` / ``_gpu_to_cpu_copy_out`` states, so timing a whole call times
the transfers with it.  ``gate_phases`` rewires the root CFG into three independently
callable phases -- copy-in, compute, copy-out -- selected by the integer symbols in
``GATES``.  The device mirrors are ``AllocationLifetime.Persistent``, so what one phase
call stages stays there for the next one, and a caller can run
``copy-in (untimed) -> compute (TIMED) -> copy-out (untimed) -> verify`` per rep.

The trailing ``_qe_sync`` state is unconditional and is the ONLY sync in the graph --
the ``sync -> t0 -> body -> sync -> t1`` shape the cloudsc/velocity GPU lanes use, so
every phase call returns exactly when its own GPU work is done.  IMPORT THIS MODULE
BEFORE dace: the ``__DACE_NO_SYNC`` pin below only takes effect if it precedes the
first ``import dace`` in the process (codegen decides sync emission at import time).
"""
import os
import time

os.environ.setdefault("__DACE_NO_SYNC", "1")

from dace import SDFG, dtypes, symbolic  # noqa: E402
from dace.sdfg import InterstateEdge  # noqa: E402

COPY_IN = "_cpu_to_gpu_copy_in"
COPY_OUT = "_gpu_to_cpu_copy_out"
GATES = ("qe_copy_in", "qe_compute", "qe_copy_out")


def _sync_state(sdfg: SDFG, label: str):
    st = sdfg.add_state(label)
    st.add_tasklet(name="_stream0_sync",
                   inputs={},
                   outputs={},
                   code="DACE_GPU_CHECK(gpuStreamSynchronize(nullptr));",
                   language=dtypes.Language.CPP,
                   side_effects=True)
    return st


def _hoist_scope_gpu_transients(sdfg: SDFG, bound: str = "dfftt_nl_d0") -> list[str]:
    """Move Scope-lifetime device transients out of the gated phases.

    DaCe emits their ``cudaMalloc`` at the first state that touches them but the matching
    ``cudaFree`` at the end of the SDFG scope (framecode.py:769-778), so once the compute
    phase can be skipped the free runs on an uninitialized pointer -> ``invalid argument``.
    Persistent lifetime puts both in ``__dace_init``/``__dace_exit``; that needs an
    init-time size, so the descriptor is widened to the G-vector axis it is indexed on
    (``newdxx_g``'s auxvc(ngm) with ngm = dfftt_ngm[0], an interstate-assigned symbol,
    against ``dfftt_nl``'s own dimension -- nl is dimensioned (ngm) too, replication
    included).
    """
    hoisted = []
    assert bound in sdfg.symbols, f"{bound} is not an SDFG symbol; no init-time size for the hoist"
    bound_sym = symbolic.pystr_to_symbolic(bound)
    for name, desc in sdfg.arrays.items():
        if not desc.transient or desc.storage != dtypes.StorageType.GPU_Global:
            continue
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            continue
        assert len(desc.shape) == 1, f"{name}: only 1-D scope transients are handled, got {desc.shape}"
        desc.set_shape((bound_sym, ))
        desc.lifetime = dtypes.AllocationLifetime.Persistent
        hoisted.append(name)
    return hoisted


def gate_phases(sdfg: SDFG) -> tuple[str, ...]:
    """Rewire the root CFG of an offloaded QE SDFG into the three gated phases."""
    _hoist_scope_gpu_transients(sdfg)
    blocks = {b.label: b for b in sdfg.nodes()}
    cin, cout = blocks[COPY_IN], blocks[COPY_OUT]
    e_in, = sdfg.out_edges(cin)
    e_out, = sdfg.in_edges(cout)
    body_start, body_end = e_in.dst, e_out.src
    assert not e_out.data.assignments, f"assignments on the copy-out edge: {e_out.data.assignments}"

    for g in GATES:
        sdfg.add_symbol(g, dtypes.int32)
    entry = sdfg.add_state("_qe_entry", is_start_block=True)
    mid = sdfg.add_state("_qe_mid")
    post = sdfg.add_state("_qe_post")
    done = _sync_state(sdfg, "_qe_sync")

    sdfg.remove_edge(e_in)
    sdfg.remove_edge(e_out)
    sdfg.add_edge(entry, cin, InterstateEdge(condition="qe_copy_in != 0"))
    sdfg.add_edge(entry, mid, InterstateEdge(condition="qe_copy_in == 0"))
    sdfg.add_edge(cin, mid, InterstateEdge())
    # The copy-in edge's assignments are host-side loop bounds/gates read before the body.
    sdfg.add_edge(mid, body_start, InterstateEdge(condition="qe_compute != 0", assignments=e_in.data.assignments))
    sdfg.add_edge(mid, post, InterstateEdge(condition="qe_compute == 0"))
    sdfg.add_edge(body_end, post, InterstateEdge())
    sdfg.add_edge(post, cout, InterstateEdge(condition="qe_copy_out != 0"))
    sdfg.add_edge(post, done, InterstateEdge(condition="qe_copy_out == 0"))
    sdfg.add_edge(cout, done, InterstateEdge())
    sdfg.validate()
    return GATES


def run_phase(csdfg, kwargs: dict, **gates: int) -> float:
    """Call one phase; returns its wall time in ms (the phase's GPU work is drained)."""
    kwargs.update(dict.fromkeys(GATES, 0))
    kwargs.update(gates)
    t0 = time.perf_counter()
    csdfg(**kwargs)
    return (time.perf_counter() - t0) * 1e3


def run_all_phases(csdfg, kwargs: dict) -> None:
    """One whole call, every phase on -- the un-gated behaviour of the shipped SDFG."""
    kwargs.update(dict.fromkeys(GATES, 1))
    csdfg(**kwargs)
