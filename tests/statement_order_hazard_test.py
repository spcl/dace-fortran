"""Consecutive Fortran statements sharing a container must stay ordered after lowering.

The bridge lowers a run of consecutive assigns into ONE SDFG state.  Two statements that touch
the same storage then land in disjoint dataflow components, so nothing in the graph orders them
and the emitted statement order is a scheduler tie-break on node insertion order.  CloudSC's
``ZQSMIX``::

    ZQSMIX(JL,JK) = ZQSMIX(JL,JK)-ZCOND1
    ZDQS(JL)      = ZQSMIX(JL,JK)-ZQOLD(JL)   ! reads the updated value
    ZQSMIX(JL,JK) = ZQOLD(JL)                 ! overwrite, disjoint component, unordered

``emit_cfg._sibling_rw_hazard`` / ``emit_assign``'s realised-graph guard already split states for
the direct-name form, so a plain three-liner on one array lowers correctly.  Both guards compare
Fortran-level NAMES, while ``access.acc`` resolves every alias onto the SOURCE container's access
node -- so a write through a POINTER alias of an array read by a neighbouring statement is invisible
to the guards and lands unordered in the same state.  Same storage, two access nodes, no path.

Reproduces WAR (read then aliased overwrite) and WAW (write then aliased overwrite).  RAW is NOT
reproducible: ``acc``'s per-state cache always hands a read the most recent write node for that
container, so a read-after-write is ordered by construction whichever name it goes through.

``test_lowers_without_intra_state_hazard`` is the graph-shape assertion.
``test_matches_gfortran`` is the semantic one: the as-built graph happens to win the tie-break in
source order today, so it is checked with the state's nodes re-inserted in reverse as well -- a
permutation that is legal on a correctly ordered graph and flips the result on a hazardous one.
"""

import numpy as np
import pytest

from _helpers import f2py, xfail
from _util import build_sdfg, have_flang
from hazard_scan import scan

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

HEAD = """
MODULE kernel_mod
CONTAINS
SUBROUTINE driver(a, b, c, d, n)
integer, intent(in) :: n
double precision, intent(in) :: b(n), c(n)
double precision, intent(inout), target :: a(n)
double precision, intent(inout) :: d(n)
double precision, pointer :: p(:)
integer i
p => a
"""

TAIL = """
END SUBROUTINE driver
END MODULE kernel_mod
"""

# d must see the INCOMING a; the aliased store to p is the later statement.
WAR_BODY = """
DO i = 1, n
    d(i) = a(i) - b(i)
    p(i) = c(i)
ENDDO
"""

# Last write wins: a must end up holding c, not b.
WAW_BODY = """
DO i = 1, n
    a(i) = b(i)
    p(i) = c(i)
    d(i) = b(i) + c(i)
ENDDO
"""

BODIES = {"war": WAR_BODY, "waw": WAW_BODY}

N = 8


def kernel_source(shape: str) -> str:
    return HEAD + BODIES[shape] + TAIL


def inputs():
    rng = np.random.default_rng(7)
    return (np.asfortranarray(rng.random(N) + 0.5), np.asfortranarray(rng.random(N) + 0.5),
            np.asfortranarray(rng.random(N) + 0.5), np.zeros(N, dtype=np.float64, order="F"))


def reverse_node_order(sdfg):
    """Re-insert every state's nodes back to front, edges unchanged.

    A pure permutation of an unordered set: on a graph whose dataflow records Fortran's statement
    order this cannot change the result.  On a hazardous state it flips the codegen tie-break.
    """
    for state in sdfg.all_states():
        nodes = list(state.nodes())
        edges = list(state.edges())
        for node in nodes:
            state.remove_node(node)
        for node in reversed(nodes):
            state.add_node(node)
        for edge in edges:
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)


def run_sdfg(sdfg, arrays):
    a, b, c, d = (x.copy(order="F") for x in arrays)
    sdfg(a=a, b=b, c=c, d=d, n=N)
    return a, d


def run_reference(shape, tmp_path, arrays):
    ref = f2py(kernel_source(shape), tmp_path / "ref", f"{shape}_order_ref")
    a, b, c, d = (x.copy(order="F") for x in arrays)
    ref.kernel_mod.driver(a, b, c, d, N)
    return a, d


@pytest.mark.parametrize("shape", sorted(BODIES))
@xfail("bridge leaves the aliased overwrite unordered in one state; frontend fix pending")
def test_lowers_without_intra_state_hazard(tmp_path, shape):
    """No two access nodes for one container may sit unordered in a single state."""
    sdfg = build_sdfg(kernel_source(shape), tmp_path / "sdfg", name=shape, entry="driver").build()
    hazards = scan(sdfg)
    detail = "\n".join(f"  {h['kind']} {h['container']} in {h['state']} roles={h['roles']}" for h in hazards)
    assert not hazards, f"{shape}: {len(hazards)} unordered same-container access pair(s)\n{detail}"


@pytest.mark.parametrize("shape", sorted(BODIES))
@xfail("statement order survives only by codegen tie-break; frontend fix pending")
def test_matches_gfortran(tmp_path, shape):
    """Bit-exact against gfortran, as built and with each state's node order reversed."""
    arrays = inputs()
    a_ref, d_ref = run_reference(shape, tmp_path, arrays)

    for variant, permute in (("asbuilt", False), ("reversed", True)):
        sdfg = build_sdfg(kernel_source(shape), tmp_path / variant, name=f"{shape}_{variant}", entry="driver").build()
        sdfg.name = f"{sdfg.name}_{variant}"
        if permute:
            reverse_node_order(sdfg)
        a_out, d_out = run_sdfg(sdfg, arrays)
        np.testing.assert_array_equal(a_out, a_ref, err_msg=f"{shape}/{variant}: a diverged from gfortran")
        np.testing.assert_array_equal(d_out, d_ref, err_msg=f"{shape}/{variant}: d diverged from gfortran")
