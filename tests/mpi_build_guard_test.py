"""``build_on_root`` must turn a rank-0 build failure into a clean all-rank error, not a
collective deadlock.

MPI e2e tests build artefacts on one rank and share via a collective (``comm.bcast`` /
``distributed_compile`` / ``comm.Barrier``).  If the build raises, pytest catches it on
the building rank -- the process stays alive, so ``mpirun`` never sees a dead rank and
every other rank blocks at the collective forever (until CI times out).
:func:`_util.build_on_root` broadcasts the failure so all ranks raise together.

Run under mpirun with >= 2 ranks::

    mpirun --oversubscribe -n 2 python -m pytest -p no:cacheprovider \\
        tests/mpi_build_guard_test.py

If this guard regresses, non-root ranks hang here instead of raising and the ``-m mpi``
CI step times out.
"""

import pytest

from _util import build_on_root

pytestmark = pytest.mark.mpi


@pytest.mark.mpi
def test_build_failure_on_root_raises_on_every_rank():
    """A build that raises only on the root rank must surface as ``RuntimeError`` on
    EVERY rank, proving no rank is left blocked at the broadcast."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("build-guard deadlock check needs >= 2 ranks")

    def _failing_build():
        # Runs only on the root rank; the others must still learn it failed.
        raise ValueError("induced build failure")

    with pytest.raises(RuntimeError, match="all ranks abort"):
        build_on_root(comm, _failing_build)

    # A successful build still returns its (broadcast) result on every rank.
    rank = comm.Get_rank()
    payload = build_on_root(comm, lambda: f"built-by-{rank}")
    assert payload == "built-by-0", f"rank {rank} saw {payload!r}, expected root's result"


@pytest.mark.mpi
def test_build_on_root_no_broadcast_returns_root_only():
    """``broadcast=False`` returns the result on root and ``None`` elsewhere (for
    non-picklable payloads consumed by a following collective); still re-raises a
    build failure on every rank."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("build-guard deadlock check needs >= 2 ranks")

    rank = comm.Get_rank()
    result = build_on_root(comm, lambda: object(), broadcast=False)
    if rank == 0:
        assert result is not None
    else:
        assert result is None

    with pytest.raises(RuntimeError, match="all ranks abort"):
        build_on_root(comm, lambda: 1 / 0, broadcast=False)
