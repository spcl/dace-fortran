"""MPI collectives (``MPI_Barrier`` / ``MPI_Allreduce`` / ``MPI_Bcast``) ->
DaCe ``dace.libraries.mpi`` library nodes -- no MPI op should be left external.

The bridge recognises the opaque ``fir.call @_QPmpi_*`` and the positional MPI
ABI; the builder lowers it to the matching DaCe node (``Barrier`` is added here;
``Allreduce`` / ``Bcast`` already exist).  Structural tests: build + validate,
assert the nodes are wired -- no ``mpirun`` needed.

The reduction op is passed as a ``use mpi``-style runtime integer handle
(``mpi_sum`` / ``mpi_prod`` / ...) so its name survives to the builder and the
op mapping is genuinely exercised.  (A Fortran ``parameter`` handle folds to an
opaque ``f__assoc_scalar_N`` temporary -- the op identity is lost upstream and
the builder must raise; see ``test_allreduce_unrecognised_op_raises``.)  The
communicator is still a local ``MPI_COMM_WORLD`` ``parameter``, which flang
lowers to a synthetic scalar the bridge treats as a (non-default) runtime
communicator -- so every collective gets a threaded ``_comm`` connector.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ``use mpi``-style runtime handles: non-``parameter`` module integers, so the
# bridge traces the op argument to its NAME (``mpi_prod`` / ``mpi_maxloc`` / ...)
# instead of a folded constant.  Real ``use mpi`` declares the handles the same
# way (runtime integers, not parameters).
_MPI_OP_MODULE = """
module mpiops
  implicit none
  integer :: mpi_sum = 1
  integer :: mpi_prod = 3
  integer :: mpi_max = 2
  integer :: mpi_min = 4
  integer :: mpi_land = 5
  integer :: mpi_lor = 7
  integer :: mpi_maxloc = 11
  integer :: mpi_minloc = 12
end module
"""

_COLLECTIVES = _MPI_OP_MODULE + """
subroutine collectives(buf, rbuf, n, root)
  use mpiops
  implicit none
  integer, intent(in) :: n, root
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Barrier, MPI_Allreduce, MPI_Bcast
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(buf, rbuf, n, MPI_DOUBLE_PRECISION, mpi_sum, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(buf, n, MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD, ierr)
end subroutine collectives
"""


def _allreduce_src(op_ref: str) -> str:
    """Single-``MPI_Allreduce`` kernel whose reduction op is ``op_ref`` (a
    ``use mpi``-style module handle)."""
    return _MPI_OP_MODULE + f"""
subroutine areduce(buf, rbuf, n)
  use mpiops
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Allreduce
  call MPI_Allreduce(buf, rbuf, n, MPI_DOUBLE_PRECISION, {op_ref}, MPI_COMM_WORLD, ierr)
end subroutine areduce
"""


def _first(sdfg, cls):
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls))


def test_resolve_mpi_op_map():
    """The op-name -> ``MPI_Op`` map is exact: ``mpi_maxloc`` / ``mpi_minloc``
    must NOT be swallowed by a ``max`` / ``min`` substring, and a non-{max,min}
    op must NOT default to ``MPI_SUM`` (the two failure modes of the old
    ``'max' in name`` heuristic).  An unrecognised op raises."""
    from dace_fortran.builder.emit_library import resolve_mpi_op

    assert resolve_mpi_op("mpi_sum") == "MPI_SUM"
    assert resolve_mpi_op("mpi_prod") == "MPI_PROD"
    assert resolve_mpi_op("mpi_land") == "MPI_LAND"
    assert resolve_mpi_op("mpi_bxor") == "MPI_BXOR"
    assert resolve_mpi_op("mpi_maxloc") == "MPI_MAXLOC"  # not MPI_MAX
    assert resolve_mpi_op("mpi_minloc") == "MPI_MINLOC"  # not MPI_MIN
    assert resolve_mpi_op("MPI_SUM") == "MPI_SUM"  # upper-case pass-through
    with pytest.raises(NotImplementedError):
        resolve_mpi_op("f__assoc_scalar_2")  # folded parameter -> identity lost


def test_collectives_lower_to_mpi_libnodes(tmp_path: Path):
    """``MPI_Barrier`` / ``MPI_Allreduce`` / ``MPI_Bcast`` become DaCe ``Barrier``
    / ``Allreduce`` / ``Bcast`` nodes (no opaque ``call`` left), and validate."""
    from dace.libraries.mpi.nodes import Allreduce, Barrier, Bcast

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_COLLECTIVES, sdfg_dir, name="collectives", entry="collectives").build()

    nodes = {
        cls: [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls)]
        for cls in (Barrier, Allreduce, Bcast)
    }
    assert len(nodes[Barrier]) == 1 and len(nodes[Allreduce]) == 1 and len(nodes[Bcast]) == 1, \
        f"expected one of each collective, got { {c.__name__: len(v) for c, v in nodes.items()} }"

    assert nodes[Allreduce][0].op == "MPI_SUM"
    # The Fortran communicator is threaded into every collective via a CommF2c
    # dataflow node feeding a ``_comm`` connector (bridge feature 93cc5f2).
    assert set(nodes[Allreduce][0].in_connectors) == {"_inbuffer", "_comm"}
    assert set(nodes[Bcast][0].in_connectors) == {"_inbuffer", "_root", "_comm"}
    assert set(nodes[Barrier][0].in_connectors) == {"_comm"}  # only the communicator

    assert "call" not in [getattr(n, "kind", None) for n, _ in sdfg.all_nodes_recursive()]
    sdfg.validate()


@pytest.mark.parametrize(
    "op_ref, expected",
    [
        ("mpi_sum", "MPI_SUM"),
        ("mpi_max", "MPI_MAX"),
        ("mpi_min", "MPI_MIN"),
        ("mpi_prod", "MPI_PROD"),  # old heuristic: silently MPI_SUM (wrong)
        ("mpi_land", "MPI_LAND"),  # old heuristic: silently MPI_SUM (wrong)
        ("mpi_maxloc", "MPI_MAXLOC"),  # old heuristic: MPI_MAX via 'max' substring (wrong)
        ("mpi_minloc", "MPI_MINLOC"),  # old heuristic: MPI_MIN via 'min' substring (wrong)
    ],
)
def test_allreduce_op_not_coerced(tmp_path: Path, op_ref: str, expected: str):
    """End-to-end (flang -> bridge -> builder): the ``MPI_Allreduce`` op is
    mapped to the correct ``MPI_Op``, never silently coerced to ``MPI_SUM`` /
    ``MPI_MAX``.  Pins the reduction-op bug."""
    from dace.libraries.mpi.nodes import Allreduce

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_allreduce_src(op_ref), sdfg_dir, name="areduce", entry="areduce").build()
    assert _first(sdfg, Allreduce).op == expected
    sdfg.validate()


def test_allreduce_unrecognised_op_raises(tmp_path: Path):
    """A Fortran ``parameter`` op folds to an opaque ``f__assoc_scalar_N`` name
    (op identity lost upstream) -> the builder raises rather than silently
    reducing with ``MPI_SUM``.  Documents the bridge-side limitation."""
    src = """
subroutine areduce_p(buf, rbuf, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_SUM = 1
  external :: MPI_Allreduce
  call MPI_Allreduce(buf, rbuf, n, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
end subroutine areduce_p
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(NotImplementedError):
        build_sdfg(src, sdfg_dir, name="areduce_p", entry="areduce_p").build()
