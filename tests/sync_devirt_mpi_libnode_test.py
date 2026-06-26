"""Devirtualize a comm-pattern dispatch, inline it down to raw MPI -> the sync's
pack/gather compute lands in the SDFG and ONLY the MPI primitives remain, as
``dace.libraries.mpi`` library nodes.

This is the ICON atmosphere ``sync_patch_array`` lowering pattern in miniature.
``sync_patch_array`` dispatches through the abstract ``t_comm_pattern`` vtable
(``p_pat%exchange_data_r3d``); its concrete arm does a local pack/gather (pure
compute) then raw ``mpi_isend``/``mpi_irecv``/``mpi_wait``. Rather than
externalising the whole sync, we:

  * ``monomorphize`` the abstract ``comm_pattern`` to its concrete arm
    (``retype`` axis) -> the dispatch becomes a static call, and
  * let ``build_sdfg``'s inline-all splice the concrete exchange body in.

The bridge then auto-recognises ``mpi_isend`` / ``mpi_irecv`` / ``mpi_wait`` and
lowers them to ``dace.libraries.mpi`` nodes -- so the pack loop is real SDFG
compute and the *only* external calls are the MPI primitives, correctly typed as
MPI library nodes.
"""
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.inliner.ast_desugaring.monomorphize import parse_program
from dace_fortran.inliner.ast_desugaring.monomorphize_rewrite import (AxisSpec, monomorphize, MonomorphizationSpec)

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

#: Abstract comm_pattern (deferred ``exchange``) + a concrete ``comm_orig`` whose
#: exchange packs (pure compute) then issues raw MPI. ``external`` MPI decls so it
#: lowers with no ``mpi.mod`` (the bridge recognises the opaque ``fir.call``).
_SRC = """
module comm_mod
  implicit none
  type, abstract :: comm_pattern
    integer :: n
  contains
    procedure(exch_i), deferred :: exchange
  end type
  abstract interface
    subroutine exch_i(this, buf, partner, tag)
      import comm_pattern
      class(comm_pattern), intent(in) :: this
      real(8), intent(inout) :: buf(:)
      integer, intent(in) :: partner, tag
    end subroutine
  end interface
  type, extends(comm_pattern) :: comm_orig
  contains
    procedure :: exchange => orig_exchange
  end type
contains
  subroutine orig_exchange(this, buf, partner, tag)
    class(comm_orig), intent(in) :: this
    real(8), intent(inout) :: buf(:)
    integer, intent(in) :: partner, tag
    integer :: i, ierr, req
    integer, parameter :: MPI_COMM_WORLD = 0
    integer, parameter :: MPI_DOUBLE_PRECISION = 17
    external :: mpi_isend, mpi_irecv, mpi_wait
    do i = 1, this%n
      buf(i) = buf(i) * 2.0d0
    end do
    call mpi_isend(buf, this%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_irecv(buf, this%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_wait(req, ierr)
  end subroutine
end module

subroutine kernel(p_pat, buf, partner, tag)
  use comm_mod
  implicit none
  class(comm_pattern), pointer, intent(in) :: p_pat
  real(8), intent(inout) :: buf(:)
  integer, intent(in) :: partner, tag
  call p_pat%exchange(buf, partner, tag)
end subroutine
"""


def test_devirtualized_sync_inlines_pack_keeps_only_mpi_libnodes(tmp_path):
    import dace
    from dace.libraries.mpi.nodes.node import MPINode

    # Devirtualize the comm-pattern vtable: CLASS(comm_pattern) -> TYPE(comm_orig).
    prog = parse_program(_SRC)
    stats = monomorphize(
        prog, MonomorphizationSpec(axes=[AxisSpec(base="comm_pattern", strategy="retype", concrete="comm_orig")]))
    assert stats.declarations_retyped >= 1
    mono = str(prog)
    # the dispatch object is now concrete (the abstract dummy keeps CLASS in the
    # deferred interface, but the pointer p_pat is retyped)
    assert "TYPE(comm_orig), POINTER" in mono

    sdfg = build_sdfg(mono, tmp_path / "sdfg", name="sync_poc", entry="kernel").build()

    # Only MPI remains external, and as dace.libraries.mpi nodes -- not opaque calls.
    mpi = sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MPINode)})
    assert mpi == ["Irecv", "Isend", "Wait"], f"expected Isend/Irecv/Wait libnodes, got {mpi}"
    # The pack loop (buf*2) is inlined as real SDFG compute, not externalised.
    assert any(isinstance(n, dace.nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive()), \
        "expected the pack-loop compute inlined as tasklets"
