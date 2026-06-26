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


#: Closer to the real ICON atmosphere layering: a ``sync_patch_array`` entry ->
#: the generic ``exchange_data`` interface -> a wrapper (``exchange_data_r3d``)
#: that does the VTABLE DISPATCH on an abstract ``CLASS(t_comm_pattern)`` dummy ->
#: the concrete arm's pack + raw MPI. The dispatch lives one level down from the
#: retyped pointer, so the bridge's inline-all + fir-polymorphic-op must resolve
#: it once the wrapper is inlined into the concrete-typed caller.
_ATMO_SRC = """
module mo_comm
  implicit none
  type, abstract :: t_comm_pattern
    integer :: n
  contains
    procedure(exch_i), deferred :: exchange_data_r3d
  end type
  abstract interface
    subroutine exch_i(p_pat, recv, partner, tag)
      import t_comm_pattern
      class(t_comm_pattern), intent(in) :: p_pat
      real(8), intent(inout) :: recv(:)
      integer, intent(in) :: partner, tag
    end subroutine
  end interface
  type, extends(t_comm_pattern) :: t_comm_pattern_orig
  contains
    procedure :: exchange_data_r3d => orig_exchange_data_r3d
  end type
  interface exchange_data
    module procedure exchange_data_r3d_wrap
  end interface
contains
  subroutine exchange_data_r3d_wrap(p_pat, recv, partner, tag)
    class(t_comm_pattern), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer, intent(in) :: partner, tag
    call p_pat%exchange_data_r3d(recv, partner, tag)
  end subroutine
  subroutine orig_exchange_data_r3d(p_pat, recv, partner, tag)
    class(t_comm_pattern_orig), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer, intent(in) :: partner, tag
    integer :: i, ierr, req
    integer, parameter :: MPI_COMM_WORLD = 0
    integer, parameter :: MPI_DOUBLE_PRECISION = 17
    external :: mpi_isend, mpi_irecv, mpi_wait
    do i = 1, p_pat%n
      recv(i) = recv(i) * 2.0d0
    end do
    call mpi_isend(recv, p_pat%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_irecv(recv, p_pat%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_wait(req, ierr)
  end subroutine
end module

subroutine sync_patch_array(p_pat, arr, partner, tag)
  use mo_comm
  implicit none
  class(t_comm_pattern), pointer, intent(in) :: p_pat
  real(8), intent(inout) :: arr(:)
  integer, intent(in) :: partner, tag
  call exchange_data(p_pat, arr, partner, tag)
end subroutine
"""


def _devirt_build_atmo(tmp_path):
    """Devirtualize the comm-pattern axis then build the sync_patch_array SDFG."""
    prog = parse_program(_ATMO_SRC)
    monomorphize(
        prog,
        MonomorphizationSpec(axes=[AxisSpec(base="t_comm_pattern", strategy="retype", concrete="t_comm_pattern_orig")]))
    return build_sdfg(str(prog), tmp_path / "sdfg", name="atmo_sync", entry="sync_patch_array").build()


def test_atmosphere_sync_patch_array_devirtualized_to_mpi_libnodes(tmp_path):
    """The atmosphere ``sync_patch_array`` layering (generic interface -> wrapper
    vtable dispatch -> concrete pack + MPI) lowers with the pack inlined and only
    the MPI primitives left, as ``dace.libraries.mpi`` library nodes."""
    import dace
    from dace.libraries.mpi.nodes.node import MPINode

    sdfg = _devirt_build_atmo(tmp_path)
    mpi = sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MPINode)})
    assert mpi == ["Irecv", "Isend", "Wait"], f"expected Isend/Irecv/Wait libnodes, got {mpi}"
    assert any(isinstance(n, dace.nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive()), \
        "expected the pack compute inlined as tasklets"


#: A dycore substep: a stencil update on a field, a halo exchange via
#: ``sync_patch_array`` (the devirtualized comm pattern), then a copy-back -- the
#: shape of one ICON dynamical-core timestep stage. ``sync_patch_array`` lives in
#: the module so a caller (``dycore_step``) has its explicit interface (a
#: polymorphic / pointer dummy requires one).
_DYCORE_STEP_SRC = """
module mo_comm
  implicit none
  type, abstract :: t_comm_pattern
    integer :: n
  contains
    procedure(exch_i), deferred :: exchange_data_r3d
  end type
  abstract interface
    subroutine exch_i(p_pat, recv, partner, tag)
      import t_comm_pattern
      class(t_comm_pattern), intent(in) :: p_pat
      real(8), intent(inout) :: recv(:)
      integer, intent(in) :: partner, tag
    end subroutine
  end interface
  type, extends(t_comm_pattern) :: t_comm_pattern_orig
  contains
    procedure :: exchange_data_r3d => orig_exchange_data_r3d
  end type
  interface exchange_data
    module procedure exchange_data_r3d_wrap
  end interface
contains
  subroutine exchange_data_r3d_wrap(p_pat, recv, partner, tag)
    class(t_comm_pattern), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer, intent(in) :: partner, tag
    call p_pat%exchange_data_r3d(recv, partner, tag)
  end subroutine
  subroutine orig_exchange_data_r3d(p_pat, recv, partner, tag)
    class(t_comm_pattern_orig), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer, intent(in) :: partner, tag
    integer :: i, ierr, req
    integer, parameter :: MPI_COMM_WORLD = 0
    integer, parameter :: MPI_DOUBLE_PRECISION = 17
    external :: mpi_isend, mpi_irecv, mpi_wait
    do i = 1, p_pat%n
      recv(i) = recv(i) * 2.0d0
    end do
    call mpi_isend(recv, p_pat%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_irecv(recv, p_pat%n, MPI_DOUBLE_PRECISION, partner, tag, MPI_COMM_WORLD, req, ierr)
    call mpi_wait(req, ierr)
  end subroutine
  subroutine sync_patch_array(p_pat, arr, partner, tag)
    class(t_comm_pattern), pointer, intent(in) :: p_pat
    real(8), intent(inout) :: arr(:)
    integer, intent(in) :: partner, tag
    call exchange_data(p_pat, arr, partner, tag)
  end subroutine
end module

subroutine dycore_step(p_pat, h, hnew, n, partner, tag)
  use mo_comm
  implicit none
  class(t_comm_pattern), pointer, intent(in) :: p_pat
  real(8), intent(inout) :: h(:), hnew(:)
  integer, intent(in) :: n, partner, tag
  integer :: i
  ! divergence-like stencil update -- real dycore compute
  do i = 2, n - 1
    hnew(i) = h(i) - 0.5d0 * (h(i + 1) - h(i - 1))
  end do
  ! halo exchange -- devirtualized + inlined, NOT an external sync
  call sync_patch_array(p_pat, hnew, partner, tag)
  do i = 1, n
    h(i) = hnew(i)
  end do
end subroutine
"""


def test_dycore_step_inlines_sync_and_devirtualizes(tmp_path):
    """End-to-end on a dycore-substep shape (stencil update -> halo sync ->
    copy-back): with the comm pattern devirtualized, ``sync_patch_array`` inlines
    so the stencil AND the sync's pack are real SDFG compute, and the only
    external calls are the MPI primitives, as ``dace.libraries.mpi`` library
    nodes -- no ExternalCall for the sync."""
    import dace
    from dace.libraries.mpi.nodes.node import MPINode
    from dace_fortran.external import ExternalCall

    prog = parse_program(_DYCORE_STEP_SRC)
    monomorphize(
        prog,
        MonomorphizationSpec(axes=[AxisSpec(base="t_comm_pattern", strategy="retype", concrete="t_comm_pattern_orig")]))
    sdfg = build_sdfg(str(prog), tmp_path / "sdfg", name="dycore_step", entry="dycore_step").build()

    # The sync's MPI primitives are libnodes; the sync itself is not external.
    mpi = sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MPINode)})
    assert mpi == ["Irecv", "Isend", "Wait"], f"expected MPI libnodes, got {mpi}"
    ext = {n.name.lower() for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ExternalCall)}
    assert not any("sync" in nm or "exchange" in nm for nm in ext), \
        f"sync must be inlined, not external; got ExternalCall names {sorted(ext)}"
    # the dycore stencil compute is in the SDFG
    assert any(isinstance(n, dace.nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive())


def test_sync_patch_array_is_not_external_anymore(tmp_path):
    """``sync_patch_array`` / ``exchange_data`` are NO LONGER externalised -- they
    are devirtualized + inlined, so the SDFG carries no ``ExternalCall`` library
    node for them. The only external boundary is the MPI primitives, as MPI
    library nodes (not opaque external calls)."""
    from dace.libraries.mpi.nodes.node import MPINode
    from dace_fortran.external import ExternalCall

    sdfg = _devirt_build_atmo(tmp_path)
    ext_names = {n.name.lower() for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ExternalCall)}
    assert not any("sync" in nm or "exchange" in nm for nm in ext_names), \
        f"sync_patch_array / exchange_data must NOT be external; got ExternalCall names {sorted(ext_names)}"
    # the only external boundary is MPI, and it is a real MPI library node
    assert any(isinstance(n, MPINode) for n, _ in sdfg.all_nodes_recursive()), \
        "the MPI primitives should remain, as dace.libraries.mpi library nodes"
