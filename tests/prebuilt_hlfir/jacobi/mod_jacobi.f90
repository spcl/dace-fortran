! Jacobi 2D update + halo exchange.  Lives in ONE file even though
! only ``halo_exchange`` uses MPI -- the point of the prebuilt-HLFIR
! test is to confirm the bridge can lower ``jacobi2d_update`` to an
! SDFG even though the enclosing module ``use mpi`` and a sibling
! procedure does call ``MPI_Sendrecv``.  flang produces one ``.hlfir``
! for the whole module; the bridge selects ``func.func
! @_QMmod_jacobiPjacobi2d_update`` out of that and never has to look
! at the MPI calls in ``halo_exchange``.
module mod_jacobi
  use mpi
  use mod_grid, only: grid_t, dp, ik
  implicit none
  private
  public :: jacobi2d_update, halo_exchange

contains

  !> Pure 5-point stencil applied to one interior point.  Called from
  !! the hot loop in :subr:`jacobi2d_update` so the bridge's
  !! ``hlfir-inline-all`` pass folds it in -- the SDFG does not get
  !! a separate ``stencil_5pt`` call site, which the test asserts.
  pure real(dp) function stencil_5pt(u_west, u_east, u_south, u_north) result(r)
    real(dp), intent(in) :: u_west, u_east, u_south, u_north
    r = 0.25_dp * (u_west + u_east + u_south + u_north)
  end function stencil_5pt

  !> One Jacobi sweep over the interior of a halo-padded field.  No
  !! MPI usage even though the enclosing module imports it.
  subroutine jacobi2d_update(u_in, u_out, nx, ny)
    integer(ik), intent(in) :: nx
    integer(ik), intent(in) :: ny
    real(dp), intent(in)    :: u_in(0:nx + 1, 0:ny + 1)
    real(dp), intent(out)   :: u_out(0:nx + 1, 0:ny + 1)
    integer :: i, j

    do j = 1, ny
      do i = 1, nx
        u_out(i, j) = stencil_5pt(u_in(i - 1, j), u_in(i + 1, j), &
                                  u_in(i, j - 1), u_in(i, j + 1))
      end do
    end do
  end subroutine jacobi2d_update

  !> Halo exchange across the 4-connected neighbour set.  This one
  !! is the MPI-heavy sibling -- never compiled into the SDFG.
  subroutine halo_exchange(g, comm, neighbours)
    type(grid_t), intent(inout) :: g
    integer, intent(in) :: comm
    integer, intent(in) :: neighbours(4)  ! N, S, E, W (MPI_PROC_NULL for boundary)
    integer :: ierr, tag, status(MPI_STATUS_SIZE)
    real(dp), allocatable :: row_send(:), row_recv(:)
    real(dp), allocatable :: col_send(:), col_recv(:)

    allocate (row_send(g%nx), row_recv(g%nx))
    allocate (col_send(g%ny), col_recv(g%ny))
    tag = 0

    row_send = g%u(1:g%nx, g%ny)
    call MPI_Sendrecv(row_send, g%nx, MPI_DOUBLE_PRECISION, neighbours(1), tag, &
                      row_recv, g%nx, MPI_DOUBLE_PRECISION, neighbours(2), tag, &
                      comm, status, ierr)
    g%u(1:g%nx, 0) = row_recv

    row_send = g%u(1:g%nx, 1)
    call MPI_Sendrecv(row_send, g%nx, MPI_DOUBLE_PRECISION, neighbours(2), tag, &
                      row_recv, g%nx, MPI_DOUBLE_PRECISION, neighbours(1), tag, &
                      comm, status, ierr)
    g%u(1:g%nx, g%ny + 1) = row_recv

    col_send = g%u(g%nx, 1:g%ny)
    call MPI_Sendrecv(col_send, g%ny, MPI_DOUBLE_PRECISION, neighbours(3), tag, &
                      col_recv, g%ny, MPI_DOUBLE_PRECISION, neighbours(4), tag, &
                      comm, status, ierr)
    g%u(0, 1:g%ny) = col_recv

    col_send = g%u(1, 1:g%ny)
    call MPI_Sendrecv(col_send, g%ny, MPI_DOUBLE_PRECISION, neighbours(4), tag, &
                      col_recv, g%ny, MPI_DOUBLE_PRECISION, neighbours(3), tag, &
                      comm, status, ierr)
    g%u(g%nx + 1, 1:g%ny) = col_recv

    deallocate (row_send, row_recv, col_send, col_recv)
  end subroutine halo_exchange

end module mod_jacobi
