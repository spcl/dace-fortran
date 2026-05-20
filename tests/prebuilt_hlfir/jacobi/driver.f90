! Distributed-Jacobi driver.  Sets up a per-rank sub-grid, runs N
! Jacobi sweeps with halo exchange between them, dumps the final
! field to a per-rank netCDF file.  This is the executable cmake
! links and runs to verify the project genuinely builds end-to-end;
! the bridge / SDFG path only consumes the prebuilt HLFIR for
! ``mod_jacobi.jacobi2d_update``.
program jacobi_driver
  use mpi
  use mod_grid, only: grid_t, dp, ik
  use mod_jacobi, only: jacobi2d_update, halo_exchange
  use mod_io, only: dump_to_netcdf
  implicit none

  integer, parameter :: nsteps = 4
  integer(ik), parameter :: nx_local = 8
  integer(ik), parameter :: ny_local = 8
  type(grid_t) :: g
  real(dp), allocatable :: u_next(:, :)
  integer :: rank, nproc, comm, ierr
  integer :: neighbours(4)
  character(len=64) :: outname
  integer :: step

  call MPI_Init(ierr)
  comm = MPI_COMM_WORLD
  call MPI_Comm_rank(comm, rank, ierr)
  call MPI_Comm_size(comm, nproc, ierr)

  ! 1D ring decomposition for simplicity: only N/S neighbours active.
  neighbours(1) = modulo(rank - 1 + nproc, nproc)
  neighbours(2) = modulo(rank + 1, nproc)
  neighbours(3) = MPI_PROC_NULL
  neighbours(4) = MPI_PROC_NULL

  g%nx = nx_local
  g%ny = ny_local
  g%halo = 1
  allocate (g%u(0:g%nx + 1, 0:g%ny + 1))
  allocate (u_next(0:g%nx + 1, 0:g%ny + 1))

  ! Deterministic init: per-rank ramp so the comparison is
  ! reproducible without an RNG dep at link time.
  block
    integer :: i, j
    do j = 0, g%ny + 1
      do i = 0, g%nx + 1
        g%u(i, j) = real(rank, dp) + 0.01_dp * real(i, dp) + 0.001_dp * real(j, dp)
      end do
    end do
  end block

  do step = 1, nsteps
    call halo_exchange(g, comm, neighbours)
    call jacobi2d_update(g%u, u_next, g%nx, g%ny)
    g%u = u_next
  end do

  write (outname, '("jacobi_out_rank", i4.4, ".nc")') rank
  call dump_to_netcdf(g, trim(outname))

  deallocate (g%u, u_next)
  call MPI_Finalize(ierr)
end program jacobi_driver
