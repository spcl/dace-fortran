! Grid-state container shared by every other module.  No external
! deps (no MPI, no netCDF) -- the leaf in the dependency graph.
module mod_grid
  use iso_c_binding, only: c_double, c_int
  implicit none
  private

  integer, parameter, public :: dp = c_double
  integer, parameter, public :: ik = c_int

  type, public :: grid_t
    integer(ik) :: nx           !< interior extent in x
    integer(ik) :: ny           !< interior extent in y
    integer(ik) :: halo         !< halo width (=1 for a 5-point stencil)
    real(dp), allocatable :: u(:, :)   !< field, shape (1-halo:nx+halo, 1-halo:ny+halo)
  end type grid_t
end module mod_grid
