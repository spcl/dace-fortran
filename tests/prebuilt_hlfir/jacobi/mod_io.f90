! NetCDF output module.  Sister-file to ``mod_jacobi`` -- never
! compiled into an SDFG by the prebuilt-HLFIR test, just here to
! prove that *other* modules in the project can keep depending on
! netCDF without the bridge needing netCDF on its own end.
module mod_io
  use netcdf
  use mod_grid, only: grid_t, dp
  implicit none
  private
  public :: dump_to_netcdf

contains

  subroutine dump_to_netcdf(g, filename)
    type(grid_t), intent(in) :: g
    character(len=*), intent(in) :: filename
    integer :: ncid, varid, status, x_dim, y_dim

    status = nf90_create(filename, NF90_CLOBBER, ncid)
    status = nf90_def_dim(ncid, "x", int(g%nx), x_dim)
    status = nf90_def_dim(ncid, "y", int(g%ny), y_dim)
    status = nf90_def_var(ncid, "u", NF90_DOUBLE, [x_dim, y_dim], varid)
    status = nf90_enddef(ncid)
    status = nf90_put_var(ncid, varid, g%u(1:g%nx, 1:g%ny))
    status = nf90_close(ncid)
  end subroutine dump_to_netcdf

end module mod_io
