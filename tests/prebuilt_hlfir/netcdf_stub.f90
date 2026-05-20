! flang-compatible netCDF stub.  See ``mpi_stub.f90`` for the
! rationale -- ``libnetcdff-dev`` 's ``netcdf.mod`` is a gfortran
! binary module flang cannot consume, so we provide just enough of
! the ``netcdf`` Fortran API to type-check the project for HLFIR
! emission.  The real executable still links the system netCDF.
module netcdf
  implicit none
  integer, parameter :: NF90_NOERR = 0
  integer, parameter :: NF90_CLOBBER = 0
  integer, parameter :: NF90_DOUBLE = 6
contains
  integer function nf90_create(path, cmode, ncid) result(status)
    character(len=*), intent(in) :: path
    integer, intent(in)  :: cmode
    integer, intent(out) :: ncid
    ncid = 0
    status = NF90_NOERR
  end function nf90_create

  integer function nf90_def_dim(ncid, name, len, dimid) result(status)
    integer, intent(in)  :: ncid, len
    character(len=*), intent(in) :: name
    integer, intent(out) :: dimid
    dimid = 0
    status = NF90_NOERR
  end function nf90_def_dim

  integer function nf90_def_var(ncid, name, xtype, dimids, varid) result(status)
    integer, intent(in)  :: ncid, xtype, dimids(:)
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid
    varid = 0
    status = NF90_NOERR
  end function nf90_def_var

  integer function nf90_enddef(ncid) result(status)
    integer, intent(in) :: ncid
    status = NF90_NOERR
  end function nf90_enddef

  integer function nf90_put_var(ncid, varid, values) result(status)
    integer, intent(in) :: ncid, varid
    real(8), intent(in) :: values(:, :)
    status = NF90_NOERR
  end function nf90_put_var

  integer function nf90_close(ncid) result(status)
    integer, intent(in) :: ncid
    status = NF90_NOERR
  end function nf90_close
end module netcdf
