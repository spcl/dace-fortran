! flang-compatible ``netcdf`` module stub for emitting ICON HLFIR.
!
! ``libnetcdff`` 's ``netcdf.mod`` is a gfortran binary module flang
! cannot parse.  ICON's ``mo_netcdf`` does ``USE netcdf`` with ``PUBLIC``
! (re-exporting the API) and is pulled into the dynamical core's
! USE-closure structurally -- ``mo_solve_nonhydro`` -> ``mo_grid_config``
! -> ``mo_netcdf`` -- not because ``solve_nh`` does any netCDF I/O.  So
! flang only needs a readable ``netcdf.mod`` to type-check that closure.
!
! This declares the surface the closure's netCDF users reference (grid /
! restart config readers).  Attribute / variable ``values`` arguments use
! ``type(*), dimension(..)`` so a call with any type and rank type-checks;
! trailing inquiry outputs are ``optional`` to match the real API's
! keyword-argument call sites.  Bodies are empty: the real executable
! links the system netCDF.
module netcdf
  implicit none
  public

  integer, parameter :: NF90_NOERR = 0
  integer, parameter :: NF90_EINVAL = -36
  integer, parameter :: NF90_GLOBAL = 0
  integer, parameter :: NF90_NOWRITE = 0
  integer, parameter :: NF90_WRITE = 1
  integer, parameter :: NF90_CLOBBER = 0
  integer, parameter :: NF90_MAX_NAME = 256
  integer, parameter :: NF90_MAX_VAR_DIMS = 32
  integer, parameter :: NF90_DOUBLE = 6
  integer, parameter :: NF90_FLOAT = 5
  integer, parameter :: NF90_INT = 4
  integer, parameter :: NF90_CHAR = 2

contains

  integer function nf90_open(path, mode, ncid) result(status)
    character(len=*), intent(in) :: path
    integer, intent(in)  :: mode
    integer, intent(out) :: ncid
    ncid = 0; status = NF90_NOERR
  end function nf90_open

  integer function nf90_close(ncid) result(status)
    integer, intent(in) :: ncid
    status = NF90_NOERR
  end function nf90_close

  function nf90_strerror(ncerr) result(string)
    integer, intent(in) :: ncerr
    character(len=80) :: string
    string = ''
  end function nf90_strerror

  integer function nf90_inquire(ncid, nDimensions, nVariables, nAttributes, &
                                unlimitedDimId, formatNum) result(status)
    integer, intent(in)            :: ncid
    integer, intent(out), optional :: nDimensions, nVariables, nAttributes
    integer, intent(out), optional :: unlimitedDimId, formatNum
    if (present(nDimensions)) nDimensions = 0
    if (present(nVariables)) nVariables = 0
    if (present(nAttributes)) nAttributes = 0
    if (present(unlimitedDimId)) unlimitedDimId = 0
    if (present(formatNum)) formatNum = 0
    status = NF90_NOERR
  end function nf90_inquire

  integer function nf90_inq_attname(ncid, varid, attnum, name) result(status)
    integer, intent(in)           :: ncid, varid, attnum
    character(len=*), intent(out) :: name
    name = ''; status = NF90_NOERR
  end function nf90_inq_attname

  integer function nf90_inq_dimid(ncid, name, dimid) result(status)
    integer, intent(in)          :: ncid
    character(len=*), intent(in) :: name
    integer, intent(out)         :: dimid
    dimid = 0; status = NF90_NOERR
  end function nf90_inq_dimid

  integer function nf90_inq_varid(ncid, name, varid) result(status)
    integer, intent(in)          :: ncid
    character(len=*), intent(in) :: name
    integer, intent(out)         :: varid
    varid = 0; status = NF90_NOERR
  end function nf90_inq_varid

  integer function nf90_inquire_dimension(ncid, dimid, name, len) result(status)
    integer, intent(in)                     :: ncid, dimid
    character(len=*), intent(out), optional :: name
    integer, intent(out), optional          :: len
    if (present(name)) name = ''
    if (present(len)) len = 0
    status = NF90_NOERR
  end function nf90_inquire_dimension

  integer function nf90_inquire_variable(ncid, varid, name, xtype, ndims, &
                                         dimids, natts) result(status)
    integer, intent(in)                     :: ncid, varid
    character(len=*), intent(out), optional :: name
    integer, intent(out), optional          :: xtype, ndims, natts
    integer, intent(out), optional          :: dimids(:)
    if (present(name)) name = ''
    if (present(xtype)) xtype = 0
    if (present(ndims)) ndims = 0
    if (present(natts)) natts = 0
    status = NF90_NOERR
  end function nf90_inquire_variable

  integer function nf90_inquire_attribute(ncid, varid, name, xtype, len, &
                                          attnum) result(status)
    integer, intent(in)            :: ncid, varid
    character(len=*), intent(in)   :: name
    integer, intent(out), optional :: xtype, len, attnum
    if (present(xtype)) xtype = 0
    if (present(len)) len = 0
    if (present(attnum)) attnum = 0
    status = NF90_NOERR
  end function nf90_inquire_attribute

  integer function nf90_put_att(ncid, varid, name, values) result(status)
    integer, intent(in)               :: ncid, varid
    character(len=*), intent(in)      :: name
    type(*), dimension(..), intent(in) :: values
    status = NF90_NOERR
  end function nf90_put_att

  integer function nf90_get_att(ncid, varid, name, values) result(status)
    integer, intent(in)           :: ncid, varid
    character(len=*), intent(in)  :: name
    type(*), dimension(..)        :: values
    status = NF90_NOERR
  end function nf90_get_att

  integer function nf90_get_var(ncid, varid, values, start, count) result(status)
    integer, intent(in)            :: ncid, varid
    type(*), dimension(..)         :: values
    integer, intent(in), optional  :: start(:), count(:)
    status = NF90_NOERR
  end function nf90_get_var

end module netcdf
