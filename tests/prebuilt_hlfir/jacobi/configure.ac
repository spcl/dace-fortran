# Autotools build for the distributed-Jacobi mock -- ICON itself is
# autoconf + a hand-written Makefile.in (no automake); we use automake
# here for a robust, standard test.  Either way the HLFIR emission
# path is the same: `bear -- make` intercepts the compiler exec()
# calls and writes compile_commands.json, which the helper consumes.
AC_INIT([jacobi_auto], [1.0])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_PROG_FC([mpif90])
# netCDF-Fortran: pkg-config on Ubuntu omits the module dir, so take
# the authoritative flags from nf-config (ships with libnetcdff-dev).
AC_CHECK_PROG([NF_CONFIG], [nf-config], [nf-config])
AS_IF([test "x$NF_CONFIG" = x],
      [AC_MSG_ERROR([nf-config not found (install libnetcdff-dev)])])
NETCDF_FFLAGS=`$NF_CONFIG --fflags`
NETCDF_FLIBS=`$NF_CONFIG --flibs`
AC_SUBST([NETCDF_FFLAGS])
AC_SUBST([NETCDF_FLIBS])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
