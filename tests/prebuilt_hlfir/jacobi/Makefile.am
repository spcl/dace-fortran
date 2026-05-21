bin_PROGRAMS = jacobi
# Listed in USE-dependency order; serial make compiles them in this
# order so each module's .mod exists before its dependents.
jacobi_SOURCES = mod_grid.f90 mod_jacobi.f90 mod_io.f90 driver.f90
AM_FCFLAGS = $(NETCDF_FFLAGS)
jacobi_LDADD = $(NETCDF_FLIBS)
