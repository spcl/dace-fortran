# Distributed-Jacobi mock project: plain MPI + netCDF Fortran build,
# the way a real codebase ships its CMakeLists.txt -- it knows
# nothing about HLFIR or the dace-fortran bridge.  The test's job is
# to verify the project compiles + links normally; emitting HLFIR
# for the bridge is a separate step the test handles by invoking
# ``python -m dace_fortran.emit_hlfir`` after this build.

cmake_minimum_required(VERSION 3.18)
project(prebuilt_hlfir_jacobi Fortran)

find_package(MPI REQUIRED COMPONENTS Fortran)
# NetCDF Fortran isn't a first-class Find module everywhere; the
# system pkg-config entry is the portable fallback (Ubuntu's
# libnetcdff-dev ships one).
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF_FORTRAN REQUIRED IMPORTED_TARGET netcdf-fortran)

add_executable(jacobi
    mod_grid.f90
    mod_jacobi.f90
    mod_io.f90
    driver.f90)
target_link_libraries(jacobi
    PRIVATE MPI::MPI_Fortran PkgConfig::NETCDF_FORTRAN)
