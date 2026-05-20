# Distributed-Jacobi mock project: builds the real MPI + netCDF
# executable (so we can prove the codebase compiles end-to-end), and
# also a separate ``emit_hlfir`` target that runs flang-new-21 with
# ``-emit-hlfir`` over every TU to drop the corresponding ``.hlfir``
# files into the build dir.  The prebuilt-HLFIR test calls cmake to
# do both, then hands the build dir to
# ``dace_fortran.build_sdfg_from_hlfir(<build_dir>,
# entry="_QMmod_jacobiPjacobi2d_update")``.

cmake_minimum_required(VERSION 3.18)
project(prebuilt_hlfir_jacobi Fortran)

find_package(MPI REQUIRED COMPONENTS Fortran)
# NetCDF Fortran isn't a first-class Find module everywhere; the
# system pkg-config entry is the portable fallback (Ubuntu's
# libnetcdff-dev ships one).
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF_FORTRAN REQUIRED IMPORTED_TARGET netcdf-fortran)

set(JACOBI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/mod_grid.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/mod_jacobi.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/mod_io.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/driver.f90)

# Real executable (gfortran/ifort/whatever -- the system Fortran,
# not flang).  Verifies the project actually compiles + links
# against real MPI + netCDF.
add_executable(jacobi ${JACOBI_SOURCES})
target_link_libraries(jacobi
    PRIVATE MPI::MPI_Fortran PkgConfig::NETCDF_FORTRAN)

# ``emit_hlfir`` target: drives flang-new-21 over the same sources
# to drop one ``.hlfir`` per TU into the build dir.  flang has its
# own ``.mod`` format (incompatible with the system Fortran's), so
# we point ``-J`` at a fresh ``hlfir_mods/`` subdir to avoid
# clobbering the real executable's ``.mod`` files.
find_program(FLANG_NEW flang-new-21)
if (NOT FLANG_NEW)
    message(WARNING "flang-new-21 not found; emit_hlfir target unavailable")
else()
    set(HLFIR_MOD_DIR ${CMAKE_CURRENT_BINARY_DIR}/hlfir_mods)
    file(MAKE_DIRECTORY ${HLFIR_MOD_DIR})

    # flang doesn't ship a flang-format mpi.mod / netcdf.mod (those
    # are gfortran binary modules from libnetcdff-dev / openmpi); we
    # use ``mpi_stub.f90`` + ``netcdf_stub.f90`` so flang has enough
    # of the interface to type-check the project.  The stubs are
    # only used for HLFIR emission, never linked into the real
    # executable above.
    set(HLFIR_STUB_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_stub.f90
        ${CMAKE_CURRENT_SOURCE_DIR}/netcdf_stub.f90)
    set(HLFIR_FILES)
    foreach(SRC ${HLFIR_STUB_SOURCES})
        get_filename_component(STEM ${SRC} NAME_WE)
        # Stubs go to .mod only (no .hlfir needed -- they're not
        # the SDFG entry).
        add_custom_command(
            OUTPUT ${HLFIR_MOD_DIR}/${STEM}_stub.tag
            COMMAND ${FLANG_NEW} -fc1 -fsyntax-only -fhermetic-module-files
                    -J ${HLFIR_MOD_DIR} -I ${HLFIR_MOD_DIR} ${SRC}
            COMMAND ${CMAKE_COMMAND} -E touch ${HLFIR_MOD_DIR}/${STEM}_stub.tag
            DEPENDS ${SRC}
            COMMENT "flang -fsyntax-only ${STEM}.f90 (stub .mod)"
            VERBATIM)
        list(APPEND HLFIR_STUB_TAGS ${HLFIR_MOD_DIR}/${STEM}_stub.tag)
    endforeach()

    foreach(SRC ${JACOBI_SOURCES})
        get_filename_component(STEM ${SRC} NAME_WE)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${STEM}.hlfir
            COMMAND ${FLANG_NEW} -fc1 -cpp -U_OPENMP -U_OPENACC
                    -fhermetic-module-files
                    -J ${HLFIR_MOD_DIR} -I ${HLFIR_MOD_DIR}
                    -emit-hlfir ${SRC}
                    -o ${CMAKE_CURRENT_BINARY_DIR}/${STEM}.hlfir
            DEPENDS ${SRC} ${HLFIR_STUB_TAGS}
            COMMENT "flang -emit-hlfir ${STEM}.f90"
            VERBATIM)
        list(APPEND HLFIR_FILES ${CMAKE_CURRENT_BINARY_DIR}/${STEM}.hlfir)
    endforeach()
    add_custom_target(emit_hlfir DEPENDS ${HLFIR_FILES})
endif()
