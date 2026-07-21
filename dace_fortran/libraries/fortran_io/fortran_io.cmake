# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
#
# Compile the shipped dace_fortran_io.f90 wrappers into one relocatable object
# and append it to DACE_OBJECTS, so the generated add_library builds and links it.

set(DACE_FORTRAN_IO_SRC ${CMAKE_CURRENT_LIST_DIR}/dace_fortran_io.f90)
set(DACE_FORTRAN_IO_OBJ ${CMAKE_CURRENT_BINARY_DIR}/dace_fortran_io.o)

add_custom_command(
  OUTPUT ${DACE_FORTRAN_IO_OBJ}
  COMMAND gfortran -c -fPIC -O2 -J${CMAKE_CURRENT_BINARY_DIR} ${DACE_FORTRAN_IO_SRC} -o ${DACE_FORTRAN_IO_OBJ}
  DEPENDS ${DACE_FORTRAN_IO_SRC}
  VERBATIM
)

set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_FORTRAN_IO_OBJ})
