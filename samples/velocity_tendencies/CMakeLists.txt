# velocity_data -- nanobind extension for the velocity_tendencies sample.
# Build pattern (Python discovery + nanobind via its cmake_dir + nanobind_add_module)
# copied from dace_fortran/CMakeLists.txt, minus its LLVM/MLIR machinery: this module
# only parses ASCII data files, so a plain host C++20 compiler is enough.
cmake_minimum_required(VERSION 3.18)
project(velocity_data CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_STATUS)
if(NOT NB_STATUS EQUAL 0)
    message(FATAL_ERROR "nanobind not found.  Install with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(velocity_data NB_STATIC velocity_data.cpp)
install(TARGETS velocity_data LIBRARY DESTINATION .)
