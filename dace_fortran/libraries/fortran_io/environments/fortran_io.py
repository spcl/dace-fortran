# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Build environment for the Fortran-I/O library nodes: compiles the shipped
``dace_fortran_io.f90`` wrappers into the program and links ``libgfortran``.
"""
import os

import dace.library

#: This library's directory, where ``dace_fortran_io.{f90,h}`` and
#: ``fortran_io.cmake`` ship together.
LIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dace.library.environment
class FortranIO:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = [LIB_DIR]
    cmake_libraries = ["gfortran"]
    cmake_compile_flags = [f"-I{LIB_DIR}"]
    cmake_link_flags = []
    cmake_files = [os.path.join(LIB_DIR, "fortran_io.cmake")]

    headers = ["dace_fortran_io.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
