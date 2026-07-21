# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Fortran external-file I/O as DaCe library nodes.

The HLFIR bridge lowers Fortran ``READ`` / ``WRITE`` and namelist I/O to the
nodes in this library.  Each node fuses the ``open`` / transfer / ``close`` of
one I/O statement and expands to a C++ tasklet calling into the shipped
``dace_fortran_io.f90`` runtime (``iso_c_binding`` wrappers, linked via
``libgfortran``), so transfers keep exact Fortran list-directed semantics.
"""
from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "fortran_io")
