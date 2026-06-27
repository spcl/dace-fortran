# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``rename_specifics``: disambiguate a specific module procedure that shares its
name with the generic interface it belongs to.

ICON's ``mo_mpi`` declares ``INTERFACE p_wait`` whose ``MODULE PROCEDURE`` list
includes a specific *also* named ``p_wait`` (the no-argument wait).  That name
sharing is legal Fortran, but externalising the generic leaves the inliner
emitting a dangling ``USE mo_mpi, ONLY: ... => p_wait`` (the bare ``p_wait`` is
now ambiguous and its specific was stubbed/dropped).  Renaming the specific to a
fresh name breaks the collision -- the generic name and the (generic-dispatched)
call sites stay, resolving to the renamed specific.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.external_functions import ExternalFunction
from dace_fortran.fparser_inliner import inline_to_ast

_SRC = """
module mo_mpi
  implicit none
  interface p_wait
    module procedure p_wait
    module procedure p_wait_1
  end interface
contains
  subroutine p_wait()
  end subroutine
  subroutine p_wait_1(req)
    integer, intent(in) :: req
  end subroutine
end module
module m
  use mo_mpi
  implicit none
contains
  subroutine kern(req)
    integer, intent(in) :: req
    call p_wait()
    call p_wait(req)
  end subroutine
end module
"""


def test_rename_clashing_specific_disambiguates_generic(tmp_path: Path):
    """With the rename, externalising the generic ``p_wait`` no longer dangles: the
    specific is renamed (so it's distinct from the generic) and the calls resolve
    to it -- the whole TU compiles."""
    ast = inline_to_ast({"s.f90": _SRC},
                        entry="m::kern",
                        external_functions=[ExternalFunction("p_wait")],
                        rename_specifics={"p_wait": "p_wait_noarg"},
                        tolerate_external_uses=True)
    out = ast.tofortran()
    low = out.lower()
    # the specific was renamed; no dangling bare `=> p_wait` import survives
    assert "p_wait_noarg" in low
    assert "=> p_wait\n" not in low and "=> p_wait," not in low and "=> p_wait " not in low
    if shutil.which("gfortran"):
        (tmp_path / "renamed.f90").write_text(out)
        subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "renamed.f90"],
                              cwd=str(tmp_path))


def test_rename_skips_non_collision_and_external_names():
    """A name that is not a source-defined generic/specific collision is left
    untouched -- so an external (``mpi_*``) name is never renamed."""
    ast = inline_to_ast({"s.f90": _SRC},
                        entry="m::kern",
                        external_functions=[ExternalFunction("p_wait")],
                        rename_specifics={
                            "mpi_wait": "mpi_wait_x",
                            "p_wait_1": "p_wait_1_x"
                        },
                        tolerate_external_uses=True)
    low = ast.tofortran().lower()
    # neither the external name nor the non-colliding specific (p_wait_1 has no
    # same-named generic) is renamed
    assert "mpi_wait_x" not in low
    assert "p_wait_1_x" not in low
