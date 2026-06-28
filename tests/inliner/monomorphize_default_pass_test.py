# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The monomorphisation pass runs in the fparser inliner BY DEFAULT and ALWAYS
(:class:`dace_fortran.fparser_inliner.ParseConfig.monomorphize`), so a single-level
abstract type-bound dispatch the bridge cannot lower is devirtualised into static
calls without anyone asking for a per-kernel spec.

This is the engine prerequisite for extracting ICON's ``solve_nonhydro`` with the
halo exchange inlined: ICON's standard build cpp-strips ``t_comm_pattern_yaxt``,
leaving ``t_comm_pattern`` with a single concrete arm, which the default pass
retypes so ``p_pat%exchange_data_*`` becomes a static call the inliner inlines.

These tests use a faithful miniature of the real dispatch chain
(``solve_nh`` -> generic wrapper -> ``CLASS(t_comm_pattern)`` dispatch -> the
``orig`` override) and drive it through the ACTUAL :func:`inline_to_ast` pipeline
(not the engine in isolation), so they pin the pass's wiring + ordering relative
to the call-resolution passes.  The SDFG-level proof (MPI libnodes, no
``ExternalCall``) lives in ``tests/sync_devirt_mpi_libnode_test.py``.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.fparser_inliner import inline_to_ast

#: A faithful miniature of ICON's halo dispatch chain, single-arm (``yaxt``
#: cpp-stripped): the entry calls a generic wrapper that takes the abstract
#: ``CLASS(t_comm_pattern)`` and dispatches ``p_pat%exchange_data_r3d``, bound in
#: the lone concrete arm ``t_comm_pattern_orig`` to ``orig_exchange``.
_CHAIN_SRC = """
module mo_comm
  implicit none
  type, abstract :: t_comm_pattern
  contains
    procedure(exch_i), deferred :: exchange_data_r3d
  end type
  abstract interface
    subroutine exch_i(p_pat, arr)
      import t_comm_pattern
      class(t_comm_pattern), intent(in) :: p_pat
      real, intent(inout) :: arr(:)
    end subroutine
  end interface
  type, extends(t_comm_pattern) :: t_comm_pattern_orig
  contains
    procedure :: exchange_data_r3d => orig_exchange
  end type
contains
  subroutine orig_exchange(p_pat, arr)
    class(t_comm_pattern_orig), intent(in) :: p_pat
    real, intent(inout) :: arr(:)
    integer :: i
    do i = 1, size(arr)
      arr(i) = arr(i) + 1.0
    end do
  end subroutine
  subroutine exchange_data_r3d_wrap(p_pat, arr)
    class(t_comm_pattern), intent(in) :: p_pat
    real, intent(inout) :: arr(:)
    call p_pat%exchange_data_r3d(arr)
  end subroutine
end module

module mo_solve
  use mo_comm
  implicit none
contains
  subroutine solve_nh(p_pat, arr)
    class(t_comm_pattern), intent(in) :: p_pat
    real, intent(inout) :: arr(:)
    arr = arr * 2.0
    call exchange_data_r3d_wrap(p_pat, arr)
  end subroutine
end module
"""


def test_default_pass_devirtualizes_single_arm_chain():
    """Through the real pipeline (default ``monomorphize=True``), the halo dispatch
    is gone: no ``Procedure_Designator`` (``%binding`` call) survives anywhere, and
    every ``CLASS(t_comm_pattern)`` outside the deferred interface signature is
    retyped to ``TYPE(t_comm_pattern_orig)``."""
    ast = inline_to_ast({"src.f90": _CHAIN_SRC}, entry="mo_solve::solve_nh")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived devirtualisation"
    text = ast.tofortran()
    low = text.lower()
    assert "type(t_comm_pattern_orig)" in low
    # the entry's passed-object is now concrete
    body = re.search(r"subroutine solve_nh.*?end subroutine", text, re.IGNORECASE | re.DOTALL).group(0).lower()
    assert "type(t_comm_pattern_orig)" in body
    assert "class(t_comm_pattern)" not in body


def test_default_pass_can_be_disabled():
    """``monomorphize=False`` is the kill-switch: the dispatch is left intact (the
    pass did not run), so the abstract ``CLASS`` dispatch still stands.  Run in
    merge mode (``optimize=False``) so the unresolved polymorphic dispatch -- which
    only the pass would have collapsed -- doesn't trip the constant-prop optimizer
    (the loud rejection of un-monomorphised dispatch on a non-fparser-optimised
    path)."""
    ast = inline_to_ast({"src.f90": _CHAIN_SRC}, entry="mo_solve::solve_nh", monomorphize=False, optimize=False)
    # with the pass off, NO retype happened: the abstract CLASS dummy is left
    # polymorphic (the concrete arm type never replaces it).
    low = ast.tofortran().lower()
    assert "type(t_comm_pattern_orig)" not in low, "monomorphize=False must not retype the abstract dummy"
    assert "class(t_comm_pattern)" in low, "the abstract CLASS dispatch dummy should survive when the pass is off"


#: ICON's hardest cross-module shape: the abstract base AND a container holding a
#: ``POINTER`` to it live in one module (``m_types``); the concrete arm is in a
#: separate downstream module (``m_orig``).  Retyping the container pointer to the
#: arm would make ``m_types`` depend on ``m_orig``, which already ``USE``s
#: ``m_types`` to EXTEND the base -- a circular module dependency.  The pass
#: resolves it by consolidating the arm (type + procedures) into the base module
#: in topological order (``t_base`` -> ``t_orig`` -> ``t_holder``).
_CYCLE_SRC = """
module m_types
  implicit none
  type, abstract :: t_base
  contains
    procedure(ie), deferred :: exch
  end type
  abstract interface
    subroutine ie(p,a)
      import t_base
      class(t_base), intent(in) :: p
      real, intent(inout) :: a(:)
    end subroutine
  end interface
  type :: t_holder
    class(t_base), pointer :: p => null()
  end type
contains
  subroutine wrap(p, a)
    class(t_base), intent(in) :: p
    real, intent(inout) :: a(:)
    call p%exch(a)
  end subroutine
end module
module m_orig
  use m_types
  implicit none
  type, extends(t_base) :: t_orig
  contains
    procedure :: exch => orig_exch
  end type
contains
  subroutine orig_exch(p, a)
    class(t_orig), intent(in) :: p
    real, intent(inout) :: a(:)
    a = a + 1.0
  end subroutine
end module
module m_use
  use m_types
  use m_orig
  implicit none
contains
  subroutine kern(h, a)
    type(t_holder), intent(in) :: h
    real, intent(inout) :: a(:)
    call wrap(h%p, a)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_retype_consolidates_arm_module_no_cycle(tmp_path: Path):
    """The cross-module retype that would create a circular module dependency is
    resolved by consolidating the arm into the base module -- and the result is
    valid, compilable Fortran (the container pointer + the dispatch are both
    statically typed to the concrete arm)."""
    ast = inline_to_ast({"s.f90": _CYCLE_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    low = out.lower()
    # the arm type + its procedure are now in the base module; the arm module is gone
    assert "module m_orig" not in low
    assert "t_orig" in low
    # the dispatch resolved to the concrete arm's override (no residual %exch)
    assert not walk(ast, f03.Procedure_Designator)
    assert "call orig_exch" in low
    # and it compiles -- no "used before defined" / circular USE
    src = tmp_path / "consolidated.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "consolidated.f90"],
                          cwd=str(tmp_path))


#: The same cycle, but the base module is TYPES-ONLY (no CONTAINS) and the halo
#: wrapper lives in a THIRD module -- so consolidation must (a) create a fresh
#: CONTAINS in the base module placed BEFORE its END MODULE (not orphaned after
#: it), and (b) import the arm into the third module's wrapper.  This is the exact
#: shape of ICON's mo_communication_types / mo_communication / mo_communication_orig.
_CYCLE_3MOD_SRC = """
module m_types
  implicit none
  type, abstract :: t_base
  contains
    procedure(ie), deferred :: exch
  end type
  abstract interface
    subroutine ie(p,a)
      import t_base
      class(t_base), intent(in) :: p
      real, intent(inout) :: a(:)
    end subroutine
  end interface
  type :: t_holder
    class(t_base), pointer :: p => null()
  end type
end module
module m_orig
  use m_types
  implicit none
  type, extends(t_base) :: t_orig
  contains
    procedure :: exch => orig_exch
  end type
contains
  subroutine orig_exch(p, a)
    class(t_orig), intent(in) :: p
    real, intent(inout) :: a(:)
    a = a + 1.0
  end subroutine
end module
module m_wrap
  use m_types
  implicit none
contains
  subroutine wrap(p, a)
    class(t_base), intent(in) :: p
    real, intent(inout) :: a(:)
    call p%exch(a)
  end subroutine
end module
module m_use
  use m_types
  use m_orig
  use m_wrap
  implicit none
contains
  subroutine kern(h, a)
    type(t_holder), intent(in) :: h
    real, intent(inout) :: a(:)
    call wrap(h%p, a)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_retype_consolidates_into_types_only_base_with_third_module_wrapper(tmp_path: Path):
    """ICON's exact module shape: a TYPES-ONLY base module (no CONTAINS), the arm
    in a second module, the halo wrapper in a third.  Consolidation must add a
    CONTAINS to the base before its END MODULE and import the arm into the wrapper
    module -- producing compilable Fortran."""
    ast = inline_to_ast({"s.f90": _CYCLE_3MOD_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    assert "module m_orig" not in out.lower()
    assert not walk(ast, f03.Procedure_Designator)
    src = tmp_path / "consolidated3.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "consolidated3.f90"],
                          cwd=str(tmp_path))


#: The arm type is referenced by a COMPONENT of a type in YET ANOTHER module
#: (``m_domain``'s ``t_patch`` holds a ``CLASS(t_base), POINTER :: comm`` -- ICON's
#: ``mo_model_domain``'s ``t_patch%comm_pat_*``).  After the retype the component
#: names the arm, and consolidation must import the arm into ``m_domain`` at the
#: MODULE level so the type definition sees it (host association then covers any
#: subprograms in that module too).
_CYCLE_COMPONENT_SRC = """
module m_types
  implicit none
  type, abstract :: t_base
  contains
    procedure(ie), deferred :: exch
  end type
  abstract interface
    subroutine ie(p,a)
      import t_base
      class(t_base), intent(in) :: p
      real, intent(inout) :: a(:)
    end subroutine
  end interface
end module
module m_orig
  use m_types
  implicit none
  type, extends(t_base) :: t_orig
  contains
    procedure :: exch => orig_exch
  end type
contains
  subroutine orig_exch(p, a)
    class(t_orig), intent(in) :: p
    real, intent(inout) :: a(:)
    a = a + 1.0
  end subroutine
end module
module m_domain
  use m_types
  implicit none
  type :: t_patch
    class(t_base), pointer :: comm => null()
  end type
end module
module m_wrap
  use m_types
  implicit none
contains
  subroutine wrap(p, a)
    class(t_base), intent(in) :: p
    real, intent(inout) :: a(:)
    call p%exch(a)
  end subroutine
end module
module m_use
  use m_types
  use m_orig
  use m_domain
  use m_wrap
  implicit none
contains
  subroutine kern(pt, a)
    type(t_patch), intent(in) :: pt
    real, intent(inout) :: a(:)
    call wrap(pt%comm, a)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_retype_imports_arm_into_module_with_component(tmp_path: Path):
    """When the arm type is used as a type COMPONENT in a separate module
    (``mo_model_domain``'s ``t_patch%comm_pat_*`` shape), consolidation imports the
    arm into that module so the type def resolves it -- producing compilable
    Fortran."""
    ast = inline_to_ast({"s.f90": _CYCLE_COMPONENT_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    assert "module m_orig" not in out.lower()
    assert not walk(ast, f03.Procedure_Designator)
    src = tmp_path / "comp.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "comp.f90"], cwd=str(tmp_path))
