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


#: The MULTI-ARM (ladder) analogue of the single-arm retype cycle: ICON's ocean
#: free-surface solver shape.  An abstract backend ``t_backend`` with a deferred
#: ``run`` and a shared non-deferred interposer ``solve`` (dispatches internally
#: on its own passed-object), a container ``t_solver`` holding a
#: ``CLASS(t_backend), ALLOCATABLE :: act``, and TWO concrete arms (``t_cg`` /
#: ``t_bicg``) in separate downstream modules.  The runtime arm is chosen at the
#: ``ALLOCATE(concrete :: s%act)`` construction site, so the pass CANNOT pin one
#: type -- it emits the tag ladder + a per-arm interposer clone.  Those clones and
#: the per-arm direct binding calls land in the base module and name concrete arm
#: types/procedures defined in the (formerly downstream) arm modules; without
#: consolidation the base module is "used before defined" + a circular USE.
#: Each module also carries a PRIVATE ``this_mod_name`` PARAMETER used in its own
#: bodies -- ICON's per-module idiom.  Consolidating the arm modules into the base
#: must rename these on collision, else the base ends up with three definitions of
#: one identifier.
_LADDER_CYCLE_SRC = """
module m_base
  implicit none
  character(len=*), parameter :: this_mod_name = 'm_base'
  type, abstract :: t_backend
  contains
    procedure(ie), deferred :: run
    procedure :: solve => backend_solve
  end type
  abstract interface
    subroutine ie(this, a)
      import t_backend
      class(t_backend), intent(inout) :: this
      real, intent(inout) :: a(:)
    end subroutine
  end interface
  type :: t_solver
    class(t_backend), allocatable :: act
    integer :: which = 1
  end type
contains
  subroutine backend_solve(this, a)
    class(t_backend), intent(inout) :: this
    real, intent(inout) :: a(:)
    character(len=*), parameter :: routine = this_mod_name // ':backend_solve'
    call this%run(a)
  end subroutine
end module
module m_cg
  use m_base
  implicit none
  character(len=*), parameter :: this_mod_name = 'm_cg'
  type, extends(t_backend) :: t_cg
  contains
    procedure :: run => cg_run
  end type
contains
  subroutine cg_run(this, a)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: a(:)
    character(len=*), parameter :: routine = this_mod_name // ':cg_run'
    a = a + 1.0
  end subroutine
end module
module m_bicg
  use m_base
  implicit none
  character(len=*), parameter :: this_mod_name = 'm_bicg'
  type, extends(t_backend) :: t_bicg
  contains
    procedure :: run => bicg_run
  end type
contains
  subroutine bicg_run(this, a)
    class(t_bicg), intent(inout) :: this
    real, intent(inout) :: a(:)
    character(len=*), parameter :: routine = this_mod_name // ':bicg_run'
    a = a + 2.0
  end subroutine
end module
module m_use
  use m_base
  use m_cg
  use m_bicg
  implicit none
contains
  subroutine kern(s, a)
    type(t_solver), intent(inout) :: s
    real, intent(inout) :: a(:)
    if (s%which == 1) then
      allocate(t_cg :: s%act)
    else
      allocate(t_bicg :: s%act)
    end if
    call s%act%solve(a)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_ladder_consolidates_every_arm_module_no_cycle(tmp_path: Path):
    """A MULTI-ARM axis (runtime-selected, no single type to pin) is laddered, and
    consolidation merges EVERY arm module into the base module -- so the per-arm
    interposer clones and direct binding calls the ladder emits into the base
    module resolve their concrete arm types/procedures locally.  The result is
    fully devirtualised (no residual dispatch) and compilable Fortran."""
    ast = inline_to_ast({"s.f90": _LADDER_CYCLE_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    low = out.lower()
    # both arm modules were consolidated away into the base module
    assert "module m_cg" not in low
    assert "module m_bicg" not in low
    # no runtime dispatch survives -- every arm is a static call
    assert not walk(ast, f03.Procedure_Designator)
    # both arms' overrides are directly called from the emitted ladder
    assert "call cg_run" in low
    assert "call bicg_run" in low
    src = tmp_path / "ladder.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ladder.f90"], cwd=str(tmp_path))


#: A ladder whose shared interposer + arm type names are long enough that the
#: per-arm clone name ``<interposer>__<arm>`` overruns Fortran's 63-char identifier
#: limit -- the shape of ICON's solver, where the shared ``construct`` interposer is
#: cloned once per composed axis (backend x agen x transfer) and the chained
#: ``__<arm>`` suffixes reach ~95 chars.  The clone name must be shortened
#: (readable prefix + stable hash) so gfortran does not reject the SUBROUTINE header
#: ("Name too long", which then cascades into "Unexpected USE in CONTAINS").
_LADDER_LONGNAME_SRC = """
module m_base
  implicit none
  type, abstract :: t_backend
  contains
    procedure(ie), deferred :: run
    procedure :: solve => backend_solve_a_deliberately_verbose_shared_interposer
  end type
  abstract interface
    subroutine ie(this, a)
      import t_backend
      class(t_backend), intent(inout) :: this
      real, intent(inout) :: a(:)
    end subroutine
  end interface
  type :: t_solver
    class(t_backend), allocatable :: act
    integer :: which = 1
  end type
contains
  subroutine backend_solve_a_deliberately_verbose_shared_interposer(this, a)
    class(t_backend), intent(inout) :: this
    real, intent(inout) :: a(:)
    call this%run(a)
  end subroutine
end module
module m_cg
  use m_base
  implicit none
  type, extends(t_backend) :: t_conjugate_gradient_solver_backend_arm
  contains
    procedure :: run => cg_run
  end type
contains
  subroutine cg_run(this, a)
    class(t_conjugate_gradient_solver_backend_arm), intent(inout) :: this
    real, intent(inout) :: a(:)
    a = a + 1.0
  end subroutine
end module
module m_bicg
  use m_base
  implicit none
  type, extends(t_backend) :: t_biconjugate_gradient_stabilized_backend_arm
  contains
    procedure :: run => bicg_run
  end type
contains
  subroutine bicg_run(this, a)
    class(t_biconjugate_gradient_stabilized_backend_arm), intent(inout) :: this
    real, intent(inout) :: a(:)
    a = a + 2.0
  end subroutine
end module
module m_use
  use m_base
  use m_cg
  use m_bicg
  implicit none
contains
  subroutine kern(s, a)
    type(t_solver), intent(inout) :: s
    real, intent(inout) :: a(:)
    if (s%which == 1) then
      allocate(t_conjugate_gradient_solver_backend_arm :: s%act)
    else
      allocate(t_biconjugate_gradient_stabilized_backend_arm :: s%act)
    end if
    call s%act%solve(a)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_ladder_clone_name_shortened_within_fortran_limit(tmp_path: Path):
    """A per-arm interposer clone whose composed name ``<interposer>__<arm>`` exceeds
    Fortran's 63-char identifier limit is shortened (readable prefix + stable hash),
    so the devirtualised TU compiles instead of emitting a rejected SUBROUTINE header.
    Mirrors ICON's multi-axis solver construct (backend x agen x transfer, ~95 chars)."""
    ast = inline_to_ast({"s.f90": _LADDER_LONGNAME_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a dispatch survived"
    out = ast.tofortran()
    # every emitted subprogram name is within the Fortran identifier limit (the
    # subprogram name is the first Name in its SUBROUTINE/FUNCTION statement)
    for stmt in walk(ast, (f03.Subroutine_Stmt, f03.Function_Stmt)):
        name = str(walk(stmt, f03.Name)[0])
        assert len(name) <= 63, f"identifier over Fortran 63-char limit: {name} ({len(name)})"
    # the naive composed name (which would overrun) is NOT emitted verbatim
    assert "backend_solve_a_deliberately_verbose_shared_interposer__t_conjugate" not in out.lower()
    src = tmp_path / "longname.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "longname.f90"], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Pointer-association tag source: ICON's ``t_lhs%trans`` / ``%agen`` are
# ``CLASS(base), POINTER`` slots a constructor binds by ``this%slot => dummy``,
# with the concrete arm arriving as the actual argument one or more call hops
# away (:func:`devirtualize_pointer_flow`).  Unlike the ``ALLOCATE(arm :: slot)``
# ladder these tests exercise the interprocedural clone that carries the concrete
# type to the association and sets the slot's tag there.
# ---------------------------------------------------------------------------

#: Single hop: a ``t_solver`` whose ``CLASS(t_op), POINTER :: op`` is bound in
#: ``setup`` (``this%op => o``), called from ``kern`` with a concrete typed local
#: chosen by SELECT CASE.  The dispatch ``this%op%apply`` reads the stored slot.
_PTR_ASSOC_SINGLE_SRC = """
module m_base
  implicit none
  type, abstract :: t_op
  contains
    procedure(app_i), deferred :: apply
  end type
  abstract interface
    subroutine app_i(this, x)
      import t_op
      class(t_op), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type :: t_solver
    class(t_op), pointer :: op => null()
  contains
    procedure :: setup => solver_setup
    procedure :: run => solver_run
  end type
contains
  subroutine solver_setup(this, o)
    class(t_solver), intent(inout) :: this
    class(t_op), target, intent(in) :: o
    this%op => o
  end subroutine
  subroutine solver_run(this, x)
    class(t_solver), intent(inout) :: this
    real, intent(inout) :: x
    call this%op%apply(x)
  end subroutine
end module
module m_diag
  use m_base
  implicit none
  type, extends(t_op) :: t_diag
  contains
    procedure :: apply => diag_apply
  end type
contains
  subroutine diag_apply(this, x)
    class(t_diag), intent(in) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
end module
module m_scale
  use m_base
  implicit none
  type, extends(t_op) :: t_scale
  contains
    procedure :: apply => scale_apply
  end type
contains
  subroutine scale_apply(this, x)
    class(t_scale), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
end module
module m_use
  use m_base
  use m_diag
  use m_scale
  implicit none
contains
  subroutine kern(x, which)
    real, intent(inout) :: x
    integer, intent(in) :: which
    type(t_solver) :: s
    type(t_diag), target :: d
    type(t_scale), target :: sc
    select case (which)
    case (1)
      call s%setup(d)
    case default
      call s%setup(sc)
    end select
    call s%run(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_pointer_assoc_single_hop_devirtualizes(tmp_path: Path):
    """A two-arm ``CLASS(base), POINTER`` slot whose arm is set by a
    pointer-association (``this%op => o``, no ``ALLOCATE``) is discovered as a
    ladder and fully devirtualised: the slot expands to a tag + per-arm pointer
    slots, ``setup`` is cloned per concrete arm (each sets the tag), the call sites
    are redirected, and no type-bound dispatch survives."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_SINGLE_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived pointer-assoc devirtualisation"
    low = ast.tofortran().lower()
    assert "op__tag" in low, "the pointer slot was not expanded to a type tag"
    src = tmp_path / "ptr1.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr1.f90"], cwd=str(tmp_path))


#: Two hops: the concrete arm flows ``kern -> t_top%setup -> t_holder%construct``,
#: where ``top_setup`` is a PASS-THROUGH (it forwards its ``CLASS(base)`` dummy to
#: the inner constructor without any association of its own).  ``holder_construct``
#: both associates the slot AND dispatches on the dummy (``call o%apply``), so the
#: fixed point must retype the dummy along the whole chain.
_PTR_ASSOC_PASSTHROUGH_SRC = """
module m_base
  implicit none
  type, abstract :: t_op
  contains
    procedure(app_i), deferred :: apply
  end type
  abstract interface
    subroutine app_i(this, x)
      import t_op
      class(t_op), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type :: t_holder
    class(t_op), pointer :: op => null()
    real :: probe
  contains
    procedure :: construct => holder_construct
    procedure :: run => holder_run
  end type
  type :: t_top
    type(t_holder) :: h
  contains
    procedure :: setup => top_setup
  end type
contains
  subroutine holder_construct(this, o)
    class(t_holder), intent(inout) :: this
    class(t_op), target, intent(in) :: o
    this%op => o
    call o%apply(this%probe)
  end subroutine
  subroutine holder_run(this, x)
    class(t_holder), intent(inout) :: this
    real, intent(inout) :: x
    call this%op%apply(x)
  end subroutine
  subroutine top_setup(this, o)
    class(t_top), intent(inout) :: this
    class(t_op), target, intent(in) :: o
    call this%h%construct(o)
  end subroutine
end module
module m_diag
  use m_base
  implicit none
  type, extends(t_op) :: t_diag
  contains
    procedure :: apply => diag_apply
  end type
contains
  subroutine diag_apply(this, x)
    class(t_diag), intent(in) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
end module
module m_scale
  use m_base
  implicit none
  type, extends(t_op) :: t_scale
  contains
    procedure :: apply => scale_apply
  end type
contains
  subroutine scale_apply(this, x)
    class(t_scale), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
end module
module m_use
  use m_base
  use m_diag
  use m_scale
  implicit none
contains
  subroutine kern(x, which)
    real, intent(inout) :: x
    integer, intent(in) :: which
    type(t_top) :: s
    type(t_diag), target :: d
    type(t_scale), target :: sc
    select case (which)
    case (1)
      call s%setup(d)
    case default
      call s%setup(sc)
    end select
    call s%h%run(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_pointer_assoc_passthrough_devirtualizes(tmp_path: Path):
    """The concrete arm reaches the association two hops away, through a
    pass-through constructor that only forwards its ``CLASS(base)`` dummy.  The
    forward fixed point clones the pass-through, which then forwards a *concrete*
    actual to the inner constructor -- so the whole chain devirtualises and the
    dummy-dispatch ``call o%apply`` becomes a static bind."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_PASSTHROUGH_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived the multi-hop pointer-assoc flow"
    src = tmp_path / "ptr2.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr2.f90"], cwd=str(tmp_path))


#: The full ocean-solver shape: TWO pointer axes (transfer + agen) on a shared
#: ``lhs_construct``, plus an ``ALLOCATE``-ladder backend (``t_cg`` / ``t_gmres``)
#: whose SHARED interposer ``backend_construct`` forwards both dummies down.  The
#: entry picks a concrete (agen, trans) pair by SELECT CASE; the backend arm by an
#: independent integer -- so the two pointer flows compose over the act ladder.
_PTR_ASSOC_TWO_AXIS_SRC = """
module m_base
  implicit none
  type, abstract :: t_transfer
    logical :: is_solver_pe = .true.
  contains
    procedure(into_i), deferred :: into
  end type
  type, abstract :: t_agen
  contains
    procedure(apply_i), deferred :: apply
  end type
  abstract interface
    subroutine into_i(this, x)
      import t_transfer
      class(t_transfer), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
    subroutine apply_i(this, x)
      import t_agen
      class(t_agen), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type :: t_lhs
    class(t_agen), pointer :: agen => null()
    class(t_transfer), pointer :: trans => null()
    real :: probe
  contains
    procedure :: construct => lhs_construct
    procedure :: run => lhs_run
  end type
  type, abstract :: t_backend
    class(t_transfer), pointer :: trans => null()
    type(t_lhs) :: lhs
  contains
    procedure :: bconstruct => backend_construct
    procedure(bsolve_i), deferred :: bsolve
  end type
  abstract interface
    subroutine bsolve_i(this, x)
      import t_backend
      class(t_backend), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
contains
  subroutine lhs_construct(this, a, t)
    class(t_lhs), intent(inout) :: this
    class(t_agen), target, intent(in) :: a
    class(t_transfer), target, intent(in) :: t
    this%agen => a
    this%trans => t
    call t%into(this%probe)
    call a%apply(this%probe)
  end subroutine
  subroutine lhs_run(this, x)
    class(t_lhs), intent(inout) :: this
    real, intent(inout) :: x
    ! stored-slot member read in an IF condition (not a Call/Assign/Ptr-assign)
    if (this%trans%is_solver_pe) then
      call this%trans%into(x)
    end if
    call this%agen%apply(x)
  end subroutine
  subroutine backend_construct(this, a, t)
    class(t_backend), intent(inout) :: this
    class(t_agen), target, intent(in) :: a
    class(t_transfer), target, intent(in) :: t
    this%trans => t
    if (this%trans%is_solver_pe) call this%lhs%construct(a, t)
  end subroutine
end module
module m_cg
  use m_base
  implicit none
  type, extends(t_backend) :: t_cg
  contains
    procedure :: bsolve => cg_bsolve
  end type
contains
  subroutine cg_bsolve(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    call this%lhs%run(x)
  end subroutine
end module
module m_gmres
  use m_base
  implicit none
  type, extends(t_backend) :: t_gmres
  contains
    procedure :: bsolve => gmres_bsolve
  end type
contains
  subroutine gmres_bsolve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    call this%lhs%run(x)
    x = x + 0.5
  end subroutine
end module
module m_triv
  use m_base
  implicit none
  type, extends(t_transfer) :: t_triv
  contains
    procedure :: into => triv_into
  end type
contains
  subroutine triv_into(this, x)
    class(t_triv), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
end module
module m_sub
  use m_base
  implicit none
  type, extends(t_transfer) :: t_sub
  contains
    procedure :: into => sub_into
  end type
contains
  subroutine sub_into(this, x)
    class(t_sub), intent(in) :: this
    real, intent(inout) :: x
    x = x * 3.0
  end subroutine
end module
module m_shl
  use m_base
  implicit none
  type, extends(t_agen) :: t_shl
  contains
    procedure :: apply => shl_apply
  end type
contains
  subroutine shl_apply(this, x)
    class(t_shl), intent(in) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
end module
module m_ppfl
  use m_base
  implicit none
  type, extends(t_agen) :: t_ppfl
  contains
    procedure :: apply => ppfl_apply
  end type
contains
  subroutine ppfl_apply(this, x)
    class(t_ppfl), intent(in) :: this
    real, intent(inout) :: x
    x = x - 5.0
  end subroutine
end module
module m_solve
  use m_base
  use m_cg
  use m_gmres
  implicit none
  type :: t_solve
    class(t_backend), allocatable :: act
  contains
    procedure :: construct => solve_construct
  end type
contains
  subroutine solve_construct(this, sel, a, t)
    class(t_solve), intent(inout) :: this
    integer, intent(in) :: sel
    class(t_agen), target, intent(in) :: a
    class(t_transfer), target, intent(in) :: t
    select case (sel)
    case (1)
      allocate(t_cg :: this%act)
    case default
      allocate(t_gmres :: this%act)
    end select
    call this%act%bconstruct(a, t)
  end subroutine
end module
module m_use
  use m_base
  use m_solve
  use m_triv
  use m_sub
  use m_shl
  use m_ppfl
  implicit none
contains
  subroutine kern(x, which, sel)
    real, intent(inout) :: x
    integer, intent(in) :: which, sel
    type(t_solve) :: s
    type(t_shl), target :: a_shl
    type(t_ppfl), target :: a_ppfl
    type(t_triv), target :: t_triv_l
    type(t_sub), target :: t_sub_l
    select case (which)
    case (1)
      call s%construct(sel, a_shl, t_triv_l)
    case (2)
      call s%construct(sel, a_shl, t_sub_l)
    case (3)
      call s%construct(sel, a_ppfl, t_triv_l)
    case default
      call s%construct(sel, a_ppfl, t_sub_l)
    end select
    call s%act%bsolve(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_pointer_assoc_two_axes_through_allocate_ladder(tmp_path: Path):
    """The ICON ocean-solver shape: TWO independent pointer axes (``trans`` and
    ``agen``) bound in one shared constructor (``lhs_construct``), reached through
    an ``ALLOCATE``-laddered backend dispatch and a shared interposer.  The two
    flows compose (a clone per ``(trans, agen)`` pair, each setting both tags
    independently), the stored-slot dispatch is laddered over both tags, and the
    dead intermediate clones + their dangling imports are cleaned -- yielding a
    fully devirtualised, compilable TU with no residual dispatch."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_TWO_AXIS_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived two-axis pointer-assoc devirt"
    low = ast.tofortran().lower()
    # both pointer axes expanded to tags, and the backend ALLOCATE axis too
    for tag in ("trans__tag", "agen__tag", "act__tag"):
        assert tag in low, f"missing expanded tag `{tag}`"
    src = tmp_path / "ptr3.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr3.f90"], cwd=str(tmp_path))


#: A DATA-CARRYING ``CLASS(t_transfer), POINTER :: trans`` slot: the abstract base
#: declares a data member (``nidx``) that the kernel reads in a DECLARATION
#: DIMENSION (``REAL :: tmp(this%trans%nidx)``) and a DO bound -- spec-part reads a
#: statement ladder cannot reach.  This is ICON ``t_lhs%trans`` (``x_t(this%trans%nidx)``):
#: expanding the slot away would leave those reads dangling on a deleted component.
#: The hybrid keeps the CLASS slot for the data reads (they lower natively, without
#: routing through a per-arm POINTER) and ladders ONLY the deferred dispatch.
_PTR_ASSOC_DATACARRY_SRC = """
module m_base
  implicit none
  type, abstract :: t_transfer
    integer :: nidx = 0
  contains
    procedure(into_i), deferred :: into
  end type
  abstract interface
    subroutine into_i(this, x)
      import t_transfer
      class(t_transfer), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type :: t_lhs
    class(t_transfer), pointer :: trans => null()
  contains
    procedure :: construct => lhs_construct
    procedure :: run => lhs_run
  end type
contains
  subroutine lhs_construct(this, t)
    class(t_lhs), intent(inout) :: this
    class(t_transfer), target, intent(in) :: t
    this%trans => t
  end subroutine
  subroutine lhs_run(this, x)
    class(t_lhs), intent(inout) :: this
    real, intent(inout) :: x
    real :: tmp(this%trans%nidx)      ! declaration-dimension data read (ladder-killer)
    integer :: i
    do i = 1, this%trans%nidx          ! DO-bound data read
      tmp(i) = x
    end do
    call this%trans%into(x)            ! deferred dispatch -> laddered per arm
    x = x + real(this%trans%nidx)      ! expression data read
  end subroutine
end module
module m_triv
  use m_base
  implicit none
  type, extends(t_transfer) :: t_triv
  contains
    procedure :: into => triv_into
  end type
contains
  subroutine triv_into(this, x)
    class(t_triv), intent(in) :: this
    real, intent(inout) :: x
    x = x + real(this%nidx)
  end subroutine
end module
module m_sub
  use m_base
  implicit none
  type, extends(t_transfer) :: t_sub
  contains
    procedure :: into => sub_into
  end type
contains
  subroutine sub_into(this, x)
    class(t_sub), intent(in) :: this
    real, intent(inout) :: x
    x = x * real(this%nidx)
  end subroutine
end module
module m_use
  use m_base
  use m_triv
  use m_sub
  implicit none
contains
  subroutine kern(x, which)
    real, intent(inout) :: x
    integer, intent(in) :: which
    type(t_lhs) :: l
    type(t_triv), target :: tv
    type(t_sub), target :: sb
    select case (which)
    case (1)
      call l%construct(tv)
    case default
      call l%construct(sb)
    end select
    call l%run(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_pointer_assoc_data_carrying_slot_kept_class(tmp_path: Path):
    """A data-carrying ``CLASS(base), POINTER`` slot read in a DECLARATION dimension
    is kept CLASS -- its data reads stay on it (they lower natively, unlike a read
    routed through a per-arm POINTER) -- while only its dispatch is laddered onto a
    concrete per-arm pointer.  This is the ICON ``t_lhs%trans`` shape: the old
    expand-the-slot-away form left ``this%trans%nidx`` dangling on a deleted
    component (a pruning ``AssertionError``); the hybrid resolves it."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_DATACARRY_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived data-carrying devirt"
    text = ast.tofortran()
    low = text.lower()
    # the CLASS slot is KEPT (not expanded away) so its data reads resolve...
    assert "class(t_transfer), pointer :: trans" in low, "the data-carrying CLASS slot was not kept"
    # ...including the declaration-dimension read (still on the CLASS slot, not a per-arm pointer)
    assert "tmp(this % trans % nidx)" in low, "the declaration-dimension read was rewritten off the CLASS slot"
    assert "do i = 1, this % trans % nidx" in low, "the DO-bound read was rewritten off the CLASS slot"
    # ...while the dispatch was laddered onto concrete per-arm pointers (tag + slots).
    assert "trans__tag" in low, "the dispatch tag was not emitted"
    assert "trans__t_triv" in low and "trans__t_sub" in low, "per-arm dispatch pointers missing"
    src = tmp_path / "ptr_dc.f90"
    src.write_text(text)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr_dc.f90"], cwd=str(tmp_path))


#: A GENERIC type-bound dispatch (ICON's ``t_transfer%into => into_2d_wp, into_idx``,
#: the deferred specifics overridden per arm) laddered onto a concrete per-arm slot.
#: The generic's specific candidates are registered on the abstract base (where they
#: are DEFERRED); resolving ``this%trans__t_triv%into(a)`` requires remapping each
#: candidate onto the concrete receiver, which overrides it.  A non-generic deferred
#: ``destruct`` seeds discovery (the live-dispatch gate keys on the specific binding
#: name, which a purely-generic axis would never expose).
_PTR_ASSOC_GENERIC_SRC = """
module m_base
  implicit none
  type, abstract :: t_transfer
    integer :: nidx = 0
  contains
    procedure(into_2d_i), deferred :: into_2d
    procedure(into_idx_i), deferred :: into_idx
    procedure(destruct_i), deferred :: destruct
    generic :: into => into_2d, into_idx
  end type
  abstract interface
    subroutine into_2d_i(this, a)
      import t_transfer
      class(t_transfer), intent(in) :: this
      real, intent(inout) :: a(:,:)
    end subroutine
    subroutine into_idx_i(this, i)
      import t_transfer
      class(t_transfer), intent(in) :: this
      integer, intent(inout) :: i
    end subroutine
    subroutine destruct_i(this)
      import t_transfer
      class(t_transfer), intent(inout) :: this
    end subroutine
  end interface
  type :: t_lhs
    integer :: mode = 0
    class(t_transfer), pointer :: trans => null()
  contains
    procedure :: construct => lhs_construct
    procedure :: run => lhs_run
  end type
contains
  subroutine lhs_construct(this, t)
    class(t_lhs), intent(inout) :: this
    class(t_transfer), target, intent(in) :: t
    this%trans => t
    ! SELECT TYPE on the dummy: the per-arm clone retypes `t` concrete, so this
    ! must resolve to its matching guard (a concrete selector is not polymorphic).
    select type (t)
    class is (t_triv)
      this%mode = 1
    class default
      this%mode = 2
    end select
  end subroutine
  subroutine lhs_run(this, a, k)
    class(t_lhs), intent(inout) :: this
    real, intent(inout) :: a(:,:)
    integer, intent(inout) :: k
    call this%trans%into(a)      ! GENERIC dispatch, 2d specific
    call this%trans%into(k)      ! GENERIC dispatch, idx specific (arg-matched)
    call this%trans%destruct()   ! non-generic deferred dispatch (discovery seed)
  end subroutine
end module
module m_triv
  use m_base
  implicit none
  type, extends(t_transfer) :: t_triv
  contains
    procedure :: into_2d => triv_into_2d
    procedure :: into_idx => triv_into_idx
    procedure :: destruct => triv_destruct
  end type
contains
  subroutine triv_into_2d(this, a)
    class(t_triv), intent(in) :: this
    real, intent(inout) :: a(:,:)
    a = a + real(this%nidx)
  end subroutine
  subroutine triv_into_idx(this, i)
    class(t_triv), intent(in) :: this
    integer, intent(inout) :: i
    i = i + this%nidx
  end subroutine
  subroutine triv_destruct(this)
    class(t_triv), intent(inout) :: this
    this%nidx = 0
  end subroutine
end module
module m_sub
  use m_base
  implicit none
  type, extends(t_transfer) :: t_sub
  contains
    procedure :: into_2d => sub_into_2d
    procedure :: into_idx => sub_into_idx
    procedure :: destruct => sub_destruct
  end type
contains
  subroutine sub_into_2d(this, a)
    class(t_sub), intent(in) :: this
    real, intent(inout) :: a(:,:)
    a = a * real(this%nidx)
  end subroutine
  subroutine sub_into_idx(this, i)
    class(t_sub), intent(in) :: this
    integer, intent(inout) :: i
    i = i * this%nidx
  end subroutine
  subroutine sub_destruct(this)
    class(t_sub), intent(inout) :: this
    this%nidx = -1
  end subroutine
end module
module m_use
  use m_base
  use m_triv
  use m_sub
  implicit none
contains
  subroutine kern(a, k, which)
    real, intent(inout) :: a(:,:)
    integer, intent(inout) :: k, which
    type(t_lhs) :: l
    type(t_triv), target :: tv
    type(t_sub), target :: sb
    select case (which)
    case (1)
      call l%construct(tv)
    case default
      call l%construct(sb)
    end select
    call l%run(a, k)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_pointer_assoc_generic_binding_resolves_per_arm(tmp_path: Path):
    """A GENERIC type-bound call laddered onto a concrete per-arm slot resolves to
    the arm's override of the matching specific.  ``deconstruct_procedure_calls``
    registers a generic's candidates on the type that DECLARES it (an abstract base,
    where the specifics are DEFERRED); it must remap each candidate onto the concrete
    receiver, which overrides it -- else ``%arm%into`` dangles (the base generic is
    pruned) and gfortran rejects the call.  This is ICON ``t_transfer%into``."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_GENERIC_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a generic type-bound dispatch survived resolution"
    low = ast.tofortran().lower()
    # generic `into(a)` resolved to the 2d specific, `into(k)` to the idx specific --
    # each on the concrete arm's override, not the base's deferred binding.
    assert "call triv_into_2d(this % trans__t_triv" in low and "call sub_into_2d(this % trans__t_sub" in low
    assert "call triv_into_idx(this % trans__t_triv" in low and "call sub_into_idx(this % trans__t_sub" in low
    # the constructor's SELECT TYPE on the retyped dummy resolved statically per arm
    # (t_triv clone -> CLASS IS branch `mode=1`; t_sub clone -> CLASS DEFAULT `mode=2`).
    assert "select type" not in low, "a SELECT TYPE on a retyped concrete dummy survived"
    assert "this % mode = 1" in low and "this % mode = 2" in low
    src = tmp_path / "ptr_gen.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr_gen.f90"], cwd=str(tmp_path))


#: An abstract base dispatched ONLY through generics: ``t_agen``'s deferred
#: ``lhs_wp`` is invoked solely as ``apply`` (``GENERIC :: apply => lhs_wp``), never
#: by its own name.  This is ICON's ``t_lhs_agen`` (the mimetic matrix operator):
#: discovery must treat the live generic ``apply`` as making its specific ``lhs_wp``
#: live, else the whole axis is missed and ``this%agen%apply`` stays polymorphic.
_GENERIC_ONLY_SRC = """
module m_base
  implicit none
  type, abstract :: t_agen
    logical :: is_const = .false.
  contains
    procedure(lhs_wp_i), deferred :: lhs_wp
    generic :: apply => lhs_wp
  end type
  abstract interface
    subroutine lhs_wp_i(this, x)
      import t_agen
      class(t_agen), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type :: t_lhs
    class(t_agen), pointer :: agen => null()
  contains
    procedure :: construct => lhs_construct
    procedure :: run => lhs_run
  end type
contains
  subroutine lhs_construct(this, a)
    class(t_lhs), intent(inout) :: this
    class(t_agen), target, intent(in) :: a
    this%agen => a
  end subroutine
  subroutine lhs_run(this, x)
    class(t_lhs), intent(inout) :: this
    real, intent(inout) :: x
    if (this%agen%is_const) x = x + 1.0   ! data read -> kept CLASS (hybrid)
    call this%agen%apply(x)               ! generic-ONLY dispatch (discovery gate)
  end subroutine
end module
module m_shl
  use m_base
  implicit none
  type, extends(t_agen) :: t_shl
  contains
    procedure :: lhs_wp => shl_wp
  end type
contains
  subroutine shl_wp(this, x)
    class(t_shl), intent(in) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
end module
module m_ppfl
  use m_base
  implicit none
  type, extends(t_agen) :: t_ppfl
  contains
    procedure :: lhs_wp => ppfl_wp
  end type
contains
  subroutine ppfl_wp(this, x)
    class(t_ppfl), intent(in) :: this
    real, intent(inout) :: x
    x = x - 5.0
  end subroutine
end module
module m_use
  use m_base
  use m_shl
  use m_ppfl
  implicit none
contains
  subroutine kern(x, which)
    real, intent(inout) :: x
    integer, intent(inout) :: which
    type(t_lhs) :: l
    type(t_shl), target :: s
    type(t_ppfl), target :: p
    select case (which)
    case (1)
      call l%construct(s)
    case default
      call l%construct(p)
    end select
    call l%run(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_generic_only_dispatch_axis_discovered(tmp_path: Path):
    """An abstract base whose deferred binding is reached ONLY via a generic
    (``apply => lhs_wp``, never ``%lhs_wp`` directly) is still discovered and
    laddered.  ``discover_axes`` keys liveness on the dispatched binding name; a
    generic dispatch names the generic, so it must propagate liveness to the
    generic's specifics -- else ICON's ``t_lhs_agen`` (``apply``/``matrix_shortcut``
    generics over deferred ``lhs_wp``/``lhs_matrix_shortcut``) is never devirtualised."""
    ast = inline_to_ast({"s.f90": _GENERIC_ONLY_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a generic-only dispatch survived (axis not discovered)"
    low = ast.tofortran().lower()
    assert "agen__tag" in low, "the generic-only axis was not discovered/laddered"
    assert "call shl_wp(this % agen__t_shl" in low and "call ppfl_wp(this % agen__t_ppfl" in low
    assert "class(t_agen), pointer :: agen" in low, "the data-carrying CLASS agen slot was not kept"
    src = tmp_path / "gen_only.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "gen_only.f90"], cwd=str(tmp_path))


#: A slot name shared across UNRELATED types: an abstract ``CLASS(t_base),POINTER :: p``
#: laddered slot (on ``t_p_wrap``) and a CONCRETE ``TYPE(t_orig),POINTER :: p`` on an
#: unrelated wrapper (``t_p_wrap_orig``).  This is ICON's ``t_p_comm_pattern`` vs
#: ``t_p_comm_pattern_orig`` (both hold a ``p``), which the 18-arm ``t_stack_op`` slot
#: ``p`` collides with: a textual ``%p`` retarget would corrupt the concrete wrapper.
_SLOT_NAME_COLLISION_SRC = """
module m_base
  implicit none
  type, abstract :: t_base
    integer :: nidx = 0
  contains
    procedure(foo_i), deferred :: foo_impl
    generic :: foo => foo_impl
  end type
  abstract interface
    subroutine foo_i(this, x)
      import t_base
      class(t_base), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(t_base) :: t_orig
  contains
    procedure :: foo_impl => orig_foo
  end type
  type :: t_p_wrap
    class(t_base), pointer :: p => null()   ! the abstract laddered slot (owner)
  end type
  type :: t_p_wrap_orig
    type(t_orig), pointer :: p => null()    ! an UNRELATED concrete `p` (collision victim)
  end type
contains
  subroutine orig_foo(this, x)
    class(t_orig), intent(in) :: this
    real, intent(inout) :: x
    x = x + real(this%nidx)
  end subroutine
end module
module m_alt
  use m_base
  implicit none
  type, extends(t_base) :: t_alt
  contains
    procedure :: foo_impl => alt_foo
  end type
contains
  subroutine alt_foo(this, x)
    class(t_alt), intent(in) :: this
    real, intent(inout) :: x
    x = x * real(this%nidx)
  end subroutine
end module
module m_use
  use m_base
  use m_alt
  implicit none
contains
  subroutine kern(x, which)
    real, intent(inout) :: x
    integer, intent(inout) :: which
    type(t_p_wrap) :: wrap
    type(t_p_wrap_orig) :: worig
    if (which == 1) then
      allocate(t_alt :: wrap%p)   ! construction seed for the t_base LADDER
    else
      allocate(t_orig :: wrap%p)
    end if
    allocate(worig%p)
    call worig%p%foo(x)           ! concrete `p` -- must be LEFT ALONE (not laddered)
    call wrap%p%foo(x)            ! owner `p` -- must be laddered per arm
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_slot_ladder_ignores_same_name_on_unrelated_type(tmp_path: Path):
    """The slot ladder retargets ``%slot`` textually, so it must be TYPE-AWARE: a
    component of the same name on an UNRELATED type (a concrete ``t_p_wrap_orig%p``
    vs the abstract, laddered ``t_p_wrap%p``) must be left alone -- else the retarget
    rewrites it to ``%p__tag`` / ``%p__arm`` on a type that has no such member.  This
    is ICON's ``t_p_comm_pattern_orig%p`` colliding with the 18-arm ``t_stack_op``
    slot ``p``.  The owner ``wrap%p`` still ladders; the concrete ``worig%p`` resolves
    as an ordinary (non-laddered) dispatch."""
    ast = inline_to_ast({"s.f90": _SLOT_NAME_COLLISION_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a dispatch survived"
    low = ast.tofortran().lower()
    # the owner slot laddered (tag + per-arm), the unrelated concrete `p` untouched
    assert "wrap % p__tag" in low and "wrap % p__t_alt" in low and "wrap % p__t_orig" in low
    assert "worig % p__" not in low, "an unrelated same-named component was wrongly retargeted"
    assert "call orig_foo" in low and "allocate(worig % p)" in low
    src = tmp_path / "collide.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "collide.f90"], cwd=str(tmp_path))


#: An arm module imports a derived TYPE that is DEFINED in one module (``m_origin``)
#: and RE-EXPORTED by another (``m_grid``), and one of its contained procedures uses
#: that type ONLY by host association -- a ``TYPE(t_subset_range)`` local declaration,
#: never a call.  Consolidating the arm into the base module gives the base a new
#: dependency on ``m_grid``/``m_origin``.  Two things must then hold or the emitted
#: TU references a type it never imports ("used before defined"): the moved ``USE``
#: must land AHEAD of the base's type defs (a ``USE`` after a declaration is illegal
#: Fortran AND invisible to ``alias_specs``), and the base module must be re-sorted
#: AFTER its new dependencies (``alias_specs`` resolves ``USE``s in document order and
#: would otherwise drop the re-exported host type).  This is ICON's ``t_subset_range``
#: (defined in ``mo_model_domain``, re-exported via ``mo_grid_subset``) host-associated
#: into ``mo_surface_height_lhs``'s ``lhs_..._matrix_wp`` helper, an arm of the
#: ``t_lhs_agen`` ladder.
_REEXPORT_HOST_TYPE_SRC = """
module m_origin
  implicit none
  type :: t_subset_range
    integer :: start_block = 1, end_block = 2
  end type
end module
module m_grid
  use m_origin, only: t_subset_range
  implicit none
  public :: t_subset_range, get_index_range
contains
  subroutine get_index_range(sr, blk, si, ei)
    type(t_subset_range), intent(in) :: sr
    integer, intent(in) :: blk
    integer, intent(out) :: si, ei
    si = sr%start_block; ei = sr%end_block
  end subroutine
end module
module m_base
  implicit none
  type, abstract :: t_agen
    logical :: is_init = .false.
  contains
    procedure(a_apply), deferred :: apply_impl
    generic :: apply => apply_impl
  end type
  abstract interface
    subroutine a_apply(this, x)
      import t_agen
      class(t_agen), intent(inout) :: this
      real, intent(inout) :: x(:)
    end subroutine
  end interface
end module
module m_arm
  use m_base, only: t_agen
  use m_grid, only: t_subset_range, get_index_range
  implicit none
  type, extends(t_agen) :: t_sfc
    integer :: n = 0
  contains
    procedure :: apply_impl => sfc_apply
  end type
contains
  subroutine sfc_apply(this, x)
    class(t_sfc), intent(inout) :: this
    real, intent(inout) :: x(:)
    call sfc_matrix(this, x)
  end subroutine
  subroutine sfc_matrix(this, x)
    class(t_sfc), intent(inout) :: this
    real, intent(inout) :: x(:)
    type(t_subset_range), pointer :: cid   ! host-associated TYPE, declaration only
    integer :: blk, si, ei
    allocate(cid)
    do blk = 1, 2
      call get_index_range(cid, blk, si, ei)
      x(si) = x(ei) + real(this%n)
    end do
    deallocate(cid)
  end subroutine
end module
module m_arm2
  use m_base, only: t_agen
  implicit none
  type, extends(t_agen) :: t_pff
    integer :: m = 0
  contains
    procedure :: apply_impl => pff_apply
  end type
contains
  subroutine pff_apply(this, x)
    class(t_pff), intent(inout) :: this
    real, intent(inout) :: x(:)
    x(1) = x(1) * real(this%m)
  end subroutine
end module
module m_use
  use m_base, only: t_agen
  use m_arm, only: t_sfc
  use m_arm2, only: t_pff
  implicit none
  type :: t_lhs
    class(t_agen), pointer :: agen => null()
  end type
contains
  subroutine driver(x, which)
    real, intent(inout) :: x(:)
    integer, intent(in) :: which
    type(t_lhs) :: lhs
    if (which == 1) then
      allocate(t_sfc :: lhs%agen)
    else
      allocate(t_pff :: lhs%agen)
    end if
    call lhs%agen%apply(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_arm_merge_keeps_reexported_host_type_resolvable(tmp_path: Path):
    """A host-associated type that is re-exported through the arm's imported module
    survives the arm->base module merge: the moved subprogram still imports it (from
    its true origin), so the emitted TU compiles instead of naming an unimported type.
    Guards both halves of the fix -- the moved ``USE`` is prepended ahead of the base's
    type defs, and the base module is re-sorted after its newly-inherited dependencies."""
    ast = inline_to_ast({"s.f90": _REEXPORT_HOST_TYPE_SRC}, entry="m_use::driver")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived devirtualisation"
    out = ast.tofortran()
    low = out.lower()
    # the arm modules are gone (merged into the base) and the ladder devirtualised
    assert "module m_arm" not in low
    # the host-associated type resolves to its TRUE origin, not the (re-exporting) m_grid
    assert "use m_origin, only: t_subset_range" in low
    # and the whole TU compiles -- no "used before defined" for t_subset_range
    src = tmp_path / "reexport.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "reexport.f90"], cwd=str(tmp_path))


#: A hybrid slot (a CLASS(base) data-carrying POINTER, kept CLASS by the component
#: ladder) is PASSED to a helper as a bare CLASS(base) dummy, and the helper dispatches
#: on it (`trans%sync`).  The slot ladder devirtualises `this%trans%binding` (a slot
#: DISPATCH) but leaves `CALL helper(this%trans)` (a slot PASS) as CLASS, so the helper's
#: `dummy%binding` stays polymorphic -> gfortran "sync is not a member of t_xfer" (the
#: base's bindings were stripped by the hybrid expansion).  This is ICON's
#: ocean_restart_gmres(trans)%sync in the solve_free_sfc solver.
_DUMMY_DISPATCH_HELPER_SRC = """
module m_base
  implicit none
  type, abstract :: t_xfer
    integer :: nblk = 0
  contains
    procedure(a_sync), deferred :: sync_impl
    generic :: sync => sync_impl
  end type
  abstract interface
    subroutine a_sync(this, x)
      import t_xfer
      class(t_xfer), intent(inout) :: this
      real, intent(inout) :: x(:)
    end subroutine
  end interface
end module
module m_triv
  use m_base, only: t_xfer
  implicit none
  type, extends(t_xfer) :: t_triv
  contains
    procedure :: sync_impl => triv_sync
  end type
contains
  subroutine triv_sync(this, x)
    class(t_triv), intent(inout) :: this
    real, intent(inout) :: x(:)
    x = x + real(this%nblk)
  end subroutine
end module
module m_sub
  use m_base, only: t_xfer
  implicit none
  type, extends(t_xfer) :: t_sub
  contains
    procedure :: sync_impl => sub_sync
  end type
contains
  subroutine sub_sync(this, x)
    class(t_sub), intent(inout) :: this
    real, intent(inout) :: x(:)
    x = x * real(this%nblk)
  end subroutine
end module
module m_solver
  use m_base, only: t_xfer
  implicit none
  type :: t_solver
    class(t_xfer), pointer :: trans => null()
  contains
    procedure :: cal => solver_cal
    procedure :: construct => solver_construct
  end type
contains
  subroutine solver_construct(this, trans)
    class(t_solver), intent(inout) :: this
    class(t_xfer), target, intent(in) :: trans
    this%trans => trans
  end subroutine
  subroutine solver_cal(this, x)
    class(t_solver), intent(inout) :: this
    real, intent(inout) :: x(:)
    call restart_helper(x, this%trans)
  end subroutine
  subroutine restart_helper(x, trans)
    real, intent(inout) :: x(:)
    class(t_xfer), pointer, intent(inout) :: trans
    integer :: i
    do i = 1, trans%nblk
      call trans%sync(x)
    end do
  end subroutine
end module
module m_use
  use m_base, only: t_xfer
  use m_triv, only: t_triv
  use m_sub, only: t_sub
  use m_solver, only: t_solver
  implicit none
contains
  subroutine driver(x, which)
    real, intent(inout) :: x(:)
    integer, intent(in) :: which
    type(t_solver) :: s
    type(t_triv), target :: tt
    type(t_sub), target :: ts
    if (which == 1) then
      call s%construct(tt)
    else
      call s%construct(ts)
    end if
    call s%cal(x)
  end subroutine
end module
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_dummy_dispatch_helper_devirtualized(tmp_path: Path):
    """A dispatch on a CLASS(base) DUMMY of a helper (`trans%sync`), where the helper is
    called with the hybrid slot `this%trans`, is devirtualised: the helper is cloned per
    arm (dummy retyped to the concrete arm) and the call becomes a tag ladder routing each
    arm to its clone with the matching per-arm slot.  Mirrors ICON's ocean_restart_gmres."""
    ast = inline_to_ast({"s.f90": _DUMMY_DISPATCH_HELPER_SRC}, entry="m_use::driver", tolerate_external_uses=True)
    assert not walk(ast, f03.Procedure_Designator), "a dispatch on the CLASS dummy survived"
    low = ast.tofortran().lower()
    # the helper was cloned per arm and the call laddered to the per-arm slots
    assert "restart_helper__t_triv" in low and "restart_helper__t_sub" in low
    assert "trans__t_triv" in low and "trans__t_sub" in low
    src = tmp_path / "dummy_dispatch.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "dummy_dispatch.f90"],
                          cwd=str(tmp_path))
