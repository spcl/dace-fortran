# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""monomorphize runs by default in the fparser inliner (ParseConfig.monomorphize),
devirtualising single-level abstract dispatch the bridge can't lower into static calls.
Tests drive the real inline_to_ast pipeline (not the engine in isolation) to pin wiring
+ ordering vs the call-resolution passes. SDFG-level proof (no ExternalCall) is in
tests/sync_devirt_mpi_libnode_test.py.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.fparser_inliner import inline_to_ast

#: ICON's halo dispatch chain, single-arm (yaxt cpp-stripped): entry -> generic
#: wrapper -> CLASS(t_comm_pattern) dispatch -> orig_exchange override.
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
    """Default monomorphize=True: no Procedure_Designator survives, and every
    CLASS(t_comm_pattern) outside the deferred interface retypes to
    TYPE(t_comm_pattern_orig)."""
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
    """monomorphize=False leaves the abstract CLASS dispatch intact. optimize=False
    too, else the constant-prop optimizer rejects the unresolved polymorphic dispatch."""
    ast = inline_to_ast({"src.f90": _CHAIN_SRC}, entry="mo_solve::solve_nh", monomorphize=False, optimize=False)
    low = ast.tofortran().lower()
    assert "type(t_comm_pattern_orig)" not in low, "monomorphize=False must not retype the abstract dummy"
    assert "class(t_comm_pattern)" in low, "the abstract CLASS dispatch dummy should survive when the pass is off"


#: Base + pointer-holder in one module, concrete arm in another: retyping the
#: pointer would create a circular USE (arm module already USEs base to EXTEND).
#: Fix: consolidate the arm into the base module (topological order).
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
    """Cross-module retype that would create a circular dependency instead
    consolidates the arm into the base module, producing compilable Fortran."""
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


#: Same cycle, but base is TYPES-ONLY (no CONTAINS) and the wrapper is in a THIRD
#: module: consolidation must add a fresh CONTAINS before END MODULE and import
#: the arm into the wrapper module. ICON's mo_communication_types/_orig shape.
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
    """TYPES-ONLY base + arm + wrapper in three modules: consolidation adds CONTAINS
    to the base before END MODULE and imports the arm into the wrapper."""
    ast = inline_to_ast({"s.f90": _CYCLE_3MOD_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    assert "module m_orig" not in out.lower()
    assert not walk(ast, f03.Procedure_Designator)
    src = tmp_path / "consolidated3.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "consolidated3.f90"],
                          cwd=str(tmp_path))


#: Arm type is referenced by a type COMPONENT in yet another module (ICON's
#: t_patch%comm_pat_*). Consolidation must import the arm into that module at
#: MODULE level so the type def resolves it.
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
    """Arm type used as a component in a separate module: consolidation imports
    the arm into that module so the type def resolves."""
    ast = inline_to_ast({"s.f90": _CYCLE_COMPONENT_SRC}, entry="m_use::kern")
    out = ast.tofortran()
    assert "module m_orig" not in out.lower()
    assert not walk(ast, f03.Procedure_Designator)
    src = tmp_path / "comp.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "comp.f90"], cwd=str(tmp_path))


#: Multi-arm ladder (ICON ocean free-surface solver): runtime ALLOCATE(concrete ::
#: s%act) picks the arm, so the pass can't pin one type -- it emits a tag ladder +
#: per-arm interposer clones into the base module, naming arm types/procedures ->
#: needs consolidation to avoid a circular USE. Each module also has a PRIVATE
#: this_mod_name PARAMETER (ICON idiom); consolidation must rename it on collision.
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
    """Multi-arm ladder merges every arm module into the base so the ladder's
    per-arm clones resolve their arm types/procedures locally: no residual
    dispatch, compilable."""
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


#: Clone name <interposer>__<arm> overruns Fortran's 63-char identifier limit
#: (ICON's composed backend x agen x transfer axis reaches ~95 chars). Must
#: shorten to a readable prefix + stable hash, else gfortran rejects the
#: SUBROUTINE header.
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
    """Clone name exceeding Fortran's 63-char identifier limit is shortened (prefix +
    stable hash) so the TU compiles instead of a rejected SUBROUTINE header. Mirrors
    ICON's multi-axis solver construct (backend x agen x transfer, ~95 chars)."""
    ast = inline_to_ast({"s.f90": _LADDER_LONGNAME_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a dispatch survived"
    out = ast.tofortran()
    # subprogram name is the first Name in its SUBROUTINE/FUNCTION statement
    for stmt in walk(ast, (f03.Subroutine_Stmt, f03.Function_Stmt)):
        name = str(walk(stmt, f03.Name)[0])
        assert len(name) <= 63, f"identifier over Fortran 63-char limit: {name} ({len(name)})"
    # the naive composed name (which would overrun) is NOT emitted verbatim
    assert "backend_solve_a_deliberately_verbose_shared_interposer__t_conjugate" not in out.lower()
    src = tmp_path / "longname.f90"
    src.write_text(out)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "longname.f90"], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Pointer-association devirt (devirtualize_pointer_flow): ICON's t_lhs%trans/%agen
# are CLASS(base) POINTER slots bound via `this%slot => dummy`, with the concrete
# arm arriving hops away as an actual argument -- unlike the ALLOCATE(arm :: slot)
# ladder above.
# ---------------------------------------------------------------------------

#: Single hop: t_solver%op (CLASS(t_op), POINTER) bound in setup via `this%op => o`,
#: the concrete local chosen by SELECT CASE. Dispatch `this%op%apply` reads the slot.
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
    """Two-arm CLASS POINTER slot set by pointer-association (no ALLOCATE) is
    discovered as a ladder: slot expands to tag + per-arm pointers, setup clones
    per arm, no dispatch survives."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_SINGLE_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived pointer-assoc devirtualisation"
    low = ast.tofortran().lower()
    assert "op__tag" in low, "the pointer slot was not expanded to a type tag"
    src = tmp_path / "ptr1.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr1.f90"], cwd=str(tmp_path))


#: Two hops: kern -> t_top%setup (pass-through, forwards CLASS(base) dummy) ->
#: t_holder%construct (associates the slot AND dispatches on the dummy) -- the
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
    """Concrete arm reaches the association two hops away through a pass-through
    constructor. Forward fixed point clones it end to end: the dummy-dispatch
    becomes a static bind."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_PASSTHROUGH_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived the multi-hop pointer-assoc flow"
    src = tmp_path / "ptr2.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr2.f90"], cwd=str(tmp_path))


#: Full ocean-solver shape: two pointer axes (transfer + agen) share lhs_construct;
#: an ALLOCATE-ladder backend (t_cg/t_gmres) forwards both dummies through a shared
#: interposer -- the two pointer flows compose over the backend ladder.
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
    """Two independent pointer axes (trans, agen) share one constructor, reached
    through an ALLOCATE-laddered backend. Flows compose into a clone per (trans,
    agen) pair; dead intermediate clones + dangling imports are cleaned."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_TWO_AXIS_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived two-axis pointer-assoc devirt"
    low = ast.tofortran().lower()
    # both pointer axes expanded to tags, and the backend ALLOCATE axis too
    for tag in ("trans__tag", "agen__tag", "act__tag"):
        assert tag in low, f"missing expanded tag `{tag}`"
    src = tmp_path / "ptr3.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr3.f90"], cwd=str(tmp_path))


#: Data-carrying CLASS(t_transfer) POINTER slot: base has a data member (nidx) read
#: in a DECLARATION DIMENSION and a DO bound -- spec-part reads a statement ladder
#: can't reach. Hybrid keeps the CLASS slot for data reads, ladders only the
#: deferred dispatch (else reads dangle on a deleted component, ICON t_lhs%trans).
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
    """Data-carrying CLASS POINTER slot read in a DECLARATION dimension stays CLASS
    (data reads lower natively); only its dispatch is laddered onto a per-arm
    pointer. Old expand-away form left the read dangling (pruning AssertionError)."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_DATACARRY_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a type-bound dispatch survived data-carrying devirt"
    text = ast.tofortran()
    low = text.lower()
    assert "class(t_transfer), pointer :: trans" in low, "the data-carrying CLASS slot was not kept"
    assert "tmp(this % trans % nidx)" in low, "the declaration-dimension read was rewritten off the CLASS slot"
    assert "do i = 1, this % trans % nidx" in low, "the DO-bound read was rewritten off the CLASS slot"
    assert "trans__tag" in low, "the dispatch tag was not emitted"
    assert "trans__t_triv" in low and "trans__t_sub" in low, "per-arm dispatch pointers missing"
    src = tmp_path / "ptr_dc.f90"
    src.write_text(text)
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr_dc.f90"], cwd=str(tmp_path))


#: GENERIC dispatch (t_transfer%into => into_2d, into_idx) laddered onto a per-arm
#: slot: specifics are registered on the abstract base, so resolving them requires
#: remapping each candidate onto the concrete receiver's override. A non-generic
#: deferred `destruct` seeds discovery (liveness gate keys on the specific binding
#: name; a purely-generic axis would never expose one).
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
    """GENERIC call laddered onto a per-arm slot resolves to the arm's override.
    deconstruct_procedure_calls registers candidates on the declaring (abstract)
    type, so it must remap each onto the concrete receiver's override, else
    %arm%into dangles once the base generic is pruned. ICON t_transfer%into."""
    ast = inline_to_ast({"s.f90": _PTR_ASSOC_GENERIC_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a generic type-bound dispatch survived resolution"
    low = ast.tofortran().lower()
    # into(a) resolves to the 2d specific, into(k) to the idx specific -- on the arm's override
    assert "call triv_into_2d(this % trans__t_triv" in low and "call sub_into_2d(this % trans__t_sub" in low
    assert "call triv_into_idx(this % trans__t_triv" in low and "call sub_into_idx(this % trans__t_sub" in low
    # SELECT TYPE on the retyped dummy resolved statically (t_triv -> mode=1, t_sub -> mode=2)
    assert "select type" not in low, "a SELECT TYPE on a retyped concrete dummy survived"
    assert "this % mode = 1" in low and "this % mode = 2" in low
    src = tmp_path / "ptr_gen.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "ptr_gen.f90"], cwd=str(tmp_path))


#: Abstract base dispatched ONLY through a generic (apply => lhs_wp, never %lhs_wp
#: directly, ICON's t_lhs_agen): discovery must treat the live generic as making
#: its specific live, else the axis is missed and %apply stays polymorphic.
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
    """Deferred binding reached only via a generic (apply => lhs_wp) is still
    discovered and laddered. discover_axes keys liveness on the dispatched name, so
    a generic dispatch must propagate liveness to its specifics (ICON t_lhs_agen)."""
    ast = inline_to_ast({"s.f90": _GENERIC_ONLY_SRC}, entry="m_use::kern")
    assert not walk(ast, f03.Procedure_Designator), "a generic-only dispatch survived (axis not discovered)"
    low = ast.tofortran().lower()
    assert "agen__tag" in low, "the generic-only axis was not discovered/laddered"
    assert "call shl_wp(this % agen__t_shl" in low and "call ppfl_wp(this % agen__t_ppfl" in low
    assert "class(t_agen), pointer :: agen" in low, "the data-carrying CLASS agen slot was not kept"
    src = tmp_path / "gen_only.f90"
    src.write_text(ast.tofortran())
    subprocess.check_call(["gfortran", "-fsyntax-only", "-ffree-line-length-none", "gen_only.f90"], cwd=str(tmp_path))


#: Slot name `p` shared across UNRELATED types: an abstract laddered CLASS(t_base)
#: POINTER (t_p_wrap%p) vs a concrete TYPE(t_orig) POINTER on an unrelated wrapper
#: (t_p_wrap_orig%p, ICON's t_p_comm_pattern_orig). A textual %p retarget would
#: corrupt the concrete wrapper.
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
    """Slot ladder retargets %slot textually, so it must be TYPE-AWARE: an unrelated
    type's same-named component (t_p_wrap_orig%p) must be left alone, else it's
    rewritten to a member that doesn't exist. ICON t_p_comm_pattern_orig%p vs
    t_stack_op%p."""
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


#: Arm module imports a TYPE defined in one module (m_origin) and re-exported by
#: another (m_grid), used only by host association (local decl, no call). Merging
#: the arm into the base adds a dependency: the moved USE must land ahead of the
#: base's type defs (else illegal Fortran + invisible to alias_specs), and the base
#: module must be re-sorted after its new deps (alias_specs resolves USEs in
#: document order). ICON's t_subset_range (mo_model_domain / mo_grid_subset).
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
    """Host-associated type re-exported through the arm's module survives the
    arm->base merge: the moved USE is prepended ahead of the base's type defs, and
    the base module is re-sorted after its new dependencies."""
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


#: Hybrid slot (CLASS(base) data-carrying POINTER, kept CLASS) passed to a helper as
#: a bare CLASS(base) dummy that dispatches on it. The slot ladder devirtualises the
#: slot DISPATCH but leaves the slot PASS as CLASS, so the helper's dummy dispatch
#: stays polymorphic -> gfortran "sync is not a member" (bindings stripped by hybrid
#: expansion). ICON's ocean_restart_gmres(trans)%sync in solve_free_sfc.
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
    """Dispatch on a helper's CLASS(base) dummy, called with the hybrid slot, is
    devirtualised: helper clones per arm (dummy retyped), call becomes a tag ladder.
    Mirrors ICON's ocean_restart_gmres."""
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
