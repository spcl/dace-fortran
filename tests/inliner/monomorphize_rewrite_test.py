# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite tests for the static-vtable engine
(:mod:`dace_fortran.inliner.ast_desugaring.monomorphize_rewrite`).

A polymorphic ``CLASS(base)`` local with a factory ``ALLOCATE`` and a
``CALL v%binding(args)`` dispatch is rewritten into the emit-all-always static
``if`` ladder.  We assert (a) the AST no longer dispatches, (b) flang lowers the
result with zero ``fir.dispatch`` (the property the bridge needs), and (c) the
rewrite is behaviour-preserving under gfortran.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from _util import _FLANG, have_flang
from dace_fortran.inliner.ast_desugaring.monomorphize import analyze, parse_program
from dace_fortran.inliner.ast_desugaring.monomorphize_rewrite import (AxisSpec, clone_shared_interposers, monomorphize,
                                                                      MonomorphizationSpec,
                                                                      monomorphize_component_dispatch,
                                                                      monomorphize_local_dispatch, retype_to_concrete)

SRC = """
module m
  type, abstract :: base
  contains
    procedure(apply_i), deferred :: apply
  end type
  abstract interface
    subroutine apply_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: apply => gmres_apply
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: apply => cg_apply
  end type
contains
  subroutine gmres_apply(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
  subroutine cg_apply(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0
  end subroutine
  subroutine run(sel, x)
    integer, intent(in) :: sel
    real, intent(inout) :: x
    class(base), allocatable :: s
    if (sel == 1) then
      allocate(t_gmres :: s)
    else
      allocate(t_cg :: s)
    end if
    call s%apply(x)
  end subroutine
end module
"""


def _rewritten() -> f03.Program:
    prog = parse_program(SRC)
    plan = analyze(prog)[0]
    assert monomorphize_local_dispatch(prog, plan) == 1
    return prog


def test_rewrite_removes_dispatch_and_emits_all_arms():
    prog = _rewritten()
    # no type-bound dispatch (`v%binding(...)`) remains anywhere
    for call in walk(prog, f03.Call_Stmt):
        designator = call.children[0]
        assert not isinstance(designator, f03.Procedure_Designator), f"residual dispatch: {call}"
    text = str(prog)
    # the polymorphic local is gone (the abstract interface's CLASS(base) dummy
    # stays -- that's a declaration, not a dispatch); a tag + one concrete slot
    # per arm replace it
    assert "CLASS(base), ALLOCATABLE" not in text
    assert "INTEGER :: s__tag" in text
    assert "TYPE(t_gmres), ALLOCATABLE :: s__t_gmres" in text
    assert "TYPE(t_cg), ALLOCATABLE :: s__t_cg" in text
    # both arms emitted as direct calls (emit-all-always)
    assert "CALL gmres_apply(s__t_gmres, x)" in text
    assert "CALL cg_apply(s__t_cg, x)" in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_rewritten_fir_has_no_dispatch(tmp_path: Path):
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    text = fir.read_text()
    assert text.count("fir.dispatch") == 0
    assert text.count("fir.select_type") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_rewrite_is_behaviour_preserving(tmp_path: Path):
    (tmp_path / "rw.f90").write_text(str(_rewritten()))
    (tmp_path / "drive.f90").write_text("program drive\n"
                                        "  use m\n"
                                        "  real :: x\n"
                                        "  x = 10.0; call run(1, x); if (abs(x - 20.0) > 1e-5) stop 1\n"
                                        "  x = 10.0; call run(2, x); if (abs(x - 30.0) > 1e-5) stop 2\n"
                                        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Component data-member access: a slot member used as a whole-statement
# assignment / pointer-assignment AND a dispatch, all routed by the same
# general per-arm statement ladder (`%act` -> `%act__arm`).
# ---------------------------------------------------------------------------
DATAMEMBER_SRC = """
module m
  type, abstract :: base
    integer :: counter = 0
    real, pointer :: b => null()
  contains
    procedure(solve_i), deferred :: solve
  end type
  abstract interface
    subroutine solve_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: solve => gmres_solve
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: solve => cg_solve
  end type
  type :: container
    class(base), allocatable :: act
    real, pointer :: rhs => null()
  contains
    procedure :: setup => container_setup
    procedure :: run   => container_run
  end type
contains
  subroutine gmres_solve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0 + real(this%counter)
  end subroutine
  subroutine cg_solve(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0 + real(this%counter)
  end subroutine
  subroutine container_setup(this, sel)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
    this%act%counter = 10
    this%act%b => this%rhs
  end subroutine
  subroutine container_run(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%solve(x)
  end subroutine
end module
"""


def _rewritten_datamember() -> f03.Program:
    prog = parse_program(DATAMEMBER_SRC)
    plan = analyze(prog)[0]
    assert monomorphize_component_dispatch(prog, plan) == 1
    return prog


def test_data_member_access_is_routed_per_arm():
    text = str(_rewritten_datamember())
    assert "this % act %" not in text  # no bare slot access survives
    # data-member assignment laddered onto each arm slot
    assert "this % act__t_gmres % counter = 10" in text
    assert "this % act__t_cg % counter = 10" in text
    # pointer-assignment: only the slot's `%act` is retargeted, the RHS is untouched
    assert "this % act__t_gmres % b => this % rhs" in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_data_member_rewritten_fir_has_no_dispatch(tmp_path: Path):
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten_datamember()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    assert fir.read_text().count("fir.dispatch") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_data_member_rewrite_is_behaviour_preserving(tmp_path: Path):
    (tmp_path / "rw.f90").write_text(str(_rewritten_datamember()))
    (tmp_path / "drive.f90").write_text("program drive\n"
                                        "  use m\n"
                                        "  type(container) :: c\n"
                                        "  real, target :: r\n"
                                        "  real :: x\n"
                                        "  c%rhs => r\n"
                                        "  call c%setup(1); x = 5.0; call c%run(x); if (abs(x - 20.0) > 1e-5) stop 1\n"
                                        "  call c%setup(2); x = 5.0; call c%run(x); if (abs(x - 25.0) > 1e-5) stop 2\n"
                                        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Component-slot case: the polymorphic slot is a *component* of a container type
# (ICON's ``t_ocean_solve%act``).  The container's component is expanded, the
# factory ALLOCATE sets the tag, and every ``obj%act%binding(...)`` is laddered.
# The tag lives in the container, so it persists across the setup/run calls.
# ---------------------------------------------------------------------------
COMPONENT_SRC = """
module m
  type, abstract :: base
  contains
    procedure(init_i),  deferred :: init
    procedure(solve_i), deferred :: solve
  end type
  abstract interface
    subroutine init_i(this)
      import base
      class(base), intent(inout) :: this
    end subroutine
    subroutine solve_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: init  => gmres_init
    procedure :: solve => gmres_solve
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: init  => cg_init
    procedure :: solve => cg_solve
  end type
  type :: container
    class(base), allocatable :: act
  contains
    procedure :: setup => container_setup
    procedure :: run   => container_run
  end type
contains
  subroutine gmres_init(this)
    class(t_gmres), intent(inout) :: this
  end subroutine
  subroutine gmres_solve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
  subroutine cg_init(this)
    class(t_cg), intent(inout) :: this
  end subroutine
  subroutine cg_solve(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0
  end subroutine
  subroutine container_setup(this, sel)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
    call this%act%init()
  end subroutine
  subroutine container_run(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%solve(x)
  end subroutine
end module
"""


def _rewritten_component() -> f03.Program:
    prog = parse_program(COMPONENT_SRC)
    plan = analyze(prog)[0]
    assert monomorphize_component_dispatch(prog, plan) == 1
    return prog


def test_component_rewrite_expands_slot_and_removes_dispatch():
    prog = _rewritten_component()
    text = str(prog)
    # the container's polymorphic component is replaced by a tag + per-arm slots
    assert "CLASS(base), ALLOCATABLE" not in text
    assert "INTEGER :: act__tag" in text
    assert "TYPE(t_gmres), ALLOCATABLE :: act__t_gmres" in text
    assert "TYPE(t_cg), ALLOCATABLE :: act__t_cg" in text
    # no bare polymorphic-slot access survives (`this % act %` is gone)
    assert "this % act %" not in text
    # factory sets the tag; dispatch is retargeted onto the concrete arm slot --
    # a static concrete-TYPE type-bound call, no longer a runtime dispatch
    assert "this % act__tag = 1" in text
    assert "CALL this % act__t_gmres % init" in text
    assert "CALL this % act__t_cg % solve(x)" in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_component_rewritten_fir_has_no_dispatch(tmp_path: Path):
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten_component()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    text = fir.read_text()
    assert text.count("fir.dispatch") == 0
    assert text.count("fir.select_type") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_component_rewrite_is_behaviour_preserving(tmp_path: Path):
    # the tag is stored in the container, so it must survive setup -> run.
    (tmp_path / "rw.f90").write_text(str(_rewritten_component()))
    (tmp_path / "drive.f90").write_text("program drive\n"
                                        "  use m\n"
                                        "  type(container) :: c\n"
                                        "  real :: x\n"
                                        "  call c%setup(1); x = 5.0; call c%run(x); if (abs(x - 10.0) > 1e-5) stop 1\n"
                                        "  call c%setup(2); x = 5.0; call c%run(x); if (abs(x - 15.0) > 1e-5) stop 2\n"
                                        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Shared-interposer clone: the slot's dispatched binding is a *non-deferred*
# shared method (`base_run`) that itself dispatches on its CLASS(base) dummy
# (`call this%doit(x)`).  After the component ladder, that buried dispatch must
# be resolved by cloning the interposer per arm with the dummy retyped to the
# concrete TYPE, redirecting each call, and dropping the now-dead original.
# ---------------------------------------------------------------------------
INTERPOSER_SRC = """
module m
  type, abstract :: base
  contains
    procedure :: run => base_run
    procedure(doit_i), deferred :: doit
  end type
  abstract interface
    subroutine doit_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: doit => gmres_doit
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: doit => cg_doit
  end type
  type :: container
    class(base), allocatable :: act
  contains
    procedure :: setup => container_setup
    procedure :: go    => container_go
  end type
contains
  subroutine base_run(this, x)
    class(base), intent(inout) :: this
    real, intent(inout) :: x
    call this%doit(x)
  end subroutine
  subroutine gmres_doit(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
  subroutine cg_doit(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0
  end subroutine
  subroutine container_setup(this, sel)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
  end subroutine
  subroutine container_go(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%run(x)
  end subroutine
end module
"""


def _rewritten_interposer() -> f03.Program:
    prog = parse_program(INTERPOSER_SRC)
    plan = analyze(prog)[0]
    assert monomorphize_component_dispatch(prog, plan) == 1
    assert clone_shared_interposers(prog, plan) == 2
    return prog


def test_shared_interposer_cloned_per_arm_and_original_dropped():
    text = str(_rewritten_interposer())
    # one concrete clone per arm, each with the passed-object retyped to TYPE(arm)
    assert "SUBROUTINE base_run__t_gmres(this, x)" in text
    assert "SUBROUTINE base_run__t_cg(this, x)" in text
    assert "TYPE(t_gmres), INTENT(INOUT) :: this" in text
    # the dead original interposer + its TBP binding are gone
    assert "SUBROUTINE base_run(this, x)" not in text
    assert "run => base_run" not in text
    # each arm's call is redirected to its clone
    assert "CALL base_run__t_gmres(this % act__t_gmres, x)" in text
    assert "CALL base_run__t_cg(this % act__t_cg, x)" in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_interposer_clone_resolves_buried_dispatch(tmp_path: Path):
    # the whole point: the buried `this%doit` inside the shared interposer is now
    # a static bind, so the FIR carries zero dispatch.
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten_interposer()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    assert fir.read_text().count("fir.dispatch") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_interposer_clone_is_behaviour_preserving(tmp_path: Path):
    (tmp_path / "rw.f90").write_text(str(_rewritten_interposer()))
    (tmp_path / "drive.f90").write_text("program drive\n"
                                        "  use m\n"
                                        "  type(container) :: c\n"
                                        "  real :: x\n"
                                        "  call c%setup(1); x = 5.0; call c%go(x); if (abs(x - 10.0) > 1e-5) stop 1\n"
                                        "  call c%setup(2); x = 5.0; call c%go(x); if (abs(x - 15.0) > 1e-5) stop 2\n"
                                        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Sibling-axis retype: an axis fixed to one concrete type at the call site
# (ICON's `trans` -> t_trivial_transfer) is specialised by retyping every
# CLASS(base) declaration -- component / dummy / local -- to TYPE(concrete);
# no ladder, no tag.  The abstract interface's polymorphic dummy is preserved.
# ---------------------------------------------------------------------------
RETYPE_SRC = """
module m
  type, abstract :: t_transfer
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
  type, extends(t_transfer) :: t_trivial
  contains
    procedure :: into => trivial_into
  end type
  type, extends(t_transfer) :: t_subset
  contains
    procedure :: into => subset_into
  end type
  type :: backend
    class(t_transfer), pointer :: trans => null()
  contains
    procedure :: use => backend_use
  end type
contains
  subroutine trivial_into(this, x)
    class(t_trivial), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
  subroutine subset_into(this, x)
    class(t_subset), intent(in) :: this
    real, intent(inout) :: x
    x = x + 2.0
  end subroutine
  subroutine backend_use(this, x)
    class(backend), intent(inout) :: this
    real, intent(inout) :: x
    call this%trans%into(x)
  end subroutine
  subroutine setup(b, tr)
    type(backend), intent(inout) :: b
    class(t_transfer), target, intent(in) :: tr
    b%trans => tr
  end subroutine
end module
"""


def _rewritten_retype() -> f03.Program:
    prog = parse_program(RETYPE_SRC)
    assert retype_to_concrete(prog, "t_transfer", "t_trivial") == 2
    return prog


def test_retype_specialises_component_and_dummy_but_not_interface():
    text = str(_rewritten_retype())
    # component pointer + the pointer-target dummy are retyped to the concrete type
    assert "TYPE(t_trivial), POINTER :: trans" in text
    assert "TYPE(t_trivial), TARGET, INTENT(IN) :: tr" in text
    # the deferred interface keeps its polymorphic dummy (signature must stay abstract)
    assert "CLASS(t_transfer), INTENT(IN) :: this" in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_retype_makes_dispatch_static(tmp_path: Path):
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten_retype()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    assert fir.read_text().count("fir.dispatch") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_retype_is_behaviour_preserving(tmp_path: Path):
    (tmp_path / "rw.f90").write_text(str(_rewritten_retype()))
    (tmp_path / "drive.f90").write_text("program drive\n"
                                        "  use m\n"
                                        "  type(backend) :: b\n"
                                        "  type(t_trivial), target :: tr\n"
                                        "  real :: x\n"
                                        "  call setup(b, tr); x = 5.0; call b%use(x); if (abs(x - 6.0) > 1e-5) stop 1\n"
                                        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Driver end-to-end: two orthogonal axes in one translation unit, nested the way
# ICON's solver TU nests them -- a *laddered* backend (`container%act`, runtime
# arm) whose shared `run` interposer dispatches both on its own deferred `doit`
# AND on a *retyped* transfer sub-component (`base%trans`, pinned to t_trivial).
# The spec wires both: retype t_transfer, ladder base.  This exercises every
# primitive through the single `monomorphize(program, spec)` entry point and the
# cross-axis interaction (the cloned interposer reads the retyped member).
# ---------------------------------------------------------------------------
COMBINED_SRC = """
module m
  type, abstract :: t_transfer
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
  type, extends(t_transfer) :: t_trivial
  contains
    procedure :: into => trivial_into
  end type
  type, extends(t_transfer) :: t_subset
  contains
    procedure :: into => subset_into
  end type
  type, abstract :: base
    class(t_transfer), pointer :: trans => null()
  contains
    procedure :: run => base_run
    procedure(doit_i), deferred :: doit
  end type
  abstract interface
    subroutine doit_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: doit => gmres_doit
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: doit => cg_doit
  end type
  type :: container
    class(base), allocatable :: act
  contains
    procedure :: setup => container_setup
    procedure :: go    => container_go
  end type
contains
  subroutine trivial_into(this, x)
    class(t_trivial), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
  subroutine subset_into(this, x)
    class(t_subset), intent(in) :: this
    real, intent(inout) :: x
    x = x + 2.0
  end subroutine
  subroutine base_run(this, x)
    class(base), intent(inout) :: this
    real, intent(inout) :: x
    call this%trans%into(x)
    call this%doit(x)
  end subroutine
  subroutine gmres_doit(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
  subroutine cg_doit(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0
  end subroutine
  subroutine container_setup(this, sel, tr)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    class(t_transfer), target, intent(in) :: tr
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
    this%act%trans => tr
  end subroutine
  subroutine container_go(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%run(x)
  end subroutine
end module
"""

COMBINED_SPEC = MonomorphizationSpec(axes=[
    AxisSpec(base="t_transfer", strategy="retype", concrete="t_trivial"),
    AxisSpec(base="base", strategy="ladder"),
])


def _rewritten_combined() -> f03.Program:
    prog = parse_program(COMBINED_SRC)
    stats = monomorphize(prog, COMBINED_SPEC)
    # retype: base%trans component + setup's tr dummy (interface dummy preserved)
    assert stats.declarations_retyped == 2
    assert stats.locals_rewritten == 0  # no CLASS(base) *local*, only the component
    assert stats.components_rewritten == 1  # container%act
    assert stats.interposers_cloned == 2  # base_run specialised for t_gmres + t_cg
    return prog


def test_driver_collapses_both_nested_axes():
    text = str(_rewritten_combined())
    # ladder axis: the polymorphic component became a tag + per-arm concrete slots
    assert "CLASS(base), ALLOCATABLE" not in text
    assert "INTEGER :: act__tag" in text
    # retype axis: the transfer sub-component is now concrete (no CLASS(t_transfer)
    # data member survives); the deferred interface keeps its polymorphic dummy
    assert "CLASS(t_transfer), POINTER" not in text
    assert "TYPE(t_trivial), POINTER :: trans" in text
    assert "CLASS(t_transfer), INTENT(IN) :: this" in text
    # the shared interposer is cloned per arm and the dead original is dropped
    assert "SUBROUTINE base_run__t_gmres(this, x)" in text
    assert "SUBROUTINE base_run(this, x)" not in text
    assert "run => base_run" not in text


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_driver_result_fir_has_no_dispatch(tmp_path: Path):
    src = tmp_path / "rw.f90"
    src.write_text(str(_rewritten_combined()))
    fir = tmp_path / "rw.fir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-fir", str(src), "-o", str(fir)], cwd=str(tmp_path))
    text = fir.read_text()
    assert text.count("fir.dispatch") == 0
    assert text.count("fir.select_type") == 0


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_driver_result_is_behaviour_preserving(tmp_path: Path):
    # gmres arm: into(+1) then doit(*2): 5 -> 6 -> 12;  cg arm: +1 then *3: 5 -> 6 -> 18.
    (tmp_path / "rw.f90").write_text(str(_rewritten_combined()))
    (tmp_path / "drive.f90").write_text(
        "program drive\n"
        "  use m\n"
        "  type(container) :: c\n"
        "  type(t_trivial), target :: tr\n"
        "  real :: x\n"
        "  call c%setup(1, tr); x = 5.0; call c%go(x); if (abs(x - 12.0) > 1e-5) stop 1\n"
        "  call c%setup(2, tr); x = 5.0; call c%go(x); if (abs(x - 18.0) > 1e-5) stop 2\n"
        "end program\n")
    subprocess.check_call(["gfortran", "rw.f90", "drive.f90", "-o", "rw_run"], cwd=str(tmp_path))
    subprocess.check_call([str(tmp_path / "rw_run")], cwd=str(tmp_path))


def test_driver_rejects_unknown_strategy():
    prog = parse_program(COMBINED_SRC)
    with pytest.raises(ValueError, match="unknown monomorphisation strategy"):
        monomorphize(prog, MonomorphizationSpec(axes=[AxisSpec(base="base", strategy="invert")]))


def test_driver_rejects_retype_without_concrete():
    prog = parse_program(COMBINED_SRC)
    with pytest.raises(ValueError, match="needs a concrete type"):
        monomorphize(prog, MonomorphizationSpec(axes=[AxisSpec(base="t_transfer", strategy="retype")]))


def test_driver_rejects_ladder_axis_absent_from_unit():
    prog = parse_program(COMBINED_SRC)
    with pytest.raises(ValueError, match="no dispatch plan"):
        monomorphize(prog, MonomorphizationSpec(axes=[AxisSpec(base="t_nonexistent", strategy="ladder")]))
