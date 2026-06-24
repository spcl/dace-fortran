"""Supportability tests for the static-vtable monomorphisation front-end
(:mod:`dace_fortran.inliner.ast_desugaring.monomorphize`).

Each fixture is a small Fortran program exercising one shape of polymorphic
type-bound-procedure dispatch.  Supported shapes (abstract base + concrete
subtypes, each providing ``%init`` and ``%solve``, single-level) must yield a
:class:`MonomorphizationPlan` enumerating every concrete arm.  Out-of-scope
shapes must be rejected -- and rather than ``xfail``, we assert the rejection
directly with ``pytest.raises(UnsupportedProgram)`` and match the reason, so the
detection is itself under test.
"""
import pytest

from dace_fortran.inliner.ast_desugaring.monomorphize import (UnsupportedProgram, analyze_source)

# ---------------------------------------------------------------------------
# Supported: an abstract solver + two concretizations, each with %init/%solve.
# (The ICON-O ocean-solver shape, minimised.)
# ---------------------------------------------------------------------------
ABSTRACT_PREAMBLE = """
module m
  type, abstract :: t_solver
    integer :: n = 0
  contains
    procedure(init_i),  deferred :: init
    procedure(solve_i), deferred :: solve
  end type
  abstract interface
    subroutine init_i(this, n)
      import t_solver
      class(t_solver), intent(inout) :: this
      integer, intent(in) :: n
    end subroutine
    subroutine solve_i(this, x)
      import t_solver
      class(t_solver), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
"""

SUPPORTED_TWO_CONCRETE = ABSTRACT_PREAMBLE + """
  type, extends(t_solver) :: t_gmres
  contains
    procedure :: init  => gmres_init
    procedure :: solve => gmres_solve
  end type
  type, extends(t_solver) :: t_cg
  contains
    procedure :: init  => cg_init
    procedure :: solve => cg_solve
  end type
contains
  subroutine gmres_init(this, n)
    class(t_gmres), intent(inout) :: this
    integer, intent(in) :: n
  end subroutine
  subroutine gmres_solve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
  end subroutine
  subroutine cg_init(this, n)
    class(t_cg), intent(inout) :: this
    integer, intent(in) :: n
  end subroutine
  subroutine cg_solve(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
  end subroutine
end module
"""

SUPPORTED_ONE_CONCRETE = ABSTRACT_PREAMBLE + """
  type, extends(t_solver) :: t_gmres
  contains
    procedure :: init  => gmres_init
    procedure :: solve => gmres_solve
  end type
contains
  subroutine gmres_init(this, n)
    class(t_gmres), intent(inout) :: this
    integer, intent(in) :: n
  end subroutine
  subroutine gmres_solve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
  end subroutine
end module
"""

# ---------------------------------------------------------------------------
# Unsupported shapes -- each must raise UnsupportedProgram with a clear reason.
# ---------------------------------------------------------------------------
UNSUPPORTED_UNLIMITED_POLY = """
module m
contains
  subroutine takes_anything(x)
    class(*), intent(in) :: x
  end subroutine
end module
"""

UNSUPPORTED_DEEP_CHAIN = """
module m
  type, abstract :: t_base
  contains
    procedure(foo_i), deferred :: foo
  end type
  abstract interface
    subroutine foo_i(this)
      import t_base
      class(t_base), intent(in) :: this
    end subroutine
  end interface
  type, extends(t_base) :: t_mid
  contains
    procedure :: foo => mid_foo
  end type
  type, extends(t_mid) :: t_leaf      ! grandchild -> depth 2
  end type
contains
  subroutine mid_foo(this)
    class(t_mid), intent(in) :: this
  end subroutine
end module
"""

UNSUPPORTED_ABSTRACT_INTERMEDIATE = """
module m
  type, abstract :: t_base
  contains
    procedure(foo_i), deferred :: foo
  end type
  abstract interface
    subroutine foo_i(this)
      import t_base
      class(t_base), intent(in) :: this
    end subroutine
  end interface
  type, abstract, extends(t_base) :: t_amid   ! abstract subtype, not a leaf
  end type
end module
"""

UNSUPPORTED_NO_CONCRETE = """
module m
  type, abstract :: t_base
  contains
    procedure(foo_i), deferred :: foo
  end type
  abstract interface
    subroutine foo_i(this)
      import t_base
      class(t_base), intent(in) :: this
    end subroutine
  end interface
end module
"""

UNSUPPORTED_MISSING_OVERRIDE = ABSTRACT_PREAMBLE + """
  type, extends(t_solver) :: t_gmres
  contains
    procedure :: init => gmres_init      ! overrides init but NOT solve
  end type
contains
  subroutine gmres_init(this, n)
    class(t_gmres), intent(inout) :: this
    integer, intent(in) :: n
  end subroutine
end module
"""


def test_supported_two_concrete_enumerates_both_arms():
    plans = analyze_source(SUPPORTED_TWO_CONCRETE)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.abstract_base == "t_solver"
    assert plan.deferred == ["init", "solve"]
    arms = {a.type_name: a.bindings for a in plan.arms}
    assert arms == {
        "t_gmres": {"init": "gmres_init", "solve": "gmres_solve"},
        "t_cg": {"init": "cg_init", "solve": "cg_solve"},
    }


def test_supported_single_concrete_is_a_one_arm_plan():
    plans = analyze_source(SUPPORTED_ONE_CONCRETE)
    assert len(plans) == 1
    assert [a.type_name for a in plans[0].arms] == ["t_gmres"]


def test_program_without_polymorphism_is_a_noop_not_an_error():
    plans = analyze_source("module m\n  integer :: x\nend module\n")
    assert plans == []


def test_unlimited_polymorphic_is_rejected():
    with pytest.raises(UnsupportedProgram, match=r"CLASS\(\*\)"):
        analyze_source(UNSUPPORTED_UNLIMITED_POLY)


def test_deep_inheritance_chain_is_rejected():
    with pytest.raises(UnsupportedProgram, match=r"depth > 1"):
        analyze_source(UNSUPPORTED_DEEP_CHAIN)


def test_abstract_intermediate_subtype_is_rejected():
    with pytest.raises(UnsupportedProgram, match=r"itself abstract"):
        analyze_source(UNSUPPORTED_ABSTRACT_INTERMEDIATE)


def test_abstract_base_without_concrete_subtype_is_rejected():
    with pytest.raises(UnsupportedProgram, match=r"nothing to dispatch to"):
        analyze_source(UNSUPPORTED_NO_CONCRETE)


def test_subtype_missing_a_deferred_override_is_rejected():
    with pytest.raises(UnsupportedProgram, match=r"does not override deferred"):
        analyze_source(UNSUPPORTED_MISSING_OVERRIDE)
