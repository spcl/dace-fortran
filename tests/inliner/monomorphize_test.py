# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra supportability tests for static-vtable monomorphisation
(:mod:`dace_fortran.inliner.ast_desugaring.monomorphize`).

This file is *complementary* to ``monomorphize_support_test.py`` (which covers
the core supported/unsupported matrix): it adds the cases that one doesn't --
multiple independent dispatch bases (ICON's orthogonal backend/transfer/lhs
axes), the depth guard hit via an abstract base that itself ``EXTENDS`` another
type, and the :class:`UnsupportedProgram` ``reason`` contract.  Rejections are
asserted with ``pytest.raises`` (not xfail) so detection is itself under test.
"""
import pytest

from dace_fortran.inliner.ast_desugaring.monomorphize import analyze_source, UnsupportedProgram

# Two independent abstract bases -- mirrors ICON's orthogonal dispatch axes
# (backend + transfer), each its own closed subtype set => two plans.
TWO_INDEPENDENT_BASES = """
module two
  type, abstract :: t_solver
  contains
    procedure(s_i), deferred :: solve
  end type
  type, abstract :: t_transfer
  contains
    procedure(t_i), deferred :: into
  end type
  abstract interface
    subroutine s_i(this)
      import t_solver
      class(t_solver), intent(in) :: this
    end subroutine
    subroutine t_i(this)
      import t_transfer
      class(t_transfer), intent(in) :: this
    end subroutine
  end interface
  type, extends(t_solver)   :: t_gmres
  contains
    procedure :: solve => gmres_solve
  end type
  type, extends(t_transfer) :: t_trivial
  contains
    procedure :: into => trivial_into
  end type
contains
  subroutine gmres_solve(this)
    class(t_gmres), intent(in) :: this
  end subroutine
  subroutine trivial_into(this)
    class(t_trivial), intent(in) :: this
  end subroutine
end module
"""

# A dispatch-root abstract base that itself EXTENDS another type (depth > 1) --
# hits the "abstract base itself extends" guard, distinct from the abstract-leaf
# subtype guard covered elsewhere.
ABSTRACT_EXTENDS_ABSTRACT = """
module m
  type, abstract :: a_root
  end type
  type, abstract, extends(a_root) :: a_disp
  contains
    procedure(g_i), deferred :: g
  end type
  abstract interface
    subroutine g_i(this)
      import a_disp
      class(a_disp), intent(in) :: this
    end subroutine
  end interface
  type, extends(a_disp) :: leaf
  contains
    procedure :: g => leaf_g
  end type
contains
  subroutine leaf_g(this)
    class(leaf), intent(in) :: this
  end subroutine
end module
"""

UNLIMITED_POLYMORPHIC = """
module m
  type :: holder
    class(*), allocatable :: anything
  end type
end module
"""


def test_two_independent_bases_yield_two_plans():
    plans = analyze_source(TWO_INDEPENDENT_BASES)
    by_base = {p.abstract_base: p for p in plans}
    assert set(by_base) == {"t_solver", "t_transfer"}
    assert [a.type_name for a in by_base["t_solver"].arms] == ["t_gmres"]
    assert [a.type_name for a in by_base["t_transfer"].arms] == ["t_trivial"]


def test_abstract_extending_abstract_is_rejected():
    with pytest.raises(UnsupportedProgram, match="itself extends"):
        analyze_source(ABSTRACT_EXTENDS_ABSTRACT)


def test_unsupported_program_carries_reason():
    with pytest.raises(UnsupportedProgram) as excinfo:
        analyze_source(UNLIMITED_POLYMORPHIC)
    # the reason is both the exception message and a dedicated attribute, so
    # callers can detect + surface *why* a program was rejected.
    assert excinfo.value.reason == str(excinfo.value)
    assert "CLASS(*)" in excinfo.value.reason
