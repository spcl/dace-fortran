# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import fparser.two.Fortran2003 as f03
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from dace_fortran import inline_to_single_tu
from dace_fortran.inliner import ast_utils
from dace_fortran.inliner.ast_desugaring import desugaring, cleanup
from inliner.fortran_test_helper import SourceCodeBuilder, parse_and_improve

# ECMWF ``fcttre``/``fccld`` idiom: a type declaration that FOLLOWS a statement
# function.  fparser opens a second ``Specification_Part`` at that boundary, and
# every downstream ``atmost_one``/``singular`` over the spec/exec parts then
# raises "must have at most 1 item".  ``fccld.func.h`` / ``fcttre.func.h`` in the
# CLOUDSC kernels interleave several of these.
_INTERLEAVED_STMT_FN_SRC = """
module m
  implicit none
contains
  subroutine stf(ptare_in, res)
    implicit none
    real(kind=8), intent(in) :: ptare_in
    real(kind=8), intent(out) :: res
    real(kind=8) :: foedelta
    real(kind=8) :: ptare
    foedelta(ptare) = max(0.0d0, ptare)
    real(kind=8) :: foeew
    foeew(ptare) = foedelta(ptare) * 2.0d0
    res = foeew(ptare_in)
  end subroutine stf
end module m
"""


def test_coalesce_split_specification_parts():
    """``coalesce_split_specification_parts`` folds a scope whose spec part
    fparser split back to a single spec + single exec part, without changing
    semantics."""
    parser = ParserFactory().create(std="f2008")
    ast = parser(FortranStringReader(_INTERLEAVED_STMT_FN_SRC))

    sub = next(iter(walk(ast, f03.Subroutine_Subprogram)))
    # Pre-condition (the bug): fparser split the spec part in two.
    assert len(list(ast_utils.children_of_type(sub, f03.Specification_Part))) == 2

    ast = desugaring.coalesce_split_specification_parts(ast)

    sub = next(iter(walk(ast, f03.Subroutine_Subprogram)))
    assert len(list(ast_utils.children_of_type(sub, f03.Specification_Part))) == 1
    assert len(list(ast_utils.children_of_type(sub, f03.Execution_Part))) == 1
    # Every declaration survives, hoisted ahead of the executable statements.
    names = {n.string.lower() for d in walk(sub, f03.Type_Declaration_Stmt) for n in walk(d, f03.Name)}
    assert {"foedelta", "foeew", "ptare", "ptare_in"} <= names
    SourceCodeBuilder().add_file(ast.tofortran()).check_with_gfortran()


def test_interleaved_statement_functions_extract_end_to_end():
    """Full inliner over the interleaved-stmt-fn source: without the coalescer
    the pipeline raises "at most 1 item"; with it the statement functions
    deconstruct into internal FUNCTIONs and the TU is emitted."""
    tu = inline_to_single_tu(sources={"m.f90": _INTERLEAVED_STMT_FN_SRC}, entry="m::stf")
    from pathlib import Path
    text = Path(tu).read_text() if not isinstance(tu, str) else tu
    low = text.lower()
    assert "function foedelta" in low and "function foeew" in low
    assert "res = foeew(ptare_in)" in low


def test_force_double_precision_collapses_real_kinds():
    """``force_double_precision`` rewrites every ``SELECTED_REAL_KIND(...)`` to
    the fp64 kind, so a kernel written for a reduced-precision kind parameter
    (the SC2026 CLOUDSC ``JPRM``/``JPRL`` family) lowers at uniform fp64 -- no
    ``REAL(KIND=4)`` survives while integer kinds are untouched."""
    from pathlib import Path
    src = """
module m
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jprm = selected_real_kind(6, 37)
  integer, parameter :: jprb = selected_real_kind(13, 300)
contains
  subroutine k(x, n)
    integer(kind=jpim), intent(in) :: n
    real(kind=jprb), intent(inout) :: x(n)
    real(kind=jprm) :: acc
    integer(kind=jpim) :: i
    acc = 0.0_jprm
    do i = 1, n
      acc = acc + x(i) * 2.0_jprb
    end do
    x(1) = acc
  end subroutine k
end module m
"""
    tu = inline_to_single_tu(sources={"m.f90": src}, entry="m::k", force_double_precision=True)
    text = (Path(tu).read_text() if not isinstance(tu, str) else tu).lower()
    assert "real(kind = 4)" not in text and "real(kind = 8)" in text
    assert "selected_real_kind" not in text
    # Integer kinds are left alone.
    assert "integer(kind = 4)" in text


_TIMER_LIKE_SRC = """
module tmod
  implicit none
  type :: perf
     integer :: c
  end type perf
contains
  subroutine p_start(self, n)
    class(perf) :: self
    integer, intent(in) :: n
    self%c = n
  end subroutine p_start
end module tmod
module dmod
contains
subroutine drv(x)
  use tmod, only: perf, p_start
  implicit none
  real(kind=8), intent(inout) :: x
  type(perf) :: t
  call p_start(t, 3)
  x = x * 2.0d0
end subroutine drv
end module dmod
"""


def test_make_noop_drops_calls_and_prunes_stub():
    """An EXPLICIT make_noop subroutine is a semantic no-op: its CALL
    statements are dropped outright and the stub -- with its derived-type
    scaffolding -- prunes away.  (A surviving module procedure with a
    derived-type dummy makes numpy f2py emit a NULL module entry, so the
    import of the f2py reference leg segfaults; CLOUDSC's PERFORMANCE_TIMER
    stubs hit exactly this.)"""
    from pathlib import Path
    tu = inline_to_single_tu(sources={"m.f90": _TIMER_LIKE_SRC}, entry="dmod::drv", make_noop=[("tmod", "p_start")])
    text = (Path(tu).read_text() if not isinstance(tu, str) else tu).lower()
    assert "p_start" not in text
    assert "type(perf)" not in text and "class(perf)" not in text
    assert "x = x * 2.0d0" in text


def test_keep_external_stub_keeps_calls():
    """The do_not_emit/keep_external path must NOT drop call statements --
    those calls are real (an external implementation or the bridge serves
    them).  Only the explicit make_noop path is a droppable no-op."""
    from pathlib import Path
    src = """
module emod
contains
subroutine ext_impl(x)
  implicit none
  real(kind=8), intent(inout) :: x
  x = x + 1.0d0
end subroutine ext_impl
end module emod
module dmod2
contains
subroutine drv2(x)
  use emod, only: ext_impl
  implicit none
  real(kind=8), intent(inout) :: x
  call ext_impl(x)
  x = x * 2.0d0
end subroutine drv2
end module dmod2
"""
    tu = inline_to_single_tu(sources={"m.f90": src}, entry="dmod2::drv2", do_not_emit=("ext_impl", ))
    text = (Path(tu).read_text() if not isinstance(tu, str) else tu).lower()
    assert "call ext_impl" in text


def test_pruned_memberless_type_gets_placeholder():
    """When pruning strips every component of a derived type that variables
    still reference, a placeholder component is kept: a zero-component type is
    questionable Fortran and makes numpy f2py's module wrapper NULL (import
    segfault)."""
    from pathlib import Path
    src = """
module pmod
  implicit none
  type :: holder
     real(kind=8), allocatable :: unused_payload(:)
  end type holder
contains
subroutine use_holder(x)
  implicit none
  real(kind=8), intent(inout) :: x
  type(holder) :: h
  x = x * 2.0d0
end subroutine use_holder
end module pmod
"""
    tu = inline_to_single_tu(sources={"m.f90": src},
                             entry="pmod::use_holder",
                             do_not_prune_type_components=False,
                             f2py_safe=True)
    text = (Path(tu).read_text() if not isinstance(tu, str) else tu).lower()
    # Either the type pruned away entirely (no variable left referencing it)
    # or, if it survived, it must not be memberless.
    if "type :: holder" in text:
        assert "pruned_type_placeholder" in text or "unused_payload" in text


def test_procedure_replacer():
    """
    Tests that type-bound procedures are correctly replaced with standard subroutine calls.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
    procedure :: area_alt => area
    procedure :: get_area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m * this%side * this%side
  end function area
  subroutine get_area(this, a)
    implicit none
    class(Square), intent(in) :: this
    real, intent(out) :: a
    a = area(this, 1.0)
  end subroutine get_area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a

  s%side = 1.0
  a = s%area(1.0)
  a = s%area_alt(1.0)
  call s%get_area(a)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
  SUBROUTINE get_area(this, a)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(OUT) :: a
    a = area(this, 1.0)
  END SUBROUTINE get_area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: get_area_deconproc_2 => get_area
  USE lib, ONLY: area_deconproc_1 => area
  USE lib, ONLY: area_deconproc_0 => area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side = 1.0
  a = area_deconproc_0(s, 1.0)
  a = area_deconproc_1(s, 1.0)
  CALL get_area_deconproc_2(s, a)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_inherited_type_bound():
    """A type-bound procedure (and a data component) inherited via ``EXTENDS``
    resolves directly on the child -- ``child_obj % gid(i)`` deconstructs to the
    *base* procedure, and ``child_obj % a`` reads the *base* component, without
    naming the parent type.

    Regression test for the alias map only registering the explicit
    parent-subobject access (``child % base % member``) and not the direct
    inherited access (``child % member``): an inherited type-bound FUNCTION call
    then crashed component resolution in ``correct_for_function_calls``.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module base_mod
  implicit none
  type :: t_base
    integer :: a
  contains
    procedure :: gid => base_gid
  end type t_base
contains
  pure integer function base_gid(this, i)
    class(t_base), intent(in) :: this
    integer, intent(in) :: i
    base_gid = this%a + i
  end function base_gid
end module base_mod

module deriv_mod
  use base_mod, only: t_base
  implicit none
  type, extends(t_base) :: t_deriv
    integer :: b
  end type t_deriv
end module deriv_mod

subroutine main(obj, i, out)
  use deriv_mod, only: t_deriv
  implicit none
  type(t_deriv), intent(in) :: obj
  integer, intent(in) :: i
  integer, intent(out) :: out
  out = obj%gid(i) + obj%a + obj%b
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    # The pass that crashed pre-fix on the inherited type-bound function call.
    ast = cleanup.correct_for_function_calls(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    # The inherited type-bound call resolves to the BASE concrete procedure.
    assert "base_gid_deconproc" in got
    # The inherited data component survives as an ordinary component read.
    assert "obj % a" in got
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_overrides_inherited_type_bound():
    """A child that overrides an inherited binding wins: ``child_obj % gid(i)``
    resolves to the child's own procedure, not the parent's.  Guards the direct
    inherited-access registration against clobbering a member the child declares
    itself."""
    sources, _ = (SourceCodeBuilder().add_file("""
module base_mod
  implicit none
  type :: t_base
    integer :: a
  contains
    procedure :: gid => base_gid
  end type t_base
contains
  pure integer function base_gid(this, i)
    class(t_base), intent(in) :: this
    integer, intent(in) :: i
    base_gid = this%a + i
  end function base_gid
end module base_mod

module deriv_mod
  use base_mod, only: t_base
  implicit none
  type, extends(t_base) :: t_deriv
    integer :: b
  contains
    procedure :: gid => deriv_gid
  end type t_deriv
contains
  pure integer function deriv_gid(this, i)
    class(t_deriv), intent(in) :: this
    integer, intent(in) :: i
    deriv_gid = this%b - i
  end function deriv_gid
end module deriv_mod

subroutine main(obj, i, out)
  use deriv_mod, only: t_deriv
  implicit none
  type(t_deriv), intent(in) :: obj
  integer, intent(in) :: i
  integer, intent(out) :: out
  out = obj%gid(i)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.correct_for_function_calls(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    # The override wins; the parent's procedure is not selected.
    assert "deriv_gid_deconproc" in got
    assert "base_gid_deconproc" not in got
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_nested():
    """
    Tests that nested type-bound procedures are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Value
    real :: val
  contains
    procedure :: get_value
  end type Value
  type Square
    type(Value) :: side
  contains
    procedure :: get_area
  end type Square
contains
  real function get_value(this)
    implicit none
    class(Value), intent(in) :: this
    get_value = this%val
  end function get_value
  real function get_area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    real :: side
    side = this%side%get_value()
    get_area = m*side*side
  end function get_area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a

  s%side%val = 1.0
  a = s%get_area(1.0)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Value
    REAL :: val
  END TYPE Value
  TYPE :: Square
    TYPE(Value) :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION get_value(this)
    IMPLICIT NONE
    CLASS(Value), INTENT(IN) :: this
    get_value = this % val
  END FUNCTION get_value
  REAL FUNCTION get_area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    REAL :: side
    side = get_value(this % side)
    get_area = m * side * side
  END FUNCTION get_area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: get_area_deconproc_0 => get_area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side % val = 1.0
  a = get_area_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_name_collision_with_exisiting_var():
    """
    Tests that procedure replacement handles name collisions with existing variables.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: area

  s%side = 1.0
  area = s%area(1.0)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: area_deconproc_0 => area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: area
  s % side = 1.0
  area = area_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_name_collision_with_another_import():
    """
    Tests that procedure replacement handles name collisions with other imports.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib_1
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib_1

module lib_2
  implicit none
  type Circle
    real :: rad
  contains
    procedure :: area
  end type Circle
contains
  real function area(this, m)
    implicit none
    class(Circle), intent(in) :: this
    real, intent(in) :: m
    area = m*this%rad*this%rad
  end function area
end module lib_2

subroutine main
  use lib_1, only: Square
  use lib_2, only: Circle
  implicit none
  type(Square) :: s
  type(Circle) :: c
  real :: area

  s%side = 1.0
  area = s%area(1.0)
  c%rad = 1.0
  area = c%area(1.0)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib_1
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib_1
MODULE lib_2
  IMPLICIT NONE
  TYPE :: Circle
    REAL :: rad
  END TYPE Circle
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Circle), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % rad * this % rad
  END FUNCTION area
END MODULE lib_2
SUBROUTINE main
  USE lib_2, ONLY: area_deconproc_1 => area
  USE lib_1, ONLY: area_deconproc_0 => area
  USE lib_1, ONLY: Square
  USE lib_2, ONLY: Circle
  IMPLICIT NONE
  TYPE(Square) :: s
  TYPE(Circle) :: c
  REAL :: area
  s % side = 1.0
  area = area_deconproc_0(s, 1.0)
  c % rad = 1.0
  area = area_deconproc_1(c, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_generic_replacer():
    """
    Tests that generic procedures are correctly replaced based on the argument types.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area_real
    procedure :: area_integer
    generic :: g_area => area_real, area_integer
  end type Square
contains
  real function area_real(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area_real = m*this%side*this%side
  end function area_real
  real function area_integer(this, m)
    implicit none
    class(Square), intent(in) :: this
    integer, intent(in) :: m
    area_integer = m*this%side*this%side
  end function area_integer
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a
  real :: mr = 1.0
  integer :: mi = 1

  s%side = 1.0
  a = s%g_area(mr)
  a = s%g_area(mi)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area_real(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area_real = m * this % side * this % side
  END FUNCTION area_real
  REAL FUNCTION area_integer(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    INTEGER, INTENT(IN) :: m
    area_integer = m * this % side * this % side
  END FUNCTION area_integer
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: area_integer_deconproc_1 => area_integer
  USE lib, ONLY: area_real_deconproc_0 => area_real
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  REAL :: mr = 1.0
  INTEGER :: mi = 1
  s % side = 1.0
  a = area_real_deconproc_0(s, mr)
  a = area_integer_deconproc_1(s, mi)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer():
    """
    Tests that the `ASSOCIATE` construct is correctly replaced by substituting
    the associated name with the original expression.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib

subroutine main
  use lib, only: Square, area
  implicit none
  type(Square) :: s
  real :: a

  associate(side => s%side)
    s%side = 0.5
    side = 1.0
    a = area(s, 1.0)
  end associate
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: Square, area
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side = 0.5
  s % side = 1.0
  a = area(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer_array_access():
    """
    Tests that `ASSOCIATE` constructs with array accesses are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: sides(2, 2)
  contains
    procedure :: area => perim
  end type Square
contains
  real function perim(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    perim = m * sum(this%sides)
  end function perim
end module lib

subroutine main
  use lib, only: Square, perim
  implicit none
  type(Square) :: s
  real :: a

  associate(sides => s%sides)
    s%sides = 0.5
    s%sides(1, 1) = 1.0
    sides(2, 2) = 1.0
    a = perim(s, 1.0)
    a = s%area(1.0)
  end associate
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_enums(ast)
    ast = desugaring.deconstruct_associations(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  CONTAINS
  REAL FUNCTION perim(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    perim = m * SUM(this % sides)
  END FUNCTION perim
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib, ONLY: Square, perim
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % sides = 0.5
  s % sides(1, 1) = 1.0
  s % sides(2, 2) = 1.0
  a = perim(s, 1.0)
  a = perim_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer_array_access_within_array_access():
    """
    Tests that `ASSOCIATE` constructs with array accesses within array accesses
    are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: sides(2, 2)
  contains
    procedure :: area => perim
  end type Square
contains
  real function perim(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    perim = m * sum(this%sides)
  end function perim
end module lib

subroutine main
  use lib, only: Square, perim
  implicit none
  type(Square) :: s
  real :: a

  associate(sides => s%sides(:, 1))
    s%sides = 0.5
    s%sides(1, 1) = 1.0
    sides(2) = 1.0
    a = perim(s, 1.0)
    a = s%area(1.0)
  end associate
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  CONTAINS
  REAL FUNCTION perim(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    perim = m * SUM(this % sides)
  END FUNCTION perim
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib, ONLY: Square, perim
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % sides = 0.5
  s % sides(1, 1) = 1.0
  s % sides(2, 1) = 1.0
  a = perim(s, 1.0)
  a = perim_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_enum_bindings_become_constants():
    """
    Tests that `ENUM` bindings are converted to integer constants.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 42
  enum, bind(c)
    enumerator :: a, b, c
  end enum
  enum, bind(c)
    enumerator :: d = a, e, f
  end enum
  enum, bind(c)
    enumerator :: g = k, h = k, i = k + 1
  end enum
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_enums(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 42
  INTEGER, PARAMETER :: a = 0 + 0
  INTEGER, PARAMETER :: b = 0 + 1
  INTEGER, PARAMETER :: c = 0 + 2
  INTEGER, PARAMETER :: d = a + 0
  INTEGER, PARAMETER :: e = a + 1
  INTEGER, PARAMETER :: f = a + 2
  INTEGER, PARAMETER :: g = k + 0
  INTEGER, PARAMETER :: h = k + 0
  INTEGER, PARAMETER :: i = k + 1 + 0
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_aliasing_through_module_procedure():
    """
    Tests that aliasing through module procedures is handled correctly.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
contains
  real function real_fun()
    implicit none
    real_fun = 1.0
  end function real_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(4)
  d(2) = fun()
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)
    ast = cleanup.correct_for_function_calls(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun()
    IMPLICIT NONE
    real_fun = 1.0
  END FUNCTION real_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: fun
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = fun()
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_module_procedures():
    """
    Tests that interfaces with module procedures are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
  interface not_fun
    module procedure not_real_fun
  end interface not_fun
  interface same_name
    module procedure same_name, real_fun
  end interface same_name
contains
  real function real_fun()
    implicit none
    real_fun = 1.0
  end function real_fun
  subroutine not_real_fun(a)
    implicit none
    real, intent(out) :: a
    a = 1.0
  end subroutine not_real_fun
  real function same_name(x)
    implicit none
    real, intent(in) :: x
    same_name = x
  end function same_name
end module lib

subroutine main
  use lib, only: fun, not_fun, same_name
  implicit none
  real d(4)
  d(2) = fun()
  call not_fun(d(3))
  d(4) = same_name(2.0)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  INTERFACE not_fun
    MODULE PROCEDURE not_real_fun
  END INTERFACE not_fun
  INTERFACE same_name
    MODULE PROCEDURE same_name, real_fun
  END INTERFACE same_name
  CONTAINS
  REAL FUNCTION real_fun()
    IMPLICIT NONE
    real_fun = 1.0
  END FUNCTION real_fun
  SUBROUTINE not_real_fun(a)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: a
    a = 1.0
  END SUBROUTINE not_real_fun
  REAL FUNCTION same_name(x)
    IMPLICIT NONE
    REAL, INTENT(IN) :: x
    same_name = x
  END FUNCTION same_name
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: not_real_fun_deconiface_1 => not_real_fun, real_fun_deconiface_0 => real_fun, same_name_deconiface_2 => same_name
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = real_fun_deconiface_0()
  CALL not_real_fun_deconiface_1(d(3))
  d(4) = same_name_deconiface_2(2.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_subroutine_decls():
    """
    Tests that interfaces with subroutine declarations are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface
    subroutine fun(z)
      implicit none
      real, intent(out) :: z
    end subroutine fun
  end interface
end module lib

subroutine main
  use lib, only: no_fun => fun
  implicit none
  real d(4)
  call no_fun(d(3))
end subroutine main

subroutine fun(z)
  implicit none
  real, intent(out) :: z
  z = 1.0
end subroutine fun
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE fun(z)
      IMPLICIT NONE
      REAL, INTENT(OUT) :: z
    END SUBROUTINE fun
  END INTERFACE
END MODULE lib
SUBROUTINE main
  IMPLICIT NONE
  REAL :: d(4)
  CALL fun(d(3))
END SUBROUTINE main
SUBROUTINE fun(z)
  IMPLICIT NONE
  REAL, INTENT(OUT) :: z
  z = 1.0
END SUBROUTINE fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_optional_args():
    """
    Tests that interfaces with optional arguments are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun, integer_fun
  end interface fun
contains
  real function real_fun(x)
    implicit none
    real, intent(in), optional :: x
    if (.not.(present(x))) then
      real_fun = 1.0
    else
      real_fun = x
    end if
  end function real_fun
  integer function integer_fun(x)
    implicit none
    integer, intent(in) :: x
    integer_fun = x * 2
  end function integer_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(4)
  d(2) = fun()
  d(3) = fun(x=4)
  d(4) = fun(x=5.0)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun, integer_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun(x)
    IMPLICIT NONE
    REAL, INTENT(IN), OPTIONAL :: x
    IF (.NOT. (PRESENT(x))) THEN
      real_fun = 1.0
    ELSE
      real_fun = x
    END IF
  END FUNCTION real_fun
  INTEGER FUNCTION integer_fun(x)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: x
    integer_fun = x * 2
  END FUNCTION integer_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: integer_fun_deconiface_1 => integer_fun, real_fun_deconiface_0 => real_fun, real_fun_deconiface_2 => real_fun
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = real_fun_deconiface_0()
  d(3) = integer_fun_deconiface_1(x = 4)
  d(4) = real_fun_deconiface_2(x = 5.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_keyworded_args():
    """
    Tests that interfaces with keyworded arguments are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
contains
  real function real_fun(w, x, y, z)
    implicit none
    real, intent(in) :: w
    real, intent(in), optional :: x
    real, intent(in) :: y
    real, intent(in), optional :: z
    if (.not.(present(x))) then
      real_fun = 1.0
    else
      real_fun = w + y
    end if
  end function real_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(3)
  d(1) = fun(1.0, 2.0, 3.0, 4.0)  ! all present, no keyword
  d(2) = fun(y=1.1, w=3.1)  ! only required ones, keyworded
  d(3) = fun(1.2, 2.2, y=3.2)  ! partially keyworded, last optional omitted.
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun(w, x, y, z)
    IMPLICIT NONE
    REAL, INTENT(IN) :: w
    REAL, INTENT(IN), OPTIONAL :: x
    REAL, INTENT(IN) :: y
    REAL, INTENT(IN), OPTIONAL :: z
    IF (.NOT. (PRESENT(x))) THEN
      real_fun = 1.0
    ELSE
      real_fun = w + y
    END IF
  END FUNCTION real_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: real_fun_deconiface_0 => real_fun, real_fun_deconiface_1 => real_fun, real_fun_deconiface_2 => real_fun
  IMPLICIT NONE
  REAL :: d(3)
  d(1) = real_fun_deconiface_0(1.0, 2.0, 3.0, 4.0)
  d(2) = real_fun_deconiface_1(y = 1.1, w = 3.1)
  d(3) = real_fun_deconiface_2(1.2, 2.2, y = 3.2)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_generic_replacer_deducing_array_types():
    """
    Tests that generic procedures with array types are correctly replaced.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type T
    real :: val(2, 2)
  contains
    procedure :: copy_matrix
    procedure :: copy_vector
    procedure :: copy_scalar
    generic :: copy => copy_matrix, copy_vector, copy_scalar
  end type T
contains
  subroutine copy_scalar(this, m)
    implicit none
    class(T), intent(in) :: this
    real, intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_scalar
  subroutine copy_vector(this, m)
    implicit none
    class(T), intent(in) :: this
    real, dimension(:), intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_vector
  subroutine copy_matrix(this, m)
    implicit none
    class(T), intent(in) :: this
    real, dimension(:, :), intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_matrix
end module lib

subroutine main
  use lib, only: T
  implicit none
  type(T) :: s, s1
  real, dimension(4, 4) :: a
  real :: b(4, 4)

  s%val = 1.0
  call s%copy(a)
  call s%copy(a(2, 2))
  call s%copy(b(:, 2))
  call s%copy(b(:, :))
  call s%copy(s1%val(:, 1))
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: T
    REAL :: val(2, 2)
  END TYPE T
  CONTAINS
  SUBROUTINE copy_scalar(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_scalar
  SUBROUTINE copy_vector(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, DIMENSION(:), INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_vector
  SUBROUTINE copy_matrix(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, DIMENSION(:, :), INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_matrix
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: copy_vector_deconproc_4 => copy_vector
  USE lib, ONLY: copy_matrix_deconproc_3 => copy_matrix
  USE lib, ONLY: copy_vector_deconproc_2 => copy_vector
  USE lib, ONLY: copy_scalar_deconproc_1 => copy_scalar
  USE lib, ONLY: copy_matrix_deconproc_0 => copy_matrix
  USE lib, ONLY: T
  IMPLICIT NONE
  TYPE(T) :: s, s1
  REAL, DIMENSION(4, 4) :: a
  REAL :: b(4, 4)
  s % val = 1.0
  CALL copy_matrix_deconproc_0(s, a)
  CALL copy_scalar_deconproc_1(s, a(2, 2))
  CALL copy_vector_deconproc_2(s, b(:, 2))
  CALL copy_matrix_deconproc_3(s, b(:, :))
  CALL copy_vector_deconproc_4(s, s1 % val(:, 1))
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_convert_data_statements_into_assignments():
    """
    Tests that DATA statements are converted to assignments.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
subroutine fun(res)
  implicit none
  real :: val = 0.0
  real, dimension(2) :: d
  real, dimension(2), intent(out) :: res
  data val/1.0/, d/2*4.2/
  data d(1:2)/2*4.2/
  data d/5.1, 5.2/
  res(:) = val*d(:)
end subroutine fun

subroutine main(res)
  implicit none
  real, dimension(2) :: res
  call fun(res)
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.convert_data_statements_into_assignments(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE fun(res)
  IMPLICIT NONE
  REAL :: val = 0.0
  REAL, DIMENSION(2) :: d
  REAL, DIMENSION(2), INTENT(OUT) :: res
  val = 1.0
  d(:) = 4.2
  d(1 : 2) = 4.2
  d(1) = 5.1
  d(2) = 5.2
  res(:) = val * d(:)
END SUBROUTINE fun
SUBROUTINE main(res)
  IMPLICIT NONE
  REAL, DIMENSION(2) :: res
  CALL fun(res)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_deconstruct_statement_functions():
    """
    Tests that statement functions are deconstructed into proper functions.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(3, 4, 5)
  double precision :: ptare, rtt(2), foedelta, foeldcp
  double precision :: ralvdcp(2), ralsdcp(2), res
  foedelta(ptare) = max(0.0, sign(1.d0, ptare - rtt(1)))
  foeldcp(ptare) = foedelta(ptare)*ralvdcp(1) + (1.0 - foedelta(ptare))*ralsdcp(1)
  rtt(1) = 4.5
  ralvdcp(1) = 4.9
  ralsdcp(1) = 5.1
  d(1, 1, 1) = foeldcp(3.d0)
  res = foeldcp(3.d0)
  d(1, 1, 2) = res
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_statement_functions(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  DOUBLE PRECISION :: d(3, 4, 5)
  DOUBLE PRECISION :: ptare, rtt(2)
  DOUBLE PRECISION :: ralvdcp(2), ralsdcp(2), res
  rtt(1) = 4.5
  ralvdcp(1) = 4.9
  ralsdcp(1) = 5.1
  d(1, 1, 1) = foeldcp(3.D0, rtt, ralvdcp, ralsdcp)
  res = foeldcp(3.D0, rtt, ralvdcp, ralsdcp)
  d(1, 1, 2) = res
  CONTAINS
  DOUBLE PRECISION FUNCTION foedelta(ptare, rtt)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: ptare
    DOUBLE PRECISION, INTENT(IN) :: rtt(2)
    foedelta = MAX(0.0, SIGN(1.D0, ptare - rtt(1)))
  END FUNCTION foedelta
  DOUBLE PRECISION FUNCTION foeldcp(ptare, rtt, ralvdcp, ralsdcp)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: ptare
    DOUBLE PRECISION, INTENT(IN) :: rtt(2)
    DOUBLE PRECISION, INTENT(IN) :: ralvdcp(2)
    DOUBLE PRECISION, INTENT(IN) :: ralsdcp(2)
    foeldcp = foedelta(ptare, rtt) * ralvdcp(1) + (1.0 - foedelta(ptare, rtt)) * ralsdcp(1)
  END FUNCTION foeldcp
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_goto_statements():
    """
    Tests that GOTO statements are correctly deconstructed into structured
    control flow, in this case by using boolean flags and IF statements.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  real, intent(inout) :: d
  integer :: i

  ! forward-only gotos
  i = 0
  if (i > 5) go to 10000  ! not taken
  i = 7
  if (i > 5) goto 10001  ! taken
  i = 1
  if (i > 5) then
    goto 10002
    i = 9
  else if (i > 6) then
    i = 10
  else
    i = 11
  end if
10001 i = 6
10000 continue
  i = 2
10002 continue
  d = 7.1*i
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    # Upstream (f2dace-windmill-qe-additions) renamed the typo'd
    # ``deconstuct_goto_statements`` to ``deconstruct_goto_statements`` and
    # reworked its output (label-numbered flags, nested guards, flag resets at
    # each label); the golden string below matches that current behaviour.
    ast = desugaring.deconstruct_goto_statements(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  REAL, INTENT(INOUT) :: d
  INTEGER :: i
  LOGICAL :: goto_10000
  LOGICAL :: goto_10001
  LOGICAL :: goto_10002
  goto_10002 = .FALSE.
  goto_10001 = .FALSE.
  goto_10000 = .FALSE.
  i = 0
  IF (i > 5) goto_10000 = .TRUE.
  IF (.NOT. goto_10000) i = 7
  IF (.NOT. goto_10000 .AND. i > 5) goto_10001 = .TRUE.
  IF (.NOT. goto_10001 .AND. .NOT. goto_10000) i = 1
  IF (.NOT. goto_10001) THEN
    IF (.NOT. goto_10000) THEN
      IF (i > 5) THEN
        goto_10002 = .TRUE.
        IF (.NOT. goto_10002) i = 9
      ELSE IF (.NOT. goto_10002 .AND. i > 6) THEN
        IF (.NOT. goto_10002) i = 10
      ELSE IF (.NOT. goto_10002) THEN
        IF (.NOT. goto_10002) i = 11
      END IF
    END IF
  END IF
10001 CONTINUE
  IF (.NOT. goto_10002) goto_10001 = .FALSE.
  IF (.NOT. goto_10002 .AND. .NOT. goto_10000) i = 6
10000 CONTINUE
  IF (.NOT. goto_10002) goto_10000 = .FALSE.
  IF (.NOT. goto_10002) i = 2
10002 CONTINUE
  goto_10002 = .FALSE.
  d = 7.1 * i
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_operator_overloading():
    """
    Tests that operator overloading is handled correctly.
    """
    sources, _ = (SourceCodeBuilder().add_file(
        """
module lib
  type cmplx
    real :: r = 1., i = 2.
  end type cmplx
  interface operator(+)
    module procedure :: add_cmplx
  end interface
contains
  function add_cmplx(a, b) result(c)
    type(cmplx), intent(in) :: a, b
    type(cmplx) :: c
    c%r = a%r + b%r
    c%i = a%i + b%i
  end function add_cmplx
end module lib

subroutine main
  use lib, only : cmplx, operator(+)
  type(cmplx) :: a, b
  b = a + a
end subroutine main
""",
        "main",
    ).check_with_gfortran().get())
    ast = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  TYPE :: cmplx
    REAL :: r = 1., i = 2.
  END TYPE cmplx
  INTERFACE OPERATOR(+)
    MODULE PROCEDURE :: add_cmplx
  END INTERFACE
  CONTAINS
  FUNCTION add_cmplx(a, b) RESULT(c)
    TYPE(cmplx), INTENT(IN) :: a, b
    TYPE(cmplx) :: c
    c % r = a % r + b % r
    c % i = a % i + b % i
  END FUNCTION add_cmplx
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: cmplx, OPERATOR(+)
  TYPE(cmplx) :: a, b
  b = a + a
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


if __name__ == "__main__":
    test_procedure_replacer()
    test_procedure_replacer_nested()
    test_procedure_replacer_name_collision_with_exisiting_var()
    test_procedure_replacer_name_collision_with_another_import()
    test_generic_replacer()
    test_association_replacer()
    test_association_replacer_array_access()
    test_association_replacer_array_access_within_array_access()
    test_enum_bindings_become_constants()
    test_aliasing_through_module_procedure()
    test_interface_replacer_with_module_procedures()
    test_interface_replacer_with_subroutine_decls()
    test_interface_replacer_with_optional_args()
    test_interface_replacer_with_keyworded_args()
    test_generic_replacer_deducing_array_types()
    test_convert_data_statements_into_assignments()
    test_deconstruct_statement_functions()
    test_goto_statements()
    test_operator_overloading()


def test_generic_resolution_matches_exact_real_kind():
    """Resolving a generic interface to a specific must match the REAL KIND
    exactly: an sp (REAL(4)) actual argument binds the sp specific, not the dp
    (REAL(8)) one -- even when the dp specific is listed first.

    Regression: the signature matcher used width subsumption (dummy width >=
    actual width), so a REAL(8) dummy also matched a REAL(4) actual; with
    first-match resolution an sp argument wrongly bound the dp specific (ICON's
    mixed-precision halo `p_isend(send_buf_sp, ...)` -> p_isend_dp type mismatch)."""
    sources, _ = (SourceCodeBuilder().add_file("""
module m
  implicit none
  integer, parameter :: sp = 4, dp = 8
  interface p_isend
    module procedure p_isend_dp
    module procedure p_isend_sp
  end interface
contains
  subroutine p_isend_dp(buf, n)
    real(dp), intent(in) :: buf(:)
    integer, intent(in) :: n
  end subroutine
  subroutine p_isend_sp(buf, n)
    real(sp), intent(in) :: buf(:)
    integer, intent(in) :: n
  end subroutine
  subroutine kern(n)
    integer, intent(in) :: n
    real(sp) :: sbuf(10)
    real(dp) :: dbuf(10)
    call p_isend(sbuf, n)
    call p_isend(dbuf, n)
  end subroutine
end module
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)
    got = ast.tofortran().lower()
    # the sp buffer call resolves to the sp specific (exact kind), the dp to dp
    assert "call p_isend_sp(sbuf" in got, got
    assert "call p_isend_dp(dbuf" in got, got


def test_interface_replacer_reverts_unresolvable_generic():
    """A call to a generic interface that CANNOT be resolved to a specific -- an
    external generic with no in-project candidate (the injected ``iso_c_binding``
    stub's empty ``INTERFACE c_loc``), or one whose signature matcher fails -- must
    have its temporary ``<name>_deconiface_tmp`` rename REVERTED, leaving the
    original generic name (valid Fortran resolved by the compiler), not a dangling
    ``_deconiface_tmp`` symbol."""
    sources, _ = (SourceCodeBuilder().add_file("""
module ext
  implicit none
  interface op  ! external generic: no candidate to resolve to
  end interface op
end module ext

module use_ext
  use ext, only: op
  implicit none
contains
  subroutine kern(x)
    real, intent(inout) :: x
    x = op(x)
  end subroutine
end module use_ext
""").get())
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)
    got = ast.tofortran().lower()
    assert "deconiface_tmp" not in got, got
    assert "x = op(x)" in got, got
    assert "only: op" in got, got
