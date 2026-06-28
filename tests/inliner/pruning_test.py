# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace_fortran.inliner.ast_desugaring import pruning, optimizations, desugaring
from inliner.fortran_test_helper import SourceCodeBuilder, parse_and_improve


def test_branch_pruning():
    """
    Tests that dead branches in `IF` constructs are pruned if the condition
    can be evaluated at compile time.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 4
  integer :: a = -1, b = -1, c = -1

  c = 5
  if (k < 2) then
    a = k
    c = 1  ! identical `c = 1` lines
    b = c + 1
  else if (k < 5) then
    b = k
    c = 1  ! identical `c = 1` lines, but this one must not be dropped.
    b = c + 1
  else
    a = k
    b = k
    c = 1  ! identical `c = 1` lines
    b = c + 1
  end if
  if (k < 5) a = 70 + k
  if (k > 5) a = 70 - k
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.prune_branches(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 4
  INTEGER :: a = - 1, b = - 1, c = - 1
  c = 5
  b = k
  c = 1
  b = c + 1
  a = 70 + k
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_object_pruning():
    """
    Tests that unused objects (variables, types, etc.) are pruned from the AST.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type used_config
    integer :: a = -1
    real :: b = -2.0
  end type used_config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
  type(used_config) :: garray(4)
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib

subroutine main
  use lib
  implicit none
  type(used_config) :: ucfg
  integer :: i = 7
  real :: a = 1
  ucfg%b = a*i
  garray(3)%b = a*i*2
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.prune_unused_objects(ast, [("main", )])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: used_config
    REAL :: b = - 2.0
  END TYPE used_config
  TYPE(used_config) :: garray(4)
  CONTAINS
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: garray, used_config
  IMPLICIT NONE
  TYPE(used_config) :: ucfg
  INTEGER :: i = 7
  REAL :: a = 1
  ucfg % b = a * i
  garray(3) % b = a * i * 2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_pointer_pruning():
    """
    Tests that unused pointers are pruned from the AST.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type T
    integer :: data(4) = 8
    integer, pointer :: ptr(:) => null()
  end type T
end module lib

subroutine main(out)
  use lib
  implicit none
  type(T), target :: cfg
  integer, pointer :: ptr(:) => null()
  integer, pointer :: unused_ptr(:) => null()
  integer, intent(out) :: out(4)
  cfg % ptr => cfg % data  ! TODO: This too should go away.
  ptr => cfg % ptr
  out = cfg % data
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.prune_unused_objects(ast, [("main", )])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: T
    INTEGER :: data(4) = 8
    INTEGER, POINTER :: ptr(:) => NULL()
  END TYPE T
END MODULE lib
SUBROUTINE main(out)
  USE lib, ONLY: T
  IMPLICIT NONE
  TYPE(T), TARGET :: cfg
  INTEGER, INTENT(OUT) :: out(4)
  cfg % ptr => cfg % data
  out = cfg % data
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_completely_unsed_modules_are_pruned_early():
    """
    Tests that completely unused modules are pruned from the AST.
    """
    sources, _ = (SourceCodeBuilder().add_file(
        """
module used
  implicit none
contains
  real function fun()
    fun = 1.
  end function fun
end module used

module unused
  implicit none
contains
  real function fun()
    fun = 2.
  end function fun
end module unused

subroutine main(d)
  use used
  implicit none
  real, intent(inout) :: d
  d = fun()
end subroutine main
""",
        "main",
    ).check_with_gfortran().get())
    ast = parse_and_improve(sources, [("main", )])

    got = ast.tofortran()
    want = """
MODULE used
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun()
    fun = 1.
  END FUNCTION fun
END MODULE used
SUBROUTINE main(d)
  USE used
  IMPLICIT NONE
  REAL, INTENT(INOUT) :: d
  d = fun()
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_uses_with_renames():
    """
    Tests that `USE` statements with renames are handled correctly, especially
    after constant evaluation, which might make the import redundant.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: pi4 = 9
  integer, parameter :: i4 = selected_int_kind(pi4)  ! `i4` will be const-evaluated to 4
end module lib

module main
contains
  subroutine fun(d)
    use lib, only: ik4 => i4  ! After const-evaluation, will be redundant.
    integer(ik4) :: i  ! `ik4` will also be const-evaluated to 4
    real, intent(out) :: d(2)
    i = 4
    d(2) = 5.5 + i
  end subroutine fun
end module main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    # A constant-evaluation will pin the constant values.
    ast = optimizations.const_eval_nodes(ast)
    # A use-consolidation will remove the now-redundant use.
    ast = pruning.consolidate_uses(ast)
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: pi4 = 9
  INTEGER, PARAMETER :: i4 = 4
END MODULE lib
MODULE main
  CONTAINS
  SUBROUTINE fun(d)
    INTEGER(KIND = 4) :: i
    REAL, INTENT(OUT) :: d(2)
    i = 4
    d(2) = 5.5 + i
  END SUBROUTINE fun
END MODULE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_use_consolidation_with_potential_ambiguity():
    """
    Tests that `USE` statement consolidation correctly handles cases where
    the same symbol is imported from different modules, avoiding ambiguity by
    using `ONLY` clauses.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module A
  integer, parameter :: mod = 1
end module A

module B
  integer, parameter :: mod = 2
  contains
  subroutine foo
    use A
    print *, mod
  end subroutine foo
end module B

subroutine main
  use B
  call foo
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.consolidate_uses(ast)

    got = ast.tofortran()
    want = """
MODULE A
  INTEGER, PARAMETER :: mod = 1
END MODULE A
MODULE B
  INTEGER, PARAMETER :: mod = 2
  CONTAINS
  SUBROUTINE foo
    USE A, ONLY: mod
    PRINT *, mod
  END SUBROUTINE foo
END MODULE B
SUBROUTINE main
  USE B, ONLY: foo
  CALL foo
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_use_consolidation_with_type_extension():
    """
    Tests that `USE` statement consolidation correctly handles type extensions,
    ensuring that the base type is imported where needed.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module A
  type, abstract :: AA
  end type AA
end module A

module B
  use A
  type, extends(AA) :: BB
  end type BB
end module B

subroutine main
  use B
  type(BB) :: c = BB()
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.consolidate_uses(ast)

    got = ast.tofortran()
    want = """
MODULE A
  TYPE, ABSTRACT :: AA
  END TYPE AA
END MODULE A
MODULE B
  USE A, ONLY: AA
  TYPE, EXTENDS(AA) :: BB
  END TYPE BB
END MODULE B
SUBROUTINE main
  USE B, ONLY: BB
  TYPE(BB) :: c = BB()
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_uses_allows_indirect_aliasing():
    """
    Tests that `USE` statements allow indirect aliasing of symbols through
    a chain of modules.
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

module lib2
  use lib
  implicit none
end module lib2

subroutine main
  use lib2, only: Square, perim
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
MODULE lib2
  USE lib
  IMPLICIT NONE
END MODULE lib2
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib2, ONLY: Square, perim
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


def test_use_consolidation_keeps_host_associated_import_for_specless_subprogram():
    """A module-level ``USE`` whose symbol is host-associated into a contained
    subprogram that has NO ``Specification_Part`` (a body of only host-associated
    references -- ICON's ``mo_mpi`` ``abort_mpi`` referencing ``mpi_comm_world``)
    must be retained at the module level: the subprogram has nowhere to host the
    import, so consolidating it away would dangle the symbol."""
    sources, _ = (SourceCodeBuilder().add_file("""
module consts
  implicit none
  integer, parameter :: kk = 7
end module consts

module lib
  use consts
  implicit none
  integer :: out
contains
  subroutine s  ! no Specification_Part: only a host-associated reference
    out = kk
  end subroutine s
end module lib

subroutine main
  use lib, only: s
  call s
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = pruning.consolidate_uses(ast)
    got = ast.tofortran()
    # `kk` stays resolvable: the module-level `use consts` survives (it is the
    # only place the host-associated `kk` can be imported), so the serialised
    # Fortran still compiles rather than referencing an undefined `kk`.
    assert "USE consts" in got
    assert "out = kk" in got
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_prune_keeps_specifics_of_referenced_generic_interface():
    """A call that references a generic INTERFACE which was NOT resolved to a
    specific (``deconstruct_interface_calls`` could not pick one -- a keyword-arg
    call to ICON's ``smooth_oncells``) must keep the interface's ``MODULE
    PROCEDURE`` specifics, so the generic stays bindable instead of emptied to a
    dangling interface.  Here the call is left generic (no deconstruct pass)."""
    sources, _ = (SourceCodeBuilder().add_file("""
module m
  implicit none
  interface gen
    module procedure spec_a
    module procedure spec_b
  end interface
contains
  subroutine spec_a(x)
    real, intent(inout) :: x(:)
    x = x + 1.0
  end subroutine
  subroutine spec_b(x)
    real, intent(inout) :: x(:, :)
    x = x + 1.0
  end subroutine
  subroutine caller(y)
    real, intent(inout) :: y(:)
    call gen(y)
  end subroutine
end module m
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    # No deconstruct_interface_calls: ``gen(y)`` stays a reference to the generic.
    ast = pruning.prune_unused_objects(ast, [("m", "caller")])
    got = ast.tofortran()
    # Both specifics survive (kept via the interface), so the generic compiles.
    assert "SUBROUTINE spec_a" in got, got
    assert "SUBROUTINE spec_b" in got, got
    SourceCodeBuilder().add_file(got).check_with_gfortran()


if __name__ == "__main__":
    test_branch_pruning()
    test_object_pruning()
    test_pointer_pruning()
    test_completely_unsed_modules_are_pruned_early()
    test_uses_with_renames()
    test_use_consolidation_with_potential_ambiguity()
    test_use_consolidation_with_type_extension()
    test_uses_allows_indirect_aliasing()
    test_use_consolidation_keeps_host_associated_import_for_specless_subprogram()
    test_prune_keeps_specifics_of_referenced_generic_interface()
