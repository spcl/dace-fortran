# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ported from upstream tests/fortran/desugaring/analysis_test.py."""
from dace_fortran.inliner.ast_desugaring import analysis
from inliner.fortran_test_helper import SourceCodeBuilder, parse_and_improve


def test_spec_mapping_of_abstract_interface():
    """Abstract-interface subroutine ``fun`` is captured in the identifier/alias maps
    under a synthetic ``__interface__`` path segment; the interface block itself is not."""
    sources, _ = (SourceCodeBuilder().add_file("""
module lib  ! should be present
  abstract interface  ! should NOT be present
    subroutine fun  ! should be present
    end subroutine fun
  end interface
end module lib
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {("lib", ), ("lib", "__interface__", "fun")}

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {("lib", ), ("lib", "__interface__", "fun")}


def test_spec_mapping_of_type_extension():
    """Type extension: base type's components appear in the extended type's alias
    map both via direct inherited access and parent-component access."""
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  type base
    integer :: a
  end type base
  type, extends(base) :: ext
    integer :: b
  end type ext
end module lib
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {
        ("lib", ),
        ("lib", "base"),
        ("lib", "base", "a"),
        ("lib", "ext"),
        ("lib", "ext", "b"),
    }

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {
        ("lib", ),
        ("lib", "base"),
        ("lib", "base", "a"),
        ("lib", "ext"),
        ("lib", "ext", "b"),
        # Direct inherited access -- Fortran flattens base components into ext, so instances reach a as x % a.
        ("lib", "ext", "a"),
        # Parent-component access -- ``x % base % a`` keeps the base type name.
        ("lib", "ext", "base"),
        ("lib", "ext", "base", "a"),
    }


def test_spec_mapping_of_procedure_pointers():
    """Spec mapping handles procedure pointers both as derived-type components and as standalone variables."""
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  type T
    procedure(fun), nopass, pointer :: fun
    procedure(fun), nopass, pointer :: nofun
  end type T
  procedure(fun), pointer :: real_fun => null()
contains
  real function fun()
    fun = 1.1
  end function fun
end module lib
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {
        ("lib", ),
        ("lib", "T"),
        ("lib", "T", "fun"),
        ("lib", "T", "nofun"),
        ("lib", "fun"),
        ("lib", "real_fun"),
    }

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {
        ("lib", ),
        ("lib", "T"),
        ("lib", "T", "fun"),
        ("lib", "T", "nofun"),
        ("lib", "fun"),
        ("lib", "real_fun"),
    }
