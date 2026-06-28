# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the dace-fortran additions to the fparser inliner:

  * ``restore_cross_module_uses`` -- re-adds the inter-module ``USE``
    statements the pipeline strips, so the merged AST serialises to valid,
    single-file-compilable Fortran (the NPB-LU multi-file fix);
  * ``strip_builtin_stub_modules`` -- drops the injected intrinsic-module
    stubs from a whole-project merge so they do not collide with the
    compiler's own;
  * the ``Proc_Component_Def_Stmt`` component-name fix and the
    ``ASSOCIATE`` fixed-dimension relative-indexing fix (exercised directly
    here, in addition to the desugaring/analysis pass tests).

Each test inlines a small multi-module project with ``inline_to_ast`` (the
merge configuration: no entry point, ``optimize=False``) and checks the
restored ``USE`` plus -- when ``gfortran`` is available -- that the single
TU actually compiles.
"""
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dace_fortran.fparser_inliner import (inline_to_ast, restore_cross_module_uses, strip_builtin_stub_modules)


def _have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


def _compiles(src_text: str) -> bool:
    """True if ``src_text`` compiles standalone with gfortran (skips if absent)."""
    if not _have_gfortran():
        pytest.skip("gfortran not on PATH")
    with TemporaryDirectory() as td:
        f = Path(td) / "single.f90"
        f.write_text(src_text)
        r = subprocess.run(["gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-c",
                            str(f)],
                           cwd=td,
                           capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode())
        return r.returncode == 0


def _merge(sources: dict, **kw) -> str:
    """Inline like the build-path merge does: whole project (no entry),
    structural passes only (``optimize=False``), intrinsic stubs available
    but stripped from the output."""
    ast = inline_to_ast(sources, None, include_builtins=True, optimize=False, **kw)
    ast = strip_builtin_stub_modules(ast)
    return ast.tofortran()


# A module-level ``use`` of ``mod`` importing exactly ``name``.
def _has_use_only(text: str, mod: str, name: str) -> bool:
    return re.search(rf"(?im)^\s*USE\s+{mod}\s*,\s*ONLY\s*:.*\b{name}\b", text) is not None


# --------------------------------------------------------------------------
# restore_cross_module_uses
# --------------------------------------------------------------------------


def test_restore_use_for_cross_module_subroutine_call():
    """A subroutine that calls another module's subroutine gets
    ``USE <mod>, ONLY: <proc>`` restored, and the single TU compiles. This is
    the NPB-LU pattern (driver module -> compute module)."""
    sources = {
        "lu.f90":
        """
module lu
  implicit none
  private
  public :: dolu
contains
  subroutine dolu()
    call ssor()
  end subroutine dolu
  subroutine ssor()
  end subroutine ssor
end module lu
""",
        "useapplu.f90":
        """
module useapplu
  use lu, only: dolu
  implicit none
contains
  subroutine call_dolu()
    call dolu()
  end subroutine call_dolu
end module useapplu
""",
    }
    out = _merge(sources)
    assert _has_use_only(out, "lu", "dolu"), out
    assert _compiles(out)


def test_restore_use_for_cross_module_function_call():
    """A function reference across modules also gets its ``USE`` restored."""
    sources = {
        "lib.f90":
        """
module lib
  implicit none
contains
  real function dbl(x)
    real, intent(in) :: x
    dbl = 2.0 * x
  end function dbl
end module lib
""",
        "drv.f90":
        """
module drv
  use lib, only: dbl
  implicit none
contains
  subroutine run(a)
    real, intent(inout) :: a
    a = dbl(a)
  end subroutine run
end module drv
""",
    }
    out = _merge(sources)
    assert _has_use_only(out, "lib", "dbl"), out
    assert _compiles(out)


def test_restore_skips_local_name_shadowing_module_proc():
    """A scope whose local array shares a name with some module procedure must
    NOT get a spurious ``USE`` for it (the local would clash with the import)."""
    sources = {
        "lib.f90":
        """
module lib
  implicit none
contains
  real function flux(x)
    real, intent(in) :: x
    flux = x
  end function flux
end module lib
""",
        "drv.f90":
        """
module drv
  implicit none
contains
  subroutine run(a)
    real, intent(inout) :: a
    real :: flux(3)        ! local array, NOT lib::flux
    flux = a
    a = flux(1)
  end subroutine run
end module drv
""",
    }
    out = _merge(sources)
    # ``flux`` is a local array here -- importing lib::flux would be wrong.
    assert not _has_use_only(out, "lib", "flux"), out
    assert _compiles(out)


def test_restore_skips_sibling_same_module():
    """A call to a sibling procedure in the same module needs no ``USE``."""
    sources = {
        "lib.f90":
        """
module lib
  implicit none
contains
  subroutine top()
    call helper()
  end subroutine top
  subroutine helper()
  end subroutine helper
end module lib
""",
    }
    out = _merge(sources)
    # ``helper`` is a sibling in the same module -> host-associated, no USE.
    assert not _has_use_only(out, "lib", "helper"), out
    assert _compiles(out)


def test_restore_no_double_import_when_whole_module_used():
    """When a scope already imports a module *whole* (``USE x`` with no
    ``ONLY:``), the pass must not add a redundant ``USE x, ONLY: proc``.

    Exercised by calling ``restore_cross_module_uses`` directly on a parsed
    AST that still carries the whole ``USE`` (the full pipeline strips every
    inter-module ``USE``, so this guard is what protects an AST that does
    retain one)."""
    from fparser.two.parser import ParserFactory
    from fparser.common.readfortran import FortranStringReader
    from dace_fortran.inliner.ast_desugaring import cleanup

    src = """
module lib
  implicit none
contains
  subroutine work()
  end subroutine work
end module lib
module drv
  use lib
  implicit none
contains
  subroutine run()
    call work()
  end subroutine run
end module drv
"""
    ast = cleanup.lower_identifier_names(ParserFactory().create(std="f2008")(FortranStringReader(src)))
    out = restore_cross_module_uses(ast).tofortran()
    assert not _has_use_only(out, "lib", "work"), out
    # The whole ``USE lib`` is still there and resolves ``work``.
    assert re.search(r"(?im)^\s*USE\s+lib\s*$", out), out


def test_restore_skips_ambiguous_multi_module_proc():
    """A procedure name defined in more than one module is ambiguous and must
    never be auto-imported by name (the call site must already disambiguate)."""
    sources = {
        "m1.f90":
        """
module m1
  implicit none
contains
  subroutine foo()
  end subroutine foo
end module m1
""",
        "m2.f90":
        """
module m2
  implicit none
contains
  subroutine foo()
  end subroutine foo
end module m2
""",
        "drv.f90":
        """
module drv
  use m1, only: foo
  implicit none
contains
  subroutine run()
    call foo()
  end subroutine run
end module drv
""",
    }
    out = _merge(sources)
    # ``foo`` is defined in both m1 and m2: the restore pass must not invent an
    # ``ONLY: foo`` for either (drv's own kept import is what disambiguates).
    assert not _has_use_only(out, "m2", "foo"), out
    assert _compiles(out)


def test_restore_idempotent():
    """Running the restore pass twice adds nothing the second time."""
    sources = {
        "lu.f90":
        """
module lu
  implicit none
contains
  subroutine dolu()
  end subroutine dolu
end module lu
""",
        "useapplu.f90":
        """
module useapplu
  use lu, only: dolu
  implicit none
contains
  subroutine call_dolu()
    call dolu()
  end subroutine call_dolu
end module useapplu
""",
    }
    ast = inline_to_ast(sources, None, include_builtins=True, optimize=False)
    once = restore_cross_module_uses(ast).tofortran()
    twice = restore_cross_module_uses(ast).tofortran()
    assert once == twice


# --------------------------------------------------------------------------
# Numeric kind suffix on literals (consolidate_uses surgery)
# --------------------------------------------------------------------------


def test_numeric_kind_literal_survives_merge():
    """A real/int literal with a *numeric* kind suffix (``0.0_8``, ``1_4`` --
    ubiquitous in real Fortran) must not crash the merge.  ``consolidate_uses``
    rewrites a *named* kind (``0.0_wp``) into a ``Name`` so it resolves, but a
    numeric kind is not a valid identifier and must be left as the plain
    literal."""
    sources = {
        "k.f90":
        """
subroutine k(v, out)
  implicit none
  real(8), intent(in) :: v(4)
  real(8), intent(out) :: out(4)
  integer :: i
  out = 0.0_8
  do i = 1, 4
    out(i) = v(i) * 2.0_8 + 1_4
  end do
end subroutine k
""",
    }
    out = _merge(sources)
    assert "0.0_8" in out and "2.0_8" in out, out
    assert _compiles(out)


def test_named_kind_literal_still_resolves():
    """A *named* kind suffix (``0.0_wp`` with ``wp`` a module parameter) is
    still handled -- the numeric-kind guard must not regress the named case."""
    sources = {
        "kinds.f90":
        """
module kinds
  implicit none
  integer, parameter :: wp = selected_real_kind(15, 307)
end module kinds
""",
        "k.f90":
        """
subroutine k(a)
  use kinds, only: wp
  implicit none
  real(wp), intent(inout) :: a
  a = a + 1.5_wp
end subroutine k
""",
    }
    out = _merge(sources)
    assert _compiles(out)


# --------------------------------------------------------------------------
# restore_intrinsic_uses -- intrinsic-module USEs survive the merge
# --------------------------------------------------------------------------


def test_intrinsic_iso_c_binding_use_preserved():
    """A ``USE, INTRINSIC :: iso_c_binding`` for C-interop kinds/types
    (``c_double_complex``, ``c_ptr``) must survive the merge so the single TU
    still compiles -- the pipeline otherwise strips it (the stub it parses
    against lacks ``c_double_complex``), leaving those names undeclared."""
    sources = {
        "kmod.f90":
        """
module kmod
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine k(x)
    complex(c_double_complex), intent(inout) :: x(4)
    x = x * 2.0_c_double
  end subroutine k
end module kmod
""",
    }
    out = _merge(sources)
    # The intrinsic USE is restored; the stub module itself is NOT emitted.
    assert "iso_c_binding" in out.lower()
    assert "MODULE ISO_C_BINDING" not in out.upper()
    assert _compiles(out)


def test_intrinsic_use_in_one_of_several_modules():
    """The intrinsic ``USE`` is restored to the right module when several are
    merged (qualified-name match), and a normal cross-module call still works."""
    sources = {
        "cmod.f90":
        """
module cmod
  use iso_c_binding, only: c_double
  implicit none
contains
  real(c_double) function scale2(x)
    real(c_double), intent(in) :: x
    scale2 = 2.0_c_double * x
  end function scale2
end module cmod
""",
        "drv.f90":
        """
module drv
  use cmod, only: scale2
  use iso_c_binding, only: c_double
  implicit none
contains
  subroutine run(a)
    real(c_double), intent(inout) :: a
    a = scale2(a)
  end subroutine run
end module drv
""",
    }
    out = _merge(sources)
    assert _has_use_only(out, "cmod", "scale2"), out  # cross-module call restored
    assert "iso_c_binding" in out.lower()  # intrinsic kind import restored
    assert _compiles(out)


# --------------------------------------------------------------------------
# restore_external_interfaces -- external-procedure INTERFACE blocks survive
# --------------------------------------------------------------------------


def test_external_interface_block_preserved():
    """An ``INTERFACE`` block declaring an *external* (C-library) procedure must
    survive the merge -- the pipeline drops it ("no candidate to resolve to"),
    leaving the call undeclared.  The single TU must compile to an object (the
    external symbol is a link-time concern the compile step never reaches)."""
    sources = {
        "kmod.f90":
        """
module kmod
  use, intrinsic :: iso_c_binding
  implicit none
  interface
     type(c_ptr) function ext_plan(n) bind(c, name='ext_plan')
       import :: c_int, c_ptr
       integer(c_int), value :: n
     end function ext_plan
  end interface
contains
  subroutine run(n, p)
    integer(c_int), intent(in) :: n
    type(c_ptr), intent(out) :: p
    p = ext_plan(n)
  end subroutine run
end module kmod
""",
    }
    out = _merge(sources)
    assert "INTERFACE" in out.upper(), out
    assert "ext_plan" in out
    assert _compiles(out)  # gfortran -c (compile to object, no link)


def test_interface_for_defined_procedure_not_duplicated():
    """An interface whose procedure IS defined in the project must not be
    restored as a duplicate declaration (the pipeline resolves it)."""
    sources = {
        "lib.f90":
        """
module lib
  implicit none
contains
  real function dbl(x)
    real, intent(in) :: x
    dbl = 2.0 * x
  end function dbl
end module lib
""",
        "drv.f90":
        """
module drv
  use lib, only: dbl
  implicit none
contains
  subroutine run(a)
    real, intent(inout) :: a
    a = dbl(a)
  end subroutine run
end module drv
""",
    }
    out = _merge(sources)
    # ``dbl`` is defined in lib -> no spurious external interface for it.
    assert "INTERFACE" not in out.upper(), out
    assert _compiles(out)


def test_external_interface_not_duplicated_after_pipeline_mangles_inplace():
    """The full single-TU path (``optimize=True``) processes a call to an
    external interface (``deconstruct_interface_calls`` renames it, const-eval
    folds the interface body's kinds, ``remove_access_and_bind_statements``
    strips its ``BIND(C)``) so the in-place copy no longer matches the verbatim
    captured original.  ``restore_external_interfaces`` must drop that mangled
    copy and re-add exactly one declaration -- and a no-argument ``BIND(C)``
    subroutine must keep its (fparser-dropped) ``()``.  This is the ICON
    ``mo_mpi`` -> ``mo_util_system`` (``util_exit`` / ``util_abort``) pattern."""
    from dace_fortran.fparser_inliner import inline_to_single_tu
    sources = {
        "s.f90":
        """
module mo_util_system
  use iso_c_binding, only: c_int
  implicit none
  interface
    subroutine util_exit(exit_no) bind(C)
      import c_int
      implicit none
      integer(c_int), value :: exit_no
    end subroutine util_exit
    subroutine util_abort() bind(C)
      implicit none
    end subroutine util_abort
  end interface
end module mo_util_system

module mo_mpi
  use mo_util_system, only: util_exit, util_abort
  implicit none
contains
  subroutine abort_mpi  ! no Specification_Part: only host-associated externals
    call util_abort()
    call util_exit(1)
  end subroutine abort_mpi
end module mo_mpi

module m_driver
contains
  subroutine run()
    use mo_mpi, only: abort_mpi
    call abort_mpi()
  end subroutine run
end module m_driver
""",
    }
    with TemporaryDirectory() as td:
        tu = inline_to_single_tu(sources,
                                 entry="m_driver::run",
                                 out_dir=Path(td),
                                 name="tu",
                                 tolerate_external_uses=True)
        out = Path(tu).read_text()
    # Exactly one declaration of the external ``util_exit`` survives.
    assert len(re.findall(r"(?im)^\s*SUBROUTINE\s+util_exit\b", out)) == 1, out
    # The no-arg ``util_abort`` keeps the parentheses ``BIND(C)`` requires.
    assert re.search(r"(?im)^\s*SUBROUTINE\s+util_abort\s*\(\s*\)\s+BIND", out), out
    assert _compiles(out)


# --------------------------------------------------------------------------
# strip_builtin_stub_modules
# --------------------------------------------------------------------------


def test_strip_builtin_stub_modules_removes_iso_c_binding():
    """A kernel that ``USE``s an intrinsic module parses via the injected stub,
    but the stub module is stripped from the output (the compiler ships its
    own ``iso_c_binding`` / ``iso_fortran_env``)."""
    sources = {
        "k.f90":
        """
subroutine k(a) bind(c, name="k")
  use iso_c_binding, only: c_double
  implicit none
  real(c_double), intent(inout) :: a
  a = a * 2.0_c_double
end subroutine k
""",
    }
    out = _merge(sources)
    assert "MODULE ISO_C_BINDING" not in out.upper(), out
    assert "MODULE ISO_FORTRAN_ENV" not in out.upper(), out
    assert _compiles(out)


# --------------------------------------------------------------------------
# Proc_Component_Def_Stmt component-name fix (find_name_of_stmt)
# --------------------------------------------------------------------------


def test_proc_pointer_component_uses_component_name_not_interface():
    """``procedure(fun), pointer :: nofun`` is the component ``nofun`` (not the
    interface ``fun``): the two same-interface components map to distinct
    specs."""
    from fparser.two.parser import ParserFactory
    from fparser.common.readfortran import FortranStringReader
    from dace_fortran.inliner.ast_desugaring import analysis, cleanup

    src = """
module lib
  type T
    procedure(fun), nopass, pointer :: fun
    procedure(fun), nopass, pointer :: nofun
  end type T
contains
  real function fun()
    fun = 1.1
  end function fun
end module lib
"""
    ast = cleanup.lower_identifier_names(ParserFactory().create(std="f2008")(FortranStringReader(src)))
    specs = set(analysis.identifier_specs(ast).keys())
    assert ("lib", "t", "fun") in specs
    assert ("lib", "t", "nofun") in specs


# --------------------------------------------------------------------------
# ASSOCIATE relative indexing with a fixed selector dimension
# --------------------------------------------------------------------------


def test_associate_section_with_fixed_dimension():
    """``associate(col => a(:, 1))`` then ``col(i) = ...`` lowers to
    ``a(i, 1) = ...`` -- the use indexes only the sectioned dimension, the
    fixed ``1`` is preserved."""
    from fparser.two.parser import ParserFactory
    from fparser.common.readfortran import FortranStringReader
    from dace_fortran.inliner.ast_desugaring import desugaring, cleanup

    src = """
subroutine main(a, i, v)
  implicit none
  real, intent(inout) :: a(4, 4)
  integer, intent(in) :: i
  real, intent(in) :: v
  associate(col => a(:, 1))
    col(2) = v
    col(i) = v
  end associate
end subroutine main
"""
    ast = cleanup.lower_identifier_names(ParserFactory().create(std="f2008")(FortranStringReader(src)))
    ast = desugaring.deconstruct_associations(ast)
    got = ast.tofortran()
    assert "a(2, 1) = v" in got, got
    assert "a(i, 1) = v" in got, got
    assert _compiles(got)
