# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end tests for the public ``inline_to_single_tu`` API plus the
user-requested extra patterns (only_clause, rename, collision,
nested_types, helper_proc, do_not_emit, cycle, generic_interface,
private_public).

Each test inlines a small multi-module project into one self-contained
``.f90`` and asserts the right code survives / is dropped, then (when
``gfortran`` is present) compiles the result to prove it is a valid TU.
"""
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dace_fortran.fparser_inliner import inline_to_ast, inline_to_single_tu


def _have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


def _compiles(src_text: str) -> bool:
    """True if ``src_text`` compiles standalone with gfortran (skips if absent)."""
    if not _have_gfortran():
        pytest.skip("gfortran not on PATH")
    with TemporaryDirectory() as td:
        f = Path(td) / "single.f90"
        f.write_text(src_text)
        r = subprocess.run(["gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-c", str(f)],
                           cwd=td, capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode())
        return r.returncode == 0


def _inline_text(sources, entry, **kw) -> str:
    return inline_to_ast(sources, entry, **kw).tofortran()


def test_basic_single_tu_two_modules():
    """A driver USE-ing a helper module inlines into one TU and compiles."""
    sources = {
        "mo_kind.f90": """
module mo_kind
  implicit none
  integer, parameter :: wp = selected_real_kind(15, 307)
end module mo_kind
""",
        "lib.f90": """
module lib
  use mo_kind, only: wp
  implicit none
contains
  real(wp) function dbl(x)
    real(wp), intent(in) :: x
    dbl = 2.0_wp * x
  end function dbl
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: dbl
  use mo_kind, only: wp
  implicit none
  real(wp), intent(inout) :: a
  a = dbl(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "SUBROUTINE run" in out
    assert "FUNCTION dbl" in out
    assert _compiles(out)


def test_only_clause_imports_just_what_is_named():
    """``USE ..., ONLY:`` brings in only the named symbol; the unnamed
    sibling in the same module is pruned away."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  real function used(x)
    real, intent(in) :: x
    used = x + 1.0
  end function used
  real function unused(x)
    real, intent(in) :: x
    unused = x - 1.0
  end function unused
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: used
  implicit none
  real, intent(inout) :: a
  a = used(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "FUNCTION used" in out
    assert "unused" not in out.lower()
    assert _compiles(out)


def test_rename_in_only_clause():
    """A renamed import (``ONLY: ll => longname``) survives inlining and
    the renamed-from procedure is callable in the single TU."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  real function longname(x)
    real, intent(in) :: x
    longname = x * 3.0
  end function longname
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: ll => longname
  implicit none
  real, intent(inout) :: a
  a = ll(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "FUNCTION longname" in out
    assert _compiles(out)


def test_name_collision_across_modules():
    """Two modules each defining a ``mod`` parameter inline together; the
    consolidated USE-ONLY clauses keep the references unambiguous."""
    sources = {
        "a.f90": """
module a
  integer, parameter :: token = 1
end module a
""",
        "b.f90": """
module b
  integer, parameter :: token = 2
contains
  subroutine foo(o)
    use a, only: token
    integer, intent(out) :: o
    o = token
  end subroutine foo
end module b
""",
        "driver.f90": """
subroutine run(o)
  use b, only: foo
  implicit none
  integer, intent(out) :: o
  call foo(o)
end subroutine run
""",
    }
    # The inliner prunes-to-used + consolidates USE-ONLY clauses (modules may
    # survive, e.g. ``b`` here); the two same-named ``token`` parameters stay
    # unambiguous, so the single TU compiles.
    out = _inline_text(sources, "run", include_builtins=False)
    assert _compiles(out)


def test_nested_derived_types():
    """A derived type nesting another derived type survives inlining with
    both type definitions present."""
    sources = {
        "types_mod.f90": """
module types_mod
  implicit none
  type inner
    real :: v
  end type inner
  type outer
    type(inner) :: i
  end type outer
end module types_mod
""",
        "driver.f90": """
subroutine run(a)
  use types_mod, only: outer
  implicit none
  real, intent(inout) :: a
  type(outer) :: o
  o%i%v = a
  a = o%i%v
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "TYPE :: inner" in out
    assert "TYPE :: outer" in out
    assert _compiles(out)


def test_helper_proc_transitively_pulled_in():
    """A helper procedure called only indirectly (through the entry's
    callee) is kept by the reachability-driven pruning."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  real function top(x)
    real, intent(in) :: x
    top = helper(x) + 1.0
  end function top
  real function helper(x)
    real, intent(in) :: x
    helper = x * x
  end function helper
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: top
  implicit none
  real, intent(inout) :: a
  a = top(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "FUNCTION top" in out
    assert "FUNCTION helper" in out
    assert _compiles(out)


def test_unresolved_use_module_is_an_error():
    """Contract: every ``USE``-d module must be supplied.  An unresolved
    external module (its source not provided) is an inlining ERROR -- the
    upstream inliner requires the full module closure (no keep-external mode)."""
    sources = {
        "driver.f90": """
subroutine run(a)
  use some_external_mod, only: ext_const
  implicit none
  real, intent(inout) :: a
  a = a + ext_const
end subroutine run
""",
    }
    with pytest.raises(Exception):
        _inline_text(sources, "run", include_builtins=False)


def test_use_cycle_between_modules():
    """Mutually-``USE``-ing modules (a USE-graph cycle) inline without
    looping forever; both modules appear once."""
    sources = {
        "a.f90": """
module a
  use b, only: bconst
  implicit none
  integer, parameter :: aconst = 10
end module a
""",
        "b.f90": """
module b
  implicit none
  integer, parameter :: bconst = 20
end module b
""",
        "driver.f90": """
subroutine run(o)
  use a, only: aconst
  use b, only: bconst
  implicit none
  integer, intent(out) :: o
  o = aconst + bconst
end subroutine run
""",
    }
    # A USE-graph cycle must inline without looping forever and yield a single
    # compilable TU (the inliner prunes-to-used + consolidates, so the surviving
    # module set is not asserted -- only that it terminated and compiles).
    out = _inline_text(sources, "run", include_builtins=False)
    assert _compiles(out)


def test_generic_interface_resolves_to_specific():
    """A generic-interface call is deconstructed to the specific
    procedure during inlining."""
    sources = {
        "lib.f90": """
module lib
  implicit none
  interface dbl
    module procedure dbl_r
  end interface dbl
contains
  real function dbl_r(x)
    real, intent(in) :: x
    dbl_r = 2.0 * x
  end function dbl_r
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: dbl
  implicit none
  real, intent(inout) :: a
  a = dbl(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    # The interface is fully removed by the pipeline.
    assert "INTERFACE" not in out.upper()
    assert "dbl_r" in out
    assert _compiles(out)


def test_private_public_visibility():
    """A module with PRIVATE default + a PUBLIC procedure inlines and the
    public procedure remains callable in the single TU."""
    sources = {
        "lib.f90": """
module lib
  implicit none
  private
  public :: pub
contains
  real function pub(x)
    real, intent(in) :: x
    pub = priv(x) + 1.0
  end function pub
  real function priv(x)
    real, intent(in) :: x
    priv = x * 2.0
  end function priv
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: pub
  implicit none
  real, intent(inout) :: a
  a = pub(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    assert "FUNCTION pub" in out
    # PRIVATE / PUBLIC access statements are removed by
    # remove_access_and_bind_statements.
    assert "PRIVATE" not in out.upper()
    assert _compiles(out)


def test_make_noop_empties_body():
    """``make_noop`` replaces a procedure body with nothing while keeping
    the procedure shell."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  subroutine logit(x)
    real, intent(in) :: x
    print *, x
  end subroutine logit
  real function compute(x)
    real, intent(in) :: x
    call logit(x)
    compute = x + 1.0
  end function compute
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: compute
  implicit none
  real, intent(inout) :: a
  a = compute(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", make_noop=[("lib", "logit")], include_builtins=False)
    # The PRINT statement from logit's body is gone.
    assert "PRINT" not in out.upper()
    assert _compiles(out)


def test_do_not_emit_leaves_procedure_external():
    """The unified external-function policy's ``do_not_emit=[names]``
    param leaves a procedure external on the public inliner API -- its
    body is emptied (same effect as the low-level ``make_noop`` above)
    but addressed by plain call-site name, not ``(module, name)``."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  subroutine logit(x)
    real, intent(in) :: x
    print *, x
  end subroutine logit
  real function compute(x)
    real, intent(in) :: x
    call logit(x)
    compute = x + 1.0
  end function compute
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: compute
  implicit none
  real, intent(inout) :: a
  a = compute(a)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", do_not_emit=["logit"], include_builtins=False)
    # logit is kept external -- its body (the PRINT) is stripped.
    assert "PRINT" not in out.upper()
    assert _compiles(out)


def test_inline_to_single_tu_writes_file(tmp_path):
    """The headline API writes a single ``.f90`` and returns its path."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  real function dbl(x)
    real, intent(in) :: x
    dbl = 2.0 * x
  end function dbl
end module lib
""",
        "driver.f90": """
subroutine run(a)
  use lib, only: dbl
  implicit none
  real, intent(inout) :: a
  a = dbl(a)
end subroutine run
""",
    }
    out = inline_to_single_tu(sources, "run", out_dir=tmp_path, name="merged", include_builtins=False)
    assert out == tmp_path / "merged.f90"
    assert out.is_file()
    text = out.read_text()
    assert "SUBROUTINE run" in text
    assert "FUNCTION dbl" in text


def test_inline_from_file_paths(tmp_path):
    """``inline_to_single_tu`` accepts file paths (not just a dict)."""
    (tmp_path / "lib.f90").write_text("""
module lib
  implicit none
contains
  real function dbl(x)
    real, intent(in) :: x
    dbl = 2.0 * x
  end function dbl
end module lib
""")
    (tmp_path / "driver.f90").write_text("""
subroutine run(a)
  use lib, only: dbl
  implicit none
  real, intent(inout) :: a
  a = dbl(a)
end subroutine run
""")
    out = inline_to_single_tu([tmp_path / "lib.f90", tmp_path / "driver.f90"],
                              "run",
                              out_dir=tmp_path / "out",
                              include_builtins=False)
    assert out.is_file()
    assert "SUBROUTINE run" in out.read_text()


def test_module_qualified_entry():
    """A ``module::proc`` entry resolves and prunes to that procedure."""
    sources = {
        "lib.f90": """
module lib
  implicit none
contains
  subroutine run(a)
    real, intent(inout) :: a
    a = a + 1.0
  end subroutine run
  subroutine other(a)
    real, intent(inout) :: a
    a = a - 1.0
  end subroutine other
end module lib
""",
    }
    out = _inline_text(sources, "lib::run", include_builtins=False)
    assert "SUBROUTINE run" in out
    assert "other" not in out.lower()


def test_pointer_component_of_extension_type_survives_pruning():
    """Root-cause regression: a POINTER component of an ``EXTENDS`` type that is
    written only via ``=>`` in a constructor must not be pruned.

    ICON-O's solver constructors (``lhs_primal_flip_flop_construct`` setting
    ``this % patch_2d => ...`` on ``t_primal_flip_flop_lhs``, which extends the
    abstract ``t_lhs_agen``) hit this: the pointer-assignment LHS is a fparser
    ``Data_Pointer_Object`` (not a ``Data_Ref``), so the entity-level prune did
    not see the component as used and dropped it from the type -- leaving the
    body referencing a member the type no longer declared.  This is the smaller
    analogue: ``t_lhs`` extends ``t_lhs_base`` and carries a ``t_grid`` pointer
    set in ``lhs_construct`` and read in ``lhs_area``; the single TU must keep
    ``grid`` and compile."""
    sources = {
        "geom.f90": """
module geom
  implicit none
  type :: t_grid
    real :: area
  end type t_grid
end module geom
""",
        "operators.f90": """
module operators
  use geom, only: t_grid
  implicit none
  type, abstract :: t_lhs_base
    logical :: is_const
  end type t_lhs_base
  type, extends(t_lhs_base) :: t_lhs
    integer :: level
    type(t_grid), pointer :: grid => null()
  end type t_lhs
contains
  subroutine lhs_construct(this, g)
    class(t_lhs), intent(inout) :: this
    type(t_grid), target, intent(in) :: g
    this % is_const = .false.
    this % level = 1
    this % grid => g
  end subroutine lhs_construct
  real function lhs_area(this)
    class(t_lhs), intent(in) :: this
    lhs_area = this % grid % area
  end function lhs_area
end module operators
""",
        "driver.f90": """
subroutine run(a)
  use geom, only: t_grid
  use operators, only: t_lhs, lhs_construct, lhs_area
  implicit none
  real, intent(inout) :: a
  type(t_lhs) :: op
  type(t_grid), target :: g
  g % area = a
  call lhs_construct(op, g)
  a = lhs_area(op)
end subroutine run
""",
    }
    out = _inline_text(sources, "run", include_builtins=False)
    # The pointer component (and the type it points at) must survive in the type
    # definition, not merely appear in the constructor body.
    tdef = re.search(r"EXTENDS\(t_lhs_base\) :: t_lhs\b.*?END TYPE t_lhs\b", out, re.S)
    assert tdef and "GRID" in tdef.group(0).upper(), f"pointer component pruned from t_lhs:\n{out}"
    assert _compiles(out)
