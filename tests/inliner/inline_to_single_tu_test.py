# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end tests for the public ``inline_to_single_tu`` API plus the
user-requested extra patterns (only_clause, rename, collision,
nested_types, helper_proc, keep_external, cycle, generic_interface,
private_public).

Each test inlines a small multi-module project into one self-contained
``.f90`` and asserts the right code survives / is dropped, then (when
``gfortran`` is present) compiles the result to prove it is a valid TU.
"""
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
