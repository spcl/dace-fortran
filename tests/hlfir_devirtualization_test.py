"""Characterisation tests pinning *why* the ICON-O solver construct/solve chain must stay
an external call, not inline into the kernel TU: an SDFG has no node for a runtime dispatch
through a vtable, so a polymorphic (``CLASS(..)``) TBP call needs flang to resolve it to a
*direct* call first -- which, against the installed flang, it never does: dispatch on a
``CLASS`` dummy or on a freshly-``ALLOCATE``d ``CLASS`` local both stay ``fir.dispatch``;
only a concrete ``TYPE(..)`` call binds directly (monomorphisation is our inliner's job, not
flang's); ``--fir-polymorphic-op`` lowers ``fir.dispatch`` but never devirtualises it.

If a future flang changes this, these ``fir.dispatch`` assertions fail -- revisit the
externalisation policy in ``tests/icon/ocean/_ocean_harness.py``."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _util import _FLANG, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Minimal abstract base + one concrete override, kept in its own TU so a direct fir.call to
# the override could only mean genuine devirtualisation (no concrete-TYPE call to muddy the
# grep). run_poly = solve's this%lhs%apply pattern; run_factory = construct's
# ALLOCATE(concrete::this%act); this%act%.. pattern.
_POLY_PREAMBLE = """
module m
  type, abstract :: base
  contains
    procedure(apply_i), deferred :: apply
  end type
  abstract interface
    subroutine apply_i(this, x)
      import base
      class(base), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: impl
  contains
    procedure :: apply => impl_apply
  end type
contains
  subroutine impl_apply(this, x)
    class(impl), intent(in) :: this
    real, intent(inout) :: x
    x = x * 2.0
  end subroutine
"""

_POLY_SOURCE = _POLY_PREAMBLE + """
  subroutine run_poly(b, x)        ! dispatch on abstract CLASS dummy
    class(base), intent(in) :: b
    real, intent(inout) :: x
    call b%apply(x)
  end subroutine
  subroutine run_factory(x)        ! ALLOCATE(concrete); dispatch in same scope
    real, intent(inout) :: x
    class(base), allocatable :: a
    allocate(impl :: a)
    call a%apply(x)
  end subroutine
end module
"""

# The monomorphised shape we would have to synthesise to avoid dispatch: a call
# on a concrete ``TYPE`` entity, kept in its own TU.
_CONCRETE_SOURCE = _POLY_PREAMBLE + """
  subroutine run_concrete(c, x)    ! call on concrete TYPE -> static bind
    type(impl), intent(in) :: c
    real, intent(inout) :: x
    call c%apply(x)
  end subroutine
end module
"""

#: mangled name of the concrete override; a *direct* fir.call to it is the unambiguous
#: signature of devirtualisation (fir.address_of / func.func def / fir.dt_entry do NOT count).
_OVERRIDE_SYMBOL = "_QMmPimpl_apply"
_DIRECT_CALL = f"fir.call @{_OVERRIDE_SYMBOL}"


def _sibling_tool(*names: str) -> str | None:
    """Resolve an LLVM companion tool (``fir-opt``/``bbc``), preferring the one
    next to the resolved flang so the versions match."""
    flang_dir = Path(_FLANG).resolve().parent if _FLANG else None
    for name in names:
        if flang_dir is not None:
            cand = flang_dir / name
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def _emit_fir(tmp_path: Path, source: str, stem: str, *, optimize: bool) -> str:
    """Emit FIR (the post-HLFIR level carrying ``fir.dispatch``) for ``source``.
    ``optimize`` runs flang's default ``-O2`` pipeline to cover the optimised path too."""
    src = tmp_path / f"{stem}.f90"
    src.write_text(source)
    out = tmp_path / f"{stem}{'_O2' if optimize else ''}.fir"
    cmd = [_FLANG, "-fc1", "-emit-fir"]
    if optimize:
        cmd.append("-O2")
    cmd += [str(src), "-o", str(out)]
    subprocess.check_call(cmd, cwd=str(tmp_path))
    return out.read_text()


def test_polymorphic_dispatch_lowers_to_fir_dispatch(tmp_path: Path):
    """Both polymorphic shapes -- TBP call on an abstract ``CLASS`` dummy (``run_poly``)
    and on a freshly-``ALLOCATE``d ``CLASS`` local (``run_factory``) -- lower to
    ``fir.dispatch``; neither yields a direct call to the override."""
    fir = _emit_fir(tmp_path, _POLY_SOURCE, "poly", optimize=False)
    # one fir.dispatch per polymorphic call site, none devirtualised.
    assert fir.count("fir.dispatch") == 2, "expected both polymorphic calls to lower to fir.dispatch"
    assert _DIRECT_CALL not in fir, ("a polymorphic call was devirtualised to a direct override call -- flang "
                                     "behaviour changed; revisit the ocean solver externalisation policy")


def test_concrete_type_call_is_a_direct_bind(tmp_path: Path):
    """Escape hatch: a TBP call on a *concrete* ``TYPE`` entity is a static bind -- a direct
    ``fir.call``, no dispatch. Reaching it requires source-level monomorphisation (our
    inliner's job), not anything flang does."""
    fir = _emit_fir(tmp_path, _CONCRETE_SOURCE, "concrete", optimize=False)
    assert _DIRECT_CALL in fir, "expected concrete TYPE call to bind directly to the override"
    assert "fir.dispatch" not in fir


def test_optimised_pipeline_keeps_dispatch(tmp_path: Path):
    """flang's default ``-O2`` pipeline (incl. ``--fir-polymorphic-op`` + inlining) does not
    turn either polymorphic shape into a direct call -- dispatch survives as a vtable indirect."""
    fir = _emit_fir(tmp_path, _POLY_SOURCE, "poly", optimize=True)
    assert _DIRECT_CALL not in fir, ("an -O2 pass devirtualised the polymorphic call -- flang behaviour "
                                     "changed; revisit the ocean solver externalisation policy")


def test_fir_polymorphic_op_lowers_dispatch_without_devirtualising(tmp_path: Path):
    """``--fir-polymorphic-op`` eliminates ``fir.dispatch`` but only by expanding it into
    the explicit vtable-load sequence (``fir.address_of`` -> indirect call) -- never a
    direct ``fir.call``. This is *lowering*, not devirtualisation."""
    fir_opt = _sibling_tool("fir-opt-21", "fir-opt")
    if fir_opt is None:
        pytest.skip("fir-opt not available alongside flang")
    raw = tmp_path / "poly.fir"
    raw.write_text(_emit_fir(tmp_path, _POLY_SOURCE, "poly", optimize=False))
    lowered = subprocess.check_output(
        [fir_opt, "--inline-all", "--fir-polymorphic-op", str(raw)], cwd=str(tmp_path)).decode()
    # The high-level dispatch op is gone ...
    assert "fir.dispatch" not in lowered
    # ... but it was replaced by a runtime vtable indirect, NOT a direct call.
    assert _DIRECT_CALL not in lowered, ("--fir-polymorphic-op produced a direct call -- it now devirtualises; "
                                         "revisit the ocean solver externalisation policy")
