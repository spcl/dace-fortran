"""Characterisation tests pinning *why* the ICON-O solver construct/solve
chain must stay an external call rather than be inlined into the kernel TU.

The dace-fortran pipeline lowers an inlined Fortran TU through flang to
HLFIR/FIR and then to a DaCe SDFG.  An SDFG is static dataflow: it has no
node for a runtime indirect call through a type descriptor's binding table.
So any Fortran type-bound-procedure (TBP) call dispatched on a *polymorphic*
(``CLASS(..)``) entity -- ``this%act%solve``, ``this%lhs%apply``,
``this%trans%into``, or the ``ALLOCATE(concrete :: this%act); this%act%..``
factory in ``ocean_solve_construct`` -- can only be lowered if flang resolves
it to a *direct* call first.

These tests demonstrate, against the installed flang, that it does **not**:

  * a dispatch on a ``CLASS`` dummy lowers to ``fir.dispatch`` (runtime vtable);
  * even a dispatch on a ``CLASS`` local whose concrete type was ``ALLOCATE``d
    one line above (the construct's factory pattern) stays ``fir.dispatch`` --
    flang does not propagate the allocated type to the call;
  * a call on a *concrete* ``TYPE(..)`` entity is the only shape that lowers to
    a direct ``fir.call`` (this is the escape hatch: source-level
    monomorphisation, which would have to happen in our inliner, never in flang);
  * the dedicated FIR pass ``--fir-polymorphic-op`` (even after ``--inline-all``)
    only *lowers* ``fir.dispatch`` into the explicit runtime vtable-load
    sequence; it never yields a direct ``fir.call`` to the override.

If a future flang gains real devirtualisation, the ``fir.dispatch`` assertions
here will start failing -- which is the signal to revisit the externalisation
policy in ``tests/icon/ocean/_ocean_harness.py`` and let the construct/solve be
inlined instead.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _util import _FLANG, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# A minimal abstract base + one concrete override.  The two *polymorphic* call
# shapes that matter for the ocean solver are kept in their own TU so that the
# only way a direct ``fir.call`` to the override could appear is genuine
# devirtualisation -- there is no concrete-``TYPE`` call to muddy a whole-module
# grep.  ``run_poly`` is the solve's ``this%lhs%apply`` pattern; ``run_factory``
# is ``ocean_solve_construct``'s ``ALLOCATE(concrete :: this%act); this%act%..``.
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

#: the mangled name flang gives the concrete override; a *direct* ``fir.call`` to
#: it is the unambiguous signature of devirtualisation.  (A ``fir.address_of`` of
#: this symbol -- loading it into a vtable indirect -- or its ``func.func``
#: definition / ``fir.dt_entry`` table slot are NOT calls and don't count.)
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
    """Emit FIR (the post-HLFIR level that carries ``fir.dispatch``) for
    ``source``.  ``optimize`` runs flang's default ``-O2`` pipeline so we cover
    the optimised path, not just the raw lowering."""
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
    """The two polymorphic shapes -- a TBP call on an abstract ``CLASS`` dummy
    (``run_poly``, the solve's ``this%lhs%apply``) and on a ``CLASS`` local whose
    concrete type was ``ALLOCATE``d one line above (``run_factory``, the
    construct's factory) -- both lower to ``fir.dispatch`` (a runtime vtable
    lookup), and neither yields a direct call to the override."""
    fir = _emit_fir(tmp_path, _POLY_SOURCE, "poly", optimize=False)
    # one fir.dispatch per polymorphic call site, none devirtualised.
    assert fir.count("fir.dispatch") == 2, "expected both polymorphic calls to lower to fir.dispatch"
    assert _DIRECT_CALL not in fir, ("a polymorphic call was devirtualised to a direct override call -- flang "
                                     "behaviour changed; revisit the ocean solver externalisation policy")


def test_concrete_type_call_is_a_direct_bind(tmp_path: Path):
    """Contrast / escape hatch: a TBP call on a *concrete* ``TYPE`` entity is a
    static bind -- a direct ``fir.call`` to the override, with no dispatch.  This
    is the only shape that avoids dispatch, and reaching it requires source-level
    monomorphisation (our inliner's job), not anything flang does."""
    fir = _emit_fir(tmp_path, _CONCRETE_SOURCE, "concrete", optimize=False)
    assert _DIRECT_CALL in fir, "expected concrete TYPE call to bind directly to the override"
    assert "fir.dispatch" not in fir


def test_optimised_pipeline_keeps_dispatch(tmp_path: Path):
    """flang's default ``-O2`` pipeline (which includes ``--fir-polymorphic-op``
    and inlining) does not turn either polymorphic shape into a direct call to
    the override -- the dispatch survives as a runtime vtable indirect."""
    fir = _emit_fir(tmp_path, _POLY_SOURCE, "poly", optimize=True)
    assert _DIRECT_CALL not in fir, ("an -O2 pass devirtualised the polymorphic call -- flang behaviour "
                                     "changed; revisit the ocean solver externalisation policy")


def test_fir_polymorphic_op_lowers_dispatch_without_devirtualising(tmp_path: Path):
    """``--fir-polymorphic-op`` (even preceded by ``--inline-all``) eliminates
    the ``fir.dispatch`` op, but only by expanding it into the explicit runtime
    vtable-load sequence (``fir.address_of`` the override -> indirect call); it
    never produces a direct ``fir.call`` to the override.  This is *lowering*,
    not devirtualisation."""
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
