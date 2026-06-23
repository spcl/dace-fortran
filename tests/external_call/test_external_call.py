"""End-to-end: a separately-compiled external **iso_c Fortran**
function lowered via the unified external-function policy (see
``DESIGN.md`` here).

The user declares ``foo`` once with
:func:`dace_fortran.apply_external_functions` -- a single
:class:`~dace_fortran.external_functions.ExternalFunction` naming the
call-site name and the ``.so`` that exports its ``bind(c)`` symbol.  The
bridge lowers ``call foo(a, n)`` to a CPP tasklet calling the
``extern "C"`` symbol, deriving the argument plan (array -> inout
pointer, scalar -> by-value) from the HLFIR call site -- the user does
NOT re-author the signature.  The SDFG ``.so`` links ``libfoo.so``
directly (rpath), so it resolves at load with no ``LD_PRELOAD``.

Contract: the target must be ``bind(c)`` -- Fortran name mangling is
compiler-specific and a ``.mod`` is not C-consumable, so a stable
``bind(c)`` symbol (native or via a hand-written shim) is the only
portable way to call a Fortran routine from the generated C++.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.external import ExternalCall, apply_external_functions, clear_external_registry
from dace_fortran.external_functions import ExternalFunction

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Separately-compiled external function: increments a whole array by 1.
# ``bind(c, name="foo")`` -> stable, unmangled C symbol callable from
# the generated C++ with ``extern "C"``.
_FOO_F90 = """
subroutine foo(a, n) bind(c, name="foo")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: a(n)
  a = a + 1.0d0
end subroutine foo
"""

# The kernel the bridge sees: it only declares foo's interface and
# calls it -- foo itself is compiled separately (only its symbol need
# resolve at load).
_KERNEL = """
module run_mod
  implicit none
contains
  subroutine run(a, n)
    use iso_c_binding
    implicit none
    integer(c_int), intent(in) :: n
    real(c_double), intent(inout) :: a(n)
    interface
      subroutine foo(a, n) bind(c, name="foo")
        use iso_c_binding
        real(c_double), intent(inout) :: a(*)
        integer(c_int), value :: n
      end subroutine foo
    end interface
    call foo(a, n)
  end subroutine run
end module run_mod
"""


def test_external_iso_c_function_increments_array(tmp_path: Path):
    clear_external_registry()
    # Build the external function as its own shared library.
    foo_f90 = tmp_path / "foo.f90"
    foo_f90.write_text(_FOO_F90)
    libfoo = tmp_path / "libfoo.so"
    # cwd=tmp_path keeps gfortran from picking up any stale .mod that
    # a prior flang invocation left in the repo root (gfortran searches
    # cwd for ``iso_c_binding.mod`` and refuses a flang-format module).
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libfoo), str(foo_f90)], cwd=str(tmp_path))

    # ONE declaration: don't-inline + emit ``foo`` as an external call,
    # bound to the ``libfoo.so`` that exports its ``bind(c)`` symbol.
    # The argument plan is derived from the HLFIR call ``foo(a, n)``
    # (array ``a`` -> inout pointer, scalar ``n`` -> by-value).
    apply_external_functions([ExternalFunction("foo", library=str(libfoo))])

    # The SDFG .so is linked against libfoo with an rpath, so it is
    # self-contained: no LD_PRELOAD / load ordering needed.
    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="run_mod::run").build()
    sdfg.name = "ext_run"

    calls = [nd for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ExternalCall)]
    assert len(calls) == 1 and calls[0].c_name == "foo"
    assert calls[0].in_connectors and calls[0].out_connectors
    sdfg.validate()

    sdfg.compile()

    n = 12
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)


def test_external_default_intent_is_inout(tmp_path: Path):
    """The HLFIR-derived plan makes an array arg ``inout`` (the safe
    default for an opaque external), so the array still gets the
    write-back edge and the mutation is modelled (a missed write would
    silently drop the increment)."""
    clear_external_registry()
    foo_f90 = tmp_path / "foo.f90"
    foo_f90.write_text(_FOO_F90)
    libfoo = tmp_path / "libfoo.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libfoo), str(foo_f90)], cwd=str(tmp_path))

    # No authored args -> emit_call derives the plan from the call site;
    # the array ``a`` is conservatively inout, so its write-back is kept.
    apply_external_functions([ExternalFunction("foo", library=str(libfoo))])

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="run_mod::run").build()
    sdfg.name = "ext_run_def"
    sdfg.compile()

    n = 7
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)
