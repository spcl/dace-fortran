"""Executable spec (xfail until implemented): a separately-compiled
external **iso_c Fortran** function lowered via a registered signature.

Design (see memory ``project_external_function_calls_design.md``):
the user registers ``foo`` 's signature; the bridge lowers
``call foo(a, n)`` to a CPP tasklet that calls the ``extern "C"``
symbol, left undefined in the SDFG ``.so`` and resolved at load by
preloading the separately-built ``libfoo.so`` (RTLD_GLOBAL).

Why iso_c / ``bind(c)``: Fortran name mangling is compiler-specific
and a ``.mod`` is not C-consumable, so the only portable way to call
a Fortran routine from the generated C++ is an ``ISO_C_BINDING``
``bind(c, name=...)`` stable symbol (or a generated ``bind(c)`` shim
when only a ``.mod`` exists -- the documented case-B follow-up).

``strict=False``: this xpasses (loudly) once the registry +
``emit_call`` land, signalling the spec is satisfied; it must not be
silently converted to a normal test until then.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

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
"""


@pytest.mark.xfail(strict=False,
                   reason="external-function registry + emit_call not yet implemented "
                   "(spec-first; see project_external_function_calls_design)")
def test_external_iso_c_function_increments_array(tmp_path: Path):
    # Future registry API -- import inside the test so its absence is
    # an xfail, not a collection error.
    from dace_fortran.external import Arg, ExternalSignature, register_external

    # Build the external function as its own shared library.
    foo_f90 = tmp_path / "foo.f90"
    foo_f90.write_text(_FOO_F90)
    libfoo = tmp_path / "libfoo.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libfoo), str(foo_f90)])

    register_external("foo", ExternalSignature(
        c_name="foo",
        args=[Arg(kind="array", dtype="float64", intent="inout"),
              Arg(kind="scalar", dtype="int32", intent="in")],
    ))

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="_QPrun").build()
    sdfg.name = "ext_run"
    sdfg.compile()

    # Resolve the undefined ``foo`` symbol by preloading libfoo
    # globally before the SDFG library is invoked.
    ctypes.CDLL(str(libfoo), mode=ctypes.RTLD_GLOBAL)

    n = 12
    a = np.arange(n, dtype=np.float64, order="F")
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)
