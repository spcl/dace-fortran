"""E2e: a separately-compiled external **iso_c Fortran** function lowered via the unified
external-function policy (see ``DESIGN.md``). ``apply_external_functions`` registers ``foo``
once; the bridge derives the arg plan (array->inout pointer, scalar->by-value) from the
HLFIR call site and lowers to a CPP tasklet calling ``extern "C" foo``, no re-authored
signature. SDFG ``.so`` links ``libfoo.so`` via rpath (no LD_PRELOAD). Contract: target must
be ``bind(c)`` -- the only portable way to call Fortran from generated C++."""
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

# The kernel the bridge sees: it only declares foo's interface and calls it -- foo itself
# is compiled separately (only its symbol needs to resolve at load).
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
    # cwd=tmp_path keeps gfortran from picking up a stale .mod that a prior flang run left
    # in repo root (gfortran refuses a flang-format iso_c_binding.mod).
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libfoo), str(foo_f90)], cwd=str(tmp_path))

    # ONE declaration: don't-inline + emit foo as an external call bound to libfoo.so.
    # Arg plan derives from the HLFIR call foo(a, n): array a -> inout ptr, scalar n -> by-value.
    apply_external_functions([ExternalFunction("foo", library=str(libfoo))])

    # SDFG .so links libfoo via rpath -- self-contained, no LD_PRELOAD needed.
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
    """HLFIR-derived plan makes an array arg ``inout`` by default (safe for an opaque
    external): the write-back edge is kept, so a missed write wouldn't silently drop it."""
    clear_external_registry()
    foo_f90 = tmp_path / "foo.f90"
    foo_f90.write_text(_FOO_F90)
    libfoo = tmp_path / "libfoo.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libfoo), str(foo_f90)], cwd=str(tmp_path))

    # No authored args -> emit_call derives the plan; array a is conservatively inout.
    apply_external_functions([ExternalFunction("foo", library=str(libfoo))])

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="run_mod::run").build()
    sdfg.name = "ext_run_def"
    sdfg.compile()

    n = 7
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)
