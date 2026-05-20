"""End-to-end coverage for :func:`dace_fortran.keep_external`.

A convenience wrapper around :func:`register_external` that surfaces
the "leave this procedure external" intent without the
``ExternalSignature`` boilerplate at the call site.  These tests
mirror the ``foo``-style coverage of ``test_external_call.py``:
compile a separately-built ``bind(c)`` Fortran subroutine into a
``.so``, mark it via ``keep_external``, build the SDFG, run it,
and assert the array was mutated as the external function expects.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.external import (Arg, ExternalCall, clear_external_registry,
                                   keep_external, lookup_external)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Increments every element of an array by 1.  ``bind(c, name="bar")``
# -> stable unmangled C symbol callable from the generated C++.
_BAR_F90 = """
subroutine bar(a, n) bind(c, name="bar")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: a(n)
  a = a + 1.0d0
end subroutine bar
"""

_KERNEL = """
subroutine run(a, n)
  use iso_c_binding
  implicit none
  integer(c_int), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  interface
    subroutine bar(a, n) bind(c, name="bar")
      use iso_c_binding
      real(c_double), intent(inout) :: a(*)
      integer(c_int), value :: n
    end subroutine bar
  end interface
  call bar(a, n)
end subroutine run
"""


def test_keep_external_lowers_to_externalcall(tmp_path: Path):
    """A ``keep_external`` registration produces the same
    :class:`ExternalCall` library node that :func:`register_external`
    does -- it is the same registry behind both."""
    clear_external_registry()
    bar_f90 = tmp_path / "bar.f90"
    bar_f90.write_text(_BAR_F90)
    libbar = tmp_path / "libbar.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libbar), str(bar_f90)])

    keep_external(
        "bar",
        args=[Arg(kind="array", dtype="float64", intent="inout"),
              Arg(kind="scalar", dtype="int32", intent="in")],
        libraries=[str(libbar)],
    )

    # Same registry -> the same signature surfaces under either lookup.
    sig = lookup_external("bar")
    assert sig is not None and sig.c_name == "bar"

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="_QPrun").build()
    sdfg.name = "ext_keep_run"

    calls = [nd for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ExternalCall)]
    assert len(calls) == 1 and calls[0].c_name == "bar"
    sdfg.validate()
    sdfg.compile()

    n = 8
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)


def test_keep_external_defaults_c_name_to_fortran_name(tmp_path: Path):
    """When ``c_name`` is omitted it falls back to the Fortran call-site
    name -- the common case where both names are identical."""
    clear_external_registry()
    bar_f90 = tmp_path / "bar.f90"
    bar_f90.write_text(_BAR_F90)
    libbar = tmp_path / "libbar.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libbar), str(bar_f90)])

    # No c_name= -> defaults to "bar".
    keep_external(
        "bar",
        args=[Arg(kind="array", dtype="float64"), Arg(kind="scalar", dtype="int32")],
        libraries=[str(libbar)],
    )
    assert lookup_external("bar").c_name == "bar"

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="_QPrun").build()
    sdfg.name = "ext_keep_default_cname"
    sdfg.compile()

    n = 5
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)


def test_keep_external_empty_args_passthrough():
    """No-args registration is legal -- the signature is stored as-is
    and the registry survives a second lookup without rewriting it."""
    clear_external_registry()
    keep_external("noop")
    sig = lookup_external("noop")
    assert sig is not None and sig.c_name == "noop"
    assert sig.args == () and sig.libraries == ()
    # Idempotent on the registry: re-registering with the same name
    # replaces the entry but does not corrupt the lookup.
    keep_external("noop", c_name="other_sym")
    assert lookup_external("noop").c_name == "other_sym"
    clear_external_registry()
    assert lookup_external("noop") is None
