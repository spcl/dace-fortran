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


# --------------------------------------------------------------------------
# kind="comm" -- MPI_Comm by-value arg.  Drives the c_decl_type +
# signature surface; full e2e (mpirun + DaCe MPI binding to materialise
# a comm at the SDFG boundary) lives in tests/mpi_comm_e2e_test.py
# (the existing Send/Recv-on-split-comm test).
# --------------------------------------------------------------------------


def test_comm_kind_c_decl_type_is_mpi_comm():
    """``Arg(kind='comm')`` declares ``MPI_Comm`` regardless of
    ``dtype`` -- the field is documented as ignored for this kind."""
    from dace_fortran.external import Arg, ExternalSignature
    a = Arg(kind="comm")
    assert a.c_decl_type() == "MPI_Comm"
    # An explicit (and irrelevant) dtype must not change the C type.
    a_explicit = Arg(kind="comm", dtype="int32")
    assert a_explicit.c_decl_type() == "MPI_Comm"

    sig = ExternalSignature(
        c_name="shim_with_comm",
        args=(Arg(kind="array", dtype="float64", intent="inout"),
              Arg(kind="scalar", dtype="int32", intent="in"),
              Arg(kind="comm")))
    decl = sig.c_declaration()
    # Argument order is preserved verbatim (left-to-right same as args).
    assert decl == 'extern "C" void shim_with_comm(double *, int, MPI_Comm);'


def test_comm_kind_rejects_unknown_dtype_only_for_data_args():
    """Unknown ``dtype`` is fatal for array/scalar (resolved via
    ``_C_TYPES``) but **not** for comm (its type is fixed)."""
    from dace_fortran.external import Arg
    with pytest.raises(ValueError, match="unsupported dtype"):
        Arg(kind="array", dtype="float16").c_decl_type()
    with pytest.raises(ValueError, match="unsupported dtype"):
        Arg(kind="scalar", dtype="complex64").c_decl_type()
    # comm: the dtype is ignored, so a nonsense one still yields MPI_Comm.
    assert Arg(kind="comm", dtype="something_irrelevant").c_decl_type() == "MPI_Comm"


def test_keep_external_with_comm_signature_round_trip():
    """``keep_external`` accepts the new ``kind='comm'`` arg and stores
    the signature unchanged (the registry is type-blind; the consumer
    in ``emit_call`` is what wires the opaque(MPI_Comm) connector)."""
    from dace_fortran.external import Arg
    clear_external_registry()
    keep_external(
        "exch_with_comm",
        c_name="exch_with_comm_c",
        args=[Arg(kind="array", dtype="float64", intent="inout"),
              Arg(kind="scalar", dtype="int32", intent="in"),
              Arg(kind="comm")])
    sig = lookup_external("exch_with_comm")
    assert sig is not None and sig.c_name == "exch_with_comm_c"
    assert tuple(a.kind for a in sig.args) == ("array", "scalar", "comm")
    assert sig.c_declaration() == \
        'extern "C" void exch_with_comm_c(double *, int, MPI_Comm);'
    clear_external_registry()
