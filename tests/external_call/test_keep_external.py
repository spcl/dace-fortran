"""End-to-end coverage for the unified external-function policy and its
rich-ABI escape hatch :func:`dace_fortran.keep_external`.

The simple cases -- a plain ``bind(c)`` callee whose argument plan the
bridge can derive from the HLFIR call site -- are declared with
:func:`dace_fortran.apply_external_functions` and one
:class:`~dace_fortran.external_functions.ExternalFunction` (call-site
name + the ``.so`` that exports the symbol).  The ``kind='comm'`` cases
at the bottom keep :func:`keep_external`'s authored
:class:`~dace_fortran.external.Arg` list: an ``MPI_Comm`` handle is an
ABI fact HLFIR cannot infer, so it stays on the rich path.  Each test
compiles a separately-built ``bind(c)`` Fortran subroutine into a
``.so``, declares it external, builds the SDFG, runs it, and asserts the
array was mutated as the external function expects.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.external import (Arg, ExternalCall, apply_external_functions, clear_external_registry, keep_external,
                                   lookup_external)
from dace_fortran.external_functions import ExternalFunction

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
module run_mod
  implicit none
contains
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
end module run_mod
"""


def test_external_registry_isolation_fixture_is_autouse(request):
    """Every test must get the external registry cleared at teardown.

    ``tests/conftest.py``'s ``isolate_external_registry`` is what stops a
    registration made here from leaking into a later test in the same process.
    Without it, this file's ``bar(a, n)`` stayed registered and rebound
    ``tests/multi_callsite_inlining_test.py``'s module-local ``CALL bar(a)`` to
    the 2-arg external: the callee read a garbage ``n`` and looped
    ``a = a + 1`` that many times -- an out-of-bounds write that segfaulted the
    interpreter (SIGSEGV), or silently returned wrong values when the garbage
    happened to be small.  Assert the fixture is wired so deleting
    ``autouse=True`` fails here, next to the cause, instead of resurfacing as
    an unrelated file's crash.
    """
    assert "isolate_external_registry" in request.fixturenames


def test_keep_external_lowers_to_externalcall(tmp_path: Path):
    """An :func:`apply_external_functions` declaration produces an
    :class:`ExternalCall` library node bound to the supplied ``.so`` --
    the same registry the rich :func:`keep_external` path populates."""
    clear_external_registry()
    bar_f90 = tmp_path / "bar.f90"
    bar_f90.write_text(_BAR_F90)
    libbar = tmp_path / "libbar.so"
    # cwd=tmp_path keeps gfortran from picking up any stale .mod that
    # a prior flang invocation left in the repo root.
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libbar), str(bar_f90)], cwd=str(tmp_path))

    # Don't-inline + emit ``bar``; the arg plan (array inout, scalar in)
    # is derived from the HLFIR call site.
    apply_external_functions([ExternalFunction("bar", library=str(libbar))])

    sig = lookup_external("bar")
    assert sig is not None and sig.c_name == "bar"

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="run_mod::run").build()
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
    """When ``c_function`` is omitted :attr:`ExternalFunction.symbol`
    falls back to the Fortran call-site name -- the common case where
    both names are identical."""
    clear_external_registry()
    bar_f90 = tmp_path / "bar.f90"
    bar_f90.write_text(_BAR_F90)
    libbar = tmp_path / "libbar.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libbar), str(bar_f90)], cwd=str(tmp_path))

    # No c_function= -> the emitted symbol defaults to "bar".
    apply_external_functions([ExternalFunction("bar", library=str(libbar))])
    assert lookup_external("bar").c_name == "bar"

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run", entry="run_mod::run").build()
    sdfg.name = "ext_keep_default_cname"
    sdfg.compile()

    n = 5
    a = np.asfortranarray(np.arange(n, dtype=np.float64))
    expected = a + 1.0
    sdfg(a=a, n=n)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)


def test_apply_external_functions_empty_args_passthrough():
    """A minimal :class:`ExternalFunction` (name only) registers a
    no-args, no-libraries signature -- the bridge derives the arg plan
    from HLFIR, so nothing is authored here.  ``c_function`` overrides
    the emitted symbol; re-declaring the same name replaces the entry."""
    clear_external_registry()
    apply_external_functions([ExternalFunction("noop")])
    sig = lookup_external("noop")
    assert sig is not None and sig.c_name == "noop"
    assert sig.args == () and sig.libraries == ()
    # Idempotent on the registry: re-declaring the same name with an
    # explicit ``c_function`` replaces the entry but does not corrupt it.
    apply_external_functions([ExternalFunction("noop", c_function="other_sym")])
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

    sig = ExternalSignature(c_name="shim_with_comm",
                            args=(Arg(kind="array", dtype="float64",
                                      intent="inout"), Arg(kind="scalar", dtype="int32",
                                                           intent="in"), Arg(kind="comm")))
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
    keep_external("exch_with_comm",
                  c_name="exch_with_comm_c",
                  args=[
                      Arg(kind="array", dtype="float64", intent="inout"),
                      Arg(kind="scalar", dtype="int32", intent="in"),
                      Arg(kind="comm")
                  ])
    sig = lookup_external("exch_with_comm")
    assert sig is not None and sig.c_name == "exch_with_comm_c"
    assert tuple(a.kind for a in sig.args) == ("array", "scalar", "comm")
    assert sig.c_declaration() == \
        'extern "C" void exch_with_comm_c(double *, int, MPI_Comm);'
    clear_external_registry()
