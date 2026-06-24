"""Phase 3 of the unified external-function policy: the bridge-side entry
point :func:`dace_fortran.external.apply_external_functions` and the
``emit_call`` *derive-from-HLFIR* path that lets a minimal
:class:`dace_fortran.external_functions.ExternalFunction` (``args=()``) lower a
call whose argument order comes entirely from the Fortran source.

Two layers:

* **contract** (no toolchain) -- ``apply_external_functions`` populates the
  registry: each :class:`ExternalFunction` becomes an *emitted* external
  (``c_name`` = its symbol, not stubbed); each ``do_not_emit`` name an *ignored*
  one (``stub=True``).  Validation rejects an inconsistent policy first.
* **e2e** (flang + gfortran) -- the linchpin: a bare ``ExternalFunction("bar",
  library=libbar)`` with NO authored ``Arg`` list produces the same working
  :class:`ExternalCall` as the authored two-``Arg`` ``keep_external`` does.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dace_fortran.external import (Arg, ExternalCall, apply_external_functions, clear_external_registry, keep_external,
                                   lookup_external, registered_names)
from dace_fortran.external_functions import ExternalFunction


# ---------------------------------------------------------------------------
# contract -- registry population (no toolchain needed)
# ---------------------------------------------------------------------------

def test_apply_registers_emitted_external():
    """An ``ExternalFunction`` becomes an emitted (non-stub) registration whose
    ``c_name`` is its symbol and whose library is linked."""
    clear_external_registry()
    try:
        apply_external_functions(
            [ExternalFunction("sync_patch_array", c_function="sync_c",
                              library="/tmp/libhalo.so")], [])
        sig = lookup_external("sync_patch_array")
        assert sig is not None
        assert sig.c_name == "sync_c"
        assert sig.libraries == ("/tmp/libhalo.so", )
        assert sig.stub is False
        assert sig.args == (), "a bare ExternalFunction authors no args"
    finally:
        clear_external_registry()


def test_apply_symbol_defaults_to_name():
    """``c_function`` omitted -> the symbol (and ``c_name``) is the Fortran name;
    ``library`` omitted -> no libraries."""
    clear_external_registry()
    try:
        apply_external_functions([ExternalFunction("exchange_data")], [])
        sig = lookup_external("exchange_data")
        assert sig.c_name == "exchange_data"
        assert sig.libraries == ()
    finally:
        clear_external_registry()


def test_apply_registers_ignored_as_stub():
    """Each ``do_not_emit`` name registers with ``stub=True`` (the call is
    dropped) -- ignore is a subset of don't-inline."""
    clear_external_registry()
    try:
        apply_external_functions([], ["finish", "message", "timer_start"])
        for nm in ("finish", "message", "timer_start"):
            sig = lookup_external(nm)
            assert sig is not None and sig.stub is True
    finally:
        clear_external_registry()


def test_apply_both_lists_together():
    """Both collections register in one call; ``registered_names`` lists all."""
    clear_external_registry()
    try:
        apply_external_functions(
            [ExternalFunction("sync_patch_array"), ExternalFunction("exchange_data")],
            ["finish", "dbg_print"])
        assert set(registered_names()) == {
            "sync_patch_array", "exchange_data", "finish", "dbg_print"}
        assert lookup_external("sync_patch_array").stub is False
        assert lookup_external("finish").stub is True
    finally:
        clear_external_registry()


def test_apply_validates_name_in_both():
    """A name that is both emitted and ignored is an inconsistent policy --
    rejected before any registration mutates the registry."""
    clear_external_registry()
    try:
        with pytest.raises(ValueError, match="both"):
            apply_external_functions([ExternalFunction("sync_patch_array")],
                                     ["sync_patch_array"])
        assert registered_names() == [], "rejected before mutating the registry"
    finally:
        clear_external_registry()


def test_apply_validates_duplicate_emit_name():
    """A duplicate emitted name is rejected by ``validate``."""
    clear_external_registry()
    try:
        with pytest.raises(ValueError, match="duplicate"):
            apply_external_functions(
                [ExternalFunction("foo"), ExternalFunction("foo")], [])
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
# e2e -- the derive-from-HLFIR linchpin: a bare ExternalFunction lowers + runs
# ---------------------------------------------------------------------------

from _util import build_sdfg, have_flang  # tests/conftest.py puts tests/ on sys.path

_e2e = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Increments every element by 1.  ``bind(c, name="bar")`` -> stable C symbol.
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


def _build_libbar(tmp_path: Path) -> Path:
    bar_f90 = tmp_path / "bar.f90"
    bar_f90.write_text(_BAR_F90)
    libbar = tmp_path / "libbar.so"
    subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", str(libbar), str(bar_f90)],
                          cwd=str(tmp_path))
    return libbar


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_e2e)])
def test_bare_external_function_lowers_and_runs(tmp_path: Path, _m):
    """The linchpin: registering only ``ExternalFunction("bar", library=...)``
    -- NO ``Arg`` list -- still lowers ``call bar(a, n)`` to a working
    ``ExternalCall``.  ``emit_call`` derives the plan from the HLFIR call site:
    ``a`` (an SDFG array) crosses as an inout pointer, ``n`` (a shape symbol) is
    referenced inline.  The result matches the authored-two-``Arg`` version."""
    clear_external_registry()
    try:
        libbar = _build_libbar(tmp_path)
        apply_external_functions([ExternalFunction("bar", library=str(libbar))], [])

        sig = lookup_external("bar")
        assert sig is not None and sig.c_name == "bar" and sig.args == ()

        sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="run",
                          entry="run_mod::run").build()
        sdfg.name = "ext_apply_bare_run"
        calls = [nd for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ExternalCall)]
        assert len(calls) == 1 and calls[0].c_name == "bar"
        sdfg.validate()
        sdfg.compile()

        n = 8
        a = np.asfortranarray(np.arange(n, dtype=np.float64))
        expected = a + 1.0
        sdfg(a=a, n=n)
        np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)
    finally:
        clear_external_registry()


def _external_call_node(tmp_path, libbar, register):
    """Build the ``_KERNEL`` SDFG under the ``register`` callback (which sets up
    the registry) and return its single :class:`ExternalCall` node."""
    register(libbar)
    sdfg = build_sdfg(_KERNEL, tmp_path / register.__name__, name="run",
                      entry="run_mod::run").build()
    calls = [nd for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ExternalCall)]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_e2e)])
def test_derived_node_matches_authored(tmp_path: Path, _m):
    """The derive-from-HLFIR node is byte-identical to the hand-authored one:
    a bare ``ExternalFunction("bar", library=...)`` and the explicit
    ``keep_external("bar", args=[Arg(array,float64,inout), Arg(scalar,int32,in)])``
    produce the SAME ``c_decl`` and ``body`` -- so the minimal registration is a
    drop-in for the verbose one (``a`` -> ``double *`` inout, ``n`` -> ``int``)."""
    libbar = _build_libbar(tmp_path)

    def bare(lib):
        clear_external_registry()
        apply_external_functions([ExternalFunction("bar", library=str(lib))], [])

    def authored(lib):
        clear_external_registry()
        keep_external("bar",
                      args=[Arg(kind="array", dtype="float64", intent="inout"),
                            Arg(kind="scalar", dtype="int32", intent="in")],
                      libraries=[str(lib)])

    try:
        derived = _external_call_node(tmp_path, libbar, bare)
        d_decl, d_body = derived.c_decl, derived.body
        explicit = _external_call_node(tmp_path, libbar, authored)
        assert d_decl == explicit.c_decl == 'extern "C" void bar(double *, int);'
        assert d_body == explicit.body
    finally:
        clear_external_registry()
