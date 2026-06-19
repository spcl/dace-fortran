# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for two robustness features of the fparser single-TU inliner that
let it ingest *real* ICON sources (rather than only cpp-clean fixtures):

1. ``expand_cpp=True`` -- run the C preprocessor (via ``flang -cpp -E -P``)
   over each source before fparser parses it, so cpp ``#include`` directives
   and ``#define`` macros (ICON declares its derived-type members through the
   DSL macro headers) are expanded into pure Fortran.  Without it fparser
   raises on the first ``#include``.

2. ``tolerate_external_uses=True`` -- do not hard-fail when a module ``USE``s
   an external library that has no Fortran source on the search path (ICON:
   ``netcdf`` / ``mpi`` / ``cdi``).  The import is left unresolved and the
   reachability pruning drops the (unused) procedures that referenced it.
   This is the mechanism that lets a kernel whose enclosing module ``USE
   netcdf`` inline cleanly as long as the kernel itself does not call netcdf.
"""
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dace_fortran.fparser_inliner import cpp_expand_sources, inline_to_ast, inline_to_single_tu
from dace_fortran.preprocess import merge_used_modules


def _have_flang() -> bool:
    return shutil.which("flang-new-21") is not None


def _have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


def _gfortran_compiles(src_text: str) -> bool:
    """True if ``src_text`` compiles standalone with gfortran."""
    with TemporaryDirectory() as td:
        f = Path(td) / "tu.f90"
        f.write_text(src_text)
        r = subprocess.run(["gfortran", "-fsyntax-only", "-ffree-line-length-none",
                            str(f)],
                           cwd=td,
                           capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode())
        return r.returncode == 0


# ---------------------------------------------------------------------------
# tolerate_external_uses -- the netcdf / mpi pruning path (no flang needed)
# ---------------------------------------------------------------------------

_NETCDF_USER = """
module mo_thing
  use netcdf, only: nf90_open, nf90_close
  implicit none
contains
  pure function kernel_add(a, b) result(c)
    real, intent(in) :: a, b
    real :: c
    c = a + b
  end function kernel_add
  subroutine writes_netcdf(path)
    character(*), intent(in) :: path
    integer :: ncid, ierr
    ierr = nf90_open(path, 0, ncid)
    ierr = nf90_close(ncid)
  end subroutine writes_netcdf
end module mo_thing
"""


def test_external_use_unreached_is_pruned():
    """A module ``USE``s netcdf but the requested entry never calls it: with
    tolerance on, the netcdf-touching procedure (and the dangling import) are
    pruned, leaving a self-contained TU."""
    out = inline_to_ast({
        "mo_thing.f90": _NETCDF_USER
    }, entry="mo_thing::kernel_add", tolerate_external_uses=True).tofortran().lower()
    assert "kernel_add" in out
    assert "writes_netcdf" not in out, "the unused netcdf procedure should be pruned"
    assert "nf90_" not in out, "the external netcdf calls should be gone"
    assert "use netcdf" not in out, "the dangling external USE should be pruned"


def test_external_use_default_strict_still_asserts():
    """Default (tolerance off) keeps the strict resolution the upstream
    desugaring relies on: an unresolved external ``USE`` is a hard error."""
    with pytest.raises(AssertionError):
        inline_to_ast({"mo_thing.f90": _NETCDF_USER}, entry="mo_thing::kernel_add")


def test_external_use_reached_does_not_crash():
    """When the entry *does* reach the external call, tolerance keeps it
    rather than crashing (the call survives as an unresolved external)."""
    out = inline_to_ast({
        "mo_thing.f90": _NETCDF_USER
    }, entry="mo_thing::writes_netcdf", tolerate_external_uses=True).tofortran().lower()
    assert "writes_netcdf" in out
    assert "nf90_open" in out


# ---------------------------------------------------------------------------
# expand_cpp -- the C-preprocessor pre-pass (needs flang)
# ---------------------------------------------------------------------------

pytestmark_flang = pytest.mark.skipif(not _have_flang(), reason="flang-new-21 not on PATH")


@pytestmark_flang
def test_cpp_expand_sources_resolves_include_and_macro(tmp_path):
    """``cpp_expand_sources`` expands a cpp ``#include`` + ``#define`` macro
    into pure Fortran (no ``#`` directives left)."""
    (tmp_path / "defs.inc").write_text("#define WP 8\n")
    src = '#include "defs.inc"\n' \
          "module mo_kindy\n  real(WP) :: x\nend module mo_kindy\n"
    out = cpp_expand_sources({"mo_kindy.F90": src}, include_dirs=[tmp_path])
    text = list(out.values())[0]
    assert "#include" not in text
    assert not any(l.lstrip().startswith("#") for l in text.splitlines())
    assert "real(8)" in text.lower()


@pytestmark_flang
def test_inline_with_cpp_include(tmp_path):
    """End-to-end: a source carrying a cpp ``#include`` inlines once
    ``expand_cpp=True`` resolves it (it would raise otherwise)."""
    (tmp_path / "kinds.inc").write_text("#define WP 8\n")
    src = '#include "kinds.inc"\n' + """
module mo_calc
  implicit none
contains
  pure function scaled(a) result(c)
    real(WP), intent(in) :: a
    real(WP) :: c
    c = 2.0 * a
  end function scaled
end module mo_calc
"""
    out = inline_to_single_tu({"mo_calc.F90": src},
                              entry="mo_calc::scaled",
                              out_dir=tmp_path / "o",
                              name="calc",
                              expand_cpp=True,
                              include_dirs=[tmp_path])
    text = out.read_text().lower().replace(" ", "")
    assert "scaled" in text
    assert "#include" not in text
    # WP -> 8 from the cpp macro; the inliner canonicalises real(8) to
    # real(kind=8).
    assert "real(kind=8)" in text or "real(8)" in text


# ---------------------------------------------------------------------------
# NAMELIST -- a supported construct (its read is an I/O node, a namelist
# variable is an ordinary variable).  Both merge engines must handle a
# namelist-bearing module; the fparser engine additionally prunes a
# namelist's dropped variables consistently (ICON's mo_ocean_nml declares
# hundreds of config variables across ~15 namelist groups, of which a kernel
# uses only a few).
# ---------------------------------------------------------------------------

_NAMELIST_MODULE = """
module mo_cfg
  implicit none
  integer :: n_zlev, ab_beta, coriolis_type
  namelist /ocean_dynamics_nml/ ab_beta, coriolis_type, n_zlev
  namelist /unused_nml/ ab_beta, coriolis_type
contains
  pure function uses_one(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x + n_zlev
  end function uses_one
end module mo_cfg
"""


def test_namelist_fparser_prunes_consistently():
    """The fparser engine keeps the namelist for the variable the kernel uses
    (``n_zlev``), prunes the unused namelist variables from both their
    declaration and the namelist object list, and drops an all-pruned group --
    so the TU never names an undeclared variable."""
    txt = inline_to_ast({
        "mo_cfg.f90": _NAMELIST_MODULE
    }, entry="mo_cfg::uses_one").tofortran().lower().replace("  ", " ")
    assert "n_zlev" in txt
    assert "namelist /ocean_dynamics_nml/ n_zlev" in txt
    assert "ab_beta" not in txt, "unused namelist variable should be pruned"
    assert "coriolis_type" not in txt
    assert "unused_nml" not in txt, "an all-pruned namelist group should be dropped"


@pytest.mark.skipif(not _have_gfortran(), reason="gfortran not on PATH")
def test_namelist_both_engines_compile(tmp_path):
    """Both merge engines turn a namelist-bearing module into compilable
    Fortran: the regex engine keeps the whole module (every variable still
    declared); the fparser engine keeps a consistent, pruned namelist."""
    # regex engine: whole-module passthrough keeps namelist + every variable.
    merged = merge_used_modules(_NAMELIST_MODULE)
    assert "namelist" in merged.lower()
    assert _gfortran_compiles(merged), "regex-merged namelist module must compile"

    # fparser engine: pruned, but the surviving namelist references only
    # declared variables, so it still compiles.
    tu = inline_to_single_tu({"mo_cfg.f90": _NAMELIST_MODULE}, entry="mo_cfg::uses_one", out_dir=tmp_path, name="cfg")
    assert _gfortran_compiles(tu.read_text()), "fparser-pruned namelist TU must compile"
