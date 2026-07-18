# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests two robustness features of the fparser single-TU inliner needed to ingest
*real* ICON sources (not just cpp-clean fixtures):

1. ``expand_cpp=True`` -- runs the C preprocessor first so ``#include``/``#define``
   (ICON's DSL macro headers) expand to pure Fortran; without it fparser raises on
   the first ``#include``.
2. ``tolerate_external_uses=True`` -- doesn't hard-fail on a ``USE`` of an external
   library with no source on the search path (netcdf/mpi/cdi); the import is left
   unresolved and reachability pruning drops procedures that referenced it.
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
    """Module USEs netcdf but the entry never calls it: with tolerance on, the
    netcdf-touching procedure (and dangling import) are pruned, leaving a self-contained TU."""
    out = inline_to_ast({
        "mo_thing.f90": _NETCDF_USER
    }, entry="mo_thing::kernel_add", tolerate_external_uses=True).tofortran().lower()
    assert "kernel_add" in out
    assert "writes_netcdf" not in out, "the unused netcdf procedure should be pruned"
    assert "nf90_" not in out, "the external netcdf calls should be gone"
    assert "use netcdf" not in out, "the dangling external USE should be pruned"


def test_external_use_default_strict_still_asserts():
    """Default (tolerance off): an unresolved external USE is a hard error."""
    with pytest.raises(AssertionError):
        inline_to_ast({"mo_thing.f90": _NETCDF_USER}, entry="mo_thing::kernel_add")


def test_external_use_reached_does_not_crash():
    """Entry reaches the external call: tolerance keeps it rather than crashing
    (survives as unresolved external)."""
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
    """``cpp_expand_sources`` expands a cpp #include + #define macro into pure
    Fortran (no # directives left)."""
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
    """End-to-end: a source with a cpp #include inlines once expand_cpp=True
    resolves it (would raise otherwise)."""
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
    # WP->8 from the cpp macro; inliner canonicalises real(8) to real(kind=8)
    assert "real(kind=8)" in text or "real(8)" in text


# ---------------------------------------------------------------------------
# NAMELIST: a namelist read is an I/O node, its variables ordinary variables.
# Both merge engines must handle it; fparser additionally prunes dropped
# variables consistently (ICON's mo_ocean_nml: ~15 groups, hundreds of vars,
# a kernel uses only a few).
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
    """fparser keeps the namelist for the used variable (n_zlev), prunes unused
    variables from both declaration and namelist list, and drops an all-pruned group."""
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
    """Both engines produce compilable Fortran: regex keeps the whole module
    (every var declared); fparser keeps a consistent, pruned namelist."""
    # regex engine: whole-module passthrough keeps namelist + every variable.
    merged = merge_used_modules(_NAMELIST_MODULE)
    assert "namelist" in merged.lower()
    assert _gfortran_compiles(merged), "regex-merged namelist module must compile"

    # fparser engine: pruned, but the surviving namelist references only declared vars, so it still compiles
    tu = inline_to_single_tu({"mo_cfg.f90": _NAMELIST_MODULE}, entry="mo_cfg::uses_one", out_dir=tmp_path, name="cfg")
    assert _gfortran_compiles(tu.read_text()), "fparser-pruned namelist TU must compile"
