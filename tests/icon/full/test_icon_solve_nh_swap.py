"""Step 4: same-signature swap of ICON's ``solve_nh`` for our binding.

``solve_nh`` in ``mo_solve_nonhydro.f90`` becomes a DIFFERENTIAL DRIVER: signature/
dummies/USE stay byte-identical, but the body deep-copies state, runs the DUT
(``solve_nh_dace_icon``, our bind-C dycore) against the REF (original body, renamed
``solve_nh_ref``), and compares bit-for-bit.  ICON's call site is untouched.

Pins: the patched file parses via ``gfortran -fsyntax-only`` against ICON's own
``-I`` set; resolves to the same external ``solve_nh`` symbol; the forwarding CALL
resolves to a linker-satisfiable ``solve_nh_dace_icon``.  Doesn't yet require the
SDFG ``.so`` to exist (runtime resolution is step 5).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from icon.full._fc import (
    FORTRAN_COMPILERS,
    cpp_flag,
    fortran_compiler_flags,
    syntax_check_argv,
)
from icon.full._icon_solve_nh_patch import (
    SOLVE_NH_WRAPPER_NAME,
    apply_solve_nh_patch,
    write_patched_solve_nh,
)

# ICON's .mod files are gfortran-built and binary-incompatible with flang's HLFIR
# .mod format (see _fc.discover_fortran_compilers).  Parametrize gfortran-only so
# the flang slot never shows up as a runtime skip; the format guard below stays defensive.
GFORTRAN_COMPILERS = [p for p in FORTRAN_COMPILERS if "gfortran" in (p.id or "")]

_HERE = Path(__file__).resolve().parent
#: Differential helper module the patched ``solve_nh`` ``USE``s; compiled first so its .mod is available.
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))
_ICON_BUILD = Path(os.environ.get("ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_PRISTINE = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
_PRISTINE_BAK = _PRISTINE.with_suffix(".f90.bak")


def _real_source() -> Path:
    return _PRISTINE_BAK if _PRISTINE_BAK.is_file() else _PRISTINE


_HAVE_ICON = _real_source().is_file()
_HAVE_ICON_MODS = (_ICON_BUILD / "mod").is_dir()

# Reads ICON's real mo_solve_nonhydro via the icon-model submodule (heavy CI lane only) -> long.
pytestmark = pytest.mark.long

# ---------------------------------------------------------------------------
# Patch-side tests (no ICON build required).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patch_preserves_signature():
    """The patched file's signature line + 14 dummy declarations are byte-identical
    to the pristine -- any drift would break ICON's mo_nh_stepping call site."""
    pristine = _real_source().read_text()
    patched = apply_solve_nh_patch(pristine)

    def _signature_surface(src: str) -> list:
        """Header line(s) + every INTENT(...) dummy decl -- the ABI surface ICON's
        callers see.  USE statements are excluded (internal; the patch adds one
        for iso_c_binding that doesn't affect callers)."""
        lines = src.splitlines()
        start = next(i for i, ln in enumerate(lines) if "SUBROUTINE solve_nh " in ln and "(" in ln)
        end = start
        while lines[end].rstrip().endswith("&"):
            end += 1
        surface = list(lines[start:end + 1])
        # Stop at the first INTERFACE block: the patch's wrapper interface has its
        # own dummy decls (c_bool/c_int types, not solve_nh's) -- internal, not ABI.
        for i in range(end + 1, len(lines)):
            stripped = lines[i].lstrip().upper()
            if stripped.startswith("END SUBROUTINE SOLVE_NH"):
                break
            if stripped.startswith("INTERFACE"):
                break
            if "INTENT" in stripped:
                surface.append(lines[i])
        return surface

    pristine_sig = "\n".join(_signature_surface(pristine))
    patched_sig = "\n".join(_signature_surface(patched))
    assert pristine_sig == patched_sig, ("patched signature drifted from pristine -- ICON callers "
                                         "would see a different surface.\nFirst differing chars:\n"
                                         f"  pristine: {pristine_sig[:200]!r}\n"
                                         f"  patched:  {patched_sig[:200]!r}")


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patched_body_calls_wrapper():
    """The patched body's first executable statement is the
    forwarding ``CALL solve_nh_dace_icon(...)``."""
    patched = apply_solve_nh_patch(_real_source().read_text())
    assert f"CALL {SOLVE_NH_WRAPPER_NAME}(" in patched
    # Forward every one of the 14 dummy args -- a dropped one silently defaults on the wrapper side.
    for arg in ("p_nh", "p_patch", "p_int", "prep_adv", "nnow", "nnew", "l_init", "l_recompute", "lsave_mflx",
                "lprep_adv", "lclean_mflx", "idyn_timestep", "jstep", "dtime", "lacc"):
        # substring match only; arg order in the CALL is pinned by the template
        assert arg in patched, f"forwarded arg {arg!r} missing"


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_differential_driver_injected():
    """The patch keeps the original body as REF (renamed ``solve_nh_ref``) and
    injects the driver (clone -> DUT -> REF -> compare -> free) as the new
    ``solve_nh`` -- the file GROWS, it does not shrink (inverts the old pure-
    forwarding design's shrink assertion).  Mirrors test_solve_nh_patch.py."""
    pristine = _real_source().read_text()
    patched = apply_solve_nh_patch(pristine)

    # Original body survives verbatim as solve_nh_ref, emitted after the driver.
    # No trailing "(" match -- ICON's header spells "solve_nh (...)" with a space.
    assert "SUBROUTINE solve_nh_ref" in patched
    assert "END SUBROUTINE solve_nh_ref" in patched

    # Differential harness is injected into the new driver, emitted before the REF body.
    assert "USE mo_solve_nh_diff" in patched
    assert "CALL clone_state_indep_prog(p_nh, nh_ref__dace)" in patched
    assert "CALL clone_prepadv_indep(prep_adv, prep_ref__dace)" in patched
    assert f"CALL {SOLVE_NH_WRAPPER_NAME}(" in patched
    assert "CALL solve_nh_ref(nh_ref__dace," in patched
    # Compare covers prog(nnew) AND prog(nnow) -- REF never assigns nnow, so a nnow
    # diff means the DUT stomped the level the next substep reads -- plus prep_adv/
    # diag, closing with the greppable TOTAL line (0 == bit-exact).
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnew," in patched
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnow," in patched
    assert "CALL compare_prepadv(prep_adv, prep_ref__dace," in patched
    assert "CALL compare_diag(p_nh % diag, nh_ref__dace % diag," in patched
    assert "[diff] solve_nh TOTAL: " in patched
    assert patched.index("CALL clone_state_indep_prog(p_nh, nh_ref__dace)") < patched.index("SUBROUTINE solve_nh_ref")

    # File grows by ~the injected driver, not shrinks; growth stays a small
    # fraction of the file so the ~3000-line body was not duplicated.
    pristine_n = len(pristine.splitlines())
    patched_n = len(patched.splitlines())
    assert patched_n > pristine_n, ("the differential patch keeps the body as the REF, so the file must "
                                    f"GROW: patched={patched_n} pristine={pristine_n}")
    growth = patched_n - pristine_n
    assert growth < pristine_n // 2, (f"patched grew by {growth} lines -- far more than the injected driver; "
                                      "the original body may have been duplicated instead of renamed")


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_write_patched_solve_nh(tmp_path: Path):
    """Round-trip the patch through the disk-writing helper."""
    out = tmp_path / "mo_solve_nonhydro.f90"
    line_count = write_patched_solve_nh(_real_source(), out)
    assert out.is_file()
    assert line_count > 200, f"patched file looks truncated ({line_count} lines)"


# ---------------------------------------------------------------------------
# Compile-side test (needs ICON's .mod files).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
@pytest.mark.parametrize("fc", GFORTRAN_COMPILERS)
def test_patched_source_parses_through_fortran_compiler(fc, tmp_path: Path, icon_build):
    """The compiler accepts the patched file (syntax-only) against ICON's own .mod
    files -- parametrized over gfortran (ICON's CPU default; other compilers' .mod
    formats are incompatible, see GFORTRAN_COMPILERS above).  Pins the INTERFACE
    block + forwarding CALL arg order; any drift breaks ICON's caller."""
    fc_name, fc_path = fc
    # .mod files are compiler-specific; probe mo_kind.mod's format up front so a
    # mismatch skips cleanly instead of reporting a bogus error.
    mod_probe = icon_build / "mod" / "mo_kind.mod"
    if mod_probe.is_file():
        header = mod_probe.read_bytes()[:64]
        is_gfortran_mod = header.startswith(b"\x1f\x8b") or b"GFORTRAN module" in header
        if is_gfortran_mod and fc_name != "gfortran":
            pytest.skip(f"$ICON_BUILD/mod carries gfortran-format .mod; "
                        f"{fc_name} can't read those")

    out = tmp_path / "mo_solve_nonhydro_patched.f90"
    write_patched_solve_nh(_real_source(), out)

    mod_dirs = [
        icon_build / "mod",
        icon_build / "externals/fortran-support/build/src/mod",
        icon_build / "externals/iconmath/build/src/support/mod",
        icon_build / "externals/iconmath/build/src/horizontal/mod",
        icon_build / "externals/iconmath/build/src/interpolation/mod",
        icon_build / "externals/memman/build/_icon/src/bindings/fortran/mod",
        icon_build / "externals/mtime/build/src/mod",
    ]
    include_flags = [f"-I{d}" for d in mod_dirs if d.is_dir()]
    include_flags.append(f"-I{_ICON_SRC}/src/include")

    # Same -D set ICON's own configure uses.
    defines = [
        "-DHAVE_CDI_GRIB2",
        "-DHAVE_FC_ATTRIBUTE_CONTIGUOUS",
        "-DICON_MPI_SUBVERSION=1",
        "-DICON_MPI_VERSION=3",
        "-D__HAVE_QUAD_PRECISION",
        "-D__ICON__",
        "-D__LOOP_EXCHANGE",
        "-D__NO_ICON_COMIN__",
        "-D__NO_ICON_OCEAN__",
        "-D__NO_ICON_TESTBED__",
        "-D__NO_ICON_WAVES__",
        "-D__NO_JSBACH_HD__",
        "-D__NO_JSBACH__",
        "-D__NO_QUINCY__",
        "-D__NO_RAGNAROK__",
    ]
    # Patched module USEs mo_solve_nh_diff, so its .mod must exist first; passing it
    # before the patched source in one invocation makes that happen (mirrors
    # test_solve_nh_patch.py's ordered multi-source syntax check).
    subprocess.check_call([
        fc_path, *syntax_check_argv(fc_name, tmp_path),
        cpp_flag(fc_name), *fortran_compiler_flags(fc_name), *include_flags, *defines,
        str(_DIFF_F90),
        str(out)
    ],
                          cwd=str(tmp_path))
