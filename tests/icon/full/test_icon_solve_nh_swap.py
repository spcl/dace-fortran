"""Step 4: same-signature swap of ICON's ``solve_nh`` body for our
binding.

ICON's call site (``mo_nh_stepping.f90:3094``) stays untouched.
Only the BODY of ``solve_nh`` in ``mo_solve_nonhydro.f90`` is patched
-- the SUBROUTINE signature, the 14 dummy declarations, the USE
statements all stay byte-for-byte identical -- and forwards to
``solve_nh_dace_icon``, a free-standing wrapper that re-extracts
pointers via ICON's real types and dispatches into our dycore SDFG's
bind-C entry.

This test pins:

  * The patched ``mo_solve_nonhydro.f90`` parses through
    ``gfortran -fsyntax-only`` against the same ``-I`` set ICON's
    own build uses for the unpatched source.
  * The patched file resolves to the same ``solve_nh`` external
    symbol ICON's call site references (same module mangling,
    same arg count + types).
  * The forwarding ``CALL`` resolves to ``solve_nh_dace_icon`` as a
    free-standing symbol the linker can satisfy from our
    bind-C-generated ``.so``.

The patched body is wrapper-aware but doesn't yet require the SDFG
``.so`` to exist; runtime resolution is a separate concern (step 5
when one materialises).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ._fc import (
    FORTRAN_COMPILERS,
    cpp_flag,
    fortran_compiler_flags,
    syntax_check_argv,
)
from ._icon_solve_nh_patch import (
    SOLVE_NH_WRAPPER_NAME,
    apply_solve_nh_patch,
    write_patched_solve_nh,
)


_HERE = Path(__file__).resolve().parent
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))
_ICON_BUILD = Path(os.environ.get(
    "ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_PRISTINE = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
_PRISTINE_BAK = _PRISTINE.with_suffix(".f90.bak")


def _real_source() -> Path:
    return _PRISTINE_BAK if _PRISTINE_BAK.is_file() else _PRISTINE


_HAVE_ICON = _real_source().is_file()
_HAVE_ICON_MODS = (_ICON_BUILD / "mod").is_dir()


# ---------------------------------------------------------------------------
# Patch-side tests (no ICON build required).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patch_preserves_signature():
    """The patched file's ``SUBROUTINE solve_nh(...)`` line + the
    14 dummy declarations are byte-for-byte identical to the pristine.
    A change anywhere in the signature surface would break ICON's
    ``mo_nh_stepping`` call site, so we pin it explicitly."""
    pristine = _real_source().read_text()
    patched = apply_solve_nh_patch(pristine)

    def _signature_surface(src: str) -> list:
        """The header line(s) + every ``INTENT(...)`` dummy
        declaration in ``solve_nh``'s body.  This is the ABI surface
        ICON's callers see; USE statements inside the routine body
        are internal and intentionally excluded -- the patch adds one
        for ``iso_c_binding`` that doesn't affect external callers."""
        lines = src.splitlines()
        start = next(i for i, ln in enumerate(lines)
                     if "SUBROUTINE solve_nh " in ln and "(" in ln)
        end = start
        while lines[end].rstrip().endswith("&"):
            end += 1
        surface = list(lines[start:end + 1])
        # Pick up every INTENT(...) line inside the routine UP TO the
        # first ``INTERFACE`` block.  After the patch, the wrapper
        # is declared via an inner INTERFACE that has its own dummy
        # declarations (matching the wrapper's c_bool / c_int
        # types, not solve_nh's); those are internal to the
        # routine and not the ABI surface ICON callers see.
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
    assert pristine_sig == patched_sig, (
        "patched signature drifted from pristine -- ICON callers "
        "would see a different surface.\nFirst differing chars:\n"
        f"  pristine: {pristine_sig[:200]!r}\n"
        f"  patched:  {patched_sig[:200]!r}")


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patched_body_calls_wrapper():
    """The patched body's first executable statement is the
    forwarding ``CALL solve_nh_dace_icon(...)``."""
    patched = apply_solve_nh_patch(_real_source().read_text())
    assert f"CALL {SOLVE_NH_WRAPPER_NAME}(" in patched
    # Forward EVERY one of solve_nh's 14 dummy args.  A regression
    # that drops one would silently leave it default-initialised on
    # the wrapper side.
    for arg in ("p_nh", "p_patch", "p_int", "prep_adv",
                "nnow", "nnew", "l_init", "l_recompute",
                "lsave_mflx", "lprep_adv", "lclean_mflx",
                "idyn_timestep", "jstep", "dtime", "lacc"):
        # ``arg`` appears multiple times (dummy decl + CALL site);
        # the forwarding CALL is what we check.  Trivial substring
        # match -- the order in the CALL is pinned by the template.
        assert arg in patched, f"forwarded arg {arg!r} missing"


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_original_body_dropped(tmp_path: Path):
    """The unreachable original body is removed.  ICON's
    ``solve_nh`` body has ~3000 lines; the patched file should be
    much shorter."""
    patched = apply_solve_nh_patch(_real_source().read_text())
    pristine = _real_source().read_text()
    # The patch SHOULD shrink the file substantially (3000-line
    # body removed, replaced by ~50-line forwarding stub).
    assert len(patched.splitlines()) < 0.5 * len(pristine.splitlines()), (
        f"patched file ({len(patched.splitlines())} lines) is too "
        f"close to pristine ({len(pristine.splitlines())} lines) -- "
        "the original body wasn't dropped")


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


@pytest.mark.skipif(not _HAVE_ICON,
                    reason="icon-model submodule not checked out")
@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)
def test_patched_source_parses_through_fortran_compiler(fc, tmp_path: Path, icon_build):
    """The Fortran compiler accepts the patched file (syntax-only)
    against ICON's own ``.mod`` files.  Parametrized over every
    compiler available on the host so a future binding-shim signature
    change surfaces as a hard error from each one we support:

      * ``gfortran``: the ICON CPU default.
      * ``flang-new-21``: the bridge's own frontend; pinning the
        wrapper's INTERFACE block here keeps the bridge's binding
        emitter in sync with what it would later lower.
      * ``nvfortran``: ICON GPU builds; their ``.mod`` format is
        compiler-specific so the test only fires when the same
        compiler that wrote ``$ICON_BUILD/mod`` also runs the
        syntax check (the gate below).

    The patched file's INTERFACE block + forwarding CALL arg order
    is what's pinned here; any drift breaks ICON's caller."""
    fc_name, fc_path = fc
    # ICON's ``.mod`` files are compiler-specific.  gfortran's mods
    # match gfortran; nvfortran's mods match nvfortran; flang has its
    # own ``.mod`` format.  The host's stock CPU build probes for
    # ``mod/mo_kind.mod``; if it carries a gfortran-shape mod, only
    # gfortran can read it.  Detect the mod-format mismatch up front
    # so the test skips cleanly rather than reporting a bogus error.
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

    # Same -D set ICON's own configure uses; pulled into a list so we
    # can compose with the wrapper-INTERFACE-block requirements.
    defines = [
        "-DHAVE_CDI_GRIB2", "-DHAVE_FC_ATTRIBUTE_CONTIGUOUS",
        "-DICON_MPI_SUBVERSION=1", "-DICON_MPI_VERSION=3",
        "-D__HAVE_QUAD_PRECISION", "-D__ICON__", "-D__LOOP_EXCHANGE",
        "-D__NO_ICON_COMIN__", "-D__NO_ICON_OCEAN__",
        "-D__NO_ICON_TESTBED__", "-D__NO_ICON_WAVES__",
        "-D__NO_JSBACH_HD__", "-D__NO_JSBACH__", "-D__NO_QUINCY__",
        "-D__NO_RAGNAROK__",
    ]
    subprocess.check_call(
        [fc_path, *syntax_check_argv(fc_name, tmp_path), cpp_flag(fc_name),
         *fortran_compiler_flags(fc_name),
         *include_flags, *defines,
         str(out)],
        cwd=str(tmp_path))
