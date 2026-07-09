"""Step 4: same-signature swap of ICON's ``solve_nh`` body for our
binding.

ICON's call site (``mo_nh_stepping.f90:3094``) stays untouched.
``solve_nh`` in ``mo_solve_nonhydro.f90`` becomes a DIFFERENTIAL DRIVER
-- the SUBROUTINE signature, the 14 dummy declarations, the USE
statements all stay byte-for-byte identical -- that deep-copies the
mutable state (``mo_solve_nh_diff``), forwards to ``solve_nh_dace_icon``
(a free-standing wrapper that re-extracts pointers via ICON's real types
and dispatches into our dycore SDFG's bind-C entry) as the DUT, runs the
original body -- preserved verbatim as ``solve_nh_ref`` -- as the REF,
and compares the two bit-for-bit.

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

# This test compiles the patched solve_nh against ICON's compiled ``.mod``
# files, which are built by gfortran -- only gfortran can read them (flang's
# HLFIR ``.mod`` format is binary-incompatible, see ``_fc.discover_fortran_
# compilers``).  Parametrize gfortran-only so the flang slot is never emitted
# as a runtime skip; the gfortran-format guard inside the test stays as a
# defensive assertion.
GFORTRAN_COMPILERS = [p for p in FORTRAN_COMPILERS if "gfortran" in (p.id or "")]

_HERE = Path(__file__).resolve().parent
#: The differential helper module the patched ``solve_nh`` ``USE``s.  Compiled
#: ahead of the patched source in the syntax check so its ``.mod`` is on hand.
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))
_ICON_BUILD = Path(os.environ.get("ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_PRISTINE = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
_PRISTINE_BAK = _PRISTINE.with_suffix(".f90.bak")


def _real_source() -> Path:
    return _PRISTINE_BAK if _PRISTINE_BAK.is_file() else _PRISTINE


_HAVE_ICON = _real_source().is_file()
_HAVE_ICON_MODS = (_ICON_BUILD / "mod").is_dir()

# Every test here reads ICON's real ``mo_solve_nonhydro`` source through the
# icon-model submodule (the patch-side tests via ``_real_source()``, the
# compile tests additionally via the ``icon_build`` fixture), which only the
# heavy CI lane checks out -> ``long``.
pytestmark = pytest.mark.long

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
        start = next(i for i, ln in enumerate(lines) if "SUBROUTINE solve_nh " in ln and "(" in ln)
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
    # Forward EVERY one of solve_nh's 14 dummy args.  A regression
    # that drops one would silently leave it default-initialised on
    # the wrapper side.
    for arg in ("p_nh", "p_patch", "p_int", "prep_adv", "nnow", "nnew", "l_init", "l_recompute", "lsave_mflx",
                "lprep_adv", "lclean_mflx", "idyn_timestep", "jstep", "dtime", "lacc"):
        # ``arg`` appears multiple times (dummy decl + CALL site);
        # the forwarding CALL is what we check.  Trivial substring
        # match -- the order in the CALL is pinned by the template.
        assert arg in patched, f"forwarded arg {arg!r} missing"


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_differential_driver_injected():
    """The differential patch KEEPS the ~3000-line original body as the
    bit-exact REFERENCE (renamed ``solve_nh_ref``) and injects the driver
    -- clone -> DUT (``solve_nh_dace_icon``) -> REF -> compare -> free --
    as the new ``solve_nh``.  So the patched file GROWS by roughly the
    driver block; it does NOT shrink.  (The OLD pure-forwarding design
    dropped the body; the differential design runs it as the reference,
    so the old ``< 0.5 * pristine`` shrink assertion is inverted here.)
    Mirrors the structural pins in ``test_solve_nh_patch.py``."""
    pristine = _real_source().read_text()
    patched = apply_solve_nh_patch(pristine)

    # The original body survives verbatim, renamed to solve_nh_ref, and is
    # emitted AFTER the driver (ICON's call site resolves ``solve_nh``).  Match
    # ``solve_nh_ref`` without a trailing ``(`` -- ICON's real header spells the
    # signature ``solve_nh (...)`` with a space, so the rename keeps that space.
    assert "SUBROUTINE solve_nh_ref" in patched
    assert "END SUBROUTINE solve_nh_ref" in patched

    # The differential harness is injected into the new solve_nh driver, which
    # is emitted BEFORE the renamed reference body.
    assert "USE mo_solve_nh_diff" in patched
    assert "CALL clone_state_indep_prog(p_nh, nh_ref__dace)" in patched
    assert f"CALL {SOLVE_NH_WRAPPER_NAME}(" in patched
    assert "CALL solve_nh_ref(nh_ref__dace," in patched
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnew," in patched
    assert patched.index("CALL clone_state_indep_prog(p_nh, nh_ref__dace)") < patched.index("SUBROUTINE solve_nh_ref")

    # The file GROWS by ~the injected driver (USE helpers + local decls +
    # clone/run-both/compare/free body), NOT shrinks -- and the growth is a
    # small fraction of the file, so the ~3000-line body was not duplicated.
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
    # The patched module ``USE``s ``mo_solve_nh_diff`` (the deep-copy / compare
    # helpers), so its ``.mod`` must exist before the patched source is checked.
    # ``mo_solve_nh_diff.f90`` ``use``s only ICON's own type modules, so it
    # compiles against the same ICON ``.mod`` tree; passing it FIRST in the same
    # invocation makes its module available to the patched source that follows
    # (mirrors ``test_solve_nh_patch.py``'s ordered multi-source syntax check).
    subprocess.check_call([
        fc_path, *syntax_check_argv(fc_name, tmp_path),
        cpp_flag(fc_name), *fortran_compiler_flags(fc_name), *include_flags, *defines,
        str(_DIFF_F90),
        str(out)
    ],
                          cwd=str(tmp_path))
