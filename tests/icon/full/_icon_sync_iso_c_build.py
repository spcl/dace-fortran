"""Build helper for ``icon_sync_iso_c.f90``.

Compiles the iso_c wrapper into ``libicon_sync_iso_c.so`` against ICON's
own ``.mod`` files (``mo_kind`` / ``mo_model_domain`` / ``mo_sync``).
Returns the ``.so`` path on success.  Used by
``test_dycore_from_icon_source.py`` to wire the wrapper into the
``keep_external`` registrations for the dycore SDFG.

A successful build needs ICON to have been configured + ``make``-d so
that its ``mod/`` directories under ``$ICON_BUILD`` exist.  When ICON's
build dir is missing, this returns ``None`` (and the test falls back
to the ``stub=True`` path that still proves the wrapper SOURCE compiles
through gfortran's parser standalone -- the per-procedure bind(c)
interfaces are pinned regardless of whether ICON has produced ``.mod``
files for them yet).
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_WRAPPER_SRC = _HERE / "icon_sync_iso_c.f90"


def build_icon_sync_iso_c_so(
    icon_build: Path,
    out_dir: Path,
    *,
    fc: Optional[str] = None,
) -> Optional[Path]:
    """Compile and link the wrapper into ``out_dir / libicon_sync_iso_c.so``.

    :param icon_build: ICON's CPU build directory (with ``mod/``
        and ``externals/*/build/.../mod/`` populated by a prior
        ``make``).  Pass ``None`` to skip the link step.
    :param out_dir: directory to write the ``.o`` and ``.so`` into.
    :param fc: Fortran compiler binary.  Defaults to ``$FC`` from
        the environment, then to ``gfortran`` -- pass an explicit
        ``nvfortran`` / ``flang-new-21`` / ``mpifort`` etc. when the
        target binding wants the wrapper bound to ICON's GPU build
        or a non-default toolchain.
    :returns: the ``.so`` path on success, ``None`` if the ICON build
        isn't there to satisfy the ``USE mo_sync`` /
        ``USE mo_model_domain`` ``.mod`` lookups, or the requested
        compiler isn't on ``PATH``.
    :raises subprocess.CalledProcessError: the compiler failed despite
        the ``.mod`` files being where they should be.
    """
    if fc is None:
        fc = os.environ.get("FC", "gfortran")
    if shutil.which(fc) is None or not icon_build.is_dir():
        return None
    mod_dirs = [
        icon_build / "mod",
        icon_build / "externals/fortran-support/build/src/mod",
        icon_build / "externals/iconmath/build/src/support/mod",
        icon_build / "externals/iconmath/build/src/horizontal/mod",
        icon_build / "externals/iconmath/build/src/interpolation/mod",
        icon_build / "externals/memman/build/_icon/src/bindings/fortran/mod",
        icon_build / "externals/mtime/build/src/mod",
    ]
    # Some of those dirs only exist after a full make; filter to ones
    # that DO so the compiler's -I list isn't littered with
    # non-existent paths (gfortran / nvfortran both warn on those).
    mod_dirs = [d for d in mod_dirs if d.is_dir()]
    if not (icon_build / "mod").is_dir():
        # No ICON-side ``mo_sync.mod`` to USE -- can't resolve the
        # wrapper's USE imports.  Caller decides what to do.
        return None
    # Resolve ``fc`` and bail if it isn't on PATH (e.g. user requested
    # nvfortran on a host without NVHPC).  Same shape as the
    # ``no .mod files`` skip.
    if shutil.which(fc) is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "icon_sync_iso_c.o"
    so_path = out_dir / "libicon_sync_iso_c.so"

    # Per-compiler flag set.  ``-fno-fast-math`` / ``-ffp-contract=off``
    # are GCC-family; nvfortran and flang ignore them or reject them,
    # so route the FP-conservative bit through compiler-specific flags.
    fc_basename = Path(fc).name.lower()
    if "nvfortran" in fc_basename:
        base_flags = ["-O0", "-fpic", "-Kieee", "-Mnofma"]
    elif "flang" in fc_basename:
        base_flags = ["-O0", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none"]
    else:  # gfortran (+ mpifort wrapping it)
        base_flags = ["-O0", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none"]
    include_flags = [f"-I{d}" for d in mod_dirs]
    # Compile
    subprocess.check_call(
        [fc, *base_flags, *include_flags, "-c",
         str(_WRAPPER_SRC), "-o", str(obj_path)], cwd=str(out_dir))
    # Link (-shared is supported by all three).
    link_pic = "-fpic" if "nvfortran" in fc_basename else "-fPIC"
    subprocess.check_call([fc, "-shared", link_pic, str(obj_path), "-o", str(so_path)], cwd=str(out_dir))
    return so_path
