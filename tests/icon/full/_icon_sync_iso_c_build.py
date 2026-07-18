"""Compiles ``icon_sync_iso_c.f90`` into ``libicon_sync_iso_c.so`` against ICON's ``.mod`` files.
Returns ``None`` if the ICON build is missing; callers fall back to a ``stub=True`` path."""
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
    """Compile and link the wrapper into ``out_dir/libicon_sync_iso_c.so`` against
    ``icon_build``'s ``.mod`` files. ``fc`` defaults to ``$FC``/gfortran; pass an explicit
    compiler to target ICON's GPU build or a non-default toolchain. Returns ``None`` if the
    ICON build or compiler isn't available; raises if the compiler fails despite the
    ``.mod`` files being present."""
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
    # Some dirs only exist after a full make; filter to existing ones so -I isn't
    # littered with non-existent paths (gfortran/nvfortran both warn).
    mod_dirs = [d for d in mod_dirs if d.is_dir()]
    if not (icon_build / "mod").is_dir():
        # No ICON-side mo_sync.mod -- can't resolve the wrapper's USE imports.
        return None
    # Bail if fc isn't on PATH (e.g. nvfortran requested on a host without NVHPC).
    if shutil.which(fc) is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "icon_sync_iso_c.o"
    so_path = out_dir / "libicon_sync_iso_c.so"

    # Per-compiler flag set: -fno-fast-math/-ffp-contract=off are GCC-family; nvfortran and
    # flang ignore or reject them, so route FP-conservative flags per compiler.
    fc_basename = Path(fc).name.lower()
    if "nvfortran" in fc_basename:
        base_flags = ["-O0", "-fpic", "-Kieee", "-Mnofma"]
    elif "flang" in fc_basename:
        base_flags = ["-O0", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none"]
    else:  # gfortran (+ mpifort wrapping it)
        base_flags = ["-O0", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none"]
    include_flags = [f"-I{d}" for d in mod_dirs]
    subprocess.check_call(
        [fc, *base_flags, *include_flags, "-c",
         str(_WRAPPER_SRC), "-o", str(obj_path)], cwd=str(out_dir))
    # Link (-shared is supported by all three).
    link_pic = "-fpic" if "nvfortran" in fc_basename else "-fPIC"
    subprocess.check_call([fc, "-shared", link_pic, str(obj_path), "-o", str(so_path)], cwd=str(out_dir))
    return so_path
