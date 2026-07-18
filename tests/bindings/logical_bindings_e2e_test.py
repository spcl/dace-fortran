"""E2e LOGICAL -> logical(c_bool) bridge tests (build+compile+run, vs emit_bindings_test.py's string-match coverage).
Covers rank-1/2/3 default LOGICAL, LOGICAL(KIND=1/4/8), c_bool pass-through, and scalar LOGICAL."""

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
    pytest.mark.skipif(shutil.which("meson") is None, reason="meson not available (f2py)"),
]


def _module_wrap(free_subroutine_src: str, module_name: str) -> str:
    """Wraps a free SUBROUTINE in a module so the SDFG build can use `module::proc` entry spelling; f2py ref path keeps the unwrapped source."""
    return f"module {module_name}\ncontains\n{free_subroutine_src}\nend module {module_name}\n"


def _build_e2e_module(
    tmp_path: Path,
    *,
    kernel_src: str,
    name: str,
    entry: str,
    outer_args: tuple,
    driver_src: str,
    module_name: str,
):
    """Build SDFG -> emit bindings -> f2py-compile bindings+driver linked to the SDFG .so; returns the loaded extension module."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(kernel_src, sdfg_dir, name=name, entry=entry).build()
    # DaCe mangles the cache-key SDFG name with the test path; force it back so bind(c) symbol
    # names in the .so match what the bindings emit.
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature
    iface = OriginalInterface(entry=name, args=outer_args)
    bindings_path = tmp_path / f"{name}_bindings.f90"
    emit_bindings(fs, iface, FlattenPlan(entries=()), str(bindings_path))

    # Driver + bindings -> Python extension via f2py; link the SDFG .so so bind(c) resolves at load time.
    driver_path = tmp_path / f"{name}_driver.f90"
    driver_path.write_text(driver_src)

    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    # f2py's meson backend rejects -Wl flags; workaround: CDLL(RTLD_GLOBAL)-preload the SDFG so its
    # symbols are visible, then compile with --unresolved-symbols=ignore-all so the linker allows
    # them to stay undefined until runtime.
    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        module_name,
        str(bindings_path),
        str(driver_path),
        "--f90flags=-Wl,--unresolved-symbols=ignore-all",
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"f2py compile failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    return __import__(module_name)


def _f2py_ref(tmp_path: Path, src: str, name: str):
    """Plain f2py reference module (no bridge) -- gfortran-built ground truth."""
    from _helpers import f2py
    return f2py(src, tmp_path / "ref", name)


# ---------------------------------------------------------------------------
# Rank-1 default LOGICAL: the cloudsc LDCUM / LLFALL pattern
# ---------------------------------------------------------------------------

_RANK1_KERNEL = """
SUBROUTINE flip_mask(mask, out, n)
integer, intent(in) :: n
logical, intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_mask
"""

_RANK1_DRIVER = """
module flip_mask_driver
  use iso_c_binding
  use flip_mask_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:)
    integer, intent(out) :: out(size(mask))
    call flip_mask_dace(mask, out, size(mask))
  end subroutine run
end module flip_mask_driver
"""


def test_e2e_rank1_default(tmp_path: Path):
    """LOGICAL, intent(in) :: mask(n) -- default kind rank 1; c_bool bridge widens np.bool_ to 4-byte default LOGICAL."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_module_wrap(_RANK1_KERNEL, "flip_mask_mod"),
        name="flip_mask",
        entry="flip_mask_mod::flip_mask",
        outer_args=outer,
        driver_src=_RANK1_DRIVER,
        module_name="flip_mask_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK1_KERNEL, "flip_mask_ref")

    mask_in = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    out_ref = ref.flip_mask(mask_in)
    # f2py auto-derives intent(out) shape into a return value -- run() returns out directly.
    out = mod.flip_mask_driver.run(mask_in)
    mod.flip_mask_dace_bindings.flip_mask_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Rank-2 default LOGICAL
# ---------------------------------------------------------------------------

_RANK2_KERNEL = """
SUBROUTINE flip_mask2(mask, out, m, n)
integer, intent(in) :: m, n
logical, intent(in) :: mask(m, n)
integer, intent(out) :: out(m, n)
integer :: i, j
DO j = 1, n
    DO i = 1, m
        IF (mask(i, j)) THEN
            out(i, j) = 1
        ELSE
            out(i, j) = 0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE flip_mask2
"""

_RANK2_DRIVER = """
module flip_mask2_driver
  use iso_c_binding
  use flip_mask2_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:, :)
    integer, intent(out) :: out(size(mask, 1), size(mask, 2))
    call flip_mask2_dace(mask, out, size(mask, 1), size(mask, 2))
  end subroutine run
end module flip_mask2_driver
"""


def test_e2e_rank2_default(tmp_path: Path):
    """``LOGICAL, intent(in) :: mask(m, n)`` -- 2D, default kind."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=2, shape=("m", "n"), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=2, shape=("m", "n"), intent="out"),
        OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_module_wrap(_RANK2_KERNEL, "flip_mask2_mod"),
        name="flip_mask2",
        entry="flip_mask2_mod::flip_mask2",
        outer_args=outer,
        driver_src=_RANK2_DRIVER,
        module_name="flip_mask2_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK2_KERNEL, "flip_mask2_ref")

    m, n = 3, 4
    rng = np.random.default_rng(7)
    mask_in = np.asfortranarray(rng.integers(0, 2, (m, n)).astype(np.bool_))
    out_ref = ref.flip_mask2(mask_in)
    out = mod.flip_mask2_driver.run(mask_in)
    mod.flip_mask2_dace_bindings.flip_mask2_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Rank-3 default LOGICAL
# ---------------------------------------------------------------------------

_RANK3_KERNEL = """
SUBROUTINE flip_mask3(mask, out, m, n, p)
integer, intent(in) :: m, n, p
logical, intent(in) :: mask(m, n, p)
integer, intent(out) :: out(m, n, p)
integer :: i, j, k
DO k = 1, p
    DO j = 1, n
        DO i = 1, m
            IF (mask(i, j, k)) THEN
                out(i, j, k) = 1
            ELSE
                out(i, j, k) = 0
            ENDIF
        ENDDO
    ENDDO
ENDDO
END SUBROUTINE flip_mask3
"""

_RANK3_DRIVER = """
module flip_mask3_driver
  use iso_c_binding
  use flip_mask3_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:, :, :)
    integer, intent(out) :: out(size(mask, 1), size(mask, 2), size(mask, 3))
    call flip_mask3_dace(mask, out, size(mask, 1), size(mask, 2), size(mask, 3))
  end subroutine run
end module flip_mask3_driver
"""


def test_e2e_rank3_default(tmp_path: Path):
    """``LOGICAL, intent(in) :: mask(m, n, p)`` -- 3D, default kind."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=3, shape=("m", "n", "p"), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=3, shape=("m", "n", "p"), intent="out"),
        OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="p", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_module_wrap(_RANK3_KERNEL, "flip_mask3_mod"),
        name="flip_mask3",
        entry="flip_mask3_mod::flip_mask3",
        outer_args=outer,
        driver_src=_RANK3_DRIVER,
        module_name="flip_mask3_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK3_KERNEL, "flip_mask3_ref")

    m, n, p = 2, 3, 4
    rng = np.random.default_rng(11)
    mask_in = np.asfortranarray(rng.integers(0, 2, (m, n, p)).astype(np.bool_))
    out_ref = ref.flip_mask3(mask_in)
    out = mod.flip_mask3_driver.run(mask_in)
    mod.flip_mask3_dace_bindings.flip_mask3_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Per-kind coverage: LOGICAL(1) / LOGICAL(4) / LOGICAL(8)
# ---------------------------------------------------------------------------


def _kind_kernel(kind: int) -> str:
    return f"""
SUBROUTINE flip_kind{kind}(mask, out, n)
integer, intent(in) :: n
logical(kind={kind}), intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_kind{kind}
"""


def _kind_driver(kind: int) -> str:
    return f"""
module flip_kind{kind}_driver
  use iso_c_binding
  use flip_kind{kind}_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical(kind={kind}), intent(in) :: mask(:)
    integer, intent(out) :: out(size(mask))
    call flip_kind{kind}_dace(mask, out, size(mask))
  end subroutine run
end module flip_kind{kind}_driver
"""


@pytest.mark.parametrize("kind", [1, 4, 8])
def test_e2e_rank1_logical_kind(tmp_path: Path, kind: int):
    """LOGICAL(KIND=1/4/8) rank-1 round-trip -- kind 1 matches c_bool size; all three bridge through the c_bool scratch."""
    src = _kind_kernel(kind)
    outer = (
        OriginalArg(name="mask", fortran_type=f"logical(kind={kind})", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=src,
        name=f"flip_kind{kind}",
        entry=f"_QPflip_kind{kind}",
        outer_args=outer,
        driver_src=_kind_driver(kind),
        module_name=f"flip_kind{kind}_e2e",
    )
    ref = _f2py_ref(tmp_path, src, f"flip_kind{kind}_ref")

    mask_in = np.array([True, False, True, False, True, False], dtype=np.bool_)
    out_ref = getattr(ref, f"flip_kind{kind}")(mask_in)
    out = getattr(mod, f"flip_kind{kind}_driver").run(mask_in)
    getattr(getattr(mod, f"flip_kind{kind}_dace_bindings"), f"flip_kind{kind}_dace_finalize")()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# logical(c_bool) outer type: pass-through (no scratch / no cast bridge)
# ---------------------------------------------------------------------------

_CBOOL_KERNEL = """
SUBROUTINE flip_cbool(mask, out, n)
use iso_c_binding, only: c_bool
integer, intent(in) :: n
logical(c_bool), intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_cbool
"""

_CBOOL_DRIVER = """
! C-callable driver that exercises the pass-through path: ``mask`` is
! ``logical(c_bool)`` matching the SDFG's bool ABI exactly, so the
! bindings wrapper allocates NO scratch and emits NO intrinsic cast.
! ``logical(c_bool)`` is not parseable by f2py (it emits ``unsigned_char``
! with underscore -- gcc rejects), so we expose this driver via plain
! ``bind(c)`` and invoke it with ctypes from Python.

subroutine run_cbool_passthrough(mask, out, n) bind(c, name='run_cbool_passthrough')
  use iso_c_binding
  use flip_cbool_dace_bindings
  implicit none
  integer(c_int), value :: n
  logical(c_bool), intent(in) :: mask(n)
  integer(c_int), intent(out) :: out(n)
  call flip_cbool_dace(mask, out, n)
  call flip_cbool_dace_finalize()
end subroutine run_cbool_passthrough
"""


def test_e2e_rank1_cbool_passthrough(tmp_path: Path):
    """logical(c_bool) matches the SDFG ABI -- pass-through, no scratch/cast bridge. f2py can't parse
    logical(c_bool) (emits an invalid `unsigned_char` cast), so this gfortran-compiles bindings+driver
    into one .so and calls via ctypes."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_module_wrap(_CBOOL_KERNEL, "flip_cbool_mod"),
                      sdfg_dir,
                      name="flip_cbool",
                      entry="flip_cbool_mod::flip_cbool").build()
    sdfg.name = "flip_cbool"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature

    outer = (
        OriginalArg(name="mask", fortran_type="logical(c_bool)", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    iface = OriginalInterface(entry=fs.entry, args=outer)
    bindings_path = tmp_path / "flip_cbool_bindings.f90"
    emit_bindings(fs, iface, FlattenPlan(entries=()), str(bindings_path))

    driver_path = tmp_path / "flip_cbool_driver.f90"
    driver_path.write_text(_CBOOL_DRIVER)

    # cwd=tmp_path avoids picking up a stale iso_c_binding.mod left by a prior flang run in the repo root.
    driver_so = tmp_path / "flip_cbool_driver.so"
    subprocess.check_call([
        "gfortran",
        "-shared",
        "-fPIC",
        str(bindings_path),
        str(driver_path),
        "-o",
        str(driver_so),
        f"-L{so_path.parent}",
        f"-Wl,-rpath,{so_path.parent}",
        f"-l:{so_path.name}",
    ],
                          cwd=str(tmp_path))

    lib = ctypes.CDLL(str(driver_so))
    lib.run_cbool_passthrough.argtypes = [
        ctypes.POINTER(ctypes.c_bool),  # mask
        ctypes.POINTER(ctypes.c_int),  # out
        ctypes.c_int,  # n (pass-by-value)
    ]
    lib.run_cbool_passthrough.restype = None

    mask_in = np.array([True, False, True, False, True, False], dtype=np.bool_)
    out = np.zeros(mask_in.size, dtype=np.int32)
    out_ref = mask_in.astype(np.int32)
    lib.run_cbool_passthrough(mask_in.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), mask_in.size)
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Scalar LOGICAL: the cloudsc LDMAINCALL / LDSLPHY pattern
# ---------------------------------------------------------------------------

_SCALAR_KERNEL = """
SUBROUTINE scalar_flag(flag, out, n)
integer, intent(in) :: n
logical, intent(in) :: flag
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (flag) THEN
        out(i) = i
    ELSE
        out(i) = -i
    ENDIF
ENDDO
END SUBROUTINE scalar_flag
"""

_SCALAR_DRIVER = """
module scalar_flag_driver
  use iso_c_binding
  use scalar_flag_dace_bindings
  implicit none
contains
  subroutine run(flag, out, n)
    integer, intent(in) :: n
    logical, intent(in) :: flag
    integer, intent(out) :: out(n)
    call scalar_flag_dace(flag, out, n)
  end subroutine run
end module scalar_flag_driver
"""


def test_e2e_scalar_logical(tmp_path: Path):
    """Scalar LOGICAL intent(in) -- cloudsc LDMAINCALL/LDSLPHY pattern; bindings emitter passes a length-1 c_bool pointer to the SDFG."""
    outer = (
        OriginalArg(name="flag", fortran_type="logical", rank=0, intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_module_wrap(_SCALAR_KERNEL, "scalar_flag_mod"),
        name="scalar_flag",
        entry="scalar_flag_mod::scalar_flag",
        outer_args=outer,
        driver_src=_SCALAR_DRIVER,
        module_name="scalar_flag_e2e",
    )
    ref = _f2py_ref(tmp_path, _SCALAR_KERNEL, "scalar_flag_ref")

    n = 5
    for flag_value in (True, False):
        out_ref = ref.scalar_flag(flag_value, n=n)
        # f2py auto-derives intent(out) array shape from ``n`` -- returns ``out``.
        out = mod.scalar_flag_driver.run(flag_value, n=n)
        np.testing.assert_array_equal(out, out_ref)
    mod.scalar_flag_dace_bindings.scalar_flag_dace_finalize()
