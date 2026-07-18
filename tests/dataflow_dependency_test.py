"""RAW/WAR/WAW/RAR dataflow-dependency pins between tasklets in one Fortran basic block.

Catches the cloudsc Section 4.5 bug (commit e85ec1f8f): codegen reordered a
write past a read it should follow, because no SDFG edge connected them. RAW=
writer-before-reader, WAR=reader-before-overwrite, WAW=later-write-wins, RAR=
no ordering needed. Each test compares the built SDFG against an f2py
reference at strict tolerance."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile inline ``src_text`` with strict FP flags."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{mod_name}.f90"
    src.write_text(src_text)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c",
            str(src), "-m", mod_name, "--quiet", "--f90flags=-O0 -fno-fast-math -ffp-contract=off"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg_args(sdfg, int_vals):
    """Route ints to Scalar-or-length-1 by SDFG descriptor."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_vals.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


def test_raw_dependency(tmp_path):
    """RAW: writer must complete before reader sees the value. y = (x_init*2)+1;
    a reorder bug would give y = x_init+1."""
    src = """
SUBROUTINE raw_kernel(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    x(i) = x(i) * 2.0_8
    y(i) = x(i) + 1.0_8
  END DO
END SUBROUTINE raw_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "raw_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="raw_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(300)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.raw_kernel(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg,
                                  x_ref,
                                  err_msg="RAW: x (in-place writer) diverged -- impossible unless tasklets reordered")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="RAW: y must equal (x_init * 2) + 1 -- bridge dropped the write-before-read")


def test_war_dependency(tmp_path):
    """WAR: reader must complete before writer overwrites. y = x_init*2, x=99;
    a reorder bug would give y = 99*2 = 198."""
    src = """
SUBROUTINE war_kernel(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    x(i) = 99.0_8
  END DO
END SUBROUTINE war_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "war_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="war_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(301)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.war_kernel(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg, x_ref, err_msg="WAR: x final values should all be 99.0")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="WAR: y must equal x_init * 2 -- bridge reordered the write "
                                  "before the read (this is the cloudsc Section 4.5 bug shape)")


def test_waw_dependency(tmp_path):
    """WAW: two writes to the same location; later write (2.0) must win, not 1.0."""
    src = """
SUBROUTINE waw_kernel(n, x)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    x(i) = 1.0_8
    x(i) = 2.0_8
  END DO
END SUBROUTINE waw_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "waw_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="waw_kernel", pipeline="hlfir-propagate-shapes").build()

    n = 32
    x_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.waw_kernel(x_ref)
    x_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg,
                                  x_ref,
                                  err_msg="WAW: final x must equal 2.0 (later write) -- bridge "
                                  "reordered the two writes")


def test_rar_dependency(tmp_path):
    """RAR: two reads of the same value, no ordering constraint -- any tasklet order is OK."""
    src = """
SUBROUTINE rar_kernel(n, x, y, z)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  REAL(KIND=8), INTENT(INOUT) :: z(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    z(i) = x(i) + 1.0_8
  END DO
END SUBROUTINE rar_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "rar_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="rar_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(303)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    y_ref = np.zeros(n, dtype=np.float64, order="F")
    z_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.rar_kernel(x, y_ref, z_ref)

    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    z_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x, y=y_sdfg, z=z_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(y_sdfg, y_ref, err_msg="RAR: y mismatch")
    np.testing.assert_array_equal(z_sdfg, z_ref, err_msg="RAR: z mismatch")


# --------------------------------------------------------------------
# cloudsc-shape regression: combines RAW + WAR, mirrors 4.5a Abel-Boutle ZCOVPTOT update.
# --------------------------------------------------------------------
def test_cloudsc_shape_war_via_division(tmp_path):
    """Reproduces cloudsc.F90:3174-3188: the ``cv`` update reads ``f`` before
    ``f -= e``. With d>f, e=f so e/f=1 exactly -> cv=MAX(g,z). If the bridge
    reorders `f -= e` first, f shrinks toward 0 and e/f_new explodes, clamping
    cv to the floor g -- the actual cloudsc bug."""
    src = """
SUBROUTINE cloudsc_war_shape(n, d, f, z, g, cv, e_out)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: d(n)         ! ZDPEVAP-like
  REAL(KIND=8), INTENT(INOUT) :: f(n)         ! ZQXFG-like
  REAL(KIND=8), INTENT(IN)    :: z(n)         ! ZA-like
  REAL(KIND=8), VALUE         :: g            ! RCOVPMIN-like floor
  REAL(KIND=8), INTENT(INOUT) :: cv(n)        ! ZCOVPTOT-like
  REAL(KIND=8), INTENT(INOUT) :: e_out(n)     ! capture evap
  INTEGER(KIND=4) :: i
  REAL(KIND=8)    :: e
  DO i = 1, n
    e = MIN(d(i), f(i))
    cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
    f(i)  = f(i) - e
    e_out(i) = e
  END DO
END SUBROUTINE cloudsc_war_shape
"""
    ref = _f2py(src, tmp_path / "ref", "cloudsc_war_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cloudsc_war_shape", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(304)
    n = 32
    # force clamp branch: d>f so e=f; post-update f~=0, reorder explodes e/f
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init + 0.5)  # always > f
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    e_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.cloudsc_war_shape(d, f_ref, z, g, cv_ref, e_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    e_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg, e_out=e_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(cv_sdfg,
                                  cv_ref,
                                  err_msg="cloudsc-WAR-shape: cv diverges -- bridge moved `f -= e` "
                                  "before the cv update so the cv tasklet reads post-update f")
    np.testing.assert_array_equal(f_sdfg, f_ref, err_msg="f final mismatch")
    np.testing.assert_array_equal(e_sdfg, e_ref, err_msg="e mismatch")


# --------------------------------------------------------------------
# DEFAULT_PIPELINE variants: same patterns through the full pipeline --
# the cloudsc divergence only manifests here, not under the minimal pipeline.
# --------------------------------------------------------------------
def test_war_dependency_default_pipeline(tmp_path):
    """Same WAR pattern under DEFAULT_PIPELINE -- a failure here (but not in the
    minimal-pipeline WAR test) means an extra pass reordered the tasklets."""
    src = """
SUBROUTINE war_default(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    x(i) = 99.0_8
  END DO
END SUBROUTINE war_default
"""
    ref = _f2py(src, tmp_path / "ref", "war_default_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="war_default").build()

    rng = np.random.default_rng(305)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.war_default(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg, x_ref, err_msg="WAR (DEFAULT_PIPELINE): x final mismatch")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="WAR (DEFAULT_PIPELINE): y mismatch -- "
                                  "DEFAULT_PIPELINE pass reordered the write before the read")


def test_cloudsc_shape_war_default_pipeline(tmp_path):
    """cloudsc-shape WAR pattern under DEFAULT_PIPELINE -- reproduces the
    bottom_upper/cloudsc_full bug; inputs chosen so MIN(d,f)=f and e/f_new
    explodes if reordered."""
    src = """
SUBROUTINE cw_default(n, d, f, z, g, cv, e_out)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: d(n)
  REAL(KIND=8), INTENT(INOUT) :: f(n)
  REAL(KIND=8), INTENT(IN)    :: z(n)
  REAL(KIND=8), VALUE         :: g
  REAL(KIND=8), INTENT(INOUT) :: cv(n)
  REAL(KIND=8), INTENT(INOUT) :: e_out(n)
  INTEGER(KIND=4) :: i
  REAL(KIND=8)    :: e
  DO i = 1, n
    e = MIN(d(i), f(i))
    cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
    f(i)  = f(i) - e
    e_out(i) = e
  END DO
END SUBROUTINE cw_default
"""
    ref = _f2py(src, tmp_path / "ref", "cw_default_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cw_default").build()

    rng = np.random.default_rng(306)
    n = 32
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init + 0.5)
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    e_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.cw_default(d, f_ref, z, g, cv_ref, e_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    e_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg, e_out=e_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(cv_sdfg,
                                  cv_ref,
                                  err_msg="cloudsc-WAR (DEFAULT_PIPELINE): cv diverges -- the cv "
                                  "tasklet reads post-update f because one of the DEFAULT_PIPELINE "
                                  "passes reordered the write-before-read.  Same bug as cloudsc_full.")
    np.testing.assert_array_equal(f_sdfg, f_ref, err_msg="f final mismatch")
    np.testing.assert_array_equal(e_sdfg, e_ref, err_msg="e mismatch")


def test_cloudsc_full_shape_nested_if(tmp_path):
    """Mirrors cloudsc.F90 4.5 structure: nested IF/DO/IF around the WAR pattern
    above (IEVAPRAIN-like outer gate, LLO1-like inner predicate)."""
    src = """
SUBROUTINE cw_nested(n, klev, mode, d, f, z, g, cv)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n, klev, mode
  REAL(KIND=8), INTENT(IN)    :: d(n, klev)
  REAL(KIND=8), INTENT(INOUT) :: f(n)
  REAL(KIND=8), INTENT(IN)    :: z(n)
  REAL(KIND=8), VALUE         :: g
  REAL(KIND=8), INTENT(INOUT) :: cv(n)
  INTEGER(KIND=4) :: i, jk
  REAL(KIND=8)    :: e, eps
  LOGICAL :: llo1
  eps = 1.0E-12_8
  IF (mode == 2) THEN
    DO jk = 1, klev
      DO i = 1, n
        llo1 = (cv(i) > eps) .AND. (f(i) > eps)
        IF (llo1) THEN
          e = MIN(d(i, jk), f(i))
          cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
          f(i) = f(i) - e
        END IF
      END DO
    END DO
  END IF
END SUBROUTINE cw_nested
"""
    ref = _f2py(src, tmp_path / "ref", "cw_nested_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cw_nested").build()

    rng = np.random.default_rng(307)
    n, klev = 8, 20
    mode = 2  # active branch (like IEVAPRAIN=2)
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init[:, None] + 0.5 * rng.random((n, klev), dtype=np.float64))
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    ref.cw_nested(mode, d, f_ref, z, g, cv_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n, klev=klev, mode=mode)))
    sdfg(**kw)

    np.testing.assert_allclose(cv_sdfg,
                               cv_ref,
                               rtol=1e-12,
                               atol=1e-12,
                               err_msg="cloudsc-nested-IF (DEFAULT_PIPELINE): cv diverges -- "
                               "the WAR ordering bug from cloudsc Section 4.5 reproduced here.")
    np.testing.assert_allclose(f_sdfg, f_ref, rtol=1e-12, atol=1e-12, err_msg="f final mismatch")
