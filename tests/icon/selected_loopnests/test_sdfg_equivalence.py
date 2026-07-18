"""End-to-end SDFG verification for the six E6 velocity-advection representative loopnests.

Complementary to ``test_equivalence.py`` (struct-vs-flat equivalence at the
Fortran level only): runs the flat kernel through HLFIR->SDFG, calls the
compiled SDFG, and compares against an f2py reference on identical seeded
inputs -- per the project's E2E rule, structural assertions alone don't catch
numerical bugs. Flat kernels are sliced out of ``icon_loopnest_N.f90`` so the
bundle (struct+flat+gfortran driver) stays the single source of truth.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_flat_kernel(bundle_path: Path) -> str:
    """Slice ``subroutine kernel_flat(...)...end subroutine`` out of a loopnest
    bundle -- f2py compiles a bare subroutine fine, so stripping the module
    wrapper is safe."""
    text = bundle_path.read_text()
    pattern = re.compile(r"(?is)(subroutine\s+kernel_flat\s*\([^)]*\).*?\bend\s+subroutine)", )
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"kernel_flat not found in {bundle_path}")
    return m.group(1)


def _f2py_build(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile ``src_text`` as ``mod_name`` into ``out_dir``, return the
    imported module. Skips if the toolchain's missing."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{mod_name}.f90"
    src.write_text(src_text)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(src), "-m", mod_name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg_from_flat(flat_src: str, tmp: Path, name: str):
    """Build the SDFG via the minimal ``hlfir-propagate-shapes`` pipeline (matches
    the ported_from_f2dace_windmill convention)."""
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(flat_src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


# ---------------------------------------------------------------------------
# Loopnest 2  --  direct stencil, partial vertical
# ---------------------------------------------------------------------------


def test_icon_loopnest_2_sdfg_matches_f2py(tmp_path: Path):
    """z_w_concorr_me = vn*ddxn + vt*ddxt  (jk = nflatlev..nlev)"""
    bundle = _HERE / "icon_loopnest_2.f90"
    flat_src = _extract_flat_kernel(bundle)

    # f2py reference -- rebuilt per test; pytest setup already imports the module on first call
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_2")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat")

    rng = np.random.default_rng(2)
    nproma, nlev, nblks_e, nflatlev = 32, 16, 8, 4

    # Fortran-order: nproma fastest-varying, matches f2py + DaCe descriptor layout
    vn = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    vt = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    ddxn = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    ddxt = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))

    z_ref = np.zeros_like(vn, order="F")
    z_sdfg = np.zeros_like(vn, order="F")

    # f2py auto-derives nproma/nlev/nblks_e from array shapes, drops them from the positional list
    ref.kernel_flat(vn, vt, ddxn, ddxt, z_ref, nflatlev, 1, nblks_e, 1, nproma)
    assert z_ref.any(), "f2py reference didn't write to z_ref"

    # bound/loop-condition integer dummies become DaCe symbols (plain ints); others
    # become length-1 Arrays (numpy arrays) -- route via sdfg.arglist(), don't hardcode
    from dace.data import Scalar
    int_args = dict(nproma=nproma,
                    nlev=nlev,
                    nblks_e=nblks_e,
                    nflatlev=nflatlev,
                    i_startblk=1,
                    i_endblk=nblks_e,
                    i_startidx=1,
                    i_endidx=nproma)
    call_kwargs = dict(vn=vn, vt=vt, ddxn=ddxn, ddxt=ddxt, z_w_concorr_me=z_sdfg)
    arglist = sdfg.arglist()
    for k, v in int_args.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            call_kwargs[k] = v  # plain int for scalars / symbols
        else:
            call_kwargs[k] = np.array([v], dtype=np.int32)  # length-1 Array
    sdfg(**call_kwargs)

    np.testing.assert_allclose(z_sdfg, z_ref, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Helpers shared by the remaining loopnest tests
# ---------------------------------------------------------------------------


def _sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each int arg to a plain int or length-1 numpy array, based on the SDFG
    descriptor's symbol/scalar vs Array classification."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# Loopnest 3  --  direct stencil with deepatmo vertical profiles
# ---------------------------------------------------------------------------


def test_icon_loopnest_3_sdfg_matches_f2py(tmp_path: Path):
    """z_v_grad_w = z_v_grad_w*gradh(jk) + vn_ie*(...) + z_vt_ie*(...)"""
    bundle = _HERE / "icon_loopnest_3.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_3")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_3")

    rng = np.random.default_rng(3)
    nproma, nlev, nblks_e = 32, 16, 8

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    vn_ie = _f((nproma, nlev, nblks_e))
    z_vt_ie = _f((nproma, nlev, nblks_e))
    ft_e = _f((nproma, nblks_e))
    fn_e = _f((nproma, nblks_e))
    gradh = _f((nlev, ))
    invr = _f((nlev, ))
    z_init = _f((nproma, nlev, nblks_e))
    z_ref = np.array(z_init, order="F")
    z_sdfg = np.array(z_init, order="F")

    ref.kernel_flat(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z_ref, 1, nblks_e, 1, nproma)

    kw = dict(vn_ie=vn_ie,
              z_vt_ie=z_vt_ie,
              ft_e=ft_e,
              fn_e=fn_e,
              gradh=gradh,
              invr=invr,
              z_v_grad_w=z_sdfg,
              nproma=nproma,
              nlev=nlev,
              nblks_e=nblks_e)
    kw.update(_sdfg_call_args(sdfg, dict(i_startblk=1, i_endblk=nblks_e, i_startidx=1, i_endidx=nproma)))
    sdfg(**kw)

    np.testing.assert_allclose(z_sdfg, z_ref, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Loopnest 5  --  vn_ie horizontal boundary (single-level + nlev+1 extrapolation)
# ---------------------------------------------------------------------------


def test_icon_loopnest_5_sdfg_matches_f2py(tmp_path: Path):
    """vn_ie(je,1,jb)=vn(je,1,jb); vn_ie(je,nlevp1,jb)=weighted sum -- literal indices
    and nlev-1/nlev-2 loop-bound arithmetic both resolve through buildIndexExpr."""
    bundle = _HERE / "icon_loopnest_5.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_5")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_5")

    rng = np.random.default_rng(5)
    nproma, nlev, nblks_e = 32, 16, 8
    nlevp1 = nlev + 1
    i_startblk, i_endblk = 1, nblks_e
    i_startidx, i_endidx = 1, nproma

    vn = np.asfortranarray(rng.random((nproma, nlev, nblks_e)))
    vt = np.asfortranarray(rng.random((nproma, nlev, nblks_e)))
    wgtfacqe = np.asfortranarray(rng.random((nproma, 3, nblks_e)))

    vn_ie_ref = np.zeros((nproma, nlevp1, nblks_e), order="F")
    z_vt_ref = np.zeros((nproma, nlevp1, nblks_e), order="F")
    z_k_ref = np.zeros((nproma, nlevp1, nblks_e), order="F")
    # f2py derives nproma/nlev/nlevp1/nblks_e from shapes; only loop-range scalars stay positional
    ref.kernel_flat(vn, vt, wgtfacqe, vn_ie_ref, z_vt_ref, z_k_ref, i_startblk, i_endblk, i_startidx, i_endidx)

    vn_ie_sdfg = np.zeros((nproma, nlevp1, nblks_e), order="F")
    z_vt_sdfg = np.zeros((nproma, nlevp1, nblks_e), order="F")
    z_k_sdfg = np.zeros((nproma, nlevp1, nblks_e), order="F")
    kw = dict(vn=vn,
              vt=vt,
              wgtfacq_e=wgtfacqe,
              vn_ie=vn_ie_sdfg,
              z_vt_ie=z_vt_sdfg,
              z_kin_hor_e=z_k_sdfg,
              nproma=nproma,
              nlev=nlev,
              nlevp1=nlevp1,
              nblks_e=nblks_e)
    kw.update(
        _sdfg_call_args(sdfg, dict(i_startblk=i_startblk, i_endblk=i_endblk, i_startidx=i_startidx, i_endidx=i_endidx)))
    sdfg(**kw)

    np.testing.assert_allclose(vn_ie_sdfg, vn_ie_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(z_vt_sdfg, z_vt_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(z_k_sdfg, z_k_ref, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Loopnest 6  --  levelmask vertical reduction
# ---------------------------------------------------------------------------


def test_icon_loopnest_6_sdfg_matches_f2py(tmp_path: Path):
    """levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))"""
    bundle = _HERE / "icon_loopnest_6.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_6")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_6")

    rng = np.random.default_rng(6)
    nlev, nblks_c = 64, 12
    i_startblk, i_endblk = 2, 10
    jk_start, jk_end = 3, nlev - 3

    # f2py's LOGICAL(4) ABI is np.int32; the SDFG uses np.bool_ (1 byte) end-to-end.
    # Build both shapes from the same random source so both backends see identical truth values.
    levmask_bool = np.asfortranarray(rng.random((nblks_c, nlev)) > 0.7)
    levmask_int = levmask_bool.astype(np.int32)
    levelmask_ref = np.zeros(nlev, dtype=np.int32)
    levelmask_sdfg = np.zeros(nlev, dtype=np.bool_)

    ref.kernel_flat(levmask_int, levelmask_ref, jk_start, jk_end, i_startblk, i_endblk)

    kw = dict(levmask=levmask_bool, levelmask=levelmask_sdfg, nlev=nlev, nblks_c=nblks_c)
    kw.update(_sdfg_call_args(sdfg, dict(jk_start=jk_start, jk_end=jk_end, i_startblk=i_startblk, i_endblk=i_endblk)))
    sdfg(**kw)

    np.testing.assert_array_equal(levelmask_sdfg.astype(np.int32), levelmask_ref)


# ---------------------------------------------------------------------------
# Loopnests 1, 4  --  indirect stencils (3D indirection)
# ---------------------------------------------------------------------------


def test_icon_loopnest_1_sdfg_matches_f2py(tmp_path: Path):
    """z_v_grad_w indirect stencil (cell+vertex indirection). Each scalar-staged index
    load (``ci0 = icidx(je,jb,1)``) is classified as a symbol; ``emit_loop`` hoists it
    onto the pre->body interstate edge so the consuming tasklet reads the live value."""
    bundle = _HERE / "icon_loopnest_1.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_1")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_1")

    rng = np.random.default_rng(1)
    nproma, nlev, nblks_e, nblks_c, nblks_v = 32, 16, 8, 8, 8

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    vn_ie = _f((nproma, nlev, nblks_e))
    inv_dual = _f((nproma, nblks_e))
    inv_primal = _f((nproma, nblks_e))
    tangent = _f((nproma, nblks_e))
    w = _f((nproma, nlev, nblks_c))
    z_vt_ie = _f((nproma, nlev, nblks_e))
    z_w_v = _f((nproma, nlev, nblks_v))

    def _idx(shape, hi):
        return np.asfortranarray(rng.integers(1, hi + 1, size=shape, dtype=np.int32))

    icidx = _idx((nproma, nblks_e, 2), nproma)
    icblk = _idx((nproma, nblks_e, 2), nblks_c)
    ividx = _idx((nproma, nblks_e, 2), nproma)
    ivblk = _idx((nproma, nblks_e, 2), nblks_v)

    z_ref = np.zeros((nproma, nlev, nblks_e), order="F")
    z_sdfg = np.zeros_like(z_ref, order="F")

    ref.kernel_flat(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, icidx, icblk, ividx, ivblk, z_ref, 1,
                    nblks_e, 1, nproma)

    kw = dict(vn_ie=vn_ie,
              inv_dual=inv_dual,
              inv_primal=inv_primal,
              tangent=tangent,
              w=w,
              z_vt_ie=z_vt_ie,
              z_w_v=z_w_v,
              icidx=icidx,
              icblk=icblk,
              ividx=ividx,
              ivblk=ivblk,
              z_v_grad_w=z_sdfg,
              nproma=nproma,
              nlev=nlev,
              nblks_e=nblks_e,
              nblks_c=nblks_c,
              nblks_v=nblks_v)
    kw.update(_sdfg_call_args(sdfg, dict(i_startblk=1, i_endblk=nblks_e, i_startidx=1, i_endidx=nproma)))
    sdfg(**kw)

    np.testing.assert_allclose(z_sdfg, z_ref, atol=1e-12, rtol=0)


def test_icon_loopnest_4_sdfg_matches_f2py(tmp_path: Path):
    """ddt_vn_apc_pc indirect stencil + (vn_ie(jk)-vn_ie(jk+1)) term -- same scalar-staged
    indirection as loopnest 1, plus a 4-D output indexed on its last dim by ``ntnd``."""
    bundle = _HERE / "icon_loopnest_4.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_4")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_4")

    rng = np.random.default_rng(4)
    nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd = 32, 16, 8, 8, 8, 3
    ntnd = 2

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    def _idx(shape, hi):
        return np.asfortranarray(rng.integers(1, hi + 1, size=shape, dtype=np.int32))

    vt = _f((nproma, nlev, nblks_e))
    vn_ie = _f((nproma, nlev + 1, nblks_e))
    f_e = _f((nproma, nblks_e))
    coeff_gradekin = _f((nproma, 2, nblks_e))
    c_lin_e = _f((nproma, 2, nblks_e))
    ddqz = _f((nproma, nlev, nblks_e))
    z_kin_hor_e = _f((nproma, nlev, nblks_e))
    z_ekinh = _f((nproma, nlev, nblks_c))
    zeta = _f((nproma, nlev, nblks_v))
    z_w_con_c_full = _f((nproma, nlev, nblks_c))
    icidx = _idx((nproma, nblks_e, 2), nproma)
    icblk = _idx((nproma, nblks_e, 2), nblks_c)
    ividx = _idx((nproma, nblks_e, 2), nproma)
    ivblk = _idx((nproma, nblks_e, 2), nblks_v)

    ddt_ref = np.zeros((nproma, nlev, nblks_e, nproma_tnd), order="F")
    ddt_sdfg = np.zeros_like(ddt_ref, order="F")

    ref.kernel_flat(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, icidx,
                    icblk, ividx, ivblk, ddt_ref, ntnd, 1, nblks_e, 1, nproma)

    kw = dict(
        vt=vt,
        vn_ie=vn_ie,
        f_e=f_e,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz=ddqz,
        z_kin_hor_e=z_kin_hor_e,
        z_ekinh=z_ekinh,
        zeta=zeta,
        z_w_con_c_full=z_w_con_c_full,
        icidx=icidx,
        icblk=icblk,
        ividx=ividx,
        ivblk=ivblk,
        ddt_vn_apc_pc=ddt_sdfg,
        nproma=nproma,
        nlev=nlev,
        nblks_e=nblks_e,
        nblks_c=nblks_c,
        nblks_v=nblks_v,
        nproma_tnd=nproma_tnd,
        # vn_ie(nproma, nlev+1, nblks_e): bridge can't resolve nlev+1 to a closed-form
        # extent, so add_descriptors synthesises vn_ie_d1; caller passes the actual value
        vn_ie_d1=nlev + 1)
    kw.update(_sdfg_call_args(sdfg, dict(ntnd=ntnd, i_startblk=1, i_endblk=nblks_e, i_startidx=1, i_endidx=nproma)))
    sdfg(**kw)

    np.testing.assert_allclose(ddt_sdfg, ddt_ref, atol=1e-12, rtol=0)
