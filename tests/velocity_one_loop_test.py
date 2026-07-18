"""Build + e2e binding + numerical-correctness test for a single velocity-tendencies
loop nest.  Source: ``velocity_one_loop.f90`` -- one DO jb/jk/je nest extracted from
ICON ``velocity_tendencies`` (upstream line 444-449), same struct-dummy shapes
(TARGET/POINTER members, USE-imported types) so the bridge exercises pointer-array
struct member flattening + dummy-arg flattening, with no indirect gather/nested calls.

Reference: pure-numpy reimplementation on flat arrays (drops the f2py struct-dummy
plumbing; the full-velocity numerical case is handled separately)."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "velocity_one_loop.f90"


def _numpy_reference(vn, wgtfac_e, vt, vn_ie, z_kin_hor_e, nproma, nlev, nblks_e):
    """Same loop nest as ``one_loop_nest``, vectorised over jb/jk/je, in-place on
    ``vn_ie``/``z_kin_hor_e``.  Pure subtractions only -- no FMA reorder risk."""
    jk = slice(1, nlev)
    jk_minus_1 = slice(0, nlev - 1)
    vn_ie[:, jk, :] = vn[:, jk, :] - vn[:, jk_minus_1, :]
    z_kin_hor_e[:, jk, :] = vt[:, jk, :] - wgtfac_e[:, jk, :]


def _make_inputs(nproma: int, nlev: int, nblks_e: int, rng: np.random.Generator):

    def rr(*shape):
        return np.asfortranarray(rng.standard_normal(shape))

    pointer_arrays = ('p_prog_vn', 'p_metrics_wgtfac_e', 'p_diag_vt', 'p_diag_vn_ie')
    kw = dict(
        # p_patch flat
        p_patch_nblks_e=np.int32(nblks_e),
        p_patch_nlev=np.int32(nlev),
        p_patch_edges_start_index=np.asfortranarray(np.ones(8, dtype=np.int32)),
        p_patch_edges_end_index=np.asfortranarray(np.full(8, nproma, dtype=np.int32)),
        p_patch_edges_start_block=np.asfortranarray(np.ones(8, dtype=np.int32)),
        p_patch_edges_end_block=np.asfortranarray(np.full(8, nblks_e, dtype=np.int32)),
        # p_prog flat
        p_prog_vn=rr(nproma, nlev, nblks_e),
        # p_metrics flat
        p_metrics_wgtfac_e=rr(nproma, nlev, nblks_e),
        # p_diag flat (INOUT)
        p_diag_vt=rr(nproma, nlev, nblks_e),
        p_diag_vn_ie=rr(nproma, nlev, nblks_e),
        # naked array (INOUT) + scalar
        z_kin_hor_e=rr(nproma, nlev, nblks_e),
        nproma=np.int32(nproma),
    )
    # deferred-shape pointer companions need ``<name>_d<i>`` extent symbols and
    # ``offset_<name>_d<i>`` lower-bound symbols (free-offset fallback when the bridge
    # can't statically infer it); all three Fortran dims are 1-based here.
    for nm in pointer_arrays:
        kw[f'{nm}_d0'] = np.int64(nproma)
        kw[f'{nm}_d1'] = np.int64(nlev)
        kw[f'offset_{nm}_d0'] = np.int64(1)
        kw[f'offset_{nm}_d1'] = np.int64(1)
        kw[f'offset_{nm}_d2'] = np.int64(1)
    return kw


def test_velocity_one_loop_builds_and_calls(tmp_path: Path):
    """Build + e2e call check.  Does NOT verify numerical correctness."""
    src = _SRC_PATH.read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(
        src,
        sdfg_dir,
        name="velocity_one_loop",
        entry="mo_velocity_one::one_loop_nest",
    ).build()
    sdfg.validate()

    nproma, nlev, nblks_e = 8, 5, 3
    rng = np.random.default_rng(0)
    kw = _make_inputs(nproma, nlev, nblks_e, rng)

    sdfg(**kw)
    assert np.all(np.isfinite(kw['p_diag_vn_ie']))
    assert np.all(np.isfinite(kw['z_kin_hor_e']))


def test_velocity_one_loop_numerical(tmp_path: Path):
    """Numerical comparison against a pure-numpy reference on flat arrays; both sides
    see the same operand order so we hold a tight tolerance."""
    src = _SRC_PATH.read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(
        src,
        sdfg_dir,
        name="velocity_one_loop",
        entry="mo_velocity_one::one_loop_nest",
    ).build()

    nproma, nlev, nblks_e = 8, 5, 3
    rng = np.random.default_rng(0)
    kw = _make_inputs(nproma, nlev, nblks_e, rng)

    # pre-call snapshot of every INOUT buffer so the numpy reference starts from the
    # same state
    vn_ref = kw['p_prog_vn'].copy(order='F')
    wgtfac_e_ref = kw['p_metrics_wgtfac_e'].copy(order='F')
    vt_ref = kw['p_diag_vt'].copy(order='F')
    vn_ie_ref = kw['p_diag_vn_ie'].copy(order='F')
    z_kin_hor_e_ref = kw['z_kin_hor_e'].copy(order='F')

    _numpy_reference(vn_ref, wgtfac_e_ref, vt_ref, vn_ie_ref, z_kin_hor_e_ref, nproma, nlev, nblks_e)

    sdfg(**kw)

    # Pure subtractions on both sides -- bit-exact.
    np.testing.assert_array_equal(kw['p_diag_vn_ie'], vn_ie_ref)
    np.testing.assert_array_equal(kw['z_kin_hor_e'], z_kin_hor_e_ref)
    # vt and vn are not written by the kernel -- should still equal the pre-call values
    np.testing.assert_array_equal(kw['p_diag_vt'], vt_ref)
    np.testing.assert_array_equal(kw['p_prog_vn'], vn_ref)
