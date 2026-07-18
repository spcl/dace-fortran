"""End-to-end test for the indirect-access stencil (``kin_to_cell``): SDFG vs
gfortran/f2py on random input, plus structural checks that each distinct
``edge_idx(jc, k)`` load mints a fresh symbol ``edge_idx_at<gid>`` and becomes an
interstate-edge assignment forcing a new state before the compute tasklet.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "indirect_stencil.f90"


def _f2py_compile(src: Path, out_dir: Path, mod_name: str) -> Path:
    """Compile `src` via f2py into `out_dir`; returns that dir for sys.path. Requires gfortran."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available (required for f2py)")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c", str(src), "-m", mod_name, "--quiet"], cwd=out_dir)
    # meson backend usually drops the .so in cwd; if it landed in a sibling build-dir, locate + symlink.
    if not list(out_dir.glob(f"{mod_name}*.so")):
        matches = list(out_dir.rglob(f"{mod_name}*.so"))
        if not matches:
            # Last resort: search /tmp tree.
            matches = list(Path("/tmp").rglob(f"{mod_name}*.so"))
            matches = [m for m in matches if m.stat().st_mtime > src.stat().st_mtime]
        if matches:
            link = out_dir / matches[0].name
            link.symlink_to(matches[0].resolve())
    return out_dir


def test_indirect_access_symbol_and_state(tmp_path):
    """Structural check: minted symbols + interstate-edge assignment."""
    b = build_sdfg(_SRC_PATH.read_text(), tmp_path, name="indirect", pipeline="hlfir-propagate-shapes")
    sdfg = b.build()
    sdfg.validate()

    idx_syms = [s for s in sdfg.symbols if s.startswith("edge_idx_at")]
    assert len(idx_syms) == 3, (f"expected three minted symbols (one per indirect load); got {idx_syms}")

    # every minted symbol must be assigned on some interstate edge and read edge_idx
    assigned = set()
    for e in sdfg.all_interstate_edges():
        for sym, expr in e.data.assignments.items():
            assigned.add(sym)
            if sym in idx_syms:
                assert "edge_idx" in expr, (f"symbol {sym} should be assigned from edge_idx, got {expr!r}")
    missing = set(idx_syms) - assigned
    assert not missing, f"symbols minted but never assigned: {missing}"


def test_indirect_access_numerical(tmp_path):
    """SDFG matches gfortran/f2py-compiled Fortran; flang is used only to emit HLFIR for the frontend."""
    f2py_dir = _f2py_compile(_SRC_PATH, tmp_path / "f2py", "ind_fort")
    sys.path.insert(0, str(f2py_dir))
    try:
        import ind_fort  # noqa: E402
    finally:
        sys.path.remove(str(f2py_dir))

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    b = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="indirect", pipeline="hlfir-propagate-shapes")
    sdfg = b.build()
    sdfg.validate()

    # frontend emits Fortran-order strides for rank>1 arrays -- pass Fortran-order numpy arrays to both sides.
    rng = np.random.default_rng(0)
    nc, ne, nk = 7, 13, 5
    edge_idx = np.asfortranarray(rng.integers(1, ne + 1, size=(nc, 3), dtype=np.int32))
    e_bln = np.asfortranarray(rng.standard_normal((nc, 3)))
    z_kin = np.asfortranarray(rng.standard_normal((ne, nk)))

    z_ekinh_fort = ind_fort.kin_to_cell(z_kin, e_bln, edge_idx)

    z_ekinh_sdfg = np.zeros((nc, nk), dtype=np.float64, order="F")
    sdfg(z_kin=z_kin, e_bln=e_bln, edge_idx=edge_idx, z_ekinh=z_ekinh_sdfg, nc=nc, ne=ne, nk=nk)

    np.testing.assert_allclose(z_ekinh_sdfg, z_ekinh_fort, rtol=1e-12, atol=1e-12)
