"""E2e numerical correctness for the AoS-of-pointer-records lift.

Each probe under ``tests/aos_pointer_records/`` builds an SDFG via the bridge
and an f2py reference from the same source, runs both with seeded random
inputs, and asserts OUTPUT arrays match elementwise. Regression gate for
``hlfir-lift-aos-pointer-records``.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

import dace_fortran

_HERE = Path(__file__).resolve().parent


def _f2py(src: Path, out_dir: Path, mod_name: str, *, kind_map: dict = None):
    """Compile ``src`` via ``numpy.f2py`` and import the resulting module.

    ``kind_map`` writes a ``.f2py_f2cmap`` for symbolic kind aliases (``wp``, ``JPRB``) f2py can't resolve itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind_map:
        # f2py reads .f2py_f2cmap from cwd; maps Fortran TypeName -> {kind_alias: ctype}.
        as_dict = {"real": {a: c for a, c in kind_map.items()}}
        (out_dir / ".f2py_f2cmap").write_text(repr(as_dict))
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c", str(src), "-m", mod_name, "--quiet"], cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg(src_text: str, out_dir: Path, entry: str, name: str):
    sdfg = dace_fortran.build_sdfg(src_text, out_dir=str(out_dir), entry=entry, name=name)
    sdfg.validate()
    return sdfg


def test_single_pointer_member_numerical(tmp_path):
    """``q(c)%x => target_c; out = q(1)%x + q(2)%x`` matches gfortran."""
    src_path = _HERE / "aos_single_pointer_member_probe.f90"
    src_text = src_path.read_text()

    mod = _f2py(src_path, tmp_path / "ref", "single_ref")
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(0)
    n, k = 8, 6
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qsum_sdfg = np.zeros((n, k), order="F")

    qsum_ref = mod.m.run(qa.copy(order="F"), qb.copy(order="F"))
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa.copy(order="F"), qb=qb.copy(order="F"), qsum=qsum_sdfg)
    np.testing.assert_allclose(qsum_sdfg, qsum_ref, rtol=1e-12, atol=1e-12)


def test_write_through_pointer_numerical(tmp_path):
    """Writes through ``q(c)%x`` propagate back to the targets after lift."""
    src_path = _HERE / "aos_write_through_pointer_probe.f90"
    src_text = src_path.read_text()
    mod = _f2py(src_path, tmp_path / "ref", "write_ref")
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(1)
    n, k = 5, 7
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qa_ref, qb_ref = qa.copy(order="F"), qb.copy(order="F")
    qa_sdfg, qb_sdfg = qa.copy(order="F"), qb.copy(order="F")

    mod.m.run(qa_ref, qb_ref)
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa_sdfg, qb=qb_sdfg)
    np.testing.assert_allclose(qa_sdfg, qa_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qb_sdfg, qb_ref, rtol=1e-12, atol=1e-12)


def test_target_and_alias_stay_coherent_numerical(tmp_path):
    """A pointer target read or written directly while ``q(c)%x`` points at it agrees with the alias.

    Covers a write through the alias read through the target, the reverse, and a whole-array write to the target
    read through the alias.  The reference is spelled out in numpy, since the values the source must produce are
    simple and the check does not depend on a Fortran build.
    """
    src_text = (_HERE / "aos_alias_coherence_probe.f90").read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(4)
    n, k = 6, 5
    qv0 = np.asfortranarray(rng.standard_normal((n, k)))
    qc0 = np.asfortranarray(rng.standard_normal((n, k)))
    qv, qc = qv0.copy(order="F"), qc0.copy(order="F")
    t, u = np.zeros((n, k), order="F"), np.zeros((n, k), order="F")

    sdfg(n=np.int32(n), k=np.int32(k), qv=qv, qc=qc, t=t, u=u)

    np.testing.assert_allclose(t, 2.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qc, qc0 + 5.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qv, 3.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u, (qc0 + 5.0) + 3.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)


def test_assumed_shape_target_numerical(tmp_path):
    """Assumed-shape pointer targets (``q(iqx)%x => t(:,:)``): gather temp's
    extents come from ``fir.box_dims``.  Regression: unbound extents used to
    default to 1 at call time, under-allocating and overflowing the heap."""
    src_path = _HERE / "aos_assumed_shape_target_probe.f90"
    src_text = src_path.read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    # Pure runtime-indexed select; reference is just the ix-th input (f2py
    # can't wrap the assumed-shape dummies this probe needs for fir.box_dims).
    rng = np.random.default_rng(3)
    n, k = 7, 5
    targets = [np.asfortranarray(rng.standard_normal((n, k))) for _ in range(4)]
    for ix in (1, 2, 3, 4):
        out_sdfg = np.zeros((n, k), order="F")
        sdfg(
            ix=np.int32(ix),
            qa=targets[0].copy(order="F"),
            qb=targets[1].copy(order="F"),
            qc=targets[2].copy(order="F"),
            qd=targets[3].copy(order="F"),
            out=out_sdfg,
        )
        np.testing.assert_allclose(out_sdfg, targets[ix - 1], rtol=1e-12, atol=1e-12, err_msg=f"mismatch for ix={ix}")


def test_wp_kind_alias_numerical(tmp_path):
    """``REAL(KIND=wp)`` compiles + runs without pre-resolving ``wp`` -- exercises ``normalize_kind_parameters``."""
    src_path = _HERE / "aos_wp_kind_probe.f90"
    src_text = src_path.read_text()
    # f2py can't evaluate SELECTED_REAL_KIND; tell it wp=double to match the
    # bridge's normalize_kind_parameters default (wp -> 8).
    mod = _f2py(src_path, tmp_path / "ref", "wp_ref", kind_map={"wp": "double"})
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(2)
    n, k = 6, 4
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qsum_sdfg = np.zeros((n, k), order="F")

    qsum_ref = mod.m.run(qa.copy(order="F"), qb.copy(order="F"))
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa.copy(order="F"), qb=qb.copy(order="F"), qsum=qsum_sdfg)
    np.testing.assert_allclose(qsum_sdfg, qsum_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import tempfile

    for test in (
        test_single_pointer_member_numerical,
        test_write_through_pointer_numerical,
        test_target_and_alias_stay_coherent_numerical,
        test_assumed_shape_target_numerical,
        test_wp_kind_alias_numerical,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
