"""Top-half-CLOUDSC reproducer for the cloudsc_full PCOVPTOT divergence (26/548 cells, 1-ulp
drift in ZQXN[JM=3] @ JK=15) -- isolates source/sink accumulation into ZSOLQA/ZSOLQB (top half
of the JK loop, before SEDIMENTATION) from the LU solver / sedimentation / flux-tendency stages.

``cloudsc.F90`` truncated before SEDIMENTATION, ZSOLQA/ZSOLQB captured per JK into INTENT(OUT)
dummies. A divergence here localizes the bug to source/sink cross-talk; a pass points to the
bottom half or the assembly/solver/clip combination instead.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang
from cloudsc.full._registries import (
    CLOUDSC_F90FLAGS,
    get_inputs_physical,
    get_outputs,
)
from cloudsc.full._harness import f2py_argnames, lower_keys, sdfg_call_args

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _module_wrap_drivers(src: str) -> str:
    """Wrap the free CLOUDSCOUTER + CLOUDSC driver subroutines in ``module cloudsc_top_half_mod`` so the SDFG entry can be ``cloudsc_top_half_mod::cloudscouter``.

    Leading derived-type modules stay outside; only the two driver subroutines move in (host
    association keeps CLOUDSCOUTER -> CLOUDSC resolving). On-disk file stays untouched for the
    f2py reference path.
    """
    marker = "SUBROUTINE CLOUDSCOUTER"
    head, _, tail = src.partition(marker)
    assert tail, "CLOUDSCOUTER driver not found in cloudsc_top_half source"
    return (f"{head}module cloudsc_top_half_mod\ncontains\n"
            f"{marker}{tail.rstrip()}\nend module cloudsc_top_half_mod\n")


@pytest.fixture(scope="module")
def _f2py_top_half(tmp_path_factory):
    src = (_HERE / "cloudsc_top_half.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_top_half_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_top_half_ref",
        # -ffree-line-length-none: gfortran-only, non-semantic parser necessity for the
        # long-line source (flang has no line limit); rest is the flang-portable FP core
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_top_half_zsolqa_zsolqb(tmp_path, _f2py_top_half, _strict_fp_cpu_args):
    """SDFG-vs-f2py equivalence on ZSOLQA/ZSOLQB after the source/sink accumulation block (top half of CLOUDSC's JK loop)."""
    src = (_HERE / "cloudsc_top_half.F90").read_text()

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    # module-wrap CLOUDSCOUTER+CLOUDSC so the SDFG entry can use cloudsc_top_half_mod::cloudscouter;
    # on-disk file stays free-subroutine for the f2py reference (resolves top-level)
    build_src = _module_wrap_drivers(src)
    sdfg = build_sdfg(build_src, sdfg_dir, name="cloudsc_top_half", entry="cloudsc_top_half_mod::cloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs_physical(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    # zsolqa_out/zsolqb_out: new INTENT(OUT) dummies, shape (KLON,KLEV,NCLV,NCLV,NBLOCKS)
    # matches ZSOLQA per JK per block
    klon = inputs["KLON"]
    klev = inputs["KLEV"]
    nclv = inputs["NCLV"]
    nblocks = inputs["NBLOCKS"]
    shape = (klon, klev, nclv, nclv, nblocks)
    zsolqa_out_ref = np.zeros(shape, dtype=np.float64, order="F")
    zsolqb_out_ref = np.zeros(shape, dtype=np.float64, order="F")
    zsolqa_out_sdfg = np.zeros_like(zsolqa_out_ref, order="F")
    zsolqb_out_sdfg = np.zeros_like(zsolqb_out_ref, order="F")

    accepted = f2py_argnames(_f2py_top_half.cloudscouter)
    all_kw_ref = {
        **lower_keys(inputs),
        **lower_keys(outputs_ref),
        "zsolqa_out": zsolqa_out_ref,
        "zsolqb_out": zsolqb_out_ref,
    }
    _f2py_top_half.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    scalar_kwargs = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(lower_keys(outputs_sdfg))
    sdfg_kwargs["zsolqa_out"] = zsolqa_out_sdfg
    sdfg_kwargs["zsolqb_out"] = zsolqb_out_sdfg
    sdfg_kwargs.update(sdfg_call_args(sdfg, scalar_kwargs))
    sdfg(**sdfg_kwargs)

    # strict tolerance to catch the ulp-level drift the cloudsc_full cascade is rooted in
    np.testing.assert_allclose(
        zsolqa_out_sdfg,
        zsolqa_out_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg="ZSOLQA diverges between SDFG and f2py top-half references",
    )
    np.testing.assert_allclose(
        zsolqb_out_sdfg,
        zsolqb_out_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg="ZSOLQB diverges between SDFG and f2py top-half references",
    )
