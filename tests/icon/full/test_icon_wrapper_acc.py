"""ICON wrapper OpenACC staging: sidecar x SDFG storage -> emitted directives.

Covers the four defined crossings of
:mod:`dace_fortran.bindings.acc_transfers` against
:func:`scripts.build_icon_dace_libs.render_icon_wrapper`, the hard errors
for every undefined one, and a compile proof of the rendered wrapper
under gfortran with OpenACC OFF and nvfortran with ``-acc`` ON.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import dace
import pytest

from _util import build_sdfg, have_flang

from dace_fortran.bindings.acc_transfers import (
    AccResidency,
    AccResidencyError,
    plan_acc_transfers,
)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_VELOCITY_SRC = _HERE / "velocity_full.f90"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_bil_spec = importlib.util.spec_from_file_location("_icon_wrapper_acc_bil",
                                                   _REPO / "scripts" / "build_icon_dace_libs.py")
_bil = importlib.util.module_from_spec(_bil_spec)
_bil_spec.loader.exec_module(_bil)

_ARGS = _bil._VELOCITY_WRAPPER_ARGS
_ARRAY_ARGS = ("p_prog", "p_patch", "p_int", "p_metrics", "p_diag", "z_w_concorr_me", "z_kin_hor_e", "z_vt_ie")
_SCALAR_ARGS = ("ntnd", "istep", "lvn_only", "dtime", "dt_linintp_ubc", "ldeepatmo")

_STUB_F90 = """\
MODULE mo_model_domain
  TYPE :: t_patch
    INTEGER :: id = 0
  END TYPE t_patch
END MODULE mo_model_domain

MODULE mo_intp_data_strc
  TYPE :: t_int_state
    INTEGER :: id = 0
  END TYPE t_int_state
END MODULE mo_intp_data_strc

MODULE mo_nonhydro_types
  USE iso_c_binding, ONLY: c_double
  TYPE :: t_nh_prog
    REAL(c_double), POINTER :: vn(:,:,:) => NULL()
  END TYPE t_nh_prog
  TYPE :: t_nh_metrics
    REAL(c_double), POINTER :: wgtfac_c(:,:,:) => NULL()
  END TYPE t_nh_metrics
  TYPE :: t_nh_diag
    REAL(c_double), POINTER :: vt(:,:,:) => NULL()
  END TYPE t_nh_diag
END MODULE mo_nonhydro_types

MODULE velocity_tendencies_dace_bindings
  USE iso_c_binding, ONLY: c_int, c_double, c_bool
  USE mo_model_domain,   ONLY: t_patch
  USE mo_intp_data_strc, ONLY: t_int_state
  USE mo_nonhydro_types, ONLY: t_nh_prog, t_nh_metrics, t_nh_diag
CONTAINS
  SUBROUTINE velocity_tendencies_dace(p_prog, p_patch, p_int, p_metrics, p_diag, &
                                      z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                                      ntnd, istep, lvn_only, &
                                      dtime, dt_linintp_ubc, ldeepatmo)
    TYPE(t_nh_prog),    INTENT(INOUT), TARGET :: p_prog
    TYPE(t_patch),      INTENT(IN),    TARGET :: p_patch
    TYPE(t_int_state),  INTENT(IN),    TARGET :: p_int
    TYPE(t_nh_metrics), INTENT(INOUT), TARGET :: p_metrics
    TYPE(t_nh_diag),    INTENT(INOUT), TARGET :: p_diag
    REAL(c_double),     INTENT(INOUT), TARGET :: z_w_concorr_me(:,:,:)
    REAL(c_double),     INTENT(INOUT), TARGET :: z_kin_hor_e(:,:,:)
    REAL(c_double),     INTENT(INOUT), TARGET :: z_vt_ie(:,:,:)
    INTEGER(c_int),     INTENT(IN),    TARGET :: ntnd
    INTEGER(c_int),     INTENT(IN),    TARGET :: istep
    LOGICAL(c_bool),    INTENT(IN),    TARGET :: lvn_only
    REAL(c_double),     INTENT(IN),    TARGET :: dtime
    REAL(c_double),     INTENT(IN),    TARGET :: dt_linintp_ubc
    LOGICAL(c_bool),    INTENT(IN),    TARGET :: ldeepatmo
    z_kin_hor_e(1, 1, 1) = z_kin_hor_e(1, 1, 1) + dtime + dt_linintp_ubc
    z_w_concorr_me(1, 1, 1) = REAL(ntnd + istep, c_double)
    z_vt_ie(1, 1, 1) = MERGE(1.0_c_double, 0.0_c_double, lvn_only .OR. ldeepatmo)
    p_patch%id = p_int%id
  END SUBROUTINE velocity_tendencies_dace
END MODULE velocity_tendencies_dace_bindings
"""


def _sidecar(device=(), host=(), unclassified=(), routine="velocity_tendencies") -> AccResidency:
    """Build a sidecar payload the way the parse-side extractor writes it."""
    args = {}
    for name in device:
        args[name] = {"residency": "device", "clause": "PRESENT", "evidence": "src.f90:1"}
    for name in host:
        args[name] = {"residency": "host", "clause": "SELF", "evidence": "src.f90:2"}
    return AccResidency.from_dict({
        "routine": routine,
        "source": "src.f90",
        "args": args,
        "unclassified": list(unclassified),
    })


def _toy_sdfg(storages, written=(), scalars=_SCALAR_ARGS) -> dace.SDFG:
    """Synthetic SDFG: one non-transient array per name, writes where asked.

    ``scalars`` land as :class:`dace.data.Scalar`, mirroring how the real
    velocity SDFG carries ``dtime`` / ``lvn_only`` and so exercising the
    scalar exemption rather than stepping around it.
    """
    sdfg = dace.SDFG("toy_wrapper")
    state = sdfg.add_state("main")
    for name, storage in storages.items():
        sdfg.add_array(name, (4, ), dace.float64, storage=storage, transient=False)
    for name in scalars:
        sdfg.add_scalar(name, dace.float64, transient=False)
    for name in written:
        tasklet = state.add_tasklet(f"w_{name}", {}, {"o"}, "o = 1.0")
        access = state.add_access(name)
        state.add_edge(tasklet, "o", access, None, dace.Memlet(f"{name}[0]"))
    return sdfg


def _lines(text: str) -> list:
    return [line.strip() for line in text.splitlines()]


# ----- crossing matrix ------------------------------------------------------


def test_no_sidecar_renders_the_cpu_only_wrapper():
    text = _bil.render_icon_wrapper()
    assert "!$ACC" not in text.upper()
    assert _bil._ICON_WRAPPER_ENTRY_READ in text
    assert _bil._ICON_WRAPPER_EXIT_READ in text
    body = _lines(text)
    entry = body.index("CALL SYSTEM_CLOCK(COUNT=velo_t0)")
    assert body[entry - 1] == "CALL SYSTEM_CLOCK(COUNT_RATE=velo_rate)"
    assert body[entry + 1].startswith("CALL velocity_tendencies_dace(")


def test_host_args_on_host_sdfg_emit_nothing():
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS})
    plan = plan_acc_transfers(sdfg, _sidecar(host=_ARRAY_ARGS + _SCALAR_ARGS), _ARGS)
    assert not plan.active
    assert "!$ACC" not in _bil.render_icon_wrapper(plan).upper()


def test_unclassified_array_without_device_args_stays_host():
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS})
    residency = _sidecar(host=_ARRAY_ARGS[1:], unclassified=(_ARRAY_ARGS[0], ))
    assert not plan_acc_transfers(sdfg, residency, _ARGS).active


def test_scalar_args_need_no_classification():
    """The shape the real extractor produces: 8 device arrays, 6 bare scalars."""
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}, written=("p_diag", ))
    plan = plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, unclassified=_SCALAR_ARGS), _ARGS)
    assert plan.update_host == _ARRAY_ARGS
    assert plan.update_device == ("p_diag", )
    assert not any(name in plan.storage for name in _SCALAR_ARGS)


def test_drift_rule_passes_the_device_pointer_through():
    storages = {name: dace.StorageType.GPU_Global for name in _ARRAY_ARGS}
    sdfg = _toy_sdfg(storages, written=("p_diag", "z_kin_hor_e"))
    plan = plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS), _ARGS)

    assert plan.update_host == ()
    assert plan.update_device == ()
    assert plan.use_device == _ARRAY_ARGS

    body = _lines(_bil.render_icon_wrapper(plan))
    assert "UPDATE" not in " ".join(body)
    call = body.index("!$ACC HOST_DATA USE_DEVICE(" + ", ".join(_ARRAY_ARGS) + ")")
    assert body[call + 1].startswith("CALL velocity_tendencies_dace(")
    close = body.index("!$ACC END HOST_DATA")
    assert body[close + 1] == "!$ACC WAIT"
    assert body[close + 2] == "CALL SYSTEM_CLOCK(COUNT=velo_t1)"


def test_mixed_device_and_host_storage_splits_the_two_paths():
    storages = {name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}
    storages["z_kin_hor_e"] = dace.StorageType.GPU_Global
    sdfg = _toy_sdfg(storages, written=("p_diag", "z_kin_hor_e", "z_vt_ie"))
    plan = plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS), _ARGS)

    assert plan.use_device == ("z_kin_hor_e", )
    assert "z_kin_hor_e" not in plan.update_host
    assert plan.update_device == ("p_diag", "z_vt_ie")


def test_render_is_deterministic():
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}, written=("p_diag", ))
    residency = _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS)
    first = _bil.render_icon_wrapper(plan_acc_transfers(sdfg, residency, _ARGS))
    second = _bil.render_icon_wrapper(plan_acc_transfers(sdfg, residency, _ARGS))
    assert first == second


# ----- forbidden crossings --------------------------------------------------


def test_host_arg_on_gpu_array_is_an_error():
    storages = {name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}
    storages["p_metrics"] = dace.StorageType.GPU_Global
    sdfg = _toy_sdfg(storages)
    with pytest.raises(AccResidencyError, match="p_metrics"):
        plan_acc_transfers(sdfg, _sidecar(host=_ARRAY_ARGS + _SCALAR_ARGS), _ARGS)


def test_unclassified_array_with_device_args_is_an_error():
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS})
    residency = _sidecar(device=_ARRAY_ARGS[1:], host=_SCALAR_ARGS, unclassified=(_ARRAY_ARGS[0], ))
    with pytest.raises(AccResidencyError, match="unclassified"):
        plan_acc_transfers(sdfg, residency, _ARGS)


def test_arg_missing_from_the_sidecar_is_an_error():
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS})
    residency = _sidecar(device=_ARRAY_ARGS[1:], host=_SCALAR_ARGS)
    with pytest.raises(AccResidencyError, match="p_prog"):
        plan_acc_transfers(sdfg, residency, _ARGS)


def test_arg_with_no_sdfg_container_is_an_error():
    storages = {name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS if name != "p_int"}
    sdfg = _toy_sdfg(storages)
    with pytest.raises(AccResidencyError, match="p_int"):
        plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS), _ARGS)


def test_arg_split_across_host_and_gpu_containers_is_an_error():
    storages = {name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}
    storages["p_diag_vt"] = dace.StorageType.GPU_Global
    sdfg = _toy_sdfg(storages)
    with pytest.raises(AccResidencyError, match="mixed host/GPU storage"):
        plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS), _ARGS)


def test_unknown_residency_word_is_an_error():
    with pytest.raises(AccResidencyError, match="residency"):
        AccResidency.from_dict({"routine": "r", "args": {"a": {"residency": "gpu"}}})


def test_missing_sidecar_file_is_an_error(tmp_path):
    with pytest.raises(AccResidencyError, match="not found"):
        AccResidency.from_file(tmp_path / "absent.acc_residency.json")


# ----- the real velocity SDFG ----------------------------------------------


@pytest.fixture(scope="session")
def velocity_sdfg():
    """The real ``velocity_tendencies`` SDFG (CPU storage throughout)."""
    cached = os.environ.get("DACE_FORTRAN_VELOCITY_SDFGZ")
    if cached:
        return dace.SDFG.from_file(cached)
    if not have_flang():
        pytest.skip("flang not on PATH and $DACE_FORTRAN_VELOCITY_SDFGZ unset")
    out = Path(dace.Config.get("default_build_folder")) / "_acc_residency_velocity"
    out.mkdir(parents=True, exist_ok=True)
    return build_sdfg(_VELOCITY_SRC.read_text(),
                      out,
                      name="velocity_tendencies",
                      entry="_QMmo_velocity_advectionPvelocity_tendencies").build()


@pytest.mark.long
def test_real_velocity_sdfg_stages_every_device_arg(velocity_sdfg, tmp_path):
    payload = {
        "routine": "velocity_tendencies",
        "source": "mo_velocity_advection.f90",
        "args": {},
        "unclassified": [],
    }
    for name in _ARRAY_ARGS:
        payload["args"][name] = {"residency": "device", "clause": "PRESENT", "evidence": "mo_velocity_advection.f90:1"}
    payload["unclassified"] = list(_SCALAR_ARGS)
    sidecar = tmp_path / "velocity_tendencies.acc_residency.json"
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")

    plan = _bil.acc_transfer_plan(velocity_sdfg, sidecar)

    assert plan.update_host == _ARRAY_ARGS
    assert plan.update_device == ("p_diag", "z_w_concorr_me", "z_kin_hor_e", "z_vt_ie")
    assert plan.use_device == ()

    body = _lines(_bil.render_icon_wrapper(plan))
    rate = body.index("CALL SYSTEM_CLOCK(COUNT_RATE=velo_rate)")
    entry = body.index("CALL SYSTEM_CLOCK(COUNT=velo_t0)")
    exit_ = body.index("CALL SYSTEM_CLOCK(COUNT=velo_t1)")
    call = next(i for i, line in enumerate(body) if line.startswith("CALL velocity_tendencies_dace("))

    assert body[rate + 1] == "!$ACC WAIT"
    assert body[entry - 1] == "!$ACC WAIT"
    assert body[exit_ - 1] == "!$ACC WAIT"
    assert body[entry + 1:call] == [f"!$ACC UPDATE HOST({name})" for name in _ARRAY_ARGS]

    after = body[call + 4:exit_ - 1]
    assert after == [f"!$ACC UPDATE DEVICE({name})" for name in plan.update_device]
    assert "!$ACC HOST_DATA" not in " ".join(body)


@pytest.mark.long
def test_real_velocity_sdfg_gpu_storage_takes_the_drift_path(velocity_sdfg):
    sdfg = dace.SDFG.from_json(velocity_sdfg.to_json())
    for name, desc in sdfg.arrays.items():
        if not desc.transient and isinstance(desc, dace.data.Array) and name.startswith("z_"):
            desc.storage = dace.StorageType.GPU_Global
    residency = _sidecar(device=("z_w_concorr_me", "z_kin_hor_e", "z_vt_ie"),
                         host=tuple(a for a in _ARGS if not a.startswith("z_")))
    plan = plan_acc_transfers(sdfg, residency, _ARGS)
    assert plan.use_device == ("z_w_concorr_me", "z_kin_hor_e", "z_vt_ie")
    assert plan.update_host == ()
    assert plan.update_device == ()


# ----- compile proofs -------------------------------------------------------


def _staged_wrapper() -> str:
    sdfg = _toy_sdfg({name: dace.StorageType.CPU_Heap for name in _ARRAY_ARGS}, written=("p_diag", "z_kin_hor_e"))
    plan = plan_acc_transfers(sdfg, _sidecar(device=_ARRAY_ARGS, host=_SCALAR_ARGS), _ARGS)
    assert plan.update_host and plan.update_device
    return _bil.render_icon_wrapper(plan)


def _compile(tmp_path: Path, compiler: str, extra: list) -> None:
    src = tmp_path / "wrapper_probe.f90"
    src.write_text(_STUB_F90 + "\n" + _staged_wrapper())
    subprocess.run([compiler, *extra, "-c", src.name, "-o", "wrapper_probe.o"],
                   cwd=str(tmp_path),
                   check=True,
                   capture_output=True,
                   text=True)


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_staged_wrapper_compiles_without_openacc(tmp_path):
    _compile(tmp_path, "gfortran", ["-ffree-line-length-none", "-Werror"])


@pytest.mark.skipif(shutil.which("nvfortran") is None, reason="nvfortran not on PATH")
def test_staged_wrapper_compiles_with_openacc(tmp_path):
    _compile(tmp_path, "nvfortran", ["-acc"])
