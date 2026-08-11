# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`dace_fortran.acc_residency`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dace_fortran.acc_residency import classify, extract_acc_residency, write_acc_residency_sidecar

REPO = Path(__file__).resolve().parents[1]
VELOCITY = REPO / "samples" / "velocity_tendencies" / "velocity_advection_acc.f90"

FIXTURE = """\
MODULE m_acc_fixture
  IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel(a_present, b_copyin, c_copyout, d_create, &
                    e_cont, f_nested, g_hostonly, h_none, n)
    REAL, INTENT(INOUT) :: a_present(:), b_copyin(:), c_copyout(:), d_create(:)
    REAL, INTENT(INOUT) :: e_cont(:), f_nested(:), g_hostonly(:), h_none(:)
    INTEGER, INTENT(IN) :: n
    INTEGER :: i
    REAL :: scratch(4)

    !$ACC DATA PRESENT(a_present) COPYIN(b_copyin) &
    !$ACC   COPYOUT(c_copyout) CREATE(d_create, scratch) &
    !$ACC   COPYIN(e_cont) COPYIN(f_nested) NO_CREATE(g_hostonly)

    !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1)
    !$ACC LOOP GANG VECTOR
    DO i = 1, n
      a_present(i) = b_copyin(i) + e_cont(i)
    END DO
    !$ACC END PARALLEL

    !$ACC DATA PRESENT(f_nested)
    !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1)
    DO i = 1, n
      c_copyout(i) = d_create(i) + f_nested(i)
    END DO
    !$ACC END PARALLEL
    !$ACC END DATA

    !$ACC END DATA
    h_none(1) = REAL(n)
  END SUBROUTINE kernel

END MODULE m_acc_fixture
"""


def _line_of(marker: str) -> int:
    """1-based fixture line containing ``marker``."""
    for i, line in enumerate(FIXTURE.splitlines(), start=1):
        if marker in line:
            return i
    raise AssertionError(f"marker {marker!r} not in fixture")


@pytest.fixture(name="payload", scope="module")
def _payload():
    return classify(FIXTURE, "kernel", "fixture.f90")


@pytest.mark.parametrize(("arg", "clause", "marker"), [
    ("a_present", "PRESENT", "PRESENT(a_present)"),
    ("b_copyin", "COPYIN", "COPYIN(b_copyin)"),
    ("c_copyout", "COPYOUT", "COPYOUT(c_copyout)"),
    ("d_create", "CREATE", "CREATE(d_create"),
])
def test_device_clauses(payload, arg, clause, marker):
    assert payload["args"][arg] == {
        "residency": "device",
        "clause": clause,
        "evidence": f"fixture.f90:{_line_of(marker)}",
        "ref": arg,
    }


def test_continuation_line_clause(payload):
    """``e_cont`` sits on the third physical line of the joined directive."""
    info = payload["args"]["e_cont"]
    assert info["residency"] == "device"
    assert info["clause"] == "COPYIN"
    assert info["evidence"] == f"fixture.f90:{_line_of('COPYIN(e_cont)')}"


def test_nested_data_region_wins(payload):
    """The inner ``PRESENT`` outranks the outer ``COPYIN`` for the same arg."""
    info = payload["args"]["f_nested"]
    assert info["residency"] == "device"
    assert info["clause"] == "PRESENT"
    assert info["evidence"] == f"fixture.f90:{_line_of('PRESENT(f_nested)')}"


def test_no_create_is_host(payload):
    info = payload["args"]["g_hostonly"]
    assert info["residency"] == "host"
    assert info["clause"] == "NO_CREATE"


def test_arg_without_clause_is_unclassified(payload):
    assert payload["unclassified"] == ["h_none", "n"]
    assert "h_none" not in payload["args"]
    assert "n" not in payload["args"]


def test_locals_are_not_reported(payload):
    """``scratch`` is CREATE-d but is not a dummy argument."""
    assert "scratch" not in payload["args"]
    assert "scratch" not in payload["unclassified"]


def test_default_present_is_not_evidence(payload):
    """``DEFAULT(PRESENT)`` must not classify anything."""
    assert all(info["evidence"] != "fixture.f90:0" for info in payload["args"].values())
    assert set(payload["args"]) | set(payload["unclassified"]) == {
        "a_present", "b_copyin", "c_copyout", "d_create", "e_cont", "f_nested", "g_hostonly", "h_none", "n"
    }


def test_routine_not_found():
    with pytest.raises(ValueError, match="not found"):
        classify(FIXTURE, "no_such_routine", "fixture.f90")


COMPONENT_FIXTURE = """\
MODULE m_acc_comp
  IMPLICIT NONE
  TYPE :: t_diag
    REAL, POINTER :: vt(:, :, :)
    REAL, POINTER :: vn_ie(:, :, :)
  END TYPE t_diag
CONTAINS
  SUBROUTINE comp_kernel(p_diag, z_vt_ie, jb, n)
    TYPE(t_diag), INTENT(INOUT) :: p_diag
    REAL, INTENT(INOUT) :: z_vt_ie(:, :, :)
    INTEGER, INTENT(IN) :: jb, n
    INTEGER :: i
    !$ACC DATA PRESENT(p_diag%vt, p_diag % vn_ie(:, :, jb)) &
    !$ACC   COPYIN(z_vt_ie(:, :, jb))
    DO i = 1, n
      p_diag%vt(i, 1, jb) = z_vt_ie(i, 1, jb)
    END DO
    !$ACC END DATA
  END SUBROUTINE comp_kernel
END MODULE m_acc_comp
"""


@pytest.fixture(name="comp_payload", scope="module")
def _comp_payload():
    return classify(COMPONENT_FIXTURE, "comp_kernel", "comp.f90")


def test_component_ref_normalises_to_base_arg(comp_payload):
    info = comp_payload["args"]["p_diag"]
    assert info["residency"] == "device"
    assert info["clause"] == "PRESENT"
    assert info["ref"] == "p_diag%vt"


def test_array_section_ref_is_retained(comp_payload):
    info = comp_payload["args"]["z_vt_ie"]
    assert info["residency"] == "device"
    assert info["ref"] == "z_vt_ie(:,:,jb)"


def test_subscript_identifiers_are_not_entities(comp_payload):
    """``jb`` appears only inside array-section subscripts; it must stay
    unclassified rather than inherit the clause of its enclosing entity."""
    assert comp_payload["unclassified"] == ["jb", "n"]


CPP_FIXTURE = """\
SUBROUTINE cpp_kernel(a, b, n)
  REAL, INTENT(INOUT) :: a(:), b(:)
  INTEGER, INTENT(IN) :: n
  INTEGER :: i
  !$ACC ENTER DATA COPYIN(a)
#ifdef _OPENACC
  !$ACC ENTER DATA CREATE(b)
  DO i = 1, n, 2
#else
  b(1) = 1.0
  DO i = 1, n
#endif
    a(i) = b(i)
  END DO
  !$ACC EXIT DATA DELETE(a, b)
END SUBROUTINE cpp_kernel
"""


def test_cpp_arm_selection_keeps_the_openacc_arm():
    """The two cpp arms carry structurally different DO statements, so the
    source only parses under arm selection; ``_OPENACC`` is on by default."""
    payload = classify(CPP_FIXTURE, "cpp_kernel", "cpp.f90")
    assert payload["args"]["b"]["clause"] == "CREATE"
    assert payload["args"]["a"]["clause"] == "COPYIN"
    assert payload["unclassified"] == ["n"]


def test_cpp_arm_selection_respects_explicit_defines():
    """Without ``_OPENACC`` the #else arm is live and ``b`` loses its CREATE;
    the unguarded EXIT DATA DELETE is then its best (device) evidence."""
    payload = classify(CPP_FIXTURE, "cpp_kernel", "cpp.f90", defines=())
    assert payload["args"]["a"]["clause"] == "COPYIN"
    assert payload["args"]["b"]["clause"] == "DELETE"


def test_write_sidecar(tmp_path):
    src = tmp_path / "fixture.f90"
    src.write_text(FIXTURE)
    out = write_acc_residency_sidecar(src, "kernel", tmp_path / "sdfg")
    assert out.name == "kernel.acc_residency.json"
    payload = json.loads(out.read_text())
    assert payload["routine"] == "kernel"
    assert payload["source"] == "fixture.f90"
    assert payload["args"]["a_present"]["residency"] == "device"


def test_cli(tmp_path):
    src = tmp_path / "fixture.f90"
    src.write_text(FIXTURE)
    out = tmp_path / "res.json"
    subprocess.run(
        [sys.executable, "-m", "dace_fortran.acc_residency",
         str(src), "--routine", "kernel", "--out",
         str(out)],
        check=True,
        cwd=REPO)
    assert json.loads(out.read_text())["args"]["b_copyin"]["clause"] == "COPYIN"


@pytest.mark.skipif(not VELOCITY.is_file(), reason="ACC-annotated velocity twin not present")
def test_velocity_tendencies_classification():
    payload = extract_acc_residency(VELOCITY, "velocity_tendencies")
    device = {a for a, i in payload["args"].items() if i["residency"] == "device"}
    assert device == {"p_prog", "p_patch", "p_int", "p_metrics", "p_diag", "z_w_concorr_me", "z_kin_hor_e", "z_vt_ie"}
    assert all(payload["args"][a]["clause"] == "PRESENT" for a in device)
    assert payload["unclassified"] == ["ntnd", "istep", "lvn_only", "dtime", "dt_linintp_ubc", "ldeepatmo"]
