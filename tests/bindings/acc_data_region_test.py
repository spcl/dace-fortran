# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone OpenACC direction: host caller, offloaded SDFG.

The mirror image of the ICON lane in ``test_icon_wrapper_acc.py``.  There the caller
already owns a device copy and the wrapper stages it DOWN to a host SDFG; here the
caller is ordinary Fortran holding host buffers and an offload pass has moved the
SDFG's copies UP to the GPU, so the wrapper opens an ``!$ACC DATA`` region instead.
No sidecar is involved -- the plan comes off the frozen signature, which remembers
both the caller's location and the kernel's.
"""

import dace
import pytest

from dace_fortran.bindings.acc_transfers import (
    plan_frozen_transfers,
    render_data_close,
    render_data_open,
    render_host_data_open,
)
from dace_fortran.bindings.block_builders import splice_acc_staging
from dace_fortran.bindings.frozen_signature import HOST_STORAGE, FrozenArg, FrozenSignature, refreeze

_ENTRY = "compute"


def _arg(name: str, intent: str, **kw) -> FrozenArg:
    return FrozenArg(fortran_name=name,
                     sdfg_name=name,
                     kind="array",
                     dtype="float64",
                     rank=1,
                     shape=("n", ),
                     intent=intent,
                     **kw)


def _signature(*args: FrozenArg) -> FrozenSignature:
    return FrozenSignature(entry=_ENTRY, mangled="_QPcompute", args=args, free_symbols=("n", ))


def _offloaded(*names: str) -> FrozenSignature:
    """Snapshot of a kernel whose ``rd``/``wr``/``rw`` buffers were moved to the GPU."""
    sdfg = dace.SDFG(_ENTRY)
    sdfg.add_symbol("n", dace.int64)
    for name in ("host", "rd", "rw", "wr"):
        sdfg.add_array(name, shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    sdfg._frozen_signature = _signature(_arg("host", "inout"), _arg("rd", "in"), _arg("rw", "inout"), _arg("wr", "out"))
    for name in names:
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    return refreeze(sdfg)


def _blocks() -> dict:
    return {
        'wrapper_body':
        "  ! body\n",
        'wrapper_tail':
        "\n".join([
            f"  call dace_program_{_ENTRY}(handle, &",
            "    a, b)",
            f"  end subroutine {_ENTRY}_dace",
        ]),
    }


# ----- planning -------------------------------------------------------------


def test_a_kernel_left_on_the_host_plans_nothing():
    plan = plan_frozen_transfers(_offloaded())
    assert not plan.active
    assert plan.data_region == ()
    assert splice_acc_staging(_blocks(), _ENTRY, plan) == _blocks()


def test_clause_follows_the_intent_of_each_relocated_arg():
    plan = plan_frozen_transfers(_offloaded("rd", "rw", "wr"))
    assert plan.copyin == ("rd", )
    assert plan.copy == ("rw", )
    assert plan.copyout == ("wr", )
    assert plan.active


def test_an_arg_left_on_the_host_gets_no_clause():
    plan = plan_frozen_transfers(_offloaded("rd"))
    assert plan.data_region == (("COPYIN", "rd"), )
    assert "host" not in plan.use_device


def test_moving_back_to_the_host_retires_the_region():
    """GPU -> CPU is a plan change, not a stale clause: refreeze clears the relocation."""
    sdfg = dace.SDFG(_ENTRY)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    sdfg._frozen_signature = _signature(_arg("a", "inout", device_storage="GPU_Global"))
    assert plan_frozen_transfers(sdfg._frozen_signature).active

    sdfg.arrays["a"].storage = dace.StorageType.CPU_Heap
    assert not plan_frozen_transfers(refreeze(sdfg)).active


def test_every_relocated_arg_is_also_use_device():
    """Inside the region the SDFG must be handed the device address, not the host one."""
    plan = plan_frozen_transfers(_offloaded("rd", "rw", "wr"))
    assert plan.use_device == ("rd", "rw", "wr")


def test_scalars_never_reach_the_region():
    frozen = _signature(_arg("a", "inout", device_storage="GPU_Global"),
                        FrozenArg(fortran_name="alpha", sdfg_name="alpha", kind="scalar", dtype="float64", rank=0))
    plan = plan_frozen_transfers(frozen)
    assert plan.data_region == (("COPY", "a"), )
    assert "alpha" not in plan.use_device


def test_storage_defaults_to_the_host_when_the_snapshot_is_silent():
    assert plan_frozen_transfers(_signature(_arg("a", "inout"))).active is False
    assert _arg("a", "inout").storage == HOST_STORAGE


# ----- rendering ------------------------------------------------------------


def test_region_is_one_clause_per_line_and_grouped():
    lines = render_data_open(plan_frozen_transfers(_offloaded("rd", "rw", "wr")), "  ")
    assert lines == [
        "  !$ACC DATA &",
        "  !$ACC   COPYIN(rd) &",
        "  !$ACC   COPY(rw) &",
        "  !$ACC   COPYOUT(wr)",
    ]


def test_only_the_last_directive_line_lacks_a_continuation():
    lines = render_data_open(plan_frozen_transfers(_offloaded("rd", "rw", "wr")), "  ")
    assert all(line.endswith("&") for line in lines[:-1])
    assert not lines[-1].endswith("&")


def test_an_inactive_plan_renders_no_directive():
    plan = plan_frozen_transfers(_offloaded())
    assert render_data_open(plan) == []
    assert render_data_close(plan) == []


def test_rendering_is_deterministic():
    a, b = (render_data_open(plan_frozen_transfers(_offloaded("rd", "rw", "wr"))) for _ in range(2))
    assert a == b


# ----- splicing -------------------------------------------------------------


def test_the_data_region_encloses_host_data_around_the_call():
    plan = plan_frozen_transfers(_offloaded("rd", "rw", "wr"))
    tail = splice_acc_staging(_blocks(), _ENTRY, plan)['wrapper_tail'].splitlines()
    order = [i for i, ln in enumerate(tail) if "!$ACC DATA" in ln or "HOST_DATA" in ln or "END DATA" in ln]
    kinds = [tail[i].strip() for i in order]

    assert kinds[0].startswith("!$ACC DATA")
    assert kinds[1].startswith("!$ACC HOST_DATA USE_DEVICE")
    assert kinds[-2] == "!$ACC END HOST_DATA"
    assert kinds[-1] == "!$ACC END DATA"


def test_the_whole_continued_call_sits_inside_the_region():
    plan = plan_frozen_transfers(_offloaded("rd"))
    tail = splice_acc_staging(_blocks(), _ENTRY, plan)['wrapper_tail'].splitlines()
    call_at = next(i for i, ln in enumerate(tail) if ln.lstrip().startswith(f"call dace_program_{_ENTRY}("))
    close_at = next(i for i, ln in enumerate(tail) if ln.strip() == "!$ACC END DATA")

    assert tail[call_at + 1].strip() == "a, b)"
    assert close_at > call_at + 1


@pytest.mark.parametrize("moved,expected", [
    (("rd", ), "COPYIN(rd)"),
    (("wr", ), "COPYOUT(wr)"),
    (("rw", ), "COPY(rw)"),
])
def test_each_direction_reaches_the_emitted_wrapper(moved, expected):
    plan = plan_frozen_transfers(_offloaded(*moved))
    tail = splice_acc_staging(_blocks(), _ENTRY, plan)['wrapper_tail']
    assert expected in tail
    assert render_host_data_open(plan, "  ")[0].endswith(f"USE_DEVICE({moved[0]})")
