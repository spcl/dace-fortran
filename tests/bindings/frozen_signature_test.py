"""``FrozenSignature`` self-contained tests -- JSON round-trip, drift detection.
No HLFIR pipeline or flang-new needed; exercises binding-side plumbing in isolation.
"""

from pathlib import Path

import dace
import pytest

from dace_fortran.bindings import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)
from dace_fortran.bindings.frozen_signature import HOST_STORAGE, refreeze


def _demo_signature() -> FrozenSignature:
    return FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="a",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="in"),
            FrozenArg(fortran_name="b",
                      sdfg_name="b",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
        ),
        free_symbols=("m", "n"),
    )


def _rich_signature() -> FrozenSignature:
    """Every field driven off its default, so comparing after a round-trip means something."""
    return FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="st%u",
                      sdfg_name="st_u",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout",
                      from_struct_member="st%u",
                      layout="transpose",
                      is_written=True,
                      aos_origin_mod="mo_becxx",
                      aos_origin_struct="becxx",
                      aos_member_path="k",
                      aos_outer_rank=1,
                      global_alloc_inside=True,
                      aos_struct_pointer=True,
                      aos_member_pointer=True,
                      module_origin_allocatable=True,
                      module_origin_pointer=True),
            FrozenArg(fortran_name="tol", sdfg_name="tol", kind="scalar", dtype="float64", rank=0, intent="in"),
            FrozenArg(fortran_name="comm", sdfg_name="comm", kind="mpi_comm", dtype="MPI_Comm", rank=0, intent="in"),
        ),
        free_symbols=("m", "n"),
        module_symbol_origins={"tol": ("mo_cfg", "tol")},
        user_comm_source="comm",
    )


def test_json_roundtrip(tmp_path: Path):
    fs = _demo_signature()
    p = tmp_path / "sig.json"
    fs.to_json(str(p))
    loaded = FrozenSignature.from_json(str(p))
    assert loaded == fs


def test_signature_survives_sdfg_roundtrip(tmp_path: Path):
    """The snapshot rides along in the SDFG file and comes back byte-for-byte identical.

    It used to be a plain Python attribute, which save/load silently dropped -- leaving the
    binding emitter with nothing to emit against unless the frontend ran in the same process.
    """
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("st_u", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _rich_signature()
    sdfg._frozen_signature = fs

    path = tmp_path / "compute.sdfgz"
    sdfg.save(str(path), compress=True)
    loaded = dace.SDFG.from_file(str(path))

    assert loaded._frozen_signature == fs
    assert loaded.frontend_metadata["frozen_signature"] == fs.to_dict()


def test_signature_absent_on_a_plain_sdfg():
    assert dace.SDFG("plain")._frozen_signature is None
    assert dace.SDFG("plain").frontend_metadata == {}


def test_refreeze_drops_specialized_scalar_and_symbol():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    # Snapshot still claims a scalar and a symbol the live SDFG has folded away -- exactly
    # what specialization leaves behind.
    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="a", sdfg_name="a", kind="array", dtype="float64", rank=1, shape=("n", )),
            FrozenArg(fortran_name="alpha", sdfg_name="alpha", kind="scalar", dtype="float64", rank=0),
            FrozenArg(fortran_name="m", sdfg_name="m", kind="symbol", dtype="int64", rank=0),
        ),
        free_symbols=("m", "n"),
    )

    new = refreeze(sdfg)
    assert [a.sdfg_name for a in new.args] == ["a"]
    assert new.free_symbols == ("n", )
    assert sdfg._frozen_signature == new


def test_refreeze_refuses_dropped_array():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="a", sdfg_name="a", kind="array", dtype="float64", rank=1, shape=("n", )),
            FrozenArg(fortran_name="b", sdfg_name="b", kind="array", dtype="float64", rank=1, shape=("n", )),
        ),
        free_symbols=("n", ),
    )

    with pytest.raises(SignatureDriftError, match="only scalars and free symbols may shrink"):
        refreeze(sdfg)


# ----- OpenACC residency ---------------------------------------------------------
#
# A Fortran caller always hands over host memory.  Offloading moves the KERNEL's copy
# to the device, and the gap between the two locations is what the binding closes with
# OpenACC data clauses -- so the snapshot has to remember both.  ``arglist()`` sorts,
# hence the alphabetical arg order in the fixtures below.


def _acc_sdfg(*names: str) -> "dace.SDFG":
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("n", dace.int64)
    for name in names:
        sdfg.add_array(name, shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    return sdfg


def _acc_arg(name: str, intent: str, **kw) -> FrozenArg:
    return FrozenArg(fortran_name=name,
                     sdfg_name=name,
                     kind="array",
                     dtype="float64",
                     rank=1,
                     shape=("n", ),
                     intent=intent,
                     **kw)


def test_storage_defaults_to_the_host():
    assert HOST_STORAGE == "CPU_Heap"
    arg = _acc_arg("a", "inout")
    assert arg.storage == HOST_STORAGE
    assert arg.device_storage == ""
    assert arg.acc_data_clause == ""


def test_refreeze_records_a_cpu_to_gpu_relocation():
    sdfg = _acc_sdfg("host", "rd", "rw", "wr")
    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(_acc_arg("host", "inout"), _acc_arg("rd", "in"), _acc_arg("rw", "inout"), _acc_arg("wr", "out")),
        free_symbols=("n", ),
    )

    for name in ("rd", "rw", "wr"):
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global

    by_name = {a.sdfg_name: a for a in refreeze(sdfg).args}
    assert by_name["rd"].storage == HOST_STORAGE
    assert by_name["rd"].device_storage == "GPU_Global"
    assert [by_name[n].acc_data_clause for n in ("rd", "wr", "rw", "host")] == \
        ["copyin", "copyout", "copy", ""]
    assert by_name["host"].device_storage == ""


def test_refreeze_clears_device_storage_on_the_way_back():
    """A pass that pulls an array back to the host retires its data clause."""
    sdfg = _acc_sdfg("a")
    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(_acc_arg("a", "inout", device_storage="GPU_Global"), ),
        free_symbols=("n", ),
    )
    assert sdfg._frozen_signature.args[0].acc_data_clause == "copy"

    sdfg.arrays["a"].storage = dace.StorageType.CPU_Heap
    back = refreeze(sdfg).args[0]
    assert back.device_storage == ""
    assert back.acc_data_clause == ""


def test_refreeze_ignores_host_side_storage_churn():
    """``CPU_Pinned`` is still the caller's pointer -- no transfer to bridge."""
    sdfg = _acc_sdfg("a")
    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(_acc_arg("a", "inout"), ),
        free_symbols=("n", ),
    )
    sdfg.arrays["a"].storage = dace.StorageType.CPU_Pinned

    assert refreeze(sdfg).args[0].acc_data_clause == ""


def test_acc_clause_is_copy_for_a_written_input():
    """``intent(in)`` on a module global the kernel writes still needs the way back."""
    arg = _acc_arg("g", "in", is_written=True, device_storage="GPU_Global")
    assert arg.acc_data_clause == "copy"


def test_scalars_and_symbols_never_get_a_data_clause():
    """They ride the argument list by value, so there is no buffer to move."""
    for kind in ("scalar", "symbol"):
        arg = FrozenArg(fortran_name="alpha",
                        sdfg_name="alpha",
                        kind=kind,
                        dtype="float64",
                        rank=0,
                        device_storage="GPU_Global")
        assert arg.acc_data_clause == ""


def test_device_relocation_survives_the_sdfg_roundtrip(tmp_path: Path):
    """The residency decision has to reach the stage that emits the binding."""
    sdfg = _acc_sdfg("a")
    sdfg.arrays["a"].storage = dace.StorageType.GPU_Global
    sdfg._frozen_signature = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(_acc_arg("a", "in"), ),
        free_symbols=("n", ),
    )
    refreeze(sdfg)

    path = tmp_path / "compute.sdfgz"
    sdfg.save(str(path), compress=True)
    arg = dace.SDFG.from_file(str(path))._frozen_signature.args[0]

    assert (arg.storage, arg.device_storage) == (HOST_STORAGE, "GPU_Global")
    assert arg.acc_data_clause == "copyin"


def test_verify_against_happy_path():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    # No raise.
    fs.verify_against(sdfg)


def test_drift_detection_arg_reordering():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    # Swap order vs the snapshot.
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    # arglist sorts alphabetically -- (a, b) either way -- so mutate the snapshot itself to flip order.
    swapped = FrozenSignature(
        entry=fs.entry,
        mangled=fs.mangled,
        args=(fs.args[1], fs.args[0]),
        free_symbols=fs.free_symbols,
    )
    with pytest.raises(SignatureDriftError):
        swapped.verify_against(sdfg)


def test_drift_detection_dtype_change():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    # Use float32 on the live SDFG; frozen says float64.
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float32, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    with pytest.raises(SignatureDriftError) as exc:
        fs.verify_against(sdfg)
    assert "dtype" in str(exc.value)


def test_drift_detection_extra_free_symbol():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("extra", dace.int64)  # not in snapshot
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    # Use `extra` somewhere so it counts as used.
    s0 = sdfg.add_state("s0")
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="extra > 0"))

    fs = _demo_signature()
    with pytest.raises(SignatureDriftError):
        fs.verify_against(sdfg)
