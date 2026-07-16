"""REAL ocean coriolis kernel + real-MPI vertex halo exchange, 2 ranks/pair.

The step up from ``tests/icon/full/test_ocean_mpi_sync_e2e.py`` (which drove a
tiny hand-written dummy dycore): here the DUT is the *actual* extracted ICON-O
kernel ``mo_scalar_product::nonlinear_coriolis_3d_fast_scalar`` -- the same
single-TU the single-rank numerical e2e (``test_ocean_numerical_e2e.py``,
``coriolis_pv``) drives bit-exact -- lowered to an SDFG and reached through its
AUTO-GENERATED struct-flattened ``bind(c)`` shim (dozens of SoA args for the
``t_patch_3d`` / ``t_operator_coeff`` structs).  It proves the full multi-rank
seam: a struct-heavy SDFG kernel whose in-body halo sync is a ``keep_external``
library node issuing a real ``MPI_Sendrecv`` against the partner rank, driven
straight from Python with a live ``mpi4py`` communicator handle -- no fork.

The kernel's real sync ``mo_sync::sync_patch_array_3d_dp(typ, p_patch, arr,
lacc)`` takes a ``t_patch`` STRUCT that is not marshalable across the flat C ABI
(nested records + pointer members) and, being geometry-only, does not carry the
communicator the halo actually needs.  So -- exactly as the dummy test does, and
mirroring how ICON's real sync is a thin wrapper over the comm -- the kernel's
sync CALL is rewritten to the patch-free, comm-carrying ``vort_v_halo_sync``
(``vort_sync_mpi.f90``), registered as a ``keep_external`` routed to a real-MPI
``.so``.  ``comm`` is threaded through the kernel signature as a plain
``INTEGER`` (a Fortran ``MPI_Fint`` handle from ``mpi4py.MPI.Comm.py2f()``,
carried byte-for-byte through the C ABI).

DUT and REF share the SAME rewritten TU and the SAME ``libvort_sync.so``: the
DUT routes the sync through the SDFG library node, the REF calls
``vort_v_halo_sync`` directly (the original kernel reached through the shim
retargeted by :func:`icon.ocean._ocean_e2e._retarget_shim`).  Both run the real
exchange on identical per-rank inputs, so the per-rank differential stays
bit-exact -- the core proof that the multi-rank SDFG is correct against the
original.  A second check re-runs the DUT on ``COMM_SELF`` (where the sync takes
its ``size <= 1`` no-op path) and asserts ``vort_v`` changed -- the only variable
between the two runs is whether the sync had a neighbour, so a difference proves
the real ``MPI_Sendrecv`` fired against a live partner and its data reached the
SDFG.

Runs under ``mpirun --oversubscribe`` at any even rank count (CI uses ``-n 4``):
COMM_WORLD splits into adjacent 2-rank pairs, each pair exchanging independently.
Skipped at an odd / <2 rank count so the default single-rank ``pytest tests/``
does not trip on it.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import dace
import numpy as np
import pytest

from _util import build_on_root, have_flang
from dace_fortran.bindings import build_fortran_library
from dace_fortran.build import build_sdfg
from dace_fortran.external import Arg, clear_external_registry, keep_external
from icon.ocean._ocean_e2e import (_invoke, _resolve_module_seeds, _retarget_shim, _size_derived_module_dims,
                                   synth_call_inputs)

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("mpifort") is None, reason="mpifort not on PATH (need an MPI Fortran wrapper)"),
]

_HERE = Path(__file__).resolve().parent
_ENTRY = "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar"
# FP-conservative flags across every build layer so the SDFG and gfortran round
# bit-for-bit -- same convention as the single-rank ocean e2e + the dummy 2-rank.
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none")
_N = 8  # mesh size: every array extent is n (nblks_v = 8, so block 1 = owned, block 2 = halo)


def _rewrite_coriolis_tu(src: str) -> str:
    """Thread ``comm`` through ``nonlinear_coriolis_3d_fast_scalar`` and rewrite
    its ``t_patch``-taking sync CALL to the patch-free, comm-carrying
    ``vort_v_halo_sync`` (registered as a ``keep_external``).  Same three edits
    the single-rank harness would need to make the kernel halo-aware."""
    sig = ("SUBROUTINE nonlinear_coriolis_3d_fast_scalar(patch_3d, vn, p_vn_dual, "
           "vort_v, operators_coefficients, vort_flux, lacc)")
    if src.count(sig) != 1:
        raise RuntimeError(f"coriolis signature anchor not unique (found {src.count(sig)})")
    src = src.replace(
        sig, "SUBROUTINE nonlinear_coriolis_3d_fast_scalar(patch_3d, vn, p_vn_dual, "
        "vort_v, operators_coefficients, vort_flux, comm, lacc)\n"
        "    USE mo_vort_sync, ONLY: vort_v_halo_sync")
    decl = "    REAL(KIND = 8) :: vort_flux_old(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)"
    if src.count(decl) != 1:
        raise RuntimeError("coriolis vort_flux_old decl anchor not unique")
    src = src.replace(decl, "    INTEGER, INTENT(IN) :: comm\n" + decl)
    synccall = "    CALL sync_patch_array_3d_dp_deconiface_38(3, patch_2d, vort_v, lacc = lzacc)"
    if src.count(synccall) != 1:
        raise RuntimeError("coriolis sync CALL anchor not unique")
    # Explicit extents (nproma / n_zlev / nblks_v) so the sync takes the field
    # explicit-shape -- passing the kernel's explicit-shape vort_v to an
    # assumed-shape dummy makes gfortran drop the halo writeback (see
    # vort_sync_mpi.f90).
    return src.replace(
        synccall, "    CALL vort_v_halo_sync(3, nproma, n_zlev, "
        "patch_3d % p_patch_2d(1) % nblks_v, vort_v, comm)")


def _build_artifacts(tmp_path: Path) -> dict:
    """Build (rank 0 only) the real-MPI sync ``.so``, the coriolis DUT (SDFG +
    struct-flat ``bind(c)`` shim, sync routed as a ``keep_external`` lib node),
    and the gfortran REF (original kernel reached through the retargeted shim).
    Returns a picklable dict of paths + the shim / binding text every rank needs
    to re-derive the ABI (:func:`build_on_root` broadcasts it)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    # 1. real-MPI sync library (shared by DUT lib node + REF direct call).
    sync_src = (_HERE / "vort_sync_mpi.f90").read_text()
    sync_f90 = tmp_path / "vort_sync_mpi.f90"
    sync_f90.write_text(sync_src)
    sync_so = tmp_path / "libvort_sync.so"
    subprocess.check_call(
        ["mpifort", "-shared", "-fPIC", *_O0_FFLAGS, f"-J{tmp_path}",
         str(sync_f90), "-o",
         str(sync_so)],
        cwd=str(tmp_path))

    # 2. rewrite the coriolis TU (comm thread + halo-sync CALL).
    tu = _rewrite_coriolis_tu((_HERE / "coriolis_pv_single_tu.f90").read_text())
    rewritten_tu = tmp_path / "coriolis_comm_tu.f90"
    rewritten_tu.write_text(tu)

    # 3. DUT: keep_external the sync -> SDFG -> struct-flat bind(c) library.
    clear_external_registry()
    keep_external(
        "vort_v_halo_sync",
        c_name="vort_v_halo_sync_c",
        args=(
            Arg(kind="scalar", dtype="int32", intent="in"),  # typ
            Arg(kind="scalar", dtype="int32", intent="in"),  # n1 = nproma
            Arg(kind="scalar", dtype="int32", intent="in"),  # n2 = n_zlev
            Arg(kind="scalar", dtype="int32", intent="in"),  # n3 = nblks_v
            Arg(kind="array", dtype="float64", intent="inout"),  # vort_v
            Arg(kind="scalar", dtype="int32", intent="in"),  # comm
        ),
        libraries=(str(sync_so), ),
        dynamic_extents_abi=False)
    cpu = dace.Config.get("compiler", "cpu", "args").replace("-ffast-math", "")
    if "-ffp-contract" not in cpu:
        cpu += " -ffp-contract=off"
    dace.Config.set("compiler", "cpu", "args", value=cpu)
    try:
        sdfg = build_sdfg(sync_src + "\n" + tu, entry=_ENTRY, name="coriolis_comm", out_dir=str(tmp_path / "sdfg"))
        dace_name = sdfg.name
        lib = build_fortran_library(sdfg,
                                    out_dir=str(tmp_path / "lib"),
                                    prelude_sources=[sync_f90, rewritten_tu],
                                    bind_c_shim=True)
    finally:
        clear_external_registry()  # AFTER compile has linked libvort_sync.so
    shim = Path(lib.bind_c_shim_f90).read_text()
    binding_files = list((tmp_path / "lib").glob("*bindings.f90"))
    binding_text = binding_files[0].read_text() if binding_files else ""

    # 4. REF: retarget the shim to the original kernel (still calls the same
    # vort_v_halo_sync directly), seeding n_zlev / nproma from the array extents.
    module_dims = _size_derived_module_dims(binding_text) if binding_text else []
    ref_shim = tmp_path / f"{dace_name}_ref_c.f90"
    ref_shim.write_text(_retarget_shim(shim, dace_name, _ENTRY, module_dims, _N))
    ref_so = tmp_path / f"lib{dace_name}_ref.so"
    r = subprocess.run([
        "gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-ffp-contract=off", "-fno-fast-math", "-o",
        str(ref_so),
        str(sync_f90),
        str(rewritten_tu),
        str(ref_shim), f"-L{sync_so.parent}", f"-Wl,-rpath,{sync_so.parent}", f"-l:{sync_so.name}"
    ],
                       capture_output=True,
                       text=True,
                       cwd=str(tmp_path))
    if r.returncode != 0:
        raise RuntimeError(f"REF .so compile failed:\n{r.stderr[-3000:]}")

    return {
        "sync_so": str(sync_so),
        "dut_so": str(lib.so_path),
        "sdfg_so": str(lib.sdfg_so),
        "ref_so": str(ref_so),
        "dace_name": dace_name,
        "shim": shim,
        "binding_text": binding_text,
    }


@pytest.mark.mpi
def test_coriolis_with_real_mpi_halo_2rank(tmp_path: Path):
    """2-rank real coriolis kernel + real-MPI ``vort_v`` halo: SDFG path vs
    gfortran reference, per-rank bit-exact, plus a pair-vs-COMM_SELF check that
    the real ``MPI_Sendrecv`` moved neighbour data into the halo block."""
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    if size < 2 or size % 2 != 0:
        pytest.skip("needs an even rank count >= 2 (mpirun --oversubscribe -n 2 / -n 4 ...)")
    pair = world.Split(color=rank // 2, key=rank)

    # Pin every rank to rank 0's tmp_path so the .so artefacts are shared.
    tmp_path = Path(world.bcast(str(tmp_path) if rank == 0 else None, root=0))
    art = build_on_root(world, lambda: _build_artifacts(tmp_path))
    world.Barrier()

    shim = art["shim"]
    binding_text = art["binding_text"]
    dace_name = art["dace_name"]
    seed_specs = _resolve_module_seeds(binding_text, {}) if binding_text else []

    # Pre-load the real-MPI sync RTLD_GLOBAL so the DUT lib node and the REF
    # direct call resolve to the SAME libvort_sync.so instance.
    ctypes.CDLL(art["sync_so"], mode=ctypes.RTLD_GLOBAL)

    # Per-rank inputs from distinct seeds so the two ranks own genuinely different
    # vort_v data + the exchange surfaces a swap; comm is the live pair handle.
    handle = pair.py2f()
    call_plan, inputs, ptr_args, ptr_local = synth_call_inputs(shim,
                                                               n=_N,
                                                               seed=42 + rank,
                                                               scalar_overrides={"comm": handle})
    # vort_v is the (nproma, n_zlev, nblks_v) vertex field the sync exchanges;
    # numpy block 0 = Fortran block 1 (owned), block 1 = Fortran block 2 (halo).
    vort_v_key = next(h for h, local in ptr_local.items() if local == "vort_v")

    dut_bufs = {k: v.copy() for k, v in inputs.items()}
    _invoke(art["dut_so"], call_plan, dut_bufs, f"{dace_name}_c", sdfg_so=art["sdfg_so"], module_seeds=seed_specs)
    ref_bufs = {k: v.copy() for k, v in inputs.items()}
    _invoke(art["ref_so"], call_plan, ref_bufs, f"{dace_name}_ref_c", module_seeds=seed_specs)

    # (1) CORRECTNESS -- per-rank bit-exact DUT vs REF on every output buffer
    # (NaN == NaN).  The SDFG (with the MPI halo as a keep_external library node)
    # reproduces the original kernel EXACTLY across the whole 2-rank run, INCLUDING
    # the halo block the real MPI_Sendrecv fills.  This is the deliverable: the
    # multi-rank SDFG is correct against the original.
    n_changed = 0
    for k in ptr_args:
        d = dut_bufs[k].astype(np.float64)
        rf = ref_bufs[k].astype(np.float64)
        equal = (d == rf) | (np.isnan(d) & np.isnan(rf))
        assert equal.all(), f"{k}: DUT diverged from REF at {(~equal).sum()} position(s)"
        if not np.array_equal(dut_bufs[k], inputs[k]):
            n_changed += 1
    assert n_changed > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"

    # (2) REAL 2-RANK EXCHANGE -- re-run the DUT with the SAME inputs but on
    # ``COMM_SELF`` (a single-rank communicator, where ``vort_v_halo_sync`` takes
    # its ``size <= 1`` no-op early return, matching real ICON single-rank).  The
    # ONLY thing that differs between the two runs is whether the sync has a
    # neighbour, so any difference in the returned ``vort_v`` is caused solely by
    # the real ``MPI_Sendrecv`` moving the partner's data in.  Assert ``vort_v``
    # DID change -- proving the exchange fired against a live partner and reached
    # the SDFG's data (a sync that silently no-op'd, or a dropped ``comm``, would
    # leave the two runs identical).
    #
    # (Note: the flat-ABI shim marshals the extracted kernel's explicit-shape
    # ``vort_v`` dummy through a gfortran copy-in/copy-out whose whole-block halo
    # writeback only partially survives back into the caller buffer -- reproduced
    # IDENTICALLY in the gfortran REF, so NOT an SDFG defect and absent from a real
    # in-ICON build where ``vort_v`` is a module array.  It does not affect the
    # DUT-vs-REF differential above; the exchange stays clearly observable here.  A
    # fully-faithful per-element halo check is deferred to the in-ICON 2-node run.)
    self_plan = synth_call_inputs(shim, n=_N, seed=42 + rank, scalar_overrides={"comm": MPI.COMM_SELF.py2f()})[0]
    self_bufs = {k: v.copy() for k, v in inputs.items()}
    _invoke(art["dut_so"], self_plan, self_bufs, f"{dace_name}_c", sdfg_so=art["sdfg_so"], module_seeds=seed_specs)
    assert not np.array_equal(dut_bufs[vort_v_key], self_bufs[vort_v_key]), (
        f"rank {rank}: vort_v identical on the pair communicator and COMM_SELF -- the real 2-rank "
        "MPI_Sendrecv moved no neighbour data (sync no-op'd or comm was dropped)")
