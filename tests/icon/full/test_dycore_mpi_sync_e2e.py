"""Dycore + real-MPI halo-exchange ``sync_patch_array`` e2e (2 ranks/pair).

Runs under ``mpirun --oversubscribe`` at any even rank count (CI uses -n 4): COMM_WORLD
splits into adjacent 2-rank pairs, each running the symmetric halo swap independently. Each
rank owns block 1 (data) of a per-rank ``(nproma, nlev, 2)`` field; the dycore fills it via
a deterministic formula, then ``sync_patch_array`` issues a real ``MPI_Sendrecv`` so block 2
ends up equal to the neighbor's block 1.

``sync_patch_array`` is ICON-shaped but tiny (no full ICON build needed). The MPI
communicator crosses as a Fortran ``MPI_Fint`` (= C ``int`` under OpenMPI) straight from
``mpi4py``'s ``comm.py2f()``, byte-for-byte, no ``MPI_Comm_f2c`` round-trip. The dycore+sync
bundle compiles as both the gfortran reference and the DaCe SDFG; comparison is 1-ULP +
bit-exact per element.

Skipped under an odd rank count or fewer than 2 ranks (default single-rank ``pytest tests/``
doesn't trip it)."""
import ctypes
import shutil
import subprocess
from pathlib import Path

import dace
import numpy as np
import pytest

from _util import build_on_root, build_sdfg, have_flang
from dace_fortran.bindings import build_fortran_library
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("mpifort") is None, reason="mpifort not on PATH (need an MPI Fortran wrapper)"),
]

# Matching FP-conservative flags across every build layer so SDFG+gfortran arithmetic match
# bit-for-bit (same convention as the standalone single-rank dycore test).
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none")
_O0_CXX_FLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-Wno-unused-parameter", "-Wno-unused-label")

# Dummy halo-exchange sync_patch_array: ICON-shaped ((tag, field, comm) signature), minimal
# body exercising a real MPI_Sendrecv. MPI constants are hard-coded to OpenMPI values (not
# `use mpi`) to avoid a flang-vs-OpenMPI .mod dependency at SDFG-bridge build time -- same
# trick as mpi_sendrecv_e2e_test.py.
_SYNC_MPI_SRC = r"""
module mo_sync_mpi
  use iso_c_binding
  implicit none
  ! OpenMPI values; matches what mpi4py.MPI.Comm.py2f() hands out
  ! (Fortran handles are plain ints under OpenMPI).
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_SIZE = 6
  integer, parameter :: MPI_STATUS_IGNORE = -1
contains
  ! Original Fortran sync -- NO bind(c).  Dycore CALLs this.
  ! Takes the MPI communicator as a Fortran handle (integer),
  ! issues an MPI_Sendrecv between the rank's owned block (block 1)
  ! and the partner rank's owned block, landing the partner's data
  ! in the rank's halo (block 2).
  subroutine sync_patch_array(tag, field, comm)
    integer(c_int), intent(in) :: tag
    real(c_double), intent(inout) :: field(:, :, :)
    integer, intent(in) :: comm
    integer :: rank, size_, neigh, ierr, count
    integer :: status_arr(MPI_STATUS_SIZE)
    real(c_double), allocatable :: send_buf(:, :), recv_buf(:, :)
    external :: MPI_Comm_rank, MPI_Comm_size, MPI_Sendrecv
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size_, ierr)
    ! Single-rank: matches real ICON's no-op behaviour.
    if (size_ <= 1) return
    ! Symmetric 2-rank exchange: rank 0 <-> rank 1.  One stderr
    ! marker per call so a missed MPI invocation surfaces without
    ! making the log unreadable.
    neigh = 1 - rank
    count = size(field, 1) * size(field, 2)
    allocate(send_buf(size(field, 1), size(field, 2)))
    allocate(recv_buf(size(field, 1), size(field, 2)))
    send_buf = field(:, :, 1)
    call MPI_Sendrecv(send_buf, count, MPI_DOUBLE_PRECISION, &
                      neigh, tag, &
                      recv_buf, count, MPI_DOUBLE_PRECISION, &
                      neigh, tag, &
                      comm, status_arr, ierr)
    write(0, '(A,I0,A,I0,A,I0)') &
        '[sync_patch_array] rank=', rank, &
        ' neigh=', neigh, &
        ' MPI_Sendrecv ierr=', ierr
    flush(0)
    field(:, :, 2) = recv_buf
    deallocate(send_buf, recv_buf)
  end subroutine sync_patch_array

  ! ``bind(c)`` wrapper the SDFG actually invokes.  Receives the
  ! flat field pointer + extents + comm as a Fortran ``MPI_Fint``
  ! integer (passed through the C ABI as ``int``); rebuilds the
  ! assumed-shape descriptor with c_f_pointer and forwards.
  subroutine sync_patch_array_c(tag, d0, d1, d2, field_p, comm) &
    bind(c, name='sync_patch_array_c')
    integer(c_int), value :: tag, d0, d1, d2, comm
    type(c_ptr), value :: field_p
    real(c_double), pointer :: field_local(:, :, :)
    call c_f_pointer(field_p, field_local, [d0, d1, d2])
    call sync_patch_array(tag, field_local, comm)
  end subroutine sync_patch_array_c
end module mo_sync_mpi
"""

# Standalone dycore: per-rank computation on block 1 (owned), then
# the sync fills block 2 (halo) from the neighbor's block 1.
_DYCORE_SRC = r"""
module mo_dycore_mpi
  use iso_c_binding
  use mo_sync_mpi, only: sync_patch_array
  implicit none
contains
  subroutine dycore_with_sync(field, alpha, comm)
    real(c_double), intent(inout) :: field(:, :, :)
    real(c_double), intent(in) :: alpha
    integer, intent(in) :: comm
    integer :: i, k
    ! Owned-block computation: deterministic + non-trivial enough
    ! that a swap-rank diff against the ref shows up immediately.
    do k = 1, size(field, 2)
      do i = 1, size(field, 1)
        field(i, k, 1) = field(i, k, 1) * alpha + &
                         sqrt(real(i + k, c_double))
      end do
    end do
    ! Halo exchange: fills field(:, :, 2) with neighbor's block 1.
    call sync_patch_array(1_c_int, field, comm)
  end subroutine dycore_with_sync
end module mo_dycore_mpi
"""

# Reference-side bind(c) driver -- same ABI as the SDFG-emitted bind_c_shim (extents-first,
# scalars-last per dynamic_extents_abi), so the same ctypes argtypes drive both.
_REF_DRIVER_SRC = r"""
subroutine dycore_with_sync_ref_c(d0, d1, d2, field_p, alpha, comm) &
  bind(c, name='dycore_with_sync_ref_c')
  use iso_c_binding
  use mo_dycore_mpi, only: dycore_with_sync
  integer(c_int), value :: d0, d1, d2, comm
  type(c_ptr), value :: field_p
  real(c_double), value :: alpha
  real(c_double), pointer :: field(:, :, :)
  call c_f_pointer(field_p, field, [d0, d1, d2])
  call dycore_with_sync(field, alpha, comm)
end subroutine dycore_with_sync_ref_c
"""


def _build_sync_mpi_lib(build_dir: Path) -> Path:
    """Compile mo_sync_mpi into a real-MPI shared library via mpifort. Output
    ``libsync_mpi.so`` carries the bind_c wrapper ``sync_patch_array_c`` that the SDFG
    kernel links + dlopens; mpifort injects the MPI include/lib paths for OpenMPI."""
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "sync_mpi.f90"
    src.write_text(_SYNC_MPI_SRC)
    so_path = build_dir / "libsync_mpi.so"
    subprocess.check_call(
        ["mpifort", "-shared", "-fPIC", *_O0_FFLAGS, f"-J{build_dir}",
         str(src), "-o", str(so_path)], cwd=build_dir)
    return so_path


def _build_ref_lib(build_dir: Path, sync_so: Path, sync_build_dir: Path) -> Path:
    """Compile the gfortran reference (``mo_dycore_mpi`` + driver), linked against the same
    real-MPI sync library the SDFG path uses. ``-I<sync_build_dir>`` resolves ``use mo_sync_mpi``."""
    build_dir.mkdir(parents=True, exist_ok=True)
    dycore_src = build_dir / "dycore.f90"
    dycore_src.write_text(_DYCORE_SRC)
    driver_src = build_dir / "driver.f90"
    driver_src.write_text(_REF_DRIVER_SRC)
    so_path = build_dir / "libdycore_ref.so"
    subprocess.check_call([
        "mpifort", "-shared", "-fPIC", *_O0_FFLAGS, f"-J{build_dir}", f"-I{sync_build_dir}",
        str(dycore_src),
        str(driver_src), f"-L{sync_so.parent}", f"-Wl,-rpath,{sync_so.parent}", f"-l:{sync_so.name}", "-o",
        str(so_path)
    ],
                          cwd=build_dir)
    return so_path


@pytest.mark.mpi
def test_dycore_with_real_mpi_sync_2rank(tmp_path: Path):
    """2-rank dycore + real-MPI ``sync_patch_array``: SDFG path vs gfortran reference,
    per-rank bit-exact. Rank 0 builds every artifact; rank 1 waits at a barrier then loads
    the same .so files from shared tmp_path. sync_patch_array is registered as
    ``keep_external`` so the bridge doesn't lower the embedded MPI_Sendrecv -- the library
    node calls libsync_mpi.so's bind(c) wrapper, which issues the real MPI exchange using
    the Fortran MPI_Fint comm mpi4py's comm.py2f() hands us."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Symmetric 2-rank swap on a 2-rank sub-communicator so the test runs (not skips) at any
    # even rank count -- CI's -n 4 splits COMM_WORLD into pairs {0,1},{2,3},... Also proves
    # the communicator is correctly scoped (no cross-pair leak).
    if size < 2 or size % 2 != 0:
        pytest.skip("needs an even rank count >= 2 "
                    "(mpirun --oversubscribe -n 2 / -n 4 ...)")
    pair = comm.Split(color=rank // 2, key=rank)
    partner_world = rank ^ 1  # the other world rank sharing this pair

    # tmp_path differs per rank (pytest fixture); pin every rank to rank 0's path so the
    # .so artefacts are shared.
    tmp_path_str = str(tmp_path) if rank == 0 else None
    tmp_path = Path(comm.bcast(tmp_path_str, root=0))

    # Build phase: rank 0 only. build_on_root broadcasts the artefact paths -- or a build
    # failure -- so other ranks never block at the barrier waiting for artefacts that won't come.
    def _build_artifacts():
        sync_build_dir = tmp_path / "sync_mpi_build"
        sync_so = _build_sync_mpi_lib(sync_build_dir)
        sync_so_str = str(sync_so)

        # --- SDFG build ---
        clear_external_registry()
        keep_external(
            "sync_patch_array",
            c_name="sync_patch_array_c",
            args=(
                Arg(kind="scalar", dtype="int32", intent="in"),  # tag
                Arg(kind="array", dtype="float64", intent="inout"),  # field
                Arg(kind="scalar", dtype="int32", intent="in"),  # comm
            ),
            libraries=(sync_so_str, ),
            dynamic_extents_abi=True,
        )
        _orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
        dace.Config.set("compiler", "cpu", "args", value=" ".join(_O0_CXX_FLAGS))
        try:
            sdfg_dir = tmp_path / "sdfg"
            sdfg_dir.mkdir(parents=True, exist_ok=True)
            full_src = _SYNC_MPI_SRC + _DYCORE_SRC
            sdfg = build_sdfg(full_src, sdfg_dir, name="dycore_with_sync", entry="dycore_with_sync").build()
            sdfg.name = "dycore_with_sync"
            sdfg.build_folder = str(sdfg_dir / "dacecache")
            iface = build_auto_interface(sdfg._fortran_interface_raw, "dycore_with_sync")
            # bind_c_shim's USE statement needs mo_sync_mpi's body as a prelude to resolve
            # at gfortran link time.
            sync_prelude = sdfg_dir / "sync_mpi.f90"
            sync_prelude.write_text(_SYNC_MPI_SRC)
            sdfg_lib = build_fortran_library(
                sdfg,
                iface=iface,
                out_dir=str(tmp_path / "sdfg_lib"),
                name="dycore_with_sync_wrap",
                prelude_sources=[sync_prelude],
                bind_c_shim=True,
                flags=_O0_FFLAGS,
            )
            sdfg_so_str = str(sdfg_lib.so_path)
        finally:
            clear_external_registry()
            dace.Config.set("compiler", "cpu", "args", value=_orig_cxx_args)

        # --- Reference build ---
        ref_so = _build_ref_lib(tmp_path / "ref", sync_so, sync_build_dir)
        ref_so_str = str(ref_so)
        return sync_so_str, sdfg_so_str, ref_so_str

    sync_so_str, sdfg_so_str, ref_so_str = build_on_root(comm, _build_artifacts)

    # Barrier so every rank only proceeds once the artefacts exist on the
    # shared filesystem (``build_on_root`` already broadcast their paths).
    comm.Barrier()

    # Both ranks load both .so files and drive them through ctypes. Pre-load sync_mpi so
    # SDFG/reference rpaths resolve to the same instance (avoids OpenMPI ODR-violation
    # diagnostics on repeat runs in the same Python session).
    ctypes.CDLL(sync_so_str, mode=ctypes.RTLD_GLOBAL)
    sdfg_lib_obj = ctypes.CDLL(sdfg_so_str)
    ref_lib_obj = ctypes.CDLL(ref_so_str)

    # Per-rank random inputs with distinct seeds so the two ranks own genuinely different
    # data -- a mis-routed comm surfaces immediately as a swap.
    nproma, nlev, nblks = 4, 3, 2
    rng = np.random.default_rng(seed=42 + rank)
    field_init = np.asfortranarray(rng.standard_normal((nproma, nlev, nblks)))
    field_sdfg = field_init.copy(order='F')
    field_ref = field_init.copy(order='F')
    alpha = 2.5
    # Hand the kernel the PAIR communicator (not COMM_WORLD): the halo
    # swap must run within {rank, partner_world}.
    mpi_comm_int = ctypes.c_int(pair.py2f())  # Fortran MPI handle

    argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    sdfg_fn = sdfg_lib_obj.dycore_with_sync_c
    sdfg_fn.restype = None
    sdfg_fn.argtypes = argtypes
    ref_fn = ref_lib_obj.dycore_with_sync_ref_c
    ref_fn.restype = None
    ref_fn.argtypes = argtypes

    sdfg_fn(nproma, nlev, nblks, field_sdfg.ctypes.data, ctypes.c_double(alpha), mpi_comm_int)
    ref_fn(nproma, nlev, nblks, field_ref.ctypes.data, ctypes.c_double(alpha), mpi_comm_int)

    # Per-rank bit-exact agreement on both block 1 (local compute) and block 2 (filled via
    # MPI Sendrecv): a mis-routed comm diverges immediately; a codegen regression shows as
    # a >1 ULP diff.
    one_ulp_rtol = 2**-52
    np.testing.assert_allclose(field_sdfg, field_ref, rtol=one_ulp_rtol, atol=0.0)
    np.testing.assert_array_equal(field_sdfg, field_ref)

    # Sanity check: rank's block 2 must equal the OTHER rank's computed block 1,
    # reconstructed locally from the neighbor's (seed 42+partner_world) initial data -- the
    # actual proof MPI ran and filled the halo correctly; at -n 4 this also catches a leak
    # across pairs.
    other_rank_init = np.asfortranarray(
        np.random.default_rng(seed=42 + partner_world).standard_normal((nproma, nlev, nblks)))
    expected_halo = other_rank_init[:, :, 0].copy()
    for k in range(nlev):
        for i in range(nproma):
            expected_halo[i, k] = (expected_halo[i, k] * alpha + np.sqrt(float((i + 1) + (k + 1))))
    np.testing.assert_allclose(field_sdfg[:, :, 1],
                               expected_halo,
                               rtol=one_ulp_rtol,
                               atol=0.0,
                               err_msg=("halo (block 2) does NOT match "
                                        "the neighbor's computed block 1 -- "
                                        "MPI_Sendrecv probably mis-fired"))
