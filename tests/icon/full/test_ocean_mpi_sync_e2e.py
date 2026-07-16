"""Ocean dycore + real-MPI halo-exchange ``sync_patch_array_3d_dp`` e2e (2 ranks/pair).

The ocean-side companion to ``test_dycore_mpi_sync_e2e.py`` (atmosphere).  Same
2-rank real-MPI shape, but the halo routine carries the ocean name the extracted
ocean kernels actually call (``mo_sync::sync_patch_array_3d_dp`` -- e.g.
``nonlinear_coriolis_3d`` syncs ``vort_v`` through it) and the owned-block
computation is an ocean-mimetic-flavoured update (a scaled field plus a
level-weighted term) instead of the atmosphere formula.

Runs under ``mpirun --oversubscribe`` at any even rank count (CI uses ``-n 4``):
COMM_WORLD splits into adjacent 2-rank pairs and each pair runs the symmetric
halo swap independently.  Each rank owns one block of a per-rank
``(nproma, n_zlev, 2)`` ``field``; block 1 is the "owned" data the ocean dycore
writes, block 2 is the "halo" the neighbour fills via the sync.  The dycore
writes block 1 from a deterministic per-rank formula, then CALLs
``sync_patch_array_3d_dp(tag, field, comm)`` which issues a real ``MPI_Sendrecv``
against the partner rank; block 2 on each rank then equals the neighbour's
block 1.

The ``sync_patch_array_3d_dp`` body is a "dummy" -- ICON-shaped but deliberately
tiny -- so the test stays self-contained (no ICON build).  As on the atmosphere
side, the communicator is a Fortran ``MPI_Fint`` integer handle
(``mpi4py.MPI.Comm.py2f()``) carried byte-for-byte through the C ABI as ``int``;
the Fortran sync uses it natively with ``MPI_Comm_rank`` / ``MPI_Sendrecv`` --
no ``MPI_Comm_f2c`` round-trip.

The dycore + sync source bundle compiles BOTH as the gfortran reference library
AND as the DaCe-bridge SDFG (``sync_patch_array_3d_dp`` registered as a
``keep_external`` so the bridge routes to the real-MPI ``.so`` instead of
lowering the embedded ``MPI_Sendrecv``).  Same input data on each rank, same MPI
exchange, same per-element comparison (1-ULP envelope + bit-exact hard check),
matching the standalone single-rank ocean convention.

Skipped automatically when run under an odd rank count or fewer than 2 ranks so
the default ``pytest tests/`` (single-rank) doesn't trip on it.
"""
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

# Matching FP-conservative flags across every build layer so SDFG + gfortran
# arithmetic match bit-for-bit -- same convention as the atmosphere 2-rank test
# and the single-rank ocean e2e.
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none")
_O0_CXX_FLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-Wno-unused-parameter", "-Wno-unused-label")

# Dummy ocean halo-exchange ``sync_patch_array_3d_dp``.  ICON-shaped (Fortran
# subroutine taking ``(tag, field, comm)``) but the body is the minimum that
# exercises a real ``MPI_Sendrecv`` against the partner rank.  The name matches
# the ocean kernels' actual sync (``mo_sync::sync_patch_array_3d_dp``).
#
# ``comm`` is a Fortran ``MPI_Fint`` integer handle (OpenMPI = C ``int``); no
# ``MPI_Comm_f2c`` on the Fortran side -- the C ABI carries the Fortran handle
# byte-for-byte, exactly the way ``mpi4py.MPI.Comm.py2f()`` hands it out.
# MPI constants are hard-coded to OpenMPI values rather than ``use mpi`` to avoid
# the flang-vs-OpenMPI ``.mod`` dependency at SDFG-bridge build time.
_SYNC_MPI_SRC = r"""
module mo_ocean_sync_mpi
  use iso_c_binding
  implicit none
  ! OpenMPI values; matches what mpi4py.MPI.Comm.py2f() hands out
  ! (Fortran handles are plain ints under OpenMPI).
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_SIZE = 6
contains
  ! Original Fortran ocean sync -- NO bind(c).  The ocean dycore CALLs this.
  ! Takes the MPI communicator as a Fortran handle (integer), issues an
  ! MPI_Sendrecv between the rank's owned block (block 1) and the partner
  ! rank's owned block, landing the partner's data in the rank's halo (block 2).
  subroutine sync_patch_array_3d_dp(tag, field, comm)
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
    ! Symmetric 2-rank exchange: rank 0 <-> rank 1.
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
        '[sync_patch_array_3d_dp] rank=', rank, &
        ' neigh=', neigh, &
        ' MPI_Sendrecv ierr=', ierr
    flush(0)
    field(:, :, 2) = recv_buf
    deallocate(send_buf, recv_buf)
  end subroutine sync_patch_array_3d_dp

  ! ``bind(c)`` wrapper the SDFG actually invokes.  Receives the flat field
  ! pointer + extents + comm as a Fortran ``MPI_Fint`` integer (through the C
  ! ABI as ``int``); rebuilds the assumed-shape descriptor and forwards.
  subroutine sync_patch_array_3d_dp_c(tag, d0, d1, d2, field_p, comm) &
    bind(c, name='sync_patch_array_3d_dp_c')
    integer(c_int), value :: tag, d0, d1, d2, comm
    type(c_ptr), value :: field_p
    real(c_double), pointer :: field_local(:, :, :)
    call c_f_pointer(field_p, field_local, [d0, d1, d2])
    call sync_patch_array_3d_dp(tag, field_local, comm)
  end subroutine sync_patch_array_3d_dp_c
end module mo_ocean_sync_mpi
"""

# Standalone ocean dycore: per-rank owned-block computation (an ocean-mimetic
# scaled field plus a level-weighted ratio -- deterministic + non-trivial), then
# the sync fills block 2 (halo) from the neighbour's block 1.
_DYCORE_SRC = r"""
module mo_ocean_dycore_mpi
  use iso_c_binding
  use mo_ocean_sync_mpi, only: sync_patch_array_3d_dp
  implicit none
contains
  subroutine ocean_dycore_with_sync(field, coeff, comm)
    real(c_double), intent(inout) :: field(:, :, :)
    real(c_double), intent(in) :: coeff
    integer, intent(in) :: comm
    integer :: i, k
    ! Owned-block update: scaled field + a level-weighted term.  The division
    ! is IEEE-deterministic under -ffp-contract=off / -fno-fast-math, so the
    ! SDFG and gfortran reference round identically.
    do k = 1, size(field, 2)
      do i = 1, size(field, 1)
        field(i, k, 1) = field(i, k, 1) * coeff + &
                         real(i, c_double) / real(k + 1, c_double)
      end do
    end do
    ! Halo exchange: fills field(:, :, 2) with the neighbour's block 1.
    call sync_patch_array_3d_dp(1_c_int, field, comm)
  end subroutine ocean_dycore_with_sync
end module mo_ocean_dycore_mpi
"""

# Reference-side ``bind(c)`` driver -- same ABI as the SDFG-emitted bind_c_shim
# (extents-first, scalars-last per ``dynamic_extents_abi``), so the same ctypes
# argtypes drive both.
_REF_DRIVER_SRC = r"""
subroutine ocean_dycore_with_sync_ref_c(d0, d1, d2, field_p, coeff, comm) &
  bind(c, name='ocean_dycore_with_sync_ref_c')
  use iso_c_binding
  use mo_ocean_dycore_mpi, only: ocean_dycore_with_sync
  integer(c_int), value :: d0, d1, d2, comm
  type(c_ptr), value :: field_p
  real(c_double), value :: coeff
  real(c_double), pointer :: field(:, :, :)
  call c_f_pointer(field_p, field, [d0, d1, d2])
  call ocean_dycore_with_sync(field, coeff, comm)
end subroutine ocean_dycore_with_sync_ref_c
"""


def _build_sync_mpi_lib(build_dir: Path) -> Path:
    """Compile mo_ocean_sync_mpi into a real-MPI shared library via ``mpifort``.
    Output ``libocean_sync_mpi.so`` carries the bind_c wrapper symbol
    ``sync_patch_array_3d_dp_c`` that the SDFG kernel links + dlopens.
    ``mpifort`` (= gfortran wrapped) injects the MPI include + lib paths so
    ``MPI_Comm_rank`` / ``MPI_Sendrecv`` resolve against the system OpenMPI."""
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "ocean_sync_mpi.f90"
    src.write_text(_SYNC_MPI_SRC)
    so_path = build_dir / "libocean_sync_mpi.so"
    subprocess.check_call(
        ["mpifort", "-shared", "-fPIC", *_O0_FFLAGS, f"-J{build_dir}",
         str(src), "-o", str(so_path)], cwd=build_dir)
    return so_path


def _build_ref_lib(build_dir: Path, sync_so: Path, sync_build_dir: Path) -> Path:
    """Compile the gfortran reference: ``mo_ocean_dycore_mpi`` +
    ``ocean_dycore_with_sync_ref_c`` driver, linked against the same real-MPI
    sync library the SDFG path uses.  ``-I<sync_build_dir>`` lets the dycore's
    ``use mo_ocean_sync_mpi`` resolve the module."""
    build_dir.mkdir(parents=True, exist_ok=True)
    dycore_src = build_dir / "ocean_dycore.f90"
    dycore_src.write_text(_DYCORE_SRC)
    driver_src = build_dir / "driver.f90"
    driver_src.write_text(_REF_DRIVER_SRC)
    so_path = build_dir / "libocean_dycore_ref.so"
    subprocess.check_call([
        "mpifort", "-shared", "-fPIC", *_O0_FFLAGS, f"-J{build_dir}", f"-I{sync_build_dir}",
        str(dycore_src),
        str(driver_src), f"-L{sync_so.parent}", f"-Wl,-rpath,{sync_so.parent}", f"-l:{sync_so.name}", "-o",
        str(so_path)
    ],
                          cwd=build_dir)
    return so_path


@pytest.mark.mpi
def test_ocean_dycore_with_real_mpi_sync_2rank(tmp_path: Path):
    """2-rank ocean dycore + real-MPI ``sync_patch_array_3d_dp``: SDFG path vs
    gfortran reference, per-rank bit-exact.

    Rank 0 builds every artifact; the other ranks wait at a barrier and then
    load the same .so files from the shared tmp_path.  The dycore SDFG is built
    via the bridge with ``sync_patch_array_3d_dp`` registered as a
    ``keep_external`` so the bridge does not lower the embedded ``MPI_Sendrecv``
    -- the library node calls into ``libocean_sync_mpi.so``'s bind(c) wrapper,
    which issues the real MPI exchange against the partner rank using the
    Fortran ``MPI_Fint`` comm that mpi4py's ``comm.py2f()`` hands us."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # The halo exchange is a symmetric 2-rank swap.  Run it on a 2-rank
    # sub-communicator so the test runs (does not skip) at any even rank count --
    # CI launches ``mpirun -n 4``: COMM_WORLD splits into adjacent pairs {0,1},
    # {2,3}, ... and each pair does an independent exchange.  This also proves
    # the communicator is correctly scoped: a rank must receive from its pair
    # partner, never leak across to the other pair.
    if size < 2 or size % 2 != 0:
        pytest.skip("needs an even rank count >= 2 "
                    "(mpirun --oversubscribe -n 2 / -n 4 ...)")
    pair = comm.Split(color=rank // 2, key=rank)
    partner_world = rank ^ 1  # the other world rank sharing this pair

    # tmp_path is a pytest fixture; on rank 0 it's freshly minted, on the other
    # ranks it's a DIFFERENT path under the same parent.  Pin every rank to
    # rank 0's path so the .so artefacts are shared.
    tmp_path_str = str(tmp_path) if rank == 0 else None
    tmp_path = Path(comm.bcast(tmp_path_str, root=0))

    # Build phase: rank 0 only.  ``build_on_root`` broadcasts the artefact paths
    # -- or a build *failure* -- so the other ranks never block at the barrier
    # below waiting for artefacts a failed build will never produce.
    def _build_artifacts():
        sync_build_dir = tmp_path / "ocean_sync_mpi_build"
        sync_so = _build_sync_mpi_lib(sync_build_dir)
        sync_so_str = str(sync_so)

        # --- SDFG build ---
        clear_external_registry()
        keep_external(
            "sync_patch_array_3d_dp",
            c_name="sync_patch_array_3d_dp_c",
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
            sdfg = build_sdfg(full_src, sdfg_dir, name="ocean_dycore_with_sync", entry="ocean_dycore_with_sync").build()
            sdfg.name = "ocean_dycore_with_sync"
            sdfg.build_folder = str(sdfg_dir / "dacecache")
            iface = build_auto_interface(sdfg._fortran_interface_raw, "ocean_dycore_with_sync")
            # The bridge needs mo_ocean_sync_mpi's body as a prelude so the
            # bind_c_shim's USE statement resolves at gfortran link time.
            sync_prelude = sdfg_dir / "ocean_sync_mpi.f90"
            sync_prelude.write_text(_SYNC_MPI_SRC)
            sdfg_lib = build_fortran_library(
                sdfg,
                iface=iface,
                out_dir=str(tmp_path / "sdfg_lib"),
                name="ocean_dycore_with_sync_wrap",
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

    # Barrier so every rank only proceeds once the artefacts exist on the shared
    # filesystem (``build_on_root`` already broadcast their paths).
    comm.Barrier()

    # Both ranks load both .so files and drive them through ctypes.  Pre-load
    # ocean_sync_mpi so the SDFG / reference rpaths resolve to the same instance.
    ctypes.CDLL(sync_so_str, mode=ctypes.RTLD_GLOBAL)
    sdfg_lib_obj = ctypes.CDLL(sdfg_so_str)
    ref_lib_obj = ctypes.CDLL(ref_so_str)

    # Per-rank random inputs: distinct seeds so the two ranks own genuinely
    # different data + the MPI exchange surfaces a swap immediately if the bridge
    # mis-routes the comm.  ``(nproma, n_zlev, 2)`` -- block 1 owned, block 2 halo.
    nproma, n_zlev, nblks = 4, 3, 2
    rng = np.random.default_rng(seed=42 + rank)
    field_init = np.asfortranarray(rng.standard_normal((nproma, n_zlev, nblks)))
    field_sdfg = field_init.copy(order='F')
    field_ref = field_init.copy(order='F')
    coeff = 2.5
    # Hand the kernel the PAIR communicator (not COMM_WORLD): the halo swap must
    # run within {rank, partner_world}.
    mpi_comm_int = ctypes.c_int(pair.py2f())  # Fortran MPI handle

    argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    sdfg_fn = sdfg_lib_obj.ocean_dycore_with_sync_c
    sdfg_fn.restype = None
    sdfg_fn.argtypes = argtypes
    ref_fn = ref_lib_obj.ocean_dycore_with_sync_ref_c
    ref_fn.restype = None
    ref_fn.argtypes = argtypes

    sdfg_fn(nproma, n_zlev, nblks, field_sdfg.ctypes.data, ctypes.c_double(coeff), mpi_comm_int)
    ref_fn(nproma, n_zlev, nblks, field_ref.ctypes.data, ctypes.c_double(coeff), mpi_comm_int)

    # Per-rank bit-exact agreement on both the owned block (block 1, local
    # compute) and the halo block (block 2, filled via the MPI Sendrecv).  A
    # mis-routed comm or a swapped neighbour immediately diverges; a codegen
    # regression shows up as a >1 ULP diff.
    one_ulp_rtol = 2**-52
    np.testing.assert_allclose(field_sdfg, field_ref, rtol=one_ulp_rtol, atol=0.0)
    np.testing.assert_array_equal(field_sdfg, field_ref)

    # Sanity check the halo: rank's block 2 must equal the OTHER rank's computed
    # block 1 (reconstructed locally by repeating the deterministic dycore
    # formula on the neighbour's initial data).  This is the actual proof that
    # MPI ran and the halo was filled correctly.  The neighbour is the rank's
    # pair partner (``partner_world``, seed ``42 + partner_world``) -- so at
    # -n 4 a leak across pairs (rank 0 receiving rank 2's data) is caught here.
    other_rank_init = np.asfortranarray(
        np.random.default_rng(seed=42 + partner_world).standard_normal((nproma, n_zlev, nblks)))
    expected_halo = other_rank_init[:, :, 0].copy()
    for k in range(n_zlev):
        for i in range(nproma):
            expected_halo[i, k] = expected_halo[i, k] * coeff + float(i + 1) / float((k + 1) + 1)
    np.testing.assert_allclose(field_sdfg[:, :, 1],
                               expected_halo,
                               rtol=one_ulp_rtol,
                               atol=0.0,
                               err_msg=("halo (block 2) does NOT match the neighbour's computed block 1 -- "
                                        "MPI_Sendrecv probably mis-fired"))
