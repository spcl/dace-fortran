"""Shared halo-exchange extraction modes for the ICON dynamical-core solvers.
``"external"``: halo generics + collectives stay a black-box ExternalCall (callback
to a real Fortran halo). ``"inlined"``: every mo_mpi wrapper is inlined to the raw
mpi_* call, which the bridge lowers to dace.libraries.mpi libnodes. Both modes must
extract to a compiling single TU for both solvers; this module is the single source
of truth so the ocean + atmosphere harnesses stay in step.
"""
from dace_fortran.external_functions import ExternalFunction

#: "external" mode: MPI ops the black box covers -- halo generics + the collectives
#: the solver calls directly (point-to-point calls live inside exchange_data).
HALO_EXTERNAL_FUNCTIONS = [
    ExternalFunction("sync_patch_array"),
    ExternalFunction("sync_patch_array_mult"),
    ExternalFunction("exchange_data"),
    ExternalFunction("p_barrier"),
    ExternalFunction("p_max"),
    ExternalFunction("p_min"),
    ExternalFunction("p_sum"),
    ExternalFunction("global_max"),
    ExternalFunction("global_min"),
    ExternalFunction("global_sum"),
]

#: ``"inlined"`` mode -- nothing MPI stays external.
HALO_INLINED_EXTERNAL_FUNCTIONS: list = []
#: Force-include the concrete comm-pattern arm (reached only via the externalised
#: factory; merge won't pull it in, but monomorphisation needs it).
HALO_INLINED_FORCE_INCLUDE = ["parallel_infrastructure/mo_communication_orig.f90"]
#: mo_mpi's INTERFACE p_wait shares its name with a specific p_wait; rename the
#: specific so resolution is unambiguous.
HALO_INLINED_RENAME_SPECIFICS = {"p_wait": "p_wait_noarg"}
#: Halo branches IF(my_process_is_mpi_seq()) <local copy> ELSE <MPI>; pin .FALSE.
#: to take the real MPI path.
HALO_INLINED_RETURN_FALSE = ["my_process_is_mpi_seq"]

#: mpi module PARAMETER constants (mpi_comm_world, mpi_status_size, ...), shared by
#: both _MPI_CONSTS_STUB and _MPI_STUB so the values never diverge. NOT closed here
#: (no ``end module``) -- each consumer appends its own tail.
_MPI_CONSTS = """\
module mpi
  implicit none
  integer, parameter :: mpi_comm_world = 0
  integer, parameter :: mpi_comm_null = 2
  integer, parameter :: mpi_status_size = 6
  integer, parameter :: mpi_status_ignore = 1
  integer, parameter :: mpi_statuses_ignore = 1
  integer, parameter :: mpi_request_null = 0
  integer, parameter :: mpi_success = 0
  integer, parameter :: mpi_undefined = -32766
  integer, parameter :: mpi_any_source = -2
  integer, parameter :: mpi_any_tag = -1
  integer, parameter :: mpi_proc_null = -1
  integer, parameter :: mpi_double_precision = 17
  integer, parameter :: mpi_real = 13
  integer, parameter :: mpi_integer = 7
  integer, parameter :: mpi_2real = 27
  integer, parameter :: mpi_2double_precision = 28
  integer, parameter :: mpi_2integer = 29
  integer, parameter :: mpi_byte = 1
  integer, parameter :: mpi_logical = 6
  integer, parameter :: mpi_character = 5
  integer, parameter :: mpi_max = 1
  integer, parameter :: mpi_min = 2
  integer, parameter :: mpi_sum = 3
  integer, parameter :: mpi_prod = 4
  integer, parameter :: mpi_maxloc = 11
  integer, parameter :: mpi_minloc = 12
  integer, parameter :: mpi_land = 5
  integer, parameter :: mpi_lor = 7
"""

#: F2008 TS 29113 type(*),dimension(..) interfaces for the inlined mo_mpi wrappers'
#: point-to-point calls: needed so a GFORTRAN stub type-checks mpi_irecv called with
#: both REAL(8) and REAL(4) buffers without -fallow-argument-mismatch. INTERFACEs
#: only -- the mpi_* calls stay undefined externals the bridge maps to libnodes.
#: fparser can't parse type(*),dimension(..), so this is EXCLUDED from the
#: fparser-fed _MPI_CONSTS_STUB and included ONLY in the gfortran-fed _MPI_STUB.
_MPI_ASSUMED_TYPE_INTERFACES = """\
  interface
    subroutine mpi_recv(buf, count, datatype, source, tag, comm, status, ierror)
      type(*), dimension(..) :: buf   ! assumed-type -> no INTENT (F2008 TS 29113)
      integer, intent(in) :: count, datatype, source, tag, comm
      integer, intent(out) :: status(*), ierror
    end subroutine mpi_recv
    subroutine mpi_irecv(buf, count, datatype, source, tag, comm, request, ierror)
      type(*), dimension(..) :: buf
      integer, intent(in) :: count, datatype, source, tag, comm
      integer, intent(out) :: request, ierror
    end subroutine mpi_irecv
    subroutine mpi_send(buf, count, datatype, dest, tag, comm, ierror)
      type(*), dimension(..) :: buf
      integer, intent(in) :: count, datatype, dest, tag, comm
      integer, intent(out) :: ierror
    end subroutine mpi_send
    subroutine mpi_isend(buf, count, datatype, dest, tag, comm, request, ierror)
      type(*), dimension(..) :: buf
      integer, intent(in) :: count, datatype, dest, tag, comm
      integer, intent(out) :: request, ierror
    end subroutine mpi_isend
  end interface
"""

#: Full mpi stub (constants + assumed-type interfaces) for a GFORTRAN reference build
#: that must compile dual-typed mpi_* calls WITHOUT -fallow-argument-mismatch. Used by
#: test_solve_nh_binding.py's prelude, NOT the fparser extraction (see _MPI_CONSTS_STUB).
_MPI_STUB = _MPI_CONSTS + _MPI_ASSUMED_TYPE_INTERFACES + "end module mpi\n"

#: Constants-only mpi stub for the FPARSER extraction: same values as _MPI_STUB but
#: without the interface block fparser can't parse (flang keeps mpi_* as externals anyway).
_MPI_CONSTS_STUB = _MPI_CONSTS + "end module mpi\n"

#: Fed to the inlined-halo EXTRACTION (fparser); constants-only so it parses.
#: Interfaces live in _MPI_STUB for the gfortran reference build.
HALO_INLINED_EXTRA_SOURCES = {"_mpi_consts_stub.f90": _MPI_CONSTS_STUB}

#: NO-OP mpi_* implementations for a SINGLE-RANK GFORTRAN REFERENCE build: _MPI_STUB
#: only declares them, but a running single-TU dycore needs them DEFINED. Single-rank
#: has no neighbours, so point-to-point calls are no-ops (every owned cell keeps its
#: value) -- matching what the DUT SDFG does by dropping the halo. External/module-less
#: so the linker resolves both mo_mpi's assumed-type calls and implicit-external
#: collectives here. Paired with the strip_deconiface reference transform.
_MPI_NOOP_IMPL = """\
subroutine mpi_recv(buf, count, datatype, source, tag, comm, status, ierror)
  type(*), dimension(..) :: buf
  integer, intent(in) :: count, datatype, source, tag, comm
  integer, intent(out) :: status(*), ierror
  ierror = 0
end subroutine
subroutine mpi_irecv(buf, count, datatype, source, tag, comm, request, ierror)
  type(*), dimension(..) :: buf
  integer, intent(in) :: count, datatype, source, tag, comm
  integer, intent(out) :: request, ierror
  request = 0; ierror = 0
end subroutine
subroutine mpi_send(buf, count, datatype, dest, tag, comm, ierror)
  type(*), dimension(..) :: buf
  integer, intent(in) :: count, datatype, dest, tag, comm
  integer, intent(out) :: ierror
  ierror = 0
end subroutine
subroutine mpi_isend(buf, count, datatype, dest, tag, comm, request, ierror)
  type(*), dimension(..) :: buf
  integer, intent(in) :: count, datatype, dest, tag, comm
  integer, intent(out) :: request, ierror
  request = 0; ierror = 0
end subroutine
subroutine mpi_waitall(count, array_of_requests, array_of_statuses, ierror)
  integer, intent(in) :: count
  integer, intent(inout) :: array_of_requests(*)
  integer, intent(out) :: array_of_statuses(*), ierror
  ierror = 0
end subroutine
subroutine mpi_barrier(comm, ierror)
  integer, intent(in) :: comm
  integer, intent(out) :: ierror
  ierror = 0
end subroutine
subroutine mpi_abort(comm, errorcode, ierror)
  integer, intent(in) :: comm, errorcode
  integer, intent(out) :: ierror
  ierror = 0
end subroutine
subroutine acc_wait_comms()
end subroutine
! ICON's C ``util_exit`` / ``util_abort`` (``BIND(C)`` error-path aborts): only
! DECLARED (interface) in the single-TU, so the reference .so has them undefined
! and won't dlopen.  A valid degenerate run never reaches the error path, so a
! no-op stub (C linkage, matching the interface) lets the library load.
subroutine util_exit(exit_no) bind(c, name="util_exit")
  use iso_c_binding, only: c_int
  integer(c_int), value :: exit_no
end subroutine
subroutine util_abort() bind(c, name="util_abort")
end subroutine
"""

#: "inlined" mode -- source-level inlining of the sync_patch_array family: these
#: wrappers select the comm pattern via a compile-time-constant `typ`, so inlining
#: lets constant-fold/branch-prune collapse the ladder to a single-source rebind the
#: bridge can lower (a runtime-selected rebind is rejected by hlfir-rewrite-pointer-assigns).
#: Names absent from a given kernel's closure are simply never matched.
#:
HALO_INLINED_SPECIALIZE_AT_SOURCE = [
    "comm_pat_of_type",
    "sync_patch_array_2d_dp",
    "sync_patch_array_2d_sp",
    "sync_patch_array_3d_dp",
    "sync_patch_array_3d_sp",
    "sync_patch_array_4d_dp",
    "sync_patch_array_4d_sp",
    "sync_patch_array_mult_f3din_dp",
    "sync_patch_array_mult_f3din_sp",
    "sync_patch_array_mult_f4din_dp",
    "sync_patch_array_mult_f4din_sp",
    "sync_patch_array_mult_mixprec",
    # NOTE: exchange_data_* is NOT inlined here -- their bodies use the OPTIONAL
    # `send` unconditionally, so a recv-only call leaves it absent in live code and
    # the inline is (correctly) abandoned. Needs recv-only semantics handled first.
]

HALO_MODES = ("external", "inlined")


def halo_config(mode: str) -> dict:
    """Extraction pieces for halo ``mode``: external_functions, force_include,
    rename_specifics, return_false, extra_sources. Callers merge these into the
    solver's own non-halo externals."""
    if mode == "external":
        return dict(external_functions=list(HALO_EXTERNAL_FUNCTIONS),
                    force_include=[],
                    rename_specifics={},
                    return_false=[],
                    extra_sources={},
                    specialize_at_source=[])
    if mode == "inlined":
        return dict(external_functions=list(HALO_INLINED_EXTERNAL_FUNCTIONS),
                    force_include=list(HALO_INLINED_FORCE_INCLUDE),
                    rename_specifics=dict(HALO_INLINED_RENAME_SPECIFICS),
                    return_false=list(HALO_INLINED_RETURN_FALSE),
                    extra_sources=dict(HALO_INLINED_EXTRA_SOURCES),
                    specialize_at_source=list(HALO_INLINED_SPECIALIZE_AT_SOURCE))
    raise ValueError(f"unknown halo mode {mode!r} (expected one of {HALO_MODES})")
