"""Shared halo-exchange extraction modes for the ICON dynamical-core solvers.

Two correct ways to treat the MPI halo when extracting a solver
(``solve_free_sfc`` ocean, ``solve_nh`` atmosphere):

  * ``"external"`` -- the halo generics (``sync_patch_array`` / ``exchange_data``)
    and the collectives (``p_barrier`` / ``p_max`` / ``p_min`` / ``p_sum``) are a
    black box; the bridge emits an ``ExternalCall``.  The callback boundary the
    bindings dispatch back to a real Fortran halo.
  * ``"inlined"`` -- NO MPI op stays external: every ``mo_mpi`` wrapper is inlined
    down to the raw ``mpi_*`` call (``p_isend`` -> ``mpi_isend``, ``p_barrier`` ->
    ``mpi_barrier``, ``p_max`` -> ``mpi_allreduce``), which the bridge lowers to
    ``dace.libraries.mpi`` libnodes.  The ``mo_mpi`` datatype / comm / error
    module variables come with ``mo_mpi``; a small ``mpi`` constants module
    resolves the ``use mpi`` parameters the wrappers reference.

Both modes must extract to a compiling single TU for both solvers.  This module
is the single source of truth so the ocean + atmosphere harnesses stay in step.
"""
from dace_fortran.external_functions import ExternalFunction

#: ``"external"`` mode -- the MPI ops the black box covers.  The point-to-point
#: ``p_isend`` &c. live inside ``exchange_data``, so the halo generics plus the
#: collectives the solver calls directly are the full external boundary.
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
#: factory, so the merge never pulls it in but monomorphisation needs it).
HALO_INLINED_FORCE_INCLUDE = ["parallel_infrastructure/mo_communication_orig.f90"]
#: ``mo_mpi``'s ``INTERFACE p_wait`` shares its name with a specific ``p_wait``;
#: rename the specific so resolution is unambiguous.
HALO_INLINED_RENAME_SPECIFICS = {"p_wait": "p_wait_noarg"}
#: The halo branches ``IF (my_process_is_mpi_seq()) <local copy> ELSE <MPI>``;
#: pin it ``.FALSE.`` to take the real MPI path.
HALO_INLINED_RETURN_FALSE = ["my_process_is_mpi_seq"]

#: Minimal ``mpi`` module so the inlined wrappers' ``use mpi`` parameters resolve
#: standalone (the raw ``mpi_*`` calls stay undefined externals the bridge maps).
_MPI_STUB = """\
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
  ! ASSUMED-TYPE interfaces for the point-to-point calls the inlined ``mo_mpi``
  ! wrappers issue.  A real MPI ``mpi`` module provides these (F2008 TS 29113
  ! ``type(*), dimension(..)``); the stub must too.  Without them gfortran sees
  ! ``mpi_irecv`` called with a REAL(8) buffer (``p_irecv_dp``) and a REAL(4)
  ! buffer (``p_irecv_sp``) and no interface, infers a fixed buffer type from the
  ! first call, and rejects the second (``Type mismatch REAL(8)/REAL(4)``).
  ! These are INTERFACEs, not bodies -- the ``mpi_*`` calls stay undefined
  ! externals the bridge maps to ``dace.libraries.mpi`` libnodes, so the SDFG
  ! side is untouched; this only lets the gfortran REFERENCE build compile
  ! type-correctly WITHOUT the unsound ``-fallow-argument-mismatch``.
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
end module mpi
"""
HALO_INLINED_EXTRA_SOURCES = {"_mpi_consts_stub.f90": _MPI_STUB}

#: ``"inlined"`` mode -- source-level procedure-body inlining of the halo
#: ``sync_patch_array`` family into their callers.  These wrappers select the comm
#: pattern via ``IF (typ==N) p_pat => p_patch%comm_pat_<X>`` (or the pointer-result
#: ``comm_pat_of_type`` FUNCTION); ``typ`` is a compile-time constant at every call
#: site, so inlining the wrapper lets the constant-fold / branch-prune collapse the
#: ladder to a SINGLE-source rebind the bridge can lower (a runtime-selected rebind
#: is rejected by ``hlfir-rewrite-pointer-assigns``).  Names not present in a given
#: kernel's closure are simply never matched.
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
    # NOTE: the ``exchange_data_*`` family is NOT inlined here.  Inlining them (to
    # connect the gather to ``p_patch%comm_pat_e``) is the right direction -- it is
    # per-call-site monomorphization of the polymorphic comm-pattern dispatch -- but
    # their bodies use the OPTIONAL ``send`` UNCONDITIONALLY (both arms of
    # ``IF(PRESENT(add))``), so a recv-only call (``exchange_data_r3d_seq(p_pat, lacc,
    # recv)``) leaves an absent ``send`` in LIVE code and the inline is (correctly)
    # abandoned.  Folding the exchange in needs the recv-only semantics handled first;
    # tracked as a focused follow-up.
]

HALO_MODES = ("external", "inlined")


def halo_config(mode: str) -> dict:
    """Extraction pieces for halo ``mode``: ``external_functions`` (the
    halo-specific subset), ``force_include`` (module relpaths), ``rename_specifics``,
    ``return_false`` and ``extra_sources`` ({name: content} spliced into the
    closure).  Callers merge these into the solver's own non-halo externals."""
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
