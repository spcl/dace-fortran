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
end module mpi
"""
HALO_INLINED_EXTRA_SOURCES = {"_mpi_consts_stub.f90": _MPI_STUB}

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
                    extra_sources={})
    if mode == "inlined":
        return dict(external_functions=list(HALO_INLINED_EXTERNAL_FUNCTIONS),
                    force_include=list(HALO_INLINED_FORCE_INCLUDE),
                    rename_specifics=dict(HALO_INLINED_RENAME_SPECIFICS),
                    return_false=list(HALO_INLINED_RETURN_FALSE),
                    extra_sources=dict(HALO_INLINED_EXTRA_SOURCES))
    raise ValueError(f"unknown halo mode {mode!r} (expected one of {HALO_MODES})")
