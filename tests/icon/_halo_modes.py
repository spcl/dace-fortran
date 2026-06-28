"""Shared halo-exchange extraction modes for the ICON dynamical-core solvers.

There are two correct ways to treat the MPI halo when extracting a solver
(``solve_free_sfc`` for the ocean, ``solve_nh`` for the atmosphere):

  * ``"external"`` -- ``sync_patch_array`` / ``exchange_data`` are a black box;
    the bridge emits an ``ExternalCall`` (the kept-external / callback boundary).
    This is the boundary the bindings can dispatch back to a real Fortran halo.
  * ``"inlined"``  -- the halo is inlined and the single-arm (post-cpp)
    ``t_comm_pattern`` dispatch is devirtualised, so the pack/gather lands inline
    and only the MPI point-to-point leaves (``p_isend`` / ``p_irecv`` / ``p_wait``
    / ``p_send`` / ``p_recv``) remain external -- "only MPI calls remain".

BOTH modes must extract to a compiling single TU for BOTH solvers -- the
external-function/callback feature is a first-class mode, not a fallback.  This
module is the single source of truth for the per-mode pieces so the ocean +
atmosphere harnesses stay in lock-step.
"""
from dace_fortran.external_functions import ExternalFunction

#: ``"external"`` mode -- the halo generics are the boundary (a black box).
HALO_EXTERNAL_FUNCTIONS = [
    ExternalFunction("sync_patch_array"),  # MPI halo exchange (generic)
    ExternalFunction("sync_patch_array_mult"),  # MPI multi-field halo exchange
    ExternalFunction("exchange_data"),  # MPI halo primitive under the syncs
]

#: ``"inlined"`` mode -- the halo is inlined; only the MPI point-to-point leaves
#: remain external (mo_mpi wrappers over mpi_isend/irecv/wait/send/recv).
HALO_INLINED_EXTERNAL_FUNCTIONS = [
    ExternalFunction("p_isend"),
    ExternalFunction("p_irecv"),
    ExternalFunction("p_wait"),
    ExternalFunction("p_send"),
    ExternalFunction("p_recv"),
]
#: Force-include the concrete comm-pattern arm: it is reached only via the
#: externalised factory, so the merge never pulls it in, but the monomorphisation
#: pass needs it to retype to.  (``t_comm_pattern_yaxt`` stays cpp'd out.)
HALO_INLINED_FORCE_INCLUDE = ["parallel_infrastructure/mo_communication_orig.f90"]
#: ``mo_mpi``'s ``INTERFACE p_wait`` shares its name with a specific ``p_wait``;
#: rename the specific so externalising the generic doesn't dangle.
HALO_INLINED_RENAME_SPECIFICS = {"p_wait": "p_wait_noarg"}
#: The halo bodies branch ``IF (my_process_is_mpi_seq()) <local copy> ELSE <MPI>``;
#: pin it ``.FALSE.`` to take the real MPI path (so "only MPI calls remain").
HALO_INLINED_RETURN_FALSE = ["my_process_is_mpi_seq"]

HALO_MODES = ("external", "inlined")


def halo_config(mode: str) -> dict:
    """The extraction pieces for halo ``mode``: ``external_functions`` (the
    halo-specific subset), ``force_include`` (module relpaths), ``rename_specifics``
    and ``return_false``.  Callers merge these into the solver's own non-halo
    externals."""
    if mode == "external":
        return dict(external_functions=list(HALO_EXTERNAL_FUNCTIONS),
                    force_include=[],
                    rename_specifics={},
                    return_false=[])
    if mode == "inlined":
        return dict(external_functions=list(HALO_INLINED_EXTERNAL_FUNCTIONS),
                    force_include=list(HALO_INLINED_FORCE_INCLUDE),
                    rename_specifics=dict(HALO_INLINED_RENAME_SPECIFICS),
                    return_false=list(HALO_INLINED_RETURN_FALSE))
    raise ValueError(f"unknown halo mode {mode!r} (expected one of {HALO_MODES})")
