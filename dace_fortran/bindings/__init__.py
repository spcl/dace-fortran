"""Fortran binding emission for HLFIR-built SDFGs.

Peer of ``builder/`` / ``intrinsics/`` under ``dace_fortran/``.
Runs AFTER the SDFG is built, consuming three inputs:

- ``FrozenSignature``  --  the SDFG's argument list snapshotted at
  build time (drift-checked at codegen).
- ``OriginalInterface``  --  the caller-facing Fortran surface of the
  entry subroutine.
- ``FlattenPlan``  --  record of every AoS -> SoA unpack performed by
  ``hlfir-flatten-structs``.

And producing one ``<entry>_bindings.f90`` module that preserves the
user's Fortran interface, aliases zero-copy where layouts agree, and
generates do-loop copy-in / copy-out where recipes demand it.

Public surface:
    FrozenArg / FrozenSignature / SignatureDriftError
         --  signature freezing + drift check
    OriginalInterface / OriginalArg / DerivedType / Member
         --  outer Fortran-facing surface
    FlattenRecipe / FlattenEntry / FlattenPlan
         --  the AoS->SoA plan from hlfir-flatten-structs
    emit_bindings(frozen, iface, plan, out_path)
         --  the top-level emitter
    build_fortran_library(sdfg, iface, plan, out_dir, ...)
         --  emit + drift-verify + link a Fortran-callable .so
"""

from dace_fortran.bindings.bind_c_shim import (
    UnsupportedShimInterfaceError,
    emit_bind_c_shim,
)
from dace_fortran.bindings.build_fortran_library import (
    FortranLibrary,
    build_fortran_library,
)
from dace_fortran.bindings.emit_bindings import emit_bindings
from dace_fortran.bindings.flatten_plan import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)
from dace_fortran.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
)
from dace_fortran.bindings.frozen_signature import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)

__all__ = [
    # Frozen signature
    "FrozenArg",
    "FrozenSignature",
    "SignatureDriftError",
    # Outer interface
    "OriginalInterface",
    "OriginalArg",
    "DerivedType",
    "Member",
    # Flatten plan
    "FlattenRecipe",
    "FlattenEntry",
    "FlattenPlan",
    "substitute_indices",
    "strip_index_args",
    # Emitter
    "emit_bindings",
    # bind(c) shim auto-gen (Phase 2.4)
    "emit_bind_c_shim",
    "UnsupportedShimInterfaceError",
    # Fortran-callable library builder
    "build_fortran_library",
    "FortranLibrary",
]
