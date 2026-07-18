"""Fortran binding emission for HLFIR-built SDFGs.

Peer of ``builder/`` / ``intrinsics/`` under ``dace_fortran/``.  Runs
AFTER the SDFG is built: takes ``FrozenSignature`` (SDFG arg list,
drift-checked), ``OriginalInterface`` (caller-facing surface), and
``FlattenPlan`` (AoS->SoA unpack record from ``hlfir-flatten-structs``),
and emits one ``<entry>_bindings.f90`` module -- aliasing zero-copy
where layouts agree, do-loop copy-in/copy-out where recipes demand it.
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
