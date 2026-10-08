# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
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
    SyntheticGlobal,
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
    "DerivedType",
    "FlattenEntry",
    "FlattenPlan",
    # Flatten plan
    "FlattenRecipe",
    "FortranLibrary",
    # Frozen signature
    "FrozenArg",
    "FrozenSignature",
    "Member",
    "OriginalArg",
    # Outer interface
    "OriginalInterface",
    "SignatureDriftError",
    "SyntheticGlobal",
    "UnsupportedShimInterfaceError",
    # Fortran-callable library builder
    "build_fortran_library",
    # bind(c) shim auto-gen (Phase 2.4)
    "emit_bind_c_shim",
    # Emitter
    "emit_bindings",
    "strip_index_args",
    "substitute_indices",
]
