# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed stand-ins for the bridge's ``VarInfo`` / ``AccessInfo`` / ``ASTNode`` records.

The emitter synthesises a few records the bridge never produced (phantom pointer-component views, bridge
scratch scalars, spliced section-alias accesses, rewritten libcalls).  Each stand-in carries the *full* field
set of the bridge type with neutral defaults, so emitter code reads ``v.role`` / ``n.options`` directly on
either kind instead of probing with ``getattr(..., default)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from dace_fortran.bridge_types import AccessRecord, NodeRecord, VarRecord


@dataclass(slots=True)
class SyntheticVar:
    """Stand-in for ``hb.VarInfo`` (same field names; every field defaulted)."""
    fortran_name: str = ''
    mangled_name: str = ''
    intent: str = ''
    rank: int = 0
    dtype: str = ''
    is_dynamic: bool = False
    is_written: bool = False
    shape_symbols: list[str] = field(default_factory=list)
    lower_bounds: list[str] = field(default_factory=list)
    role: str = ''
    const_data: bool = False
    view_source: str = ''
    view_subset: list[str] = field(default_factory=list)
    view_dim_map: list[str] = field(default_factory=list)
    module_origin_mod: str = ''
    module_origin_name: str = ''
    module_origin_allocatable: bool = False
    module_origin_pointer: bool = False
    bounds_remap_view: bool = False
    bounds_remap_source: str = ''
    bounds_remap_total_extent: str = ''
    bounds_remap_source_subset: list[str] = field(default_factory=list)
    unbindable_section: bool = False
    aos_origin_mod: str = ''
    aos_origin_struct: str = ''
    aos_member_path: str = ''
    aos_outer_rank: int = 0
    global_alloc_inside: bool = False
    aos_struct_pointer: bool = False
    aos_member_pointer: bool = False


@dataclass(slots=True)
class ComplexAliasSpec(SyntheticVar):
    """Same-dtype COMPLEX view of a complex-as-2-reals component alias; ``shape`` is its descriptor extent."""
    shape: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SyntheticAccess:
    """Stand-in for ``hb.AccessInfo``."""
    array_name: str = ''
    index_vars: list[str] = field(default_factory=list)
    index_exprs: list[str] = field(default_factory=list)
    is_read: bool = False
    is_write: bool = False


@dataclass(slots=True)
class SyntheticNode:
    """Stand-in for ``hb.ASTNode`` (statement-level fields only)."""
    kind: str = ''
    loop_iter: str = ''
    loop_bound: str = ''
    loop_lower: int = -1
    loop_lower_expr: str = ''
    loop_step: int = 1
    loop_step_expr: str = ''
    target: str = ''
    expr: str = ''
    accesses: list = field(default_factory=list)
    pos_indices: list[int] = field(default_factory=list)
    target_is_array: bool = False
    condition: str = ''
    callee: str = ''
    call_args: list[str] = field(default_factory=list)
    call_arg_subsets: list[str] = field(default_factory=list)
    aos_marshal_groups: list[int] = field(default_factory=list)
    reduce_src: str = ''
    reduce_wcr: str = ''
    reduce_identity: str = ''
    reduce_axes: list[int] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    else_children: list = field(default_factory=list)


#: A variable record: bridge-produced or emitter-synthesised (both satisfy :class:`VarRecord`).
VarLike: TypeAlias = VarRecord
#: An access record: bridge-produced or emitter-synthesised.
AccessLike: TypeAlias = AccessRecord
#: A statement node: bridge-produced or emitter-synthesised.
NodeLike: TypeAlias = NodeRecord
