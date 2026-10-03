# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``NamelistRead`` library node: read named members from a Fortran namelist.

A fixed ``NAMELIST`` read needs its group declared at compile time, so this
node expands to a C++ tasklet calling the generic ``dace_nml_*`` wrappers
instead: open ``(file, group)``, fetch each member by name, close.
"""

from __future__ import annotations
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace.transformation.transformation import ExpandTransformation

from .node import FortranIONode, fio_type
from .write import c_string
from .. import environments
from typing import Any, Sequence, cast

from dace import SDFG, SDFGState
from dace_fortran.dace_types import library_node


@dace.library.expansion
class ExpandNamelistReadFortranIO(ExpandTransformation):
    environments = [environments.FortranIO]

    @staticmethod
    def expansion(
        node: LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, *args: Any, **kwargs: Any
    ) -> nodes.Tasklet:
        assert isinstance(node, NamelistRead)
        items = node.ordered_items(parent_sdfg, parent_state, "_out_", edges_in=False, num_items=node.num_items)
        if len(node.members) != len(items):
            raise ValueError(
                f"NamelistRead '{node.name}': {len(node.members)} member names for {len(items)} connected outputs"
            )
        path, group = c_string(node.filename), c_string(node.group)
        lines = [
            f'int _h = dace_nml_open("{path}", {len(node.filename.encode())}, "{group}", {len(node.group.encode())});'
        ]
        for (conn, desc, count, is_value), member in zip(items, node.members):
            suffix, ctype = fio_type(desc.dtype)
            name_arg = f'"{c_string(member)}", {len(member.encode())}'
            if is_value:
                lines.append(f"dace_nml_get_{suffix}(_h, {name_arg}, ({ctype} *)&{conn});")
            else:
                lines.append(f"dace_nml_get_{suffix}_arr(_h, {name_arg}, ({ctype} *){conn}, {count});")
        lines.append("dace_nml_close(_h);")
        return nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            "\n".join(lines),
            language=dtypes.Language.CPP,
            side_effects=True,
        )


@library_node
class NamelistRead(FortranIONode):
    """Read ``members`` of namelist ``group`` from ``filename`` into outputs."""

    implementations = {"FortranIO": ExpandNamelistReadFortranIO}
    default_implementation = "FortranIO"

    filename = dace.properties.Property(dtype=str, default="", desc="Namelist file path")
    group = dace.properties.Property(dtype=str, default="", desc="Namelist group name")
    # dace types ``ListProperty(element_type=str)`` as ``list[type[str]]``.
    members = cast(
        list[str],
        dace.properties.ListProperty(element_type=str, default=[], desc="Member names, in output-connector order"),
    )

    def __init__(
        self, name: str, filename: str = "", group: str = "", members: Sequence[str] | None = None, **kwargs: Any
    ) -> None:
        members = list(members or [])
        super().__init__(name, inputs=set(), outputs={f"_out_{i}" for i in range(len(members))}, **kwargs)
        self.filename = filename
        self.group = group
        self.members = members

    @property
    def num_items(self) -> int:
        return len(self.members)
