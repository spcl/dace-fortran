# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``READ`` library node: list-directed read from a file into its outputs.

Lowers a Fortran ``read`` statement (fused with its ``open``/``close``) to a
C++ tasklet calling the shipped ``dace_fio_*`` wrappers, so the real Fortran
runtime performs the transfer.
"""

from __future__ import annotations
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .node import FortranIONode, fio_type
from .write import c_string
from .. import environments
from typing import Any

from dace import SDFG, SDFGState
from dace_fortran.dace_types import library_node


@dace.library.expansion
class ExpandReadFortranIO(ExpandTransformation):

    environments = [environments.FortranIO]

    @staticmethod
    def expansion(node: Read, parent_state: SDFGState, parent_sdfg: SDFG) -> nodes.Tasklet:  # type: ignore[override]  # dace's own library nodes narrow ``node`` the same way
        items = node.ordered_items(parent_sdfg, parent_state, "_out_", edges_in=False, num_items=node.num_items)
        path = c_string(node.filename)
        lines = [f'int _u = dace_fio_open("{path}", {len(node.filename.encode())}, 0);']
        for conn, desc, count, is_value in items:
            suffix, ctype = fio_type(desc.dtype)
            if is_value:
                lines.append(f'dace_fio_read_{suffix}(_u, ({ctype} *)&{conn});')
            else:
                lines.append(f'dace_fio_read_{suffix}_arr(_u, ({ctype} *){conn}, {count});')
        lines.append("dace_fio_close(_u);")
        return nodes.Tasklet(node.name,
                             node.in_connectors,
                             node.out_connectors,
                             "\n".join(lines),
                             language=dtypes.Language.CPP,
                             side_effects=True)


@library_node
class Read(FortranIONode):
    """Read ``num_items`` connected outputs from ``filename`` (list-directed)."""

    implementations = {"FortranIO": ExpandReadFortranIO}
    default_implementation = "FortranIO"

    filename = dace.properties.Property(dtype=str, default="", desc="Input file path")
    num_items: int = dace.properties.Property(dtype=int, default=0, desc="Number of items read")  # type: ignore[assignment]  # dace types a Property descriptor as returning its dtype argument

    def __init__(self, name: str, filename: str = "", num_items: int = 0, **kwargs: Any) -> None:
        super().__init__(name, inputs=set(), outputs={f"_out_{i}" for i in range(num_items)}, **kwargs)
        self.filename = filename
        self.num_items = num_items
