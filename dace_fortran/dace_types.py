# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Type aliases for DaCe call signatures whose declared parameter types are invariant containers."""
from __future__ import annotations

from typing import Callable, Iterable, TypeAlias, TypeVar, cast

import dace.library
from dace import dtypes
from dace.sdfg.nodes import LibraryNode
from dace.subsets import Subset
from dace.transformation import transformation

T = TypeVar("T")

#: ``map_ranges`` / ``ndrange`` argument of ``add_map`` / ``add_mapped_tasklet``: dict is invariant in its value
#: type, so a ``dict[str, str]`` literal needs this declared value type to be accepted.
MapRanges: TypeAlias = dict[str, str | Subset]


def connectors(names: Iterable[str]) -> dict[str, dtypes.typeclass]:
    """Ordered untyped tasklet connectors (``name -> None``; DaCe infers the type from the memlet).

    ``add_tasklet`` annotates the value as a required ``typeclass`` although ``None`` is its untyped-connector value;
    a set would lose the deterministic connector order that code generation depends on.
    """
    return cast(dict[str, dtypes.typeclass], dict.fromkeys(names))


def library_node(cls: type[T]) -> type[T]:
    """``@dace.library.node`` that keeps the decorated class's type.

    DaCe leaves the decorator unannotated, so type checkers lose the class and see a bare ``LibraryNode``.
    """
    return cast(type[T], dace.library.node(cls))


def register_expansion(node_cls: type[LibraryNode], name: str) -> Callable[[type[T]], type[T]]:
    """``@dace.library.register_expansion`` that keeps the decorated expansion class's type (DaCe types ``node_cls``
    as an instance and leaves the result unannotated)."""
    return cast(Callable[[type[T]], type[T]], dace.library.register_expansion(node_cls, name))  # type: ignore[arg-type]


def explicit_cf_compatible(cls: type[T]) -> type[T]:
    """``@explicit_cf_compatible`` taking and returning the pass class (DaCe annotates it as taking an instance)."""
    return cast(type[T], transformation.explicit_cf_compatible(cls))  # type: ignore[type-var]
