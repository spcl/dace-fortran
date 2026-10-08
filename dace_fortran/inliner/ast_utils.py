# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""fparser-tree helpers used by the vendored ``ast_desugaring`` package.

Minimal slice of upstream ``dace.frontend.fortran.ast_utils`` -- only the
three pure-fparser helpers the desugaring/pruning/cleanup passes call
(``singular``, ``atmost_one``, ``children_of_type``); the SDFG-construction
helpers there aren't needed here.  Kept import-compatible so the copied
``ast_desugaring`` modules need no source edits.
"""

from collections.abc import Iterator
from typing import TypeVar, Union, overload

from fparser.two.utils import Base

T = TypeVar("T")


def singular(items: Iterator[T]) -> T:
    """Asserts `items` has exactly 1 item and returns it."""
    it = atmost_one(items)
    assert it is not None, "`items` must not be empty."
    return it


def atmost_one(items: Iterator[T]) -> T | None:
    """Asserts `items` has at most 1 item; returns it or `None`."""
    try:
        it = next(items)
    except StopIteration:
        return None
    try:
        nit = next(items)
    except StopIteration:
        return it
    raise ValueError(f"`items` must have at most 1 item, got: {it}, {nit}, ...")


@overload
def children_of_type(node: Base, typ: str) -> Iterator[Base]: ...


@overload
def children_of_type(node: Base, typ: type[T]) -> Iterator[T]: ...


@overload
def children_of_type(node: Base, typ: tuple[type[T], ...]) -> Iterator[T]: ...


def children_of_type(node: Base, typ: Union[str, type[T], tuple[type[T], ...]]) -> Iterator[Base] | Iterator[T]:
    """Generator over `node`'s children that are of type `typ`."""
    if isinstance(typ, str):
        return (c for c in node.children if type(c).__name__ == typ)
    return (c for c in node.children if isinstance(c, typ))
