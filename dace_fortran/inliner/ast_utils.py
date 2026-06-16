# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""fparser-tree helpers used by the vendored ``ast_desugaring`` package.

This is a deliberately minimal slice of the upstream
``dace.frontend.fortran.ast_utils`` -- only the three pure-fparser helper
functions that the desugaring / pruning / cleanup passes actually call
(``singular``, ``atmost_one``, ``children_of_type``).  The upstream module
also exposes a large body of SDFG-construction helpers that pull in the
whole ``dace`` SDFG stack; none of those are needed for the source-text
single-TU inliner, so they are intentionally not vendored here.

Kept import-compatible with the upstream so the copied ``ast_desugaring``
modules (which do ``from .. import ast_utils``) need no source edits.
"""
from typing import Iterator, Optional, Tuple, Type, TypeVar, Union

from fparser.two.utils import Base

T = TypeVar("T")


def singular(items: Iterator[T]) -> T:
    """
    Asserts that any given iterator or generator `items` has exactly 1 item and returns that.
    """
    it = atmost_one(items)
    assert it is not None, "`items` must not be empty."
    return it


def atmost_one(items: Iterator[T]) -> Optional[T]:
    """
    Asserts that any given iterator or generator `items` has at most 1 item and returns that (or `None`).
    """
    # We might get one item.
    try:
        it = next(items)
    except StopIteration:
        # No items found.
        return None
    # But not another one.
    try:
        nit = next(items)
    except StopIteration:
        # I.e., we must have exhausted the iterator.
        return it
    raise ValueError(f"`items` must have at most 1 item, got: {it}, {nit}, ...")


def children_of_type(node: Base, typ: Union[str, Type[T], Tuple[Type, ...]]) -> Iterator[T]:
    """
    Returns a generator over the children of `node` that are of type `typ`.
    """
    if isinstance(typ, str):
        return (c for c in node.children if type(c).__name__ == typ)
    return (c for c in node.children if isinstance(c, typ))
