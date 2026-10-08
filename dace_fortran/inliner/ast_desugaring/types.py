# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeGuard, Union

import fparser.two.Fortran2003 as f03
import numpy as np

if TYPE_CHECKING:
    from dace_fortran.inliner.ast_desugaring.utils import NAMED_STMTS_OF_INTEREST_TYPES

# fparser node type aliases live in the modules that use them; this file is for our own custom types.

# SPEC: tuple of strings uniquely identifying an AST object, e.g. ('my_program', 'my_module', 'my_subroutine', 'my_variable').
SPEC = tuple[str, ...]

# Maps a SPEC to its defining node; string is a forward ref to NAMED_STMTS_OF_INTEREST_TYPES in utils.
SPEC_TABLE = dict[SPEC, "NAMED_STMTS_OF_INTEREST_TYPES"]

# Type Aliases for numpy types used in constant evaluation
NUMPY_INTS_TYPES = Union[np.int8, np.int16, np.int32, np.int64]
NUMPY_REALS_TYPES = Union[np.float32, np.float64]
NUMPY_TYPES = Union[NUMPY_INTS_TYPES, NUMPY_REALS_TYPES, np.bool_]

# Type Aliases for fparser literal nodes.
LITERAL_TYPES = Union[
    f03.Real_Literal_Constant,
    f03.Signed_Real_Literal_Constant,
    f03.Int_Literal_Constant,
    f03.Signed_Int_Literal_Constant,
    f03.Logical_Literal_Constant,
]
LITERAL_CLASSES = (
    f03.Real_Literal_Constant,
    f03.Signed_Real_Literal_Constant,
    f03.Int_Literal_Constant,
    f03.Signed_Int_Literal_Constant,
    f03.Logical_Literal_Constant,
)


class TYPE_SPEC:
    """Parses a Fortran variable's attribute string (e.g. 'DIMENSION(..)', 'INTENT(IN)') into shape/intent/etc properties."""

    NO_ATTRS = ""

    __slots__ = ("alloc", "const", "inp", "keyword", "optional", "out", "pointer", "shape", "spec")

    spec: SPEC
    shape: tuple[str, ...]
    optional: bool
    pointer: bool
    inp: bool
    out: bool
    alloc: bool
    const: bool
    keyword: str | None

    def __init__(self, spec: Union[str, SPEC], attrs: str = NO_ATTRS, is_arg: bool = False) -> None:
        if isinstance(spec, str):
            spec = (spec,)
        self.spec = spec
        self.shape = self._parse_shape(attrs)
        self.optional = "OPTIONAL" in attrs
        self.pointer = "POINTER" in attrs
        self.inp = "INTENT(IN)" in attrs or "INTENT(INOUT)" in attrs
        self.out = "INTENT(OUT)" in attrs or "INTENT(INOUT)" in attrs
        self.alloc = "ALLOCATABLE" in attrs
        self.const = "PARAMETER" in attrs
        self.keyword = None
        if is_arg and not self.inp and not self.out:
            # Argument with no explicit intent is both in and out.
            self.inp, self.out = True, True

    def copy(self) -> TYPE_SPEC:
        """Independent copy; use before mutating shape/keyword in place so a shared
        sentinel like the match-anything MATCH_ALL is never corrupted."""
        other = TYPE_SPEC(self.spec)
        other.shape = self.shape
        other.optional = self.optional
        other.pointer = self.pointer
        other.inp = self.inp
        other.out = self.out
        other.alloc = self.alloc
        other.const = self.const
        other.keyword = self.keyword
        return other

    @staticmethod
    def _parse_shape(attrs: str) -> tuple[str, ...]:
        """Parses the DIMENSION attribute into per-dimension strings."""
        if "DIMENSION" not in attrs:
            return tuple()
        parts = []
        dims = attrs.split("DIMENSION")[1]
        assert dims[0] == "("
        paren_count, part_start = 1, 1
        for i in range(1, len(dims)):
            if dims[i] == "(":
                paren_count += 1
            elif dims[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    parts.append(dims[part_start:i])
                    break
            elif dims[i] == "," and paren_count == 1:
                parts.append(dims[part_start:i])
                part_start = i + 1
        return tuple(p.strip().lower() for p in parts)

    def __repr__(self) -> str:
        attrs = []
        if self.pointer:
            attrs.append("*")
        if self.shape:
            attrs.append(f"shape={self.shape}")
        if self.optional:
            attrs.append("optional")
        if not attrs:
            return f"{self.spec}"
        return f"{self.spec}[{' | '.join(attrs)}]"

    def to_decl(self, var: str) -> str:
        """Generates a Fortran declaration string for `var` with this type spec."""
        TYPE_MAP = {
            "INTEGER1": "INTEGER(kind=1)",
            "INTEGER2": "INTEGER(kind=2)",
            "INTEGER4": "INTEGER(kind=4)",
            "INTEGER8": "INTEGER(kind=8)",
            "INTEGER": "INTEGER(kind=4)",
            "REAL4": "REAL(kind=4)",
            "REAL8": "REAL(kind=8)",
            "REAL": "REAL(kind=4)",
            "COMPLEX4": "COMPLEX(kind=4)",
            "COMPLEX8": "COMPLEX(kind=8)",
            "COMPLEX": "COMPLEX(kind=4)",
            "LOGICAL": "LOGICAL",
        }
        typ = self.spec[-1]
        typ = TYPE_MAP.get(typ, f"type({typ})")

        bits: list[str] = [typ]
        if self.alloc:
            bits.append("allocatable")
        if self.optional:
            bits.append("optional")
        if self.inp and self.out:
            bits.append("intent(inout)")
        elif self.inp:
            bits.append("intent(in)")
        elif self.out:
            bits.append("intent(out)")
        if self.const:
            bits.append("parameter")
        bits_str: str = ", ".join(bits)
        shape_str: str = ", ".join(self.shape) if self.shape else ""
        shape_str = f"({shape_str})" if shape_str else ""
        return f"{bits_str} :: {var}{shape_str}"


@dataclass(slots=True)
class ConstTypeInjection:
    """Constant-value injection for a derived-type component, applied everywhere that type is used (optionally scoped)."""

    scope_spec: SPEC | None  # Only replace within this scope object.
    type_spec: SPEC  # The root config derived type's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


@dataclass(slots=True)
class ConstInstanceInjection:
    """Constant-value injection for one variable instance's component (not all instances of its type)."""

    scope_spec: SPEC | None  # Only replace within this scope object.
    root_spec: SPEC  # The root config object's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


ConstInjection = Union[ConstTypeInjection, ConstInstanceInjection]


def is_numpy_int(val: object) -> TypeGuard[NUMPY_INTS_TYPES]:
    """Whether ``val`` is one of the numpy signed-integer scalars the constant evaluator produces."""
    return isinstance(val, np.integer)


def is_numpy_real(val: object) -> TypeGuard[NUMPY_REALS_TYPES]:
    """Whether ``val`` is one of the numpy floating scalars the constant evaluator produces."""
    return isinstance(val, np.floating)


def numpy_type_to_literal(val: NUMPY_TYPES) -> LITERAL_TYPES:
    """Converts a numpy scalar (int/float/bool) to its fparser literal node."""
    if isinstance(val, np.bool_):
        return f03.Logical_Literal_Constant(".true." if val else ".false.")
    if is_numpy_int(val):
        bytez = count_bytes(type(val))
        if val < 0:
            return f03.Signed_Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
        return f03.Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
    if is_numpy_real(val):
        bytez = count_bytes(type(val))
        valstr = str(val)
        if bytez == 8:
            valstr = valstr.replace("e", "D") if "e" in valstr else f"{valstr}D0"
        if val < 0:
            return f03.Signed_Real_Literal_Constant(valstr)
        return f03.Real_Literal_Constant(valstr)
    raise TypeError(f"not a numpy scalar the constant evaluator produces: {type(val)}")


def count_bytes(t: type[NUMPY_TYPES]) -> int:
    """Byte size of a numpy numeric type."""
    if t is np.int8:
        return 1
    if t is np.int16:
        return 2
    if t is np.int32:
        return 4
    if t is np.int64:
        return 8
    if t is np.float32:
        return 4
    if t is np.float64:
        return 8
    if t is np.bool_:
        return 1
    raise ValueError(f"{t} is not an expected type; expected {NUMPY_TYPES}")
