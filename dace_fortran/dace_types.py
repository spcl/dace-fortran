# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Type aliases for DaCe call signatures whose declared parameter types are invariant containers."""
from __future__ import annotations

from typing import Iterable, TypeAlias, cast

from dace import dtypes
from dace.subsets import Subset

#: ``map_ranges`` / ``ndrange`` argument of ``add_map`` / ``add_mapped_tasklet``: dict is invariant in its value
#: type, so a ``dict[str, str]`` literal needs this declared value type to be accepted.
MapRanges: TypeAlias = dict[str, str | Subset]


def connectors(names: Iterable[str]) -> dict[str, dtypes.typeclass]:
    """Ordered untyped tasklet connectors (``name -> None``; DaCe infers the type from the memlet).

    ``add_tasklet`` annotates the value as a required ``typeclass`` although ``None`` is its untyped-connector value;
    a set would lose the deterministic connector order that code generation depends on.
    """
    return cast(dict[str, dtypes.typeclass], dict.fromkeys(names))
