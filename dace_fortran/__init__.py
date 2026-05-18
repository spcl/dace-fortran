# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fortran (HLFIR) frontend for DaCe.

``import dace_fortran`` registers the Fortran frontend; it builds
SDFGs by lowering Fortran through ``flang-new``'s HLFIR and an
MLIR/C++ bridge into a ``dace.SDFG`` (DaCe stays the dependency, this
package is the frontend plugin on top of it).

The public entry points are exposed lazily so ``import dace_fortran``
stays cheap: the C++ HLFIR bridge is compiled on first *use* (first
access of ``SDFGBuilder`` / ``generate_sdfg``), not at import time.

    import dace_fortran
    sdfg = dace_fortran.generate_sdfg("kernel.hlfir")   # -> dace.SDFG

    # or from Fortran source via the preprocess / builder API, e.g.
    from dace_fortran.preprocess import merge_used_modules
"""

_LAZY = {
    "SDFGBuilder": "dace_fortran.hlfir_to_sdfg",
    "generate_sdfg": "dace_fortran.hlfir_to_sdfg",
    "DEFAULT_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "MULTI_FILE_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "merge_used_modules": "dace_fortran.preprocess",
    "preprocess_fortran": "dace_fortran.preprocess",
    "rewrite_integer_powers": "dace_fortran.preprocess",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """PEP 562 lazy attribute access  --  defer importing the builder
    (and the C++-bridge build it triggers) until an entry point is
    actually referenced."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(__all__)
