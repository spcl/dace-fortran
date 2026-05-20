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

    # Inline Fortran source -> built dace.SDFG:
    sdfg = dace_fortran.build_sdfg(src, entry="_QPrun")

    # A multi-file project (driver + the modules it USEs, any order):
    sdfg = dace_fortran.build_sdfg_from_files([drv, mod], entry="_QPrun")

    # A kernel that CALLs a separately-compiled bind(c) function:
    dace_fortran.register_external("foo", dace_fortran.ExternalSignature(
        c_name="foo",
        args=[dace_fortran.Arg("array", "float64")],   # intent defaults to inout
        libraries=["/path/libfoo.so"]))

    # Low level: an already-emitted .hlfir file:
    sdfg = dace_fortran.generate_sdfg("kernel.hlfir")
"""

_LAZY = {
    # Public build entry points (the documented surface).
    "build_sdfg": "dace_fortran.build",
    "build_sdfg_from_files": "dace_fortran.build",
    "register_external": "dace_fortran.external",
    "keep_external": "dace_fortran.external",
    "ExternalSignature": "dace_fortran.external",
    "Arg": "dace_fortran.external",
    "clear_external_registry": "dace_fortran.external",
    # Lower-level / advanced.
    "SDFGBuilder": "dace_fortran.hlfir_to_sdfg",
    "generate_sdfg": "dace_fortran.hlfir_to_sdfg",
    "DEFAULT_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "MULTI_FILE_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "preprocess_fortran_source": "dace_fortran.preprocess",
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
