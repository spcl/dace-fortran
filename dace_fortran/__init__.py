# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fortran (HLFIR) frontend for DaCe: lowers Fortran through flang-new's
HLFIR and an MLIR/C++ bridge into a ``dace.SDFG``.

Entry points below are exposed lazily so ``import dace_fortran`` stays
cheap -- the C++ bridge builds on first use, not at import time. See
README "Building an SDFG from a real project" for worked examples.
"""

_LAZY = {
    # Public build entry points (the documented surface).
    "build_sdfg": "dace_fortran.build",
    "build_sdfg_from_files": "dace_fortran.build",
    "build_sdfg_from_hlfir": "dace_fortran.build",
    "build_sdfg_from_project": "dace_fortran.build",
    # emit_hlfir is a module (CLI or direct import), not a single function -- not in this facade.
    "register_external": "dace_fortran.external",
    "keep_external": "dace_fortran.external",
    "apply_external_functions": "dace_fortran.external",
    "ExternalSignature": "dace_fortran.external",
    "Arg": "dace_fortran.external",
    "clear_external_registry": "dace_fortran.external",
    # Distinct from the internal ExternalCall libnode / ExternalSignature ABI record -- no name conflict.
    "ExternalFunction": "dace_fortran.external_functions",
    # Lower-level / advanced.
    "SDFGBuilder": "dace_fortran.hlfir_to_sdfg",
    "generate_sdfg": "dace_fortran.hlfir_to_sdfg",
    "DEFAULT_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "MULTI_FILE_PIPELINE": "dace_fortran.hlfir_to_sdfg",
    "preprocess_fortran_source": "dace_fortran.preprocess",
    "merge_used_modules": "dace_fortran.preprocess",
    # fparser-based single-TU inliner (opt-in alternative to the regex merge_used_modules splicer).
    "inline_to_single_tu": "dace_fortran.fparser_inliner",
    "inline_to_ast": "dace_fortran.fparser_inliner",
    "preprocess_fortran": "dace_fortran.preprocess",
    "rewrite_integer_powers": "dace_fortran.preprocess",
    "normalize_kind_parameters": "dace_fortran.preprocess",
    "replace_external_with_modules": "dace_fortran.preprocess",
    "rewrite_string_enum_to_integer": "dace_fortran.preprocess",
    # Real-world-codebase helpers (ICON / IFS / ECRAD etc.).
    "prepare_flang_translation_unit": "dace_fortran.flang_codebase",
    "emit_hlfir_from_codebase": "dace_fortran.flang_codebase",
    "extract_make_compile_args": "dace_fortran.flang_codebase",
    "vendor_netcdf_fortran": "dace_fortran.flang_codebase",
    "mpi_stub_source": "dace_fortran.flang_codebase",
    "find_openmpi_include": "dace_fortran.flang_codebase",
    "LIBRARY_STUBS": "dace_fortran.flang_codebase",
    "FLANG_BUG_PATCHES": "dace_fortran.flang_codebase",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """PEP 562 lazy attribute access -- defers importing the builder (and its C++-bridge build) until first use."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(__all__)
