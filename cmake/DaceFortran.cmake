# ============================================================================
# DaceFortran.cmake -- one-line CMake integration for the dace-fortran
# preprocessing pipeline.
# ============================================================================
#
# Goal: minimal manual steps.  A project that wants the bridge's source
# rewrites (USE-module merging, EXTERNAL -> module imports, string-enum
# -> integer, precision-alias substitution) before compiling Fortran
# adds two lines:
#
#   include(DaceFortran)
#   dace_fortran_preprocess(TARGET mylib
#                            SOURCES src/kernel.f90 src/utils.f90
#                            SEARCH_DIRS src/utils
#                            PASSES merge_modules normalize_kind
#                                   rewrite_external rewrite_string_enum)
#
# Each input .f90 gets a sibling .f90 under
# ${CMAKE_CURRENT_BINARY_DIR}/dace_fortran_preprocessed/<rel-path>.f90
# that the actual library target compiles.  CMake's dependency graph
# rebuilds the preprocessed source whenever the input or any sidecar
# module under SEARCH_DIRS changes.
#
# Idempotent: a second call against the same SOURCES is a no-op (custom
# commands keyed by output path).
#
# Requirements: a Python 3 interpreter (find_package(Python3) or the
# DACE_FORTRAN_PYTHON cache variable) reachable from the build, plus
# the ``dace_fortran`` package importable from that interpreter.
#
# ----------------------------------------------------------------------------
# Public API:
#
#   dace_fortran_preprocess(
#     TARGET   <target_name>
#     SOURCES  <abs-or-rel paths to .f90 / .F90 files>
#     [SEARCH_DIRS <dirs of sidecar modules>]
#     [PASSES  <pass-name ...>]              # default: all_defaults
#     [OUTPUT_DIR <path>]                    # default: ${CMAKE_CURRENT_BINARY_DIR}/dace_fortran_preprocessed
#     [KIND_MAP <NAME=N ...>]                # forwarded as --kind-map
#     [ENUM_MAPS_DIR <path>]                 # default: <OUTPUT_DIR>/enum_maps
#   )
#
# Side effects:
#   * Creates one ``add_custom_command`` per input source.
#   * Creates a target ``<target_name>_preprocessed_sources`` that
#     groups every custom command (downstream consumers ``add_dependencies``
#     against it).
#   * Returns the preprocessed paths in the variable
#     ``<target_name>_PREPROCESSED_SOURCES`` (parent scope) -- callers
#     pass that variable to ``add_library(... ${...})`` /
#     ``target_sources(...)`` to compile the rewritten Fortran.
#   * For ``rewrite_string_enum``, sidecar ``.enum_maps.json`` files
#     land in ENUM_MAPS_DIR and are tracked as BYPRODUCTS of the same
#     custom commands so they regenerate alongside the source.
#
# Pass names recognised in PASSES (match the CLI flags):
#   merge_modules, strip_openmp, rewrite_integer_powers,
#   normalize_kind, rewrite_external, rewrite_string_enum,
#   rewrite_if_intvar, all_defaults
#
# ============================================================================

if(NOT DEFINED DACE_FORTRAN_PYTHON)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(DACE_FORTRAN_PYTHON ${Python3_EXECUTABLE} CACHE FILEPATH
        "Python interpreter the dace_fortran preprocess CLI runs under")
endif()

# Verify the CLI is reachable.  ``-c`` form so the message-on-failure
# is the helpful ImportError, not a flag-parsing complaint.
execute_process(
    COMMAND ${DACE_FORTRAN_PYTHON} -c
        "import dace_fortran.preprocess_cli"
    RESULT_VARIABLE _DACE_FORTRAN_PROBE
    OUTPUT_QUIET
    ERROR_VARIABLE _DACE_FORTRAN_PROBE_ERR)
if(NOT _DACE_FORTRAN_PROBE EQUAL 0)
    message(FATAL_ERROR
        "DaceFortran.cmake: the dace_fortran Python package is not "
        "importable from ${DACE_FORTRAN_PYTHON}.\n"
        "Install it (``pip install dace-fortran`` from a checkout) or "
        "set DACE_FORTRAN_PYTHON to an interpreter that has it.\n"
        "ImportError was:\n${_DACE_FORTRAN_PROBE_ERR}")
endif()


function(dace_fortran_preprocess)
    set(_options)
    set(_one_value_args TARGET OUTPUT_DIR ENUM_MAPS_DIR)
    set(_multi_value_args SOURCES SEARCH_DIRS PASSES KIND_MAP)
    cmake_parse_arguments(_DFP "${_options}" "${_one_value_args}"
                          "${_multi_value_args}" ${ARGN})

    if(NOT _DFP_TARGET)
        message(FATAL_ERROR "dace_fortran_preprocess: TARGET required")
    endif()
    if(NOT _DFP_SOURCES)
        message(FATAL_ERROR "dace_fortran_preprocess: SOURCES required")
    endif()
    if(NOT _DFP_OUTPUT_DIR)
        set(_DFP_OUTPUT_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/dace_fortran_preprocessed")
    endif()
    if(NOT _DFP_ENUM_MAPS_DIR)
        set(_DFP_ENUM_MAPS_DIR "${_DFP_OUTPUT_DIR}/enum_maps")
    endif()
    if(NOT _DFP_PASSES)
        set(_DFP_PASSES all_defaults)
    endif()

    # Translate the friendly pass list to ``--<flag>`` form.
    set(_pass_flags)
    foreach(_pass IN LISTS _DFP_PASSES)
        if(_pass STREQUAL "all_defaults")
            list(APPEND _pass_flags "--all-defaults")
        else()
            # ``rewrite_string_enum`` -> ``--rewrite-string-enum``
            string(REPLACE "_" "-" _flag "${_pass}")
            list(APPEND _pass_flags "--${_flag}")
        endif()
    endforeach()

    # SEARCH_DIRS -> one ``--search-dir`` per entry.
    set(_search_flags)
    foreach(_d IN LISTS _DFP_SEARCH_DIRS)
        # CMake normalises paths; absolutize so a relative
        # SEARCH_DIRS works no matter where the caller cmakes from.
        get_filename_component(_d_abs "${_d}" ABSOLUTE)
        list(APPEND _search_flags "--search-dir" "${_d_abs}")
    endforeach()

    # KIND_MAP -> one ``--kind-map NAME=N`` per entry.
    set(_kind_flags)
    foreach(_km IN LISTS _DFP_KIND_MAP)
        list(APPEND _kind_flags "--kind-map" "${_km}")
    endforeach()

    set(_processed_sources)
    foreach(_src IN LISTS _DFP_SOURCES)
        get_filename_component(_src_abs "${_src}" ABSOLUTE)
        file(RELATIVE_PATH _src_rel
             "${CMAKE_CURRENT_SOURCE_DIR}" "${_src_abs}")
        # ``../`` in the relative path -- the input lives above the
        # current source dir; fall back to basename so the output stays
        # under OUTPUT_DIR.
        if(_src_rel MATCHES "\\.\\./")
            get_filename_component(_src_rel "${_src_abs}" NAME)
        endif()
        set(_out "${_DFP_OUTPUT_DIR}/${_src_rel}")
        get_filename_component(_out_dir "${_out}" DIRECTORY)
        file(MAKE_DIRECTORY "${_out_dir}")

        # Track every module file under SEARCH_DIRS so a change to a
        # sidecar module regenerates dependent preprocessed sources.
        set(_search_globs)
        foreach(_d IN LISTS _DFP_SEARCH_DIRS)
            get_filename_component(_d_abs "${_d}" ABSOLUTE)
            file(GLOB_RECURSE _glob CONFIGURE_DEPENDS
                "${_d_abs}/*.f90" "${_d_abs}/*.F90" "${_d_abs}/*.incf")
            list(APPEND _search_globs ${_glob})
        endforeach()

        # If the user enabled rewrite_string_enum, the sidecar JSON
        # is a byproduct in ENUM_MAPS_DIR.
        set(_enum_maps_byproduct)
        if(_DFP_PASSES MATCHES "rewrite_string_enum")
            file(MAKE_DIRECTORY "${_DFP_ENUM_MAPS_DIR}")
            set(_enum_maps_byproduct
                "${_DFP_ENUM_MAPS_DIR}/${_src_rel}.enum_maps.json")
        endif()

        add_custom_command(
            OUTPUT "${_out}"
            BYPRODUCTS ${_enum_maps_byproduct}
            COMMAND ${DACE_FORTRAN_PYTHON} -m dace_fortran.preprocess_cli
                ${_pass_flags}
                ${_search_flags}
                ${_kind_flags}
                --in "${_src_abs}"
                --out "${_out}"
            DEPENDS "${_src_abs}" ${_search_globs}
            COMMENT "[dace-fortran] preprocess ${_src_rel}"
            VERBATIM
        )
        list(APPEND _processed_sources "${_out}")

        # Sidecar JSON lives next to the rewritten source under
        # ENUM_MAPS_DIR (different tree so the JSON doesn't pollute
        # the Fortran-source glob the actual library target uses).
        if(_enum_maps_byproduct)
            # Symlink / copy the byproduct into the predictable JSON
            # location so binding-emission downstream finds them by
            # a stable path.
            get_filename_component(_emap_dir "${_enum_maps_byproduct}" DIRECTORY)
            file(MAKE_DIRECTORY "${_emap_dir}")
        endif()
    endforeach()

    add_custom_target(${_DFP_TARGET}_preprocessed_sources ALL
                      DEPENDS ${_processed_sources})

    # Return the preprocessed source list in parent scope so the caller
    # can pass it to ``add_library`` / ``target_sources``.
    set("${_DFP_TARGET}_PREPROCESSED_SOURCES"
        ${_processed_sources}
        PARENT_SCOPE)
endfunction()
