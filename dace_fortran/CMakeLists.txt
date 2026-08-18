# ============================================================================
# CMakeLists.txt  --  hlfir_bridge
# ============================================================================
# Build targets:
#   hlfir_bridge_core     --  static lib  (trace/extract)
#   hlfir_bridge_passes   --  static lib  (MLIR passes)
#   hlfir_bridge          --  Python ext  (nanobind; links both above)
#
# MLIR linking strategy (cf. pixel-druid.com/articles/how-to-link-against-mlir-with-cmake):
#   Debian/Ubuntu MLIR cmake configs are often broken (missing tool binaries
#   cause CMake export-set errors), so find_package(MLIR) is used only on a
#   prefix that has no monolithic libMLIR.  Otherwise find_package(LLVM)  --
#   which always works  --  locates MLIR headers and libraries through LLVM's
#   install prefix.  MLIR is always co-installed alongside LLVM.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# --- LLVM version ------------------------------------------------------------
# -DLLVM_VERSION pins a major; empty means "probe LLVM_SUPPORTED_VERSIONS and take
# the first one installed".  Keep this list in sync with llvm_toolchain.py's
# SUPPORTED_LLVM_VERSIONS and with llvm_compat.h's LLVM_VERSION_MAJOR arms.
set(LLVM_SUPPORTED_VERSIONS "21;22" CACHE STRING "LLVM major versions the bridge supports, in preference order")
set(LLVM_VERSION "" CACHE STRING "LLVM major version to use (empty: auto-detect)")

# Debian/Ubuntu ship flang-new-<major>/flang-<major>; spack and upstream installs
# ship only unversioned flang-new/flang, which have to self-identify.
function(_flang_reported_major out_var binary)
    execute_process(COMMAND "${binary}" --version OUTPUT_VARIABLE _out ERROR_QUIET)
    string(REGEX MATCH "flang(-new)? version ([0-9]+)" _match "${_out}")
    set(${out_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
endfunction()

function(_find_flang_for out_bin out_version)
    set(_versions "${LLVM_SUPPORTED_VERSIONS}")
    if(LLVM_VERSION)
        set(_versions "${LLVM_VERSION}")
    endif()
    foreach(_v IN LISTS _versions)
        unset(_probe CACHE)
        find_program(_probe NAMES flang-new-${_v} flang-${_v})
        if(NOT _probe)
            unset(_probe CACHE)
            find_program(_probe NAMES flang-new flang)
            if(_probe)
                _flang_reported_major(_major "${_probe}")
                if(NOT _major STREQUAL _v)
                    set(_probe "_probe-NOTFOUND")
                endif()
            endif()
        endif()
        if(_probe)
            set(${out_bin} "${_probe}" PARENT_SCOPE)
            set(${out_version} "${_v}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_bin} "" PARENT_SCOPE)
    set(${out_version} "" PARENT_SCOPE)
endfunction()

_find_flang_for(_FLANG_BIN _FLANG_VERSION)
if(_FLANG_VERSION)
    set(LLVM_VERSION "${_FLANG_VERSION}")
elseif(NOT LLVM_VERSION)
    # No flang on PATH: fall back to the newest supported major so LLVM_DIR /
    # the distro paths below still get a version to substitute.
    list(GET LLVM_SUPPORTED_VERSIONS -1 LLVM_VERSION)
    message(STATUS "No flang found for LLVM ${LLVM_SUPPORTED_VERSIONS}; assuming ${LLVM_VERSION}")
endif()

# --- Discover LLVM prefix from flang -----------------------------------------
set(_LLVM_PREFIX "" CACHE PATH "LLVM installation prefix (auto-detected)")

if(NOT _LLVM_PREFIX AND _FLANG_BIN)
    get_filename_component(_FLANG_REAL "${_FLANG_BIN}" REALPATH)
    get_filename_component(_FLANG_BINDIR "${_FLANG_REAL}" DIRECTORY)
    get_filename_component(_LLVM_PREFIX "${_FLANG_BINDIR}" DIRECTORY)
    message(STATUS "Derived LLVM ${LLVM_VERSION} prefix from ${_FLANG_BIN}: ${_LLVM_PREFIX}")
endif()

# --- Default compilers -------------------------------------------------------
# Prefer the clang shipped in the same prefix as the LLVM we link against, under
# either the versioned (distro) or unversioned (spack/upstream) name.
macro(_default_compiler lang envvar stem)
    if(NOT DEFINED CMAKE_${lang}_COMPILER AND NOT DEFINED ENV{${envvar}})
        unset(_found CACHE)
        if(_LLVM_PREFIX)
            find_program(_found NAMES ${stem}-${LLVM_VERSION} ${stem} PATHS "${_LLVM_PREFIX}/bin" NO_DEFAULT_PATH)
        endif()
        if(NOT _found)
            unset(_found CACHE)
            find_program(_found NAMES ${stem}-${LLVM_VERSION})
        endif()
        if(_found)
            set(CMAKE_${lang}_COMPILER "${_found}" CACHE FILEPATH "")
        endif()
        unset(_found CACHE)
    endif()
endmacro()
_default_compiler(C CC clang)
_default_compiler(CXX CXX clang++)

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    if(_FLANG_BIN)
        set(CMAKE_Fortran_COMPILER "${_FLANG_BIN}" CACHE FILEPATH "")
    endif()
endif()

project(hlfir_bridge C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- LLVM (the only find_package we use) -------------------------------------
if(NOT LLVM_DIR)
    if(_LLVM_PREFIX AND EXISTS "${_LLVM_PREFIX}/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "${_LLVM_PREFIX}/lib/cmake/llvm" CACHE PATH "")
    else()
        set(LLVM_DIR "/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm"
            CACHE PATH "Path to LLVMConfig.cmake")
    endif()
endif()

find_package(LLVM REQUIRED CONFIG)
if(NOT LLVM_VERSION_MAJOR IN_LIST LLVM_SUPPORTED_VERSIONS)
    message(FATAL_ERROR "LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR} is unsupported; "
        "the bridge builds against LLVM ${LLVM_SUPPORTED_VERSIONS}.")
endif()
set(LLVM_VERSION "${LLVM_VERSION_MAJOR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM library dir:  ${LLVM_LIBRARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Debian builds LLVM with RTTI, upstream/spack default to without.  Subclassing an
# MLIR class from an -frtti translation unit against an -fno-rtti LLVM leaves the
# base's typeinfo undefined at import time, so the MLIR-facing libraries follow
# whatever the prefix was built with.  The nanobind module cannot: nanobind's own
# headers use typeid.  Hence installPassTracer lives in the passes library.
if(LLVM_ENABLE_RTTI)
    set(BRIDGE_MLIR_RTTI_FLAGS "")
else()
    set(BRIDGE_MLIR_RTTI_FLAGS -fno-rtti)
endif()

# --- MLIR --------------------------------------------------------------------
# A monolithic libMLIR tells the two prefix shapes apart.  Debian ships one and
# splits the cmake config across packages so that including it is a hard error,
# not a QUIET miss -- its export set names tool binaries the packages omit; MLIR
# is located through the LLVM prefix there.  A source or spack install has no
# dylib but a working MLIRConfig.cmake, and the static component closure has to
# come from that package: its imported targets pre-empt the manual ones below.
# Search the prefix ONLY: a dylib from the system path can belong to a different
# point release than the prefix's static libs, which links and then misbehaves.
find_library(MLIR_MONO_LIB MLIR PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH)
if(NOT MLIR_MONO_LIB)
    find_package(MLIR CONFIG QUIET HINTS "${LLVM_LIBRARY_DIR}/cmake/mlir")
endif()
if(NOT MLIR_FOUND)
    set(MLIR_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" CACHE PATH "MLIR include dirs")
endif()
message(STATUS "MLIR cmake package: ${MLIR_FOUND}")

# Helper: find one MLIR (or Flang) library and create an imported target.
# Usage: find_mlir_lib(MLIRIR) -> creates target MLIRIR if libMLIRIR.a exists.
macro(find_mlir_lib name)
    if(NOT TARGET ${name})
        find_library(_${name}_LIB ${name} HINTS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(_${name}_LIB)
            add_library(${name} STATIC IMPORTED)
            set_target_properties(${name} PROPERTIES IMPORTED_LOCATION "${_${name}_LIB}")
        else()
            message(WARNING "Cannot find lib${name} in ${LLVM_LIBRARY_DIR}")
        endif()
        unset(_${name}_LIB CACHE)
    endif()
endmacro()

# The MLIR libraries we actually use.
set(_MLIR_LIBS
    MLIRIR MLIRParser MLIRSupport
    MLIRPass MLIRTransforms
    MLIRFuncDialect MLIRArithDialect MLIRSCFDialect
    MLIRControlFlowToSCF
    MLIROptLib
)
foreach(_lib ${_MLIR_LIBS})
    find_mlir_lib(${_lib})
endforeach()

# Monolithic shared libs (Debian ships these; handles transitive deps).
# Debian names it libLLVM-<major>.so and drops it on the default link path; spack
# and upstream keep it in the LLVM prefix, sometimes as a plain libLLVM.so -- so
# resolve a full path rather than emitting a bare -lLLVM-<major>.
# Search the prefix ONLY, for the reason given at the MLIR_MONO_LIB probe above.
find_library(LLVM_MONO_LIB NAMES LLVM-${LLVM_VERSION_MAJOR} LLVM
    PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
# What the bridge finally links for MLIR/LLVM.  Two shapes:
#   * dylib prefix (Debian): the two monolithic libs resolve everything;
#   * static-only prefix (source / spack builds): no dylib exists, so the whole
#     archive set stands in for it.  The flang dialect libs are pulled with
#     --whole-archive ahead of these, and they reference far more of MLIR than
#     the components this bridge names itself, so nothing narrower resolves.
if(MLIR_MONO_LIB)
    set(MLIR_LINK_LIBS "${MLIR_MONO_LIB}")
else()
    file(GLOB MLIR_LINK_LIBS "${LLVM_LIBRARY_DIR}/libMLIR*.a")
endif()
if(LLVM_MONO_LIB)
    set(LLVM_LINK_LIBS "${LLVM_MONO_LIB}")
elseif(MLIR_LINK_LIBS)
    file(GLOB LLVM_LINK_LIBS "${LLVM_LIBRARY_DIR}/libLLVM*.a")
else()
    set(LLVM_LINK_LIBS "LLVM-${LLVM_VERSION_MAJOR}")
endif()
list(LENGTH MLIR_LINK_LIBS _n_mlir_link)
list(LENGTH LLVM_LINK_LIBS _n_llvm_link)
message(STATUS "MLIR link: ${_n_mlir_link} lib(s), LLVM link: ${_n_llvm_link} lib(s)")

# --- Flang dialect libraries --------------------------------------------------
set(FLANG_DIALECT_LIBS
    HLFIRDialect FIRDialect FIRDialectSupport FIRSupport
    FIRBuilder FIRAnalysis CUFDialect CUFAttrs
    # ---- flang transforms / codegen ---------------------------------
    # Needed so the bridge can run ``fir-polymorphic-op``, which
    # statically devirtualises ``fir.dispatch`` / ``fir.select_type``
    # ops whose target is resolvable at compile time.  Surviving
    # dispatches after the conversion are caught loudly by our own
    # ``hlfir-reject-polymorphism`` pass.  The user-visible benefit:
    # CLASS-typed code that doesn't actually USE polymorphism (every
    # call site has a static target) lowers cleanly.
    #
    # The cascade of libs here is dictated by what
    # PolymorphicOpConversion.cpp actually references.  Each one is
    # noted with the symbol that pulls it in:
    #   FIRTransforms       fir::createPolymorphicOpConversion
    #   FIRCodeGenDialect   fir.cg.XDeclareOp (used by select_type
    #                                          -> cf branch lowering)
    #   FIRCodeGen          fir::LLVMTypeConverter::convertBoxTypeAsStruct
    #   FortranSupport      Fortran::common::getFlangFullVersion (codegen
    #                                                              produced-by metadata)
    #   FIROpenACCSupport   fir::acc::registerTransformationalAttrs
    #   FIROpenMPSupport    fir::omp::registerTransformationalAttrs
    #                       (both are defensive dialect registrations
    #                       FIRCodeGen.a calls unconditionally; the
    #                       bridge doesn't use OpenACC or OpenMP, this
    #                       is purely transitive linker-cleanup.)
    FIRTransforms FIRCodeGenDialect FIRCodeGen FortranSupport
    FIROpenACCSupport FIROpenMPSupport
)

# LLVM 22 added the ``mif`` (multi-image FIR / coarray) dialect --
# flang/lib/Optimizer/Dialect/MIF, absent in 21 -- and FIRTransforms.a now
# references its op TypeIDs, so the module fails to dlopen without it.
if(LLVM_VERSION_MAJOR GREATER_EQUAL 22)
    list(APPEND FLANG_DIALECT_LIBS MIFDialect)
endif()

foreach(_lib ${FLANG_DIALECT_LIBS})
    find_mlir_lib(${_lib})
endforeach()

# --- Flang include dir --------------------------------------------------------
if(NOT FLANG_INCLUDE_DIR)
    set(FLANG_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}")
endif()

# --- Python + nanobind --------------------------------------------------------
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_STATUS)
if(NOT NB_STATUS EQUAL 0)
    message(FATAL_ERROR "nanobind not found.  Install with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Common include paths -----------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${FLANG_INCLUDE_DIR}
)
add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_compile_definitions(${LLVM_DEFINITIONS_LIST})

# --- Core: trace + extraction -------------------------------------------------
add_library(hlfir_bridge_core STATIC
    bridge/trace_utils.cpp
    bridge/extract_vars.cpp
    bridge/extract_ast.cpp
    bridge/ast/expressions.cpp
    bridge/ast/assigns.cpp
    bridge/ast/elementals.cpp
    bridge/ast/control_flow.cpp
    bridge/ast/dispatch.cpp
)
target_link_libraries(hlfir_bridge_core PUBLIC
    MLIRIR MLIRParser MLIRSupport MLIRFuncDialect MLIRArithDialect
)
target_compile_options(hlfir_bridge_core PRIVATE ${BRIDGE_MLIR_RTTI_FLAGS})

# --- Passes library -----------------------------------------------------------
add_library(hlfir_bridge_passes STATIC
    passes/Passes.cpp
    passes/PropagateShapes.cpp
    passes/VersionShapeScalars.cpp
    passes/InlineAll.cpp
    passes/LowerFirSelectCase.cpp
    passes/FlattenStructs.cpp
    passes/FlattenGlobalScalarReads.cpp
    passes/EliminateDoubleBufferToggle.cpp
    passes/DefaultIntent.cpp
    passes/VerifyNoUnresolvedCalls.cpp
    passes/FoldElementAliases.cpp
    passes/DropStubCalls.cpp
    passes/FoldCopyInOut.cpp
    passes/ExpandVectorSubscriptGather.cpp
    passes/ExpandVectorSubscriptScatter.cpp
    passes/RejectPolymorphism.cpp
    passes/RewritePointerAssigns.cpp
    passes/RewriteSequenceAssociation.cpp
    passes/LiftReductionOperands.cpp
    passes/LiftAllocArrayOfRecords.cpp
    passes/LiftAosPointerRecords.cpp
    passes/PruneUnreachable.cpp
    passes/MarshalExternalStructs.cpp
    passes/UnwrapEvalInMem.cpp
    passes/StripErrorHelpers.cpp
    passes/StripRuntimeIo.cpp
    passes/StripCharacterRuntime.cpp
    passes/PreserveMutableGlobals.cpp
    passes/MarkBoundsRemapViews.cpp
    passes/FoldAssumedRankQueries.cpp
    passes/PruneNeverAllocatedMemberDeref.cpp
)
target_link_libraries(hlfir_bridge_passes PUBLIC
    hlfir_bridge_core
    MLIRPass MLIRTransforms
    MLIRControlFlowToSCF
)
target_compile_options(hlfir_bridge_passes PRIVATE ${BRIDGE_MLIR_RTTI_FLAGS})

# --- Python extension ---------------------------------------------------------
nanobind_add_module(hlfir_bridge NB_STATIC bridge/bridge.cpp)
target_link_libraries(hlfir_bridge PRIVATE
    hlfir_bridge_core
    hlfir_bridge_passes
    -Wl,--whole-archive
    ${FLANG_DIALECT_LIBS}
    -Wl,--no-whole-archive
    -Wl,--start-group
    ${MLIR_LINK_LIBS}
    ${LLVM_LINK_LIBS}
    -Wl,--end-group
)
set_target_properties(hlfir_bridge PROPERTIES
    INSTALL_RPATH "${LLVM_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE)

# --- Install ------------------------------------------------------------------
install(TARGETS hlfir_bridge      LIBRARY DESTINATION .)
