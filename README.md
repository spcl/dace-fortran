# HLFIR -> DaCe Fortran frontend

Turn a Fortran kernel into an optimisable [DaCe](https://github.com/spcl/dace)
SDFG and get back a Fortran-callable shared library that preserves the
caller's original interface.

```
 kernel.f90
     |  (0) preprocess + flang -fc1 -emit-hlfir
     v
 kernel.hlfir            MLIR (HLFIR dialect)  --  flang did the parsing,
     |  (1-3) C++ bridge   name binding, type inference, intrinsic lowering,
     v                     inlining, normalisation
 normalised single-TU IR
     |  (4) walk -> DaCe
     v
 dace.SDFG  --(you optimise it)-->  any DaCe transformation
     |  (5) emit binding
     v
 <entry>_bindings.f90  +  lib<entry>.so   <- the caller links here
```

Flang owns the front-end (parsing, name binding, type inference, intrinsic
lowering). This frontend consumes Flang's already-elaborated **HLFIR** (an
MLIR dialect), normalises it into one narrow IR shape, walks it into an
SDFG, and regenerates a Fortran wrapper around the optimised result.

## Quick start

```python
import dace_fortran
from dace_fortran.bindings import build_fortran_library

src = open("kernel.f90").read()

# Build the SDFG.  ``entry`` is the mangled flang symbol --
# ``_QPname`` for a free subroutine, ``_QM<mod>Pname`` for a module
# procedure.  Omit it only when the source has exactly one procedure.
sdfg = dace_fortran.build_sdfg(src, entry="_QPkernel", name="kernel")

# ... optimise the SDFG here ...

# Emit + link a Fortran-callable .so.  Caller interface and flatten
# plan are auto-derived from the SDFG.
lib = build_fortran_library(sdfg, out_dir="build", name="kernel")
```

For a multi-file kernel with `USE` imports across files, pass
`search_dirs=[...]` to `build_sdfg` so `merge_used_modules` inlines the
dependencies. For a CMake / Autotools project, the bridge reads
`compile_commands.json` directly -- see [Build-system integration](#build-system-integration).

## Preprocess passes

Source-text rewrites that run before flang. All are SED-style regex
transforms with shared comment/string awareness (`_scan_line`), not a
Fortran parser. Each is narrow and idempotent: a second invocation finds
nothing to rewrite and returns the input verbatim.

| Pass | Default | What it does | When you need it |
|---|---|---|---|
| `merge_used_modules` | on | Inlines every externally-`USE`-d module's source so flang sees one self-contained TU. | Multi-file projects with module dependencies. |
| `strip_openmp_directives` | on | Drops `!$OMP` / `!$ACC` / `!$` sentinels, the ICON `#include "omp_definitions.inc"`, and `#ifdef _OPENMP` / `_OPENACC` blocks. | Any accelerator-annotated legacy code. |
| `normalize_kind_parameters` | on | Substitutes precision aliases (`wp`, `sp`, `dp`, `qp`) with literal kind integers when the alias isn't locally bound. | Sources that `USE` a constants module the bridge can't compile alongside. |
| `rewrite_integer_powers` | on | Expands integer-valued REAL-literal powers (`x**2.0` -> `(x*x)`). | Removes a backend-dependent `pow(x, 2.0)` vs `x*x` rounding difference against the gfortran reference. |
| `replace_external_with_modules` | opt-in | Resolves `EXTERNAL :: name` to `USE mod, ONLY: name` against modules in `search_dirs`. | Legacy code that declares procedures `EXTERNAL` even when the defining module is available. |
| `rewrite_string_enum_to_integer` | opt-in | Converts `CHARACTER` enum-style dummies into `INTEGER`. Returns a `{procedure: {arg: {literal: int}}}` map for binding generation. | Kernels using strings as enum switches (QE's `flag == 'c' .OR. flag == 'C'`). |
| `preprocess_fortran` (IF-intvar) | opt-in | Rewrites `IF (intvar)` -> `IF (intvar /= 0)` for INTEGER scalars. | Legacy code that uses integer guards (flang-new-21 only accepts LOGICAL). |

Each pass is importable standalone from `dace_fortran.preprocess`. For build-system integration via a CLI, see [Build-system integration](#build-system-integration).

## QE end-to-end example -- `ast_v1_vexx_bp_k_gpu.f90`

A worked-example for a real Quantum Espresso kernel that exercises every
preprocess pass and the bridge's full HLFIR pipeline. The fixture
(`tests/qe/exx_bp/ast_v1_vexx_bp_k_gpu.f90`) is a flat single-TU
checkpoint emitted by `f2dace-qe-source`'s pruner for the
`exx_bp::vexx_bp_k_gpu` entry: ~2k lines, every USE-closure module
inlined.

```python
import re, pathlib, dace_fortran
from dace_fortran.preprocess import (
    merge_used_modules, normalize_kind_parameters,
    replace_external_with_modules, rewrite_string_enum_to_integer,
)

src = pathlib.Path("tests/qe/exx_bp/ast_v1_vexx_bp_k_gpu.f90").read_text()

# 1. Source-side restore for an upstream-pruner quirk: the pruner emits
#    empty ``INTERFACE invfft / fwfft`` blocks before ``MODULE fft_types``
#    is in scope.  Delete the empty block and re-emit the upstream
#    specifics AFTER fft_types.  See _restore_fft_interfaces in
#    tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py for the verbatim helper.
src = restore_fft_interfaces(src)

# 2. Apply preprocess passes.  ``build_sdfg`` runs every default-on pass
#    automatically; we list them explicitly here for clarity.
#
#    QE uses every pass:
#     * USE-closure already inlined (single TU), merge is a no-op.
#     * No OpenMP sentinels in this kernel; strip is a no-op.
#     * ``REAL(KIND=dp)`` -- normalize_kind substitutes dp -> 8 since the
#       precision module isn't co-compiled.
#     * Several library helpers declared EXTERNAL with the defining
#       module already inlined.
#     * ``addusxx_g`` / ``newdxx_g`` take a CHARACTER(LEN=1) ``flag``
#       compared against ``'c' .OR. 'C'`` etc. -- string-enum pattern.
src, enum_maps = rewrite_string_enum_to_integer(src)
src = normalize_kind_parameters(src)
src = replace_external_with_modules(src, search_dirs=["tests/qe/exx_bp/"])

# 3. Build the SDFG.  ``build_sdfg`` runs the default preprocess pass
#    composition again internally -- safe because all passes are
#    idempotent.  ``entry`` is the QE module-procedure path.
sdfg = dace_fortran.build_sdfg(
    src,
    entry="exx_bp::vexx_bp_k_gpu",
    name="vexx_bp_k_gpu",
)

# 4. Emit + link.  The binding emitter uses the enum_maps to expose the
#    original string surface at the Python boundary; under the hood the
#    SDFG receives the integer value.
from dace_fortran.bindings import build_fortran_library
lib = build_fortran_library(
    sdfg, out_dir="build", name="vexx_bp_k_gpu",
    enum_maps=enum_maps,
)
```

The full SDFG build is currently xfail-strict-false; see
`tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py` for the live progression.
The four QE-driving HLFIR-pipeline fixes that landed in the bridge are
called out in the table below.

## HLFIR pipeline

Run in this order by `DEFAULT_PIPELINE` in `builder/__init__.py`. The
QE end-to-end above touches every entry. Passes added or substantially
extended to support the QE / ICON / CLOUDSC corpus are marked with a
`(*)`.

| Pass | Purpose |
|---|---|
| `hlfir-prune-unreachable` | Erase dispatch-table bindings the entry never dynamically invokes. |
| `symbol-dce` (early) | Drop private functions never reached by the entry. |
| `lower-fir-select-case` | `fir.select_case` -> `cf.cond_br` BEFORE inlining (the inliner's block-operand remap segfaults on a callee containing select-case). |
| `lift-cf-to-scf` (first) | Structurise callees: fold early `RETURN` / in-callee CFG into `scf.if` so `hlfir-inline-all` cannot corrupt a structured region. |
| `hlfir-strip-error-helpers` (*) | Delete `CALL errore` / `CALL finish` / `CALL abor1` etc. -- those abort the program; their `IF (ierr <= 0) RETURN ... STOP 1` shape stays multi-block past structurisation and crashes the inliner. |
| `hlfir-strip-runtime-io` (*) | Delete `_FortranAio*` calls (`WRITE`, `PRINT`, `OPEN`, `CLOSE`, `FLUSH`). The SDFG models numerical dataflow; diagnostic prints are orthogonal. |
| `hlfir-inline-all` | Splice every callee body into the entry. Refuses multi-block callees as a safety net (the strip-error-helpers pass eats the common case). |
| `hlfir-unwrap-eval-in-mem` | `hlfir.eval_in_mem` -> `fir.alloca` + body + plain reads. |
| `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elementals. |
| `hlfir-expand-vector-subscript-{gather,scatter}` | `hlfir.elemental` noncontiguous gather temps + scatter destinations -> explicit `do` loops. |
| `symbol-dce` (late) | Drop private callees once their bodies are inlined. |
| `fir-polymorphic-op` | Statically devirtualise resolvable `fir.dispatch` / `fir.select_type`. |
| `hlfir-reject-polymorphism` | Loud-fail on surviving virtual dispatch -- the bridge supports CLASS-as-monomorphic-box only. |
| `hlfir-rewrite-sequence-association` | Collapse sequence-association adapters into section designates. |
| `hlfir-lift-alloc-array-of-records` | Lift `type(t), allocatable :: f(:)` into top-level companions with a leading runtime-extent dim. |
| `hlfir-lift-aos-pointer-records` (*) | Materialise concat companions for ICON's `TYPE(t_qx_ptr) :: q(N)` AoS-of-pointer-records (Graupel pattern). |
| `hlfir-split-aor-dummies` | Split allocatable-array-of-records dummies into per-member descriptors. |
| `hlfir-marshal-external-structs` | Expand registered-external `Arg(kind="aos")` calls into per-member arguments. |
| `hlfir-flatten-structs` | AoS -> SoA. Emits the `hlfir.flatten_plan` attribute. |
| `hlfir-mark-bounds-remap-views` (*) | Tag Fortran 2003 `ptr(1:N*K) => target(:, slice)` bounds-remapping pointer assigns so descriptors.py emits a DaCe View instead of rejecting. |
| `hlfir-rewrite-pointer-assigns` | Collapse plain `ptr => target` rebinds under strict-no-alias. Skips marks from the prior pass. |
| `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols. |
| `hlfir-lift-reduction-operands` (*) | Lift inline reductions (`max(x, MAXVAL(slice))`) into a preceding scalar-temp assign. Skips dimensional reductions (`SUM(arr, DIM=k)` -> `!hlfir.expr<NxT>`). |
| `hlfir-default-intent` | Intent-less dummies default to `intent_inout`. |
| `lift-cf-to-scf` (late) | Raw-CFG loops (`DO WHILE`, `DO...EXIT`) -> `scf.while` + `scf.if`. |
| `sccp,canonicalize,cse` | Fold + simplify + dedupe. |

The pipeline runs on a dedicated 2 GB-stack worker thread; MLIRContext
multithreading is disabled for the duration so nested
`OperationPass<FuncOp>` runs serially on the same big-stack worker.

## Build-system integration

Three integration paths, in increasing rigor:

**No build-system glue.** Run the preprocess CLI once in-place over your
source tree; your existing compiler builds the result with zero changes
to your build files:

```bash
python -m dace_fortran.preprocess_cli \
    --all-defaults --rewrite-external --rewrite-string-enum \
    --search-dir src/utils \
    --inplace \
    --in src/kernel.f90 --in src/helper.f90
```

`--inplace` rewrites each file via atomic tempfile + rename. Optional
`--backup-suffix .orig` keeps a diff-able backup. No-op when no pass
touches the source (mtime preserved).

**CMake.** Two lines:

```cmake
include(DaceFortran)
dace_fortran_preprocess(
    TARGET mylib
    SOURCES src/kernel.f90 src/helper.f90
    SEARCH_DIRS src/utils
    PASSES all_defaults rewrite_external rewrite_string_enum)
add_library(mylib ${mylib_PREPROCESSED_SOURCES})
```

**Autotools.** One `DACE_FORTRAN_PREPROCESS` in `configure.ac`, one
`include $(top_srcdir)/dace_fortran.mk` in `Makefile.am`:

```makefile
include $(top_srcdir)/dace_fortran.mk

DACE_FORTRAN_PASSES      = all_defaults rewrite_external
DACE_FORTRAN_SEARCH_DIRS = $(srcdir)/utils
mylib_a_SOURCES          = $(call dace_fortran_preprocess, kernel.f90 util.f90)
```

See `cmake/DaceFortran.cmake` and `autotools/dace_fortran.m4` for full
docs.

## Supported / not supported

[OK] supported, [!] planned, [X] never (out of scope).

### Types

| Feature | Status | Notes |
|---|---|---|
| `INTEGER(1/2/4/8)` | [OK] | |
| `REAL(4/8)` | [OK] | |
| `LOGICAL` (any kind) | [OK] | binding bridges the caller's kind width |
| `COMPLEX(4/8)` | [OK] | arrays only; scalar by-value is a DaCe-core gap |
| `CHARACTER` | [!] | as enum switch only (via `rewrite_string_enum_to_integer`); string content unsupported |
| Derived type, flat / nested | [OK] | path-flattened name `base_m1_m2_leaf` |
| Array-of-struct with array members | [OK] | `A(i)%w(j,k)` -> `A_w(i,j,k)` |
| Array-of-struct with allocatable members | [OK] | flat allocatable companion + per-allocate-site rename |
| AoS-of-pointer-records (ICON Graupel) | [OK] | `hlfir-lift-aos-pointer-records` |
| Polymorphic / `SELECT TYPE` / `CLASS(*)` | [X] | requires runtime type discrimination |
| Circular type definitions | [X] | |

### Control flow

| Feature | Status | Notes |
|---|---|---|
| `DO` / `DO WHILE` / `DO CONCURRENT` | [OK] | LoopRegion + scf.while |
| `IF` / `ELSE IF` / `ELSE` | [OK] | scf.if |
| `SELECT CASE` (incl. on rewritten string-enum) | [OK] | `lower-fir-select-case` |
| `EXIT`, `CYCLE` | [OK] | |
| Symbolic loop step / bounds | [OK] | hoisted to `loopbegin_<N>` / `loopend_<N>` / `loopstep_<N>` symbols |
| `GOTO` (unstructured) | [X] | doesn't lift to scf |
| `SELECT TYPE` | [X] | |

### Allocatable / Pointer

| Feature | Status | Notes |
|---|---|---|
| `ALLOCATE` / `DEALLOCATE` (local + dummy) | [OK] | buffer reaching-definitions model |
| Conditional `IF/ELSE` allocate (branch extent) | [OK] | merged class, `<buf>_d<i>` PHI symbol |
| Sequential realloc chain | [OK] | distinct transients per epoch |
| Plain `ptr => target` rebind | [OK] | collapsed under strict-no-alias (warns) |
| `ptr(1:N*K) => target(:, slice)` bounds-remap | [OK] | emitted as DaCe View aliasing the parent |
| `MOVE_ALLOC`, allocatable assignment | [X] | |

### Subprograms / linkage

| Feature | Status | Notes |
|---|---|---|
| Module-contained `SUBROUTINE` / `FUNCTION` | [OK] | inlined by `hlfir-inline-all` |
| Internal subprograms (`CONTAINS`) | [OK] | |
| `INTERFACE` blocks, `USE`, `USE ... ONLY:` | [OK] | resolved at flang time |
| `OPTIONAL` dummy + `PRESENT` | [OK] | folded statically post-inline |
| `EXTERNAL` declarations resolvable to modules | [OK] | `replace_external_with_modules` |
| Separately-compiled `bind(c)` external | [OK] | `register_external` / `keep_external` |
| MPI send/recv (incl. non-default communicator) | [OK] | `dace.libraries.mpi`, recognised automatically |
| Error helpers (`errore`, `finish`, `abor1`, ...) | [OK] | stripped by `hlfir-strip-error-helpers` |
| I/O (`WRITE`, `PRINT`, `OPEN`, `CLOSE`) | [OK] | stripped by `hlfir-strip-runtime-io` |

### Slicing / array ops

| Feature | Status | Notes |
|---|---|---|
| Contiguous slice, whole-array assign, elementwise intrinsics | [OK] | |
| Reductions (sum/product/min/max/all/any/count/minval/maxval, scalar) | [OK] | |
| Reductions, dimensional (`SUM(arr, DIM=k)`) | [!] | bridge pipeline now safe; AST emit gap remains |
| BLAS/LAPACK (matmul, transpose) | [OK] | dense -> libnode |
| Noncontiguous gather (rank-1+), scatter | [OK] | gather/scatter expand passes |
| Noncontiguous slice, symbolic extent | [X] | DaCe can't express runtime-sized symbol arrays |

### Codegen targets

| Feature | Status |
|---|---|
| CPU C++ tasklets | [OK] |
| GPU CUDA / Native `!$OMP` / COARRAY | [X] |

## Testing

Every supported construct has a seeded numerical test against
gfortran / f2py. Binding-specific tests live in `tests/bindings/`; QE
loopnest kernels in `tests/qe/selected_loopnests/`; ICON velocity-advection
loopnests in `tests/icon/selected_loopnests/`; build-system integration
tests in `tests/buildsys_integration/`.

```bash
# Main sweep -- excludes the multi-rank mpi-marked tests.
python3 -m pytest -n 4 -m "not mpi" tests/

# Multi-rank MPI tests; --oversubscribe lets the 4-rank tests run on
# laptops with fewer than 4 cores.
mpirun --oversubscribe -n 4 python3 -m pytest -m mpi -p no:cacheprovider tests/

# Dump built SDFGs for inspection.
__DACE_HLFIR_GEN_TEST_SDFGS=1        python3 -m pytest tests/
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/
```

All executable-Fortran tests compile with `gfortran` (Ubuntu's
`flang-new-21` ships without `libflang_rt`, so flang is emit-HLFIR-only).

## Non-goals

- Re-parsing Fortran in Python. Flang is authoritative.
- GPU target bindings (would need OpenACC shim emission).
- Cross-kernel fusion across TU boundaries. Inline-all handles intra-TU
  fusion; cross-TU is the binding emitter's problem.
- `CHARACTER` string content (only enum-as-integer is supported).
