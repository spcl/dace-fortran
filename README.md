# dace-fortran — a Fortran (HLFIR) frontend for DaCe

Lowers Fortran HPC kernels to optimisable [DaCe](https://github.com/spcl/dace) SDFGs; hands back a **Fortran-callable shared library** preserving the caller's original interface (SDFG drops in for the kernel it replaces).

- Flang (`flang-new`) owns the front-end (parse / name-bind / type-infer / intrinsic-lower).
- Consumes Flang's elaborated **HLFIR** (MLIR dialect) → MLIR pass pipeline normalises to one narrow IR shape → walks into `dace.SDFG` → regenerates `bind(c)` Fortran wrapper.
- Targets: ICON (ocean + atmosphere dycore, graupel, velocity advection), Quantum ESPRESSO (`exx`), CLOUDSC, NPB, FV3, LULESH, + a large construct-level test corpus.

```
 kernel.f90
     │  (0) preprocess (source-text rewrites) + flang -fc1 -emit-hlfir
     ▼
 kernel.hlfir            MLIR (HLFIR dialect) — flang did the parsing,
     │  (1) C++/MLIR bridge   name binding, type inference, intrinsic lowering
     ▼
 normalised single-TU IR (struct-flattened, inlined, shape-propagated, …)
     │  (2) walk → DaCe
     ▼
 dace.SDFG  ──(you optimise it with any DaCe transformation)──▶
     │  (3) emit binding
     ▼
 <entry>_bindings.f90  +  lib<entry>.so   ◀── the caller links here
```

## Key features

- **Flang-authoritative**: no Fortran re-parsing in Python; bridge consumes flang-elaborated HLFIR.
- **MLIR pass pipeline**: struct flatten (AoS→SoA), whole-kernel inline, pointer-assign rewrite, static devirtualisation + loud-reject surviving polymorphism, shape propagation, reduction lifting, non-numerical-noise deletion (error helpers / runtime I/O / character runtime).
- **AoS / nested derived types**: path-flattened to per-member arrays; array-of-struct (array + allocatable members), ICON's array-of-pointer-records (Graupel) pattern.
- **Fortran binding generation**: emits `<entry>_bindings.f90`, preserves caller interface, zero-copy alias where layouts agree, copy-in/copy-out do-loops otherwise. Optional flat-C-ABI shim (`bind(c)`) for a stable C entry point.
- **External-call policy**: a kernel `CALL` to a separately-compiled `bind(c)` procedure (e.g. ICON's MPI halo exchange `sync_patch_array`) stays external → DaCe library node, or is dropped — one declaration drives both the inliner and the bridge.
- **Build-system integration**: standalone preprocess CLI + CMake module + Autotools macros run the source-text rewrites in place — existing build emits flang-consumable Fortran with no other changes.

## Architecture / pipeline

### Source-text preprocess passes (before flang)

Run before flang, on raw source. SED-style regex transforms with shared comment/string awareness (not a Fortran parser); narrow + idempotent. Importable standalone from `dace_fortran.preprocess`:

| Pass | Default | What it does |
|---|---|---|
| `merge_used_modules` | on | Inlines `USE`-d module sources so flang sees one self-contained TU. Regex text-splice by default; an **fparser** AST engine is available via `merge_engine="fparser"`. |
| `strip_openmp_directives` | on | Drops `!$OMP` / `!$ACC` / `!$` sentinels and `#ifdef _OPENMP` / `_OPENACC` blocks. |
| `normalize_kind_parameters` | on | Substitutes precision aliases (`wp`, `sp`, `dp`, `qp`) with literal kind integers when the alias isn't locally bound. |
| `rewrite_integer_powers` | on | Expands integer-valued REAL-literal powers (`x**2.0` → `x*x`). |
| `replace_external_with_modules` | opt-in | Resolves `EXTERNAL :: name` to `USE mod, ONLY: name` against `search_dirs`. |
| `rewrite_string_enum_to_integer` | opt-in | Converts `CHARACTER` enum-style dummies into `INTEGER`, returning a map for binding generation. |
| `preprocess_fortran` (IF-intvar) | opt-in | Rewrites `IF (intvar)` → `IF (intvar /= 0)` for INTEGER scalars (flang-21 accepts only LOGICAL). |

`dace_fortran.fparser_inliner` — alternative single-TU inliner on an **fparser AST**: parses whole project, resolves `USE`/`ONLY:`/`=>` renames, prunes to the entry's reachability, consolidates surviving `USE` clauses, desugars (ASSOCIATE/GOTO/statement-functions, removes interfaces), restores cross-module `USE` so the output is legal single-file Fortran, re-emits one `.f90`. Requires `fparser > 0.2`.

### HLFIR pass pipeline (inside the bridge)

Order = `DEFAULT_PIPELINE` in `dace_fortran/builder/__init__.py` (`MULTI_FILE_PIPELINE` = multi-file variant). Runs on a dedicated 2 GB-stack worker thread, MLIRContext multithreading disabled.

| # | Pass | Purpose |
|---|---|---|
| 1 | `hlfir-prune-unreachable` | Erase dispatch-table bindings the entry never dynamically invokes. |
| 2 | `symbol-dce` (early) | Drop private functions the entry never reaches. |
| 3 | `lower-fir-select-case` | `fir.select_case` → `cf.cond_br` before inlining (the inliner segfaults on select-case callees). |
| 4 | `lift-cf-to-scf` (first) | Structurise callees (fold early `RETURN` / CFG into `scf.if`) so inlining can't corrupt a structured region. |
| 5 | `hlfir-strip-error-helpers` | Delete `CALL errore` / `finish` / `abor1` etc. — their `STOP`-terminated shape stays multi-block and crashes the inliner. |
| 6 | `hlfir-strip-runtime-io` | Delete diagnostic `_FortranAio*` calls (`WRITE`/`PRINT`/…); file-bound chains are preserved as `dace.libraries.fortran_io` nodes. |
| 7 | `hlfir-strip-character-runtime` | Delete `_FortranACharacter*` calls (compare/Trim/Adjust) — the bridge models no character data. |
| 8 | `hlfir-inline-all` | Splice every callee body into the entry; refuses multi-block callees as a safety net. |
| 9 | `hlfir-unwrap-eval-in-mem` | `hlfir.eval_in_mem` → `fir.alloca` + body + plain reads. |
| 10 | `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elementals. |
| 11–12 | `hlfir-expand-vector-subscript-{gather,scatter}` | Noncontiguous gather temps / scatter destinations → explicit `do` loops. |
| 13 | `symbol-dce` (late) | Drop private callees once inlined. |
| 14 | `fir-polymorphic-op` | Statically devirtualise resolvable `fir.dispatch` / `fir.select_type`. |
| 15 | `hlfir-reject-polymorphism` | Loud-fail on surviving virtual dispatch (CLASS-as-monomorphic-box only). |
| 16 | `hlfir-rewrite-sequence-association` | Collapse sequence-association adapters into section designates. |
| 17 | `hlfir-fold-copy-in-out` | Fold flang's copy-in/copy-out temporaries. |
| 18 | `hlfir-lift-alloc-array-of-records` | Lift `type(t), allocatable :: f(:)` into top-level companions. |
| 19 | `hlfir-lift-aos-pointer-records` | Materialise concat companions for ICON's AoS-of-pointer-records (Graupel). |
| 20 | `hlfir-split-aor-dummies` | Split allocatable-array-of-records dummies into per-member descriptors. |
| 21 | `hlfir-marshal-external-structs` | Expand registered-external `aos` calls into per-member arguments. |
| 22 | `hlfir-flatten-structs` | AoS → SoA; emits the `hlfir.flatten_plan` attribute. |
| 23 | `hlfir-mark-bounds-remap-views` | Tag F2003 bounds-remapping pointer assigns so a DaCe View is emitted. |
| 24 | `hlfir-rewrite-pointer-assigns` | Collapse plain `ptr => target` rebinds under strict-no-alias. |
| 25 | `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols. |
| 26 | `hlfir-version-shape-scalars` | SSA-version a straight-line reassigned scalar used as an array extent. |
| 27 | `hlfir-lift-reduction-operands` | Lift inline reductions (`max(x, MAXVAL(slice))`) into a preceding scalar temp. |
| 28 | `hlfir-default-intent` | Intent-less dummies default to `intent_inout`. |
| 29 | `lift-cf-to-scf` (late) | Raw-CFG loops (`DO WHILE`, `DO…EXIT`) → `scf.while` + `scf.if`. |
| 30 | `hlfir-preserve-mutable-globals` | Clear init bodies of caller-mutable BSS globals so `sccp` can't fold their loads. |
| 31 | `hlfir-fold-assumed-rank-queries` | Fold `fir.box_rank` / `fir.is_assumed_size` when the box's rank/shape is statically known. |
| 32 | `sccp,canonicalize,cse` | Fold + simplify + dedupe. |

### Bridge (HLFIR → SDFG)

- `dace_fortran/bridge/` — nanobind Python extension (`hlfir_bridge`). `bridge.cpp` owns an `MLIRContext`+`ModuleOp`, delegates to `trace_utils.cpp` (declaration tracing), `extract_vars.cpp` (variable/descriptor extraction), `extract_ast.cpp` + `bridge/ast/` (`expressions`, `assigns`, `elementals`, `control_flow`, `dispatch`) for the IR walk.
- Passes live under `dace_fortran/passes/`, link into the `hlfir_bridge_passes` static library.
- Python side: `dace_fortran/hlfir_to_sdfg.py` (`SDFGBuilder`) + `dace_fortran/builder/` construct the SDFG; `dace_fortran/intrinsics/` lowers Fortran intrinsics (elementwise, reductions, BLAS/LAPACK).

### Binding generation (SDFG → Fortran-callable .so)

`dace_fortran/bindings/` runs after the SDFG is built. Inputs: a `FrozenSignature` (SDFG arglist snapshot, drift-checked at codegen), an `OriginalInterface` (caller-facing Fortran surface), a `FlattenPlan` (AoS→SoA record from `hlfir-flatten-structs`). Emits `<entry>_bindings.f90`, zero-copy alias where layouts agree, copy-in/copy-out loops otherwise, optionally a `bind(c)` flat-C-ABI shim, then compiles + links a `.so`.

## Key design decisions

Mechanisms turning idiomatic Fortran into a flat, monomorphic SDFG:

### 1. Devirtualisation / monomorphisation (removing CLASS dispatch)

DaCe SDFGs are monomorphic — no runtime type dispatch. Two layers guarantee every call site is statically resolved before SDFG construction.

**Source-level** — `dace_fortran/inliner/ast_desugaring/monomorphize_rewrite.py` (+ analyzer `monomorphize.py`):
- A polymorphic `CLASS(base)` slot (local var, or a container-type component e.g. `t_ocean_solve%act`) expands into `INTEGER :: <var>__tag` discriminator + one concrete companion per arm `TYPE(arm) [, ALLOCATABLE] :: <var>__<arm>`.
- `ALLOCATE(concrete :: v)` → `v__tag = <k>` (+ `allocate(v__<arm>)`).
- Virtual dispatch `CALL v%binding(args)` → static emit-all-always ladder: `IF (v__tag==k) THEN; CALL <arm-proc>(v__<arm>, args); ELSE IF ...`.
- Each arm calls a concrete proc on a concrete `TYPE` → only direct `fir.call`s; `<var>__tag` reads become free SDFG symbols (e.g. `this_act__tag`).
- 4 composable primitives over the per-TU `MonomorphizationSpec`: local-dispatch ladder, component-dispatch ladder, shared-interposer cloning (specialise an inherited `solve`/`construct` per arm), `RETYPE` (axis pinned to one concrete type at its construction site → rewrite `CLASS(base)`→`TYPE(concrete)` on declarations, no tag needed).
- `stack_slots=True`: arms become plain stack objects, not allocatable — the bridge cannot lower an allocatable derived-type scalar.

**Bridge-level guard** — `dace_fortran/passes/RejectPolymorphism.cpp` (`hlfir-reject-polymorphism`): after flang's own `fir-polymorphic-op` statically devirtualises the resolvable cases, walks for any surviving `fir.dispatch` / `fir.select_type` / `fir.box_tdesc` (type-info read) and loud-fails with a source-located error. Non-polymorphic `CLASS(t)` boxes (member access, no virtual dispatch) are supported, peeled like `fir.box<T>` — only genuine runtime type discrimination is rejected.

### 2. Struct flattening (AoS → SoA)

`dace_fortran/passes/FlattenStructs.cpp` (`hlfir-flatten-structs`) eliminates Fortran derived types from the IR before SDFG construction — DaCe handles flat arrays well, structs awkwardly. Post-SDFG mirror of DaCe-core's `StructToContainerGroups`: recursive walk over record members → one flat per-member companion array, SoA naming, outer-shape concatenation.

Three shapes handled:
- scalar struct w/ flat members: `t%u(M)` → `t_u(M)`
- array-of-struct w/ array members (outer+inner extents concatenate): `type(t), dimension(K) :: A` → `A_u(K,M)`; `A(i)%u(j)` → `A_u(i,j)`
- nested records unfold recursively to flat leaf: `o%inner%x(j)` → `o_inner_x(j)`

Struct *dummy arguments* get the same treatment — `replaceStructArg` inserts one block arg per member/leaf, `_soa`-suffixes the function; inlined alias chains from `hlfir-inline-all` are followed transparently. The pass records a `hlfir.flatten_plan` attribute consumed by the bindings emitter.

The bind(c) wrapper marshals host AoS ⇄ SDFG SoA with copy-in/out gather loops — `_render_aos_copy_in` / `_render_aos_copy_out` / `_aos_loop_pieces` in `dace_fortran/bindings/block_builders.py`. SoA buffer layout: `[element-dims…, member-dims…]` (N-D record array → N leading element-index loops, then the member's own dims). Handles N-dim record arrays and both member kinds: allocatable/pointer (extent = per-element `max` cap, guarded by `allocated`/`associated`, zero-filled where unallocated) and fixed-shape value members (e.g. `t_cartesian_coordinates%x(3)` — literal extents, always present, cap-scan/presence-guard skipped). Copy-out scatters back only when the argument is written.

### 3. Allocation-buffer SSA (the unifying ALLOCATABLE model)

The bridge's model for `ALLOCATABLE` arrays under arbitrary `ALLOCATE`/`DEALLOCATE`/conditional-allocate (consolidated from the former `ALLOC_BUFFER_SSA_DESIGN.md`).

**Semantics modelled:** an `ALLOCATABLE` at routine/`BLOCK` scope has one name bound to ≤1 current buffer; allocation status persists across control flow within scope (`ALLOCATE` in a taken `IF` branch stays allocated after the `IF`); referencing an unallocated allocatable is prohibited. The bridge never *proves* allocation — it models "the current buffer at each point", trusting the program conforms, and may safely over-allocate on a path where Fortran would leave the name unallocated (a conforming program never reads it there).

**Abstraction — buffer reaching-definitions:** each `ALLOCATE` site is a buffer *definition*, each `DEALLOCATE` a *kill*. Two sites belong to the **same DaCe transient** iff their buffers can reach a common use/join as alternatives (both `IF` arms allocate → live to the join); sites never simultaneously reaching are **distinct transients** (sequential re-allocation). Build a merge relation `s ~ t` (both reaching at some join/use), take its union-find equivalence classes — **each class is one DaCe transient**. Reproduces every pattern: `IF/ELSE` both-allocate → one buffer; sequential `alloc;dealloc;alloc` → two buffers; conditional + later realloc → two classes; realloc-chain inside one branch → the right split.

**Per-class shape (the PHI):** a class whose sites share an extent gets a concrete shape; a class whose sites differ (a real conditional) gets a branch-dependent extent symbol `<buf>_d<i>` (each site assigns it on its own path, DaCe binds it from whichever path ran). Classes named in first-definition order: class 0 keeps the base name, later classes get `<name>_alloc1`, `<name>_alloc2`, …; the bridge's alias map routes reads/writes to the current class buffer as it walks the IR, and both `IF` arms set the *same* merged-class buffer so post-join reads need no special handling. Grouping = a structured recursive walk over `scf`/`fir.if` regions (no general iterative dataflow), lives in the bridge's `extract_vars` allocation-site analysis.

Out of scope: `MOVE_ALLOC`/allocatable-assignment auto-realloc; buffer-reuse storage aliasing (a DaCe-core concern).

**Related hazard — shape-scalar versioning** — `dace_fortran/passes/VersionShapeScalars.cpp` (`hlfir-version-shape-scalars`). A local integer scalar used as an array extent (`ALLOCATE(x(m))`) may be reassigned (`m = m + 3`) before another array is sized from it — both extents resolve to the bare name `m` (a mutable SDFG symbol), so a whole-array op over `x`'s shape after the reassignment iterates `m`'s *new* value over a buffer allocated to its old one (OOB / heap corruption). Fix: for a scalar feeding an `fir.allocmem` extent, reassigned *after* that allocation in straight-line code, the pass SSA-versions it (`m`, `m_2`, …) so each array binds the version live at its allocation. Reassignment inside a loop or branch is refused with a clear error rather than silently emitting a mutable shape. Non-hazards left untouched: accumulate-then-allocate-once; loop bounds/subscripts that mint a `fir.shape` but never an `fir.allocmem` extent.

### 4. External-call policy (calls left un-inlined)

Some `CALL`s should not be pulled into the TU — internals unlowerable (MPI, polymorphic dispatch, string scans) or target a separately-compiled `bind(c)` library. One declaration drives both halves: the inliner stubs the named procedure's body (keeps its interface) so its internals never enter the TU; the bridge then either **emits** the surviving `CALL` as an `ExternalCall` library node (`dace_fortran/external.py`) bound to a C-ABI symbol, or **drops** it. Three behaviours: *inline* (default), *don't-inline + emit*, *don't-inline + ignore* (invariant `ignore ⊆ don't-inline`).

Declare once with `apply_external_functions(EXTERNAL, IGNORE)`: `EXTERNAL` = list of `ExternalFunction(name, c_function=…, library=…)`, `IGNORE` = drop list (`finish`, `message`, timers, …). The emitted call's argument plan is **derived from the HLFIR call site** (array → `inout` pointer, scalar/free-symbol → by-value) — you supply only the `extern "C"` symbol and the `.so` that exports it (linked via rpath, resolved at load time). Contract: an emitted target must be `bind(c, name="…")` — Fortran name mangling is compiler-specific and a `.mod` is not C-consumable, so a stable C symbol (native or a thin forwarding shim) is the only portable handle. When the C ABI carries facts HLFIR cannot infer (whole derived-type/AoS struct args, an `MPI_Comm` handle, per-leaf dynamic extents, cross-library module-global forwarding, intent narrowing) — register an authored `ExternalSignature` via `keep_external(name, c_name=…, args=…, libraries=…)` (same registry `apply_external_functions` uses). See `tests/external_call/` and `tests/external_aos_test.py`.

## Prerequisites

- **LLVM/Flang 21** — `flang-new-21` (validated LLVM 21.1.8; default set by `LLVM_VERSION="21"` in `dace_fortran/CMakeLists.txt` + `build_bridge.py`, override via `LLVM_VERSION` env var). Debian/Ubuntu: `llvm-21-dev libmlir-21-dev mlir-21-tools libflang-21-dev flang-21 clang-21` from apt.llvm.org. `libflang-21-dev` provides both the FIR/HLFIR static libs the bridge links and the flang headers it includes.
- **Python** 3.10–3.14 (CI runs 3.12).
- **DaCe** — pinned `dace @ git+https://github.com/spcl/dace.git@FaCe` (FaCe branch carries DaCe-core pieces the frontend needs; see `pyproject.toml`). Plus `fparser > 0.2`, `networkx`, `numpy`.
- **nanobind** — bridge is a nanobind extension (`pip install nanobind`).
- **CMake ≥ 3.18**, C++17 compiler (clang-21 auto-selected if present), **gfortran** (binding tests + numerical references compile with gfortran; Ubuntu's `flang-new-21` ships without `libflang_rt` → flang is emit-HLFIR-only).

Bridge locates LLVM/MLIR by deriving the install prefix from `flang-new-21`, using `find_package(LLVM)` only (avoids the often-broken Debian `find_package(MLIR)` cmake config; finds MLIR headers/libs via LLVM's prefix).

## Install & build

```bash
pip install -e ".[testing]"     # editable install + test deps
```

The C++ bridge compiles on first **use** (first reference of `SDFGBuilder` / `build_sdfg`), not at import — `dace_fortran/build_bridge.py` runs cmake + build and symlinks the resulting `hlfir_bridge*.so` into the package. To build it explicitly, or force a clean rebuild:

```bash
# Auto-detect LLVM, configure, and build:
python -m dace_fortran.build_bridge            # build if stale
python -m dace_fortran.build_bridge --clean    # wipe build dir and rebuild

# Or drive cmake directly:
cd dace_fortran/build
cmake .. -DLLVM_VERSION=21 -DCMAKE_BUILD_TYPE=Release
make -j8
```

Override LLVM discovery with the `LLVM_VERSION` / `LLVM_DIR` env vars if auto-detection misses.

## Quick start

```python
import dace_fortran
from dace_fortran.bindings import build_fortran_library

src = open("kernel.f90").read()

# Build the SDFG.  ``entry`` selects the target procedure; it accepts a
# plain Fortran name (``kernel``), a ``module::proc`` qualifier
# (``mo_x::kernel``), or a mangled Flang symbol (``_QPkernel`` /
# ``_QMmo_xPkernel``).  Omit it only when the source has exactly one
# procedure.
sdfg = dace_fortran.build_sdfg(src, entry="kernel", name="kernel")

# ... optimise the SDFG here with any DaCe transformation ...

# Emit + drift-verify + link a Fortran-callable .so.  The caller-facing
# interface and AoS→SoA flatten plan are auto-derived from the SDFG.
lib = build_fortran_library(sdfg, out_dir="build", name="kernel")
```

Other entry points (all return a built, validated `dace.SDFG`):

```python
# A multi-file project (driver + the modules it USEs, in any order):
sdfg = dace_fortran.build_sdfg_from_files([driver, mod], entry="mo_x::kernel")

# A large / dependency-tangled project: emit .hlfir from your own build,
# then consume compile_commands.json directly (tier 3):
sdfg = dace_fortran.build_sdfg_from_project(
    "build/compile_commands.json", entry="_QMmymodPmysub")

# A kernel that CALLs a separately-compiled bind(c) function — keep it
# external and bind it to a C-ABI symbol / library:
dace_fortran.register_external("foo", dace_fortran.ExternalSignature(
    c_name="foo",
    args=[dace_fortran.Arg("array", "float64")],   # intent defaults to inout
    libraries=["/path/libfoo.so"]))
```

Real-codebase recipes (ICON from source, Quantum ESPRESSO `exx`): `docs/ICON_INTEGRATION.md`, `docs/CODEBASE_HELPERS.md`, the external-call policy above (§4); worked examples under `tests/external_call/`, `tests/icon/full/`, `tests/qe/`.

## Build-system integration

Three integration paths run the source-text preprocess passes in place so your existing compiler builds the result:

```bash
# Standalone CLI — atomic in-place rewrite, no build-file changes:
python -m dace_fortran.preprocess_cli \
    --all-defaults --rewrite-external --rewrite-string-enum \
    --search-dir src/utils --inplace \
    --in src/kernel.f90 --in src/helper.f90
```

```cmake
# CMake (cmake/DaceFortran.cmake):
include(DaceFortran)
dace_fortran_preprocess(
    TARGET mylib SOURCES src/kernel.f90 src/helper.f90
    SEARCH_DIRS src/utils
    PASSES all_defaults rewrite_external rewrite_string_enum)
add_library(mylib ${mylib_PREPROCESSED_SOURCES})
```

Autotools is supported via `autotools/dace_fortran.m4` + an included `dace_fortran.mk`.

## Testing

```bash
# Main sweep — excludes multi-rank MPI and slow ICON-build tests:
python3 -m pytest -n 4 -m "not mpi and not long" tests/

# Multi-rank MPI tests (run under mpirun; --oversubscribe for <4 cores):
mpirun --oversubscribe -n 4 python3 -m pytest -m mpi -p no:cacheprovider tests/

# Dump built SDFGs for inspection:
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/
```

`tests/conftest.py` sets test-env defaults automatically (via `setdefault` — explicit override still wins): `HWLOC_COMPONENTS=-gl` (stop hwloc's GL/X11 probe hanging `MPI_Init` on a desktop X display), `UCX_VFS_ENABLE=n` + `OMPI_MCA_pml=ob1`/`OMPI_MCA_btl=self,vader` (in-node transports, so UCX/PMIx finalize can't abort xdist workers), raises the stack soft-limit to its hard limit for deeply-inlined kernels. Pytest markers: `mpi`, `long`, `sequential`, `xdist_group` (see `pyproject.toml`).

`TMPDIR` controls where scratch `.f90`/`.hlfir`/`.dacecache` build artifacts land. Executable-Fortran tests compile+run with `gfortran`/`f2py` against a seeded numerical reference.

Validated corpora: construct-level suite (types, control flow, allocatable/pointer, slicing/intrinsics, reductions, BLAS/LAPACK, derived types, MPI send/recv); ICON ocean + atmosphere dycore/graupel/velocity-advection + full ICON-from-source integration (`tests/icon/`); Quantum ESPRESSO `exx` (`tests/qe/`); CLOUDSC (`tests/cloudsc/`); NPB LU (`tests/npb/`); FV3; LULESH; build-system integration (`tests/buildsys_integration/`); binding-specific tests (`tests/bindings/`).

## Repository layout

```
dace_fortran/
  build.py                 public build_sdfg* entry points
  hlfir_to_sdfg.py         SDFGBuilder + DEFAULT_PIPELINE re-export
  build_bridge.py          auto-build + import the C++ bridge
  preprocess.py            source-text preprocess passes
  preprocess_cli.py        CLI for the preprocess passes
  fparser_inliner.py       fparser-AST single-TU inliner
  flang_codebase.py        real-codebase flang driver helpers (ICON/IFS/…)
  external.py / external_functions.py  external-call policy + registry
  emit_hlfir.py            tier-3 .hlfir emission helper
  CMakeLists.txt           bridge build (LLVM 21, nanobind)
  bridge/                  C++/MLIR bridge (nanobind ext: hlfir_bridge)
    bridge.cpp             nanobind boundary
    extract_vars.cpp / extract_ast.cpp / trace_utils.cpp
    ast/                   expressions, assigns, elementals, control_flow, dispatch
  passes/                  the MLIR passes (one .cpp per pass + Passes.cpp)
  builder/                 SDFG construction (access, descriptors, emit_*)
  bindings/                Fortran bind(c) binding generator + C-ABI shim
  intrinsics/              Fortran intrinsic lowering (elementwise/reduction/linalg)
  inliner/                 fparser-based module inliner / ast_desugaring
  data/                    distributed-data helpers
cmake/                     DaceFortran.cmake (CMake integration)
autotools/                 dace_fortran.m4 + dace_fortran.mk
scripts/                   ICON build/configure helpers
docs/                      CODEBASE_HELPERS, ICON_INTEGRATION, …
tests/                     test corpora (see Testing)
```

## Future work / roadmap

- **GPU target bindings** — binding generator marshals host (CPU) arrays only; accepting GPU device-pointer inputs (an SDFG compiled for GPU callable with device memory, no host round-trip) is future work. GPU codegen itself is a DaCe capability — the missing piece is device-pointer marshalling at the Fortran/C-ABI boundary.
- **Dimensional reductions in AST emit** — the bridge pipeline safely lifts `SUM(arr, DIM=k)`-style reductions; the AST emit path for the dimensional case still has a gap.
- **`CHARACTER` string content** — only the enum-as-integer pattern (`rewrite_string_enum_to_integer`) is supported; arbitrary character data is not modelled.
- **Polymorphism beyond monomorphic CLASS** — `SELECT TYPE`/`CLASS(*)` with genuine runtime type discrimination is rejected; static devirtualisation handles only the resolvable case.
- **Caller-mutable global classification** — `PreserveMutableGlobals` treats every `fir.zero_bits` (BSS-default) global as caller-mutable; a genuinely zero-initialised mutable global is indistinguishable from an uninitialised input in the IR (see `WORK_PACKAGES.md`).

Tracked (+ per-construct support matrix) in `WORK_PACKAGES.md` and the `docs/` planning notes.

## Non-goals

- Re-parsing Fortran in Python — Flang is authoritative.
- Cross-kernel fusion across translation-unit boundaries (inline-all handles intra-TU fusion; cross-TU is the binding emitter's concern).
- Unstructured `GOTO` (it does not lift to `scf`).

## Transformation examples (Fortran→Fortran)

Before→after for the key source-text transforms (conceptual tables above; every snippet below is exercised by a test in `tests/inliner/`).

**Type-bound call → free call** (`deconstruct_procedure_calls`) — the type loses its `CONTAINS` block (becomes pure data); each `obj%bind(a)` becomes a free call with `obj` threaded as the first argument:

```fortran
type Square ; real::side ; contains ; procedure::area ; end type   ! before
a = s%area(1.0)
! after:  TYPE :: Square ; REAL :: side ; END TYPE   (no CONTAINS)
a = area(s, 1.0)
```

**Local constant propagation** (`exploit_locally_constant_variables`) — constant and pointer values are folded into later uses, but a pointer passed as an **actual argument is left intact** (a POINTER dummy needs a pointer actual, not its target expression):

```fortran
ptr => data ; x = ptr + 1.    →    x = data + 1.        ! folded in an expression
ptr => data ; call f(ptr)      →    call f(ptr)          ! NOT f(data) — pointer kept
```

**Stubbing & the external-call / IO-binding policy** — the callee shell (and its call sites) always survives; only the body is rewritten. Pick the variant by what the procedure *is*:

```fortran
make_noop / do_not_emit  : SUBROUTINE log(x) ; END           ! body emptied; bridge keeps/DROPs the call
make_return_false        : LOGICAL FUNCTION isRestart() ; isRestart = .FALSE. ; END
ExternalFunction("sync_patch_array", c_function=…, library=…) ! body emptied; bridge EMITS a bind(c) call
```

`do_not_emit`: pure side-effects (IO, timers, `finish`). `ExternalFunction`: calls the runtime must still make (MPI halo) — bound to a `bind(c)` symbol, typically via a thin hand-written shim. One generic name stubs the whole specific family (`sync_patch_array` ⇒ `sync_patch_array_3d_dp`, …).

**AoS → SoA flatten** (`hlfir-flatten-structs`) — a derived-type access becomes a plain per-member array indexed by the record index (7 variants: scalar dummy, array-of-records, multi-dim, nested connectivity, pointer/allocatable box-section, local never-allocated, double-buffer):

```fortran
type(t) :: p(:) ; ... = p(i)%w     →     ... = p_w(i)
```

**Monomorphization** (`monomorphize(program, spec)`) — removes Fortran virtual dispatch (`fir.dispatch`, which has no SDFG node) so a CLASS-heavy kernel like the ICON-O solver becomes lowerable. Two strategies:

```fortran
! retype: pin a CLASS to one concrete type at its construction site
CLASS(t_transfer), POINTER :: trans   →   TYPE(t_trivial_transfer), POINTER :: trans

! ladder: a runtime factory over a closed arm-set → static type-tag if-ladder
call s%apply(x)   →   IF (s__tag==1) CALL gmres_apply(s__t_gmres, x)
                      ELSE IF (s__tag==2) CALL cg_apply(s__t_cg, x) ...
```

## License

BSD 3-Clause. Copyright ETH Zurich and the DaCe authors (see `AUTHORS` / `LICENSE`).
