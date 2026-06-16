# ICON-O kernel extraction via an fparser single-TU inliner — phased implementation plan

**Status:** PLAN ONLY. Nothing in this document is implemented. It is the design/sequencing
artifact for (A) porting an fparser-based single-translation-unit (single-TU) module inliner into
`dace-fortran`, (B) using it to extract four numerically-critical ICON-ocean (ICON-O) kernels into
self-contained single-file Fortran, and (C) adding all four as `dace-fortran` unit tests that parse
to a DaCe SDFG (eventual goal: emit GPU; ICON-O does not currently run on GPU).

Author target audience: the implementer who picks this up after the QE-designate /
Graupel-scalar-shadow holds are released (the memory hold-condition for the inliner port).

All paths are absolute. Cited `file:line` is against the working tree as read on 2026-06-15.

---

## 0. Executive summary / orientation

### 0.1 What already exists (do NOT rebuild it)

There is **already a single-TU inliner in the repo**, and it is the thing the windmill port is meant
to supersede, not duplicate:

- `merge_used_modules(source, *, search_dirs=())` —
  `/home/primrose/Work/dace-fortran/dace_fortran/preprocess.py:788`. Its own docstring
  (`preprocess.py:789-807`) calls it *"A minimal, fparser-free port of the f2dace single-TU
  concept."* It is **regex/string based** (no fparser, no MLIR): it extracts top-level
  `MODULE … END MODULE` blocks (`_module_blocks`, `preprocess.py:723-765`), `rglob`s `search_dirs`
  for `*.f90`/`*.F90`/`*.incf` and indexes module-name → verbatim block (first-found wins,
  `preprocess.py:815-825`), resolves the `USE` graph with a post-order DFS (`preprocess.py:833-847`,
  via `_used_modules` `preprocess.py:768-785`), and **prepends each needed module's verbatim block in
  dependency order** (`preprocess.py:851-862`).
- It is **default-on** in the build path: `preprocess_fortran_source(..., merge=True)`
  (`preprocess.py` default arg; invoked from `dace_fortran/build.py:237`) and is **pass-through** for
  self-contained input (so the whole inline-source test suite is unaffected).
- Companion source passes in the same file the fparser inliner will sit beside:
  `strip_openmp_directives` (`preprocess.py:348` — drops `!$ACC`/`!$OMP`/`!$` + the
  `#include "*omp_definitions*.inc"`), `normalize_kind_parameters` (`preprocess.py:502` — `wp`→`8`
  etc.), `rewrite_integer_powers` (`preprocess.py:194`), `replace_external_with_modules`
  (`preprocess.py:964`), `_balance_cpp` (`preprocess.py:692`), and the string-enum lowering further
  down the file.

**Critical limitation of `merge_used_modules` that the fparser port fixes:** it splices **whole
module bodies verbatim** and **punts every `ONLY:`/rename clause and every cross-module name
collision to flang** (`_USE_RE` at `preprocess.py:650` captures only the module *name*; `ONLY:`
contents are never parsed). That is fine for the small synthetic multi-module tests, but for an
ICON-O kernel that `USE`s 15–18 modules transitively reaching ~45–55 modules and ~60k LoC of I/O /
CDI / MPI infrastructure (see §3), verbatim whole-module splice pulls in vast unrelated code
(var_list, GRIB2, netCDF, the communication stack) that flang then has to parse and that the bridge
cannot lower. The fparser inliner's job is **selective, entity-level extraction with collision
handling** so the emitted TU contains only the kernel + the parameters / derived types / helper
procedures it actually needs.

### 0.2 How the fparser inliner relates to the existing flang-level merge (`fb945d684`)

The commit `fb945d684` ("hlfir: merge USE-d modules into one TU as a default-on preprocess pass",
author Y.K. Budanaz) is recorded in `/home/primrose/Work/d-face/.git/logs/HEAD:244`. The HLFIR
frontend was subsequently **stripped out of d-face** (`.git/logs/HEAD:257`, commit c5a186fe "Strip
Fortran HLFIR frontend (lives in spcl/dace-fortran)"). Therefore:

- The "flang-level USE-merge pass" of `fb945d684` **is** `merge_used_modules` — it operates at the
  **Fortran source-text level as a preprocess step before flang**, NOT as an MLIR/HLFIR pass. (The
  fixed C++ bridge pass set contains no merge pass: confirmed against
  `/home/primrose/Work/d-face/dace/frontend/hlfir/build/CMakeFiles/hlfir_bridge_passes.dir/DependInfo.cmake:11-25`
  — DefaultIntent, ExpandVectorSubscript{Gather,Scatter}, FlattenStructs, FoldElementAliases,
  InlineAll, LiftAllocArrayOfRecords, LiftReductionOperands, LowerFirSelectCase, PropagateShapes,
  RejectPolymorphism, RewritePointerAssigns, RewriteSequenceAssociation, VerifyNoUnresolvedCalls.)
- **They live at the same layer** (source preprocess feeding `flang -fc1 -emit-hlfir`). The fparser
  inliner is the **robust successor** to the regex `merge_used_modules`, not a different stage.

**Decision (must be stated in the README, WP4):** the fparser inliner **supersedes**
`merge_used_modules` as the engine, but is introduced **side-by-side** behind a selector so the
regex pass stays the default for the existing self-contained inline-source tests (where both are
no-ops anyway) and the fparser engine is opted into for genuine multi-file ICON-O extraction.
Concretely: keep `merge_used_modules` as-is; add `merge_engine: Literal["regex","fparser"]="regex"`
to `preprocess_fortran_source` and route to the new function when `"fparser"` is selected (or when a
`search_dirs` scan detects an ICON-style `ONLY:`-heavy graph — but prefer an explicit opt-in to
avoid surprising the existing suite). The two must produce byte-identical output on the synthetic
multi-module fixtures (`tests/module_merge_test.py`) — that equality is a WP2 regression gate.

### 0.3 The other build route (relevant to test design, §5)

There is a **separate, heavier route** that does NOT use source-text merge at all:
`emit_hlfir_from_codebase(entry_source, out_path, *, search_dirs, library_stubs, defines, ...)`
(`/home/primrose/Work/dace-fortran/dace_fortran/flang_codebase.py:425`) →
`build_sdfg_from_hlfir(hlfir, entry=)` (`/home/primrose/Work/dace-fortran/dace_fortran/build.py:413`).
This is what the full-ICON parse tests use (`tests/icon/full/test_velocity_from_icon_source.py`,
`test_dycore_from_icon_source.py`). Its docstring states the key property: *"Inlining is intra-TU
only … a procedure USE-d from another TU stays an external symbol reference in the SDFG"*
(`build.py:429-432`). That is exactly why we need single-TU: anything left in a sibling module
becomes an unresolved external. For the four kernels we deliberately keep a *small* set of routines
external (`sync_patch_array`, `rot_vertex_ocean_3d`, `get_index_range`) via `keep_external`
(`dace_fortran/external.py:309`) and inline everything else.

**Two viable test routes, both used below:**
1. **Single-TU + lightweight f2py oracle** (preferred for the extracted kernels): the fparser
   inliner emits one self-contained `.f90`; the test drives `build_sdfg_from_files([the_tu], entry=…)`
   (`build.py:525`) and compares the SDFG numerically against an f2py build of the *same* `.f90`.
2. **Codebase route, parse-only** (fallback / first maturity stage when struct-flattening or a gather
   is not yet lowering): `emit_hlfir_from_codebase(...)` + `keep_external` + `build_sdfg_from_hlfir`,
   asserting the SDFG validates — mirrors the existing ICON-full tests.

### 0.4 The four kernels, ranked by extraction difficulty (cleanest → hardest)

| Rank | Kernel | File:line | LoC | Hazard class |
|---|---|---|---|---|
| 1 (PRIMARY) | `upwind_vflux_ppm_onBlock` | `ocean/tracer_transport/mo_ocean_tracer_transport_vert.f90:213-551` | 339 | Ragged `dolic` columns only; **has a built-in oracle** (`upwind_vflux_ppm_vector:554-902`) |
| 2 | `velocity_diffusion_vertical_implicit_onBlock` | `ocean/dynamics/mo_ocean_velocity_diffusion.f90:1078-1272` | 195 | Clean column Thomas solve; serial vertical recurrence; cpp `#ifdef` twin |
| 3 | `limiter_ocean_zalesak_horizontal_onTriangles` | `ocean/tracer_transport/mo_ocean_limiter.f90:587-1059` | 473 | Unstructured gather (3 neighbors / 3 edges / cellOfEdge); **two halo-sync barriers**; ragged dolic |
| 4 | `nonlinear_coriolis_3d_fast_scalar` | `ocean/math/mo_scalar_product.f90:350-622` | 273 | **Variable-length gather** over `verts%num_edges`; data-dependent `MIN(dolic)`; calls `rot_vertex_ocean_3d` + halo sync |

(LoC counts are the `_onBlock`/`_scalar` body spans the user gave; the surrounding host modules are
much larger — `mo_scalar_product.f90` alone is ~4015 lines with 70+ public procedures.)

### 0.5 Cross-cutting hazard shared by all four — the DSL `.inc` macros

The derived types every kernel touches are declared with **cpp macros** from
`/home/primrose/Work/dace-fortran/tests/icon/full/icon-model/src/include/iconfor_dsl_definitions.inc`.
Examples actually used by the kernels (all read from that file):
- `onCellsBlock` → `REAL(wp), POINTER, DIMENSION(:,:)` (`iconfor_dsl_definitions.inc:124,56`) — the 9
  members of `t_verticalAdvection_ppm_coefficients` (`mo_ocean_types.f90:61-69`).
- `onCells` → `REAL(wp), POINTER, DIMENSION(:,:,:)` (`:116,101,20`), `onEdges_3D_Int` →
  `INTEGER, POINTER, DIMENSION(:,:,:)` (`:39`), `mapEdgesToEdges` →
  `REAL(wp), POINTER, DIMENSION(:,:,:,:)` (`:128,126`), `onGrid_1D` →
  `REAL(wp), POINTER, DIMENSION(:)` (`:135,94`).

`mo_ocean_types.f90:14` does `#include "iconfor_dsl_definitions.inc"`. **These macros must be
expanded before fparser sees the type definitions** (fparser parses standard Fortran, not cpp). Two
options, decided in WP1 §2.3:
- (Preferred) Run cpp/`flang -E` (or the bridge's existing `-cpp -D…` path; the codebase route
  already drives `flang -fc1 -cpp -U_OPENMP -U_OPENACC` with `-I include_dirs` and `-D defines`,
  `flang_codebase.py:474-478`) to expand `.inc` macros **into** the staged source the fparser inliner
  reads, with `-I .../src/include`.
- (Fallback) A tiny macro-substitution pre-pass in the inliner that loads
  `iconfor_dsl_definitions.inc`, builds the `#define NAME EXPANSION` table, and textually expands
  member declarations of the form `<macro> :: <names>` (and `<macro>(tp) :: …`) before parsing.
  Cheaper than full cpp, but must handle the function-like macros (`onCells_3D_Type(tp)`).

Note `mo_ocean_types.f90:43,46,49` also declare members like `onCells :: p` (a bare macro as the
whole type-spec) — the substitution must run at the *line* level on member-declaration lines, not
just inside `TYPE() ::` heads.

---

## PART A — fparser single-TU inliner port (WP1–WP4)

### 1. WP1 — port the inliner + add the fparser dependency (inliner-only)

#### 1.1 Where it lands

- New module: `/home/primrose/Work/dace-fortran/dace_fortran/fparser_inliner.py` (sits beside
  `preprocess.py`; same package). Keep it **self-contained** — only `preprocess.py` imports it (from
  `preprocess_fortran_source`, gated by the `merge_engine="fparser"` selector). Do **not** entangle
  it with the C++ bridge or the rest of the windmill source.
- Public Python entry point (mirror `merge_used_modules`'s signature so it is drop-in):
  ```python
  def inline_to_single_tu(source: str, *, search_dirs=(), include_dirs=(),
                          keep_external=(), expand_macros=True) -> str
  ```
  Returns one self-contained `.f90` text. `keep_external` names procedures NOT to inline
  (`sync_patch_array`, `rot_vertex_ocean_3d`, `get_index_range`, …) so they stay external symbols
  the bridge resolves via its own `keep_external` registry (`external.py:309`).
- CLI entry point (for offline kernel extraction, WP3): add a console script /
  `python -m dace_fortran.fparser_inliner --entry <name> --src-root <dir>... --out <file.f90>` that
  calls `inline_to_single_tu` and writes the result. This is how the four kernel `.f90` artifacts in
  §4 get generated and checked into `tests/icon/ocean/`.

#### 1.2 Locating the windmill source

The inliner originates from a **`dace-windmill` / `f2dace-windmill`** branch that is **NOT checked out
locally**. WP1 step 0 is to fetch/locate it:
- It is a sibling of the `spcl/dace` lineage; check `git remote -v` in `/home/primrose/Work/d-face`
  for a windmill remote, otherwise clone `f2dace-windmill` separately. (Git is currently sandbox-
  blocked in this planning session, so the exact ref must be resolved by the implementer.)
- Identify the inliner module in that branch (look for the same vocabulary as `merge_used_modules`:
  `Use_Stmt`, `Module_Stmt`, single-TU, `ParserFactory`). Port **only** that module + its direct
  helpers; do not pull the windmill build system.

#### 1.3 fparser API surface (cross-check against what's vendored)

The Fortran frontend that already depends on fparser is in d-face (now-stripped HLFIR aside), at
`/home/primrose/Work/d-face/dace/frontend/fortran/`:
- **Version pin discrepancy to resolve (IMPORTANT):** d-face pins **`fparser==0.1.4`**
  (`/home/primrose/Work/d-face/requirements.txt:3`) with a loose floor `'fparser >= 0.1.3'`
  (`setup.py:70`). The task statement and project memory reference **fparser 0.2.3**. WP1 must pick
  ONE and pin it in `dace-fortran/pyproject.toml`. Recommend **0.2.3** (newer node coverage, and it is
  what the rest of the ICON tooling expects per memory), and verify the ported inliner's node-class
  names against 0.2.3 (the F2003/F2008 class names are stable across this range, but confirm
  `Use_Stmt`, `Only_List`, `Rename`, `Module_Stmt`, `Derived_Type_Def`, `Type_Declaration_Stmt`).
- API actually exercised by the existing frontend (use the same surface; cite for the port):
  - `from fparser.two.parser import ParserFactory` → `ParserFactory().create(std="f2008")`
    (`d-face/dace/frontend/fortran/fortran_parser.py:21,1071`).
  - `from fparser.common.readfortran import FortranStringReader, FortranFileReader`
    (`fortran_parser.py:22-23`); `FortranStringReader(text)` (`:1072`).
  - Node classes: `from fparser.two import Fortran2003 as f03` / `Fortran2008 as f08`
    (`ast_components.py:2-4`); `f03.Use_Stmt` (`ast_components.py:1006`), `f03.Module*`,
    `f08.Type_Declaration_Stmt` (`ast_components.py:1002`), `f03.Entity_Decl`,
    `f03.Derived_Type_Def`, `f03.Component_Decl`. Tree walking via `fparser.two.utils.walk`.
  - `from fparser.two import symbol_table` (`ast_components.py:4`) — fparser's own symbol table; the
    inliner can use it to resolve `ONLY:` names to their declarations, OR roll its own name index
    (simpler, fewer fparser-version surprises). Prefer rolling its own for portability.
- **py3.14 circular-import fix:** project memory records the fix (commit 74110d690) = *import
  `Fortran2003` before `Fortran2008`* in `ast_components.py` to break a circular import that
  fparser 0.2.3 introduced under Python 3.14. The current d-face tree imports F2008 **before** F2003
  (`ast_components.py:2-3`) and does **not** carry the fix. **WP1 must apply the same import-order
  fix at the top of `fparser_inliner.py`** (import `Fortran2003` first, then `Fortran2008`) and add a
  one-line comment citing 74110d690, so the inliner imports cleanly on py3.14 with fparser 0.2.3.

#### 1.4 How the inliner resolves the graph (algorithm)

This is the substance of WP1. The fparser engine must do what `merge_used_modules` does **plus** the
three things it punts. Stages:

1. **Index pass (over `search_dirs`, recursive `*.f90`/`*.F90`/`*.incf`):** for each file, expand DSL
   macros (§0.5), parse with fparser, and build a global index keyed by **entity name** →
   `(module, kind, node)` where kind ∈ {parameter, type, interface, procedure, variable}. This is
   finer-grained than `merge_used_modules`'s module-level index (`preprocess.py:815-825`) — the whole
   point is entity-level selection. Record each module's own `USE`/`ONLY` edges for transitive
   resolution.
2. **Entry resolution:** parse the entry source, find the target subroutine, collect its `USE`
   statements with their `ONLY:` lists and `=>` renames.
3. **Reachability (the closure):** starting from the entry's `ONLY:` names (and any names referenced
   in the entry body that resolve to a module entity), transitively pull in:
   - PARAMETERs / module variables it reads (`n_zlev`, `nproma`, `dtime`, `dbl_eps`,
     `sea_boundary`, `min_dolic`, `l_ANTICIPATED_VORTICITY`, `no_dual_edges`,
     `eliminate_upper_diag`, …),
   - **derived-type definitions** it uses (and, recursively, every type those nest — see §3.2:
     `t_patch_3d` → `t_patch` → `t_grid_cells`/`t_grid_edges`/`t_grid_vertices` → `t_subset_range`;
     `t_patch_vert`; `t_operator_coeff` → `t_verticalAdvection_ppm_coefficients`,
     `t_cartesian_coordinates`). **Types reaching the kernel MUST survive into the TU** — the bridge
     flattens them later (§6.2); the inliner must not drop or stub them.
   - **helper procedures** it actually calls and that are NOT in `keep_external`
     (`v_ppm_slimiter_mo_onBlock` for PPM — same module `mo_ocean_limiter`;
     `set_acc_host_or_device` from `mo_fortran_tools`), recursing into *their* `USE` closures.
   - Stop recursion at `keep_external` names and at intrinsic modules
     (`preprocess.py:635-646` already lists `iso_c_binding`, `iso_fortran_env`, `ieee_*`,
     `omp_lib*`, `openacc`, `mpi*`).
4. **`ONLY:`/rename handling:** honor `ONLY:` — only the named entities are imported, not the whole
   module (this is the core win over the regex pass). For `local => orig` renames, emit the entity
   under `orig` and add a local alias (or rename references in the kernel body) so the kernel still
   compiles.
5. **Name-collision handling:** when two different modules export the same name (e.g. `top`,
   `idt_src`, `str_module` are PARAMETER/var names that recur across ICON ocean modules; `init`
   appears in `mo_fortran_tools`), the inliner must detect the clash in the closure and **rename**
   one (deterministic suffix, e.g. `__from_<module>`), rewriting every reference within the emitted
   TU. Because everything is collapsed into one program unit (or a single synthetic module), there is
   no module namespace left to disambiguate — collisions are real and must be resolved by the
   inliner, not flang.
6. **Emit:** topologically order the surviving entities (parameters and types before the procedures
   that use them; deps before dependents — reuse the post-order-DFS shape from
   `preprocess.py:833-847`) and write one `.f90`: a single synthetic `MODULE` (or bare program-unit
   scoping) containing the surviving parameters, type defs, helper procedures, then the kernel.
   Strip `PRIVATE`/`PUBLIC` access statements that no longer apply. Run
   `strip_openmp_directives` (`preprocess.py:348`) on the result so `!$ACC` is dropped (the user
   requires `!$ACC` ignored; flang treats it as a comment anyway without `-fopenmp`).

#### 1.5 cpp conditionals inside the kernels (must be handled)

Several kernels sit inside cpp `#ifdef` blocks that the inliner must resolve consistently with how
flang will be invoked:
- `mo_ocean_velocity_diffusion.f90:1273` has `#else` — the tridiag `_onBlock` is the
  `#ifdef __LVECTOR__` (or similar) arm with a non-vector twin after `#else`
  (`mo_ocean_velocity_diffusion.f90:1274`). The inliner / cpp step must pick **one** arm
  deterministically (match whatever `defines=` the test passes) so only one
  `velocity_diffusion_vertical_implicit_onBlock` survives.
- `mo_ocean_limiter.f90:111` (`#ifdef __LVECTOR__`) and `:686` (`#ifdef NAGFOR`),
  `:1029` etc. — same treatment. The dispatcher `limiter_ocean_zalesak_horizontal:79-157` is NOT the
  extraction target; we extract `_onTriangles` (`:587`) directly, so the dispatcher's `#ifdef` is
  moot as long as we name the `_onTriangles` entry.
- Reuse `strip_openmp_directives`'s cpp handling for `_OPENMP`/`_OPENACC` (`preprocess.py:336-438`)
  and `_balance_cpp` (`preprocess.py:692`) for stray guards split across module-block boundaries;
  extend the macro set as needed for `__LVECTOR__`/`NAGFOR`.

#### 1.6 WP1 deliverables / acceptance

- `dace_fortran/fparser_inliner.py` with `inline_to_single_tu(...)` + CLI.
- fparser pinned in `pyproject.toml` (decide 0.2.3 vs 0.1.4 per §1.3; recommend 0.2.3) with the
  py3.14 import-order fix applied.
- Selector wired into `preprocess_fortran_source` (`preprocess.py`) — default stays `regex`.
- Acceptance: on the synthetic fixtures (WP2) the fparser engine output compiles to the same SDFG
  numerics as the regex engine; on each of the four kernels it emits a `.f90` that `flang -fc1
  -emit-hlfir` accepts (PPM first — §4.1).

### 2. WP2 — the inliner's own unit-test subfolder

- New dir: `/home/primrose/Work/dace-fortran/tests/inliner/` with `__init__.py`, mirroring the
  package-test convention (no `sys.path` hacks, no absolute paths — per repo rule; import via the
  `tests.` package).
- Fixtures: small synthetic multi-module Fortran projects under `tests/inliner/fixtures/`, each
  exercising one feature, structured like the existing `tests/external_to_modules/` fixture dir
  (`external_basic_example.f90`, `utils_mod.f90`, …) and `tests/module_merge_test.py`'s on-disk
  `_proj()` helper (`module_merge_test.py:94-100`):
  1. `only_clause/` — module exporting 4 entities, kernel `USE`s 2 via `ONLY:`; assert the other 2
     do NOT appear in the emitted TU (the regex pass would include the whole module — this is the
     differentiating test).
  2. `rename/` — `USE m, ONLY: a => b`; assert the kernel body still compiles and `b` is emitted.
  3. `collision/` — two modules both export `top`; assert one is renamed and references rewired,
     and the TU compiles.
  4. `nested_types/` — `t_outer` nests `t_inner` nests a leaf; kernel takes `TYPE(t_outer)`; assert
     **all three** type defs survive in dependency order (types-before-use).
  5. `helper_proc/` — kernel calls a module subroutine that itself `USE`s another module; assert the
     helper + its transitive closure are inlined, recursively.
  6. `keep_external/` — kernel calls `sync_stub`; pass `keep_external=("sync_stub",)`; assert it is
     NOT inlined and remains a bare `CALL`.
  7. `dsl_macro/` — a fixture with a local `defs.inc` (`#define onX REAL(8),DIMENSION(:,:)`) and a
     type using `onX :: p`; assert macro expansion produces a parseable type (§0.5).
  8. `cycle/` — mutual `USE` between two modules (the `mo_ocean_types` ↔
     `mo_ocean_tracer_transport_types` shape, §3.1); assert the back-edge is dropped and the TU is
     emitted once (parity with `merge_used_modules`'s cycle note `preprocess.py:830-832`).
- Driving + assertions per fixture:
  - **Parse gate:** `inline_to_single_tu(...)` output → `compile_to_hlfir(...)` (`tests/_util.py:182`)
    succeeds (flang accepts the TU).
  - **Numeric gate (where a result is computable):** `build_sdfg_from_files([emitted_tu], entry=…)`
    (`build.py:525`) and compare against an `f2py_compile` (`tests/_util.py:118`) of the emitted TU,
    `np.testing.assert_allclose(rtol=1e-12, atol=1e-12)` (the `weird_offsets_e2e_test.py` /
    `cross_tu_function_call_test.py` shape).
  - **Equivalence gate:** for fixtures that both engines can handle, assert
    `inline_to_single_tu(...)` and `merge_used_modules(...)` give SDFGs with equal numerics.
- Skip-guards: `have_flang()` (`tests/_util.py:85`) and the gfortran/meson skips already baked into
  `f2py_compile` (`tests/_util.py:152-155`).

### 3. (Reference for WP1/WP3) The transitive USE graph + heavy/cyclic modules

Mapped from source. Defining file paths are under
`/home/primrose/Work/dace-fortran/tests/icon/full/icon-model/`.

#### 3.1 Shared "leaf" parameter/kind/constant modules (cheap — inline freely)

| Module | File | Note |
|---|---|---|
| `mo_kind` | `src/shared/mo_kind.f90` | `wp`, `sp`; 0 USEs (root) |
| `mo_impl_constants` | `src/shared/mo_impl_constants.f90` | `sea_boundary`, `min_dolic`, `max_char_length`, … |
| `mo_math_constants` | `externals/iconmath/src/support/mo_math_constants.f90` | `dbl_eps` (EXTERNAL tree) |
| `mo_math_types` | `externals/iconmath/src/support/mo_math_types.f90` | `t_cartesian_coordinates` (`:31-33`, `x(3)`, `BIND(C)`) |
| `mo_parallel_config` | `src/configure_model/mo_parallel_config.f90` | `nproma`, `p_test_run` (large file but only consts needed) |
| `mo_run_config` | `src/configure_model/mo_run_config.f90` | `dtime`, `ltimer` |
| `mo_ocean_nml` | `src/ocean/config/mo_ocean_nml.f90` | `n_zlev`, `l_ANTICIPATED_VORTICITY`, many knobs; itself USEs 15-20 (only the consts are needed) |
| `mo_dynamics_config` | `src/configure_model/mo_dynamics_config.f90` | `nold`, `nnew` |
| `mo_grid_subset` | `src/shared/mo_grid_subset.f90` | `t_subset_range`, `get_index_range` (the latter → `keep_external`) |
| `mo_fortran_tools` | `src/shared/mo_fortran_tools.f90` | `set_acc_host_or_device`, `init` |
| `mo_timer` | `src/shared/mo_timer.f90` | timer symbols (calls are dead once `ltimer`-guarded; can drop) |
| `mo_util_dbg_prnt` | `src/shared/mo_util_dbg_prnt.f90` | `dbg_print` (debug-only → `keep_external` or drop) |
| `mo_exception` | `externals/fortran-support/src/mo_exception.F90` | `finish`, `message` (error path → `keep_external` or stub) |
| `mo_sync` | `src/parallel_infrastructure/mo_sync.f90` | `sync_*`, `sync_patch_array(_mult)` — **always `keep_external`** |
| `mo_mpi` | `src/parallel_infrastructure/mo_mpi.f90` | `global_mpi_barrier` (Zalesak) → `keep_external` |

#### 3.2 Heavy / structurally-central modules (selective extraction — types only)

| Module | File | LoC | Directly USEs (heads) |
|---|---|---|---|
| `mo_model_domain` | `src/grid/mo_model_domain.f90` | ~1035 | `mo_kind` (no ONLY), `mo_math_types`, `mo_impl_constants`, **`mo_communication`** (t_comm_pattern…), `mo_io_units`, `mo_util_uuid_types`, `mo_lib_grid_geometry_info`, `mo_decomposition_tools`, `mo_read_netcdf_types`. Defines `t_patch`, `t_patch_3d`, `t_patch_vert`, `t_grid_cells/edges/vertices`, `t_subset_range`. |
| `mo_ocean_types` | `src/ocean/dynamics/mo_ocean_types.f90` | ~594 | `mo_kind`, `mo_impl_constants`, `mo_math_types`, `mo_ocean_diagnostics_types`, `mo_model_domain`, **`mo_ocean_tracer_transport_types` (no ONLY)**. Defines `t_verticalAdvection_ppm_coefficients` (`:56-71`), `t_operator_coeff` (`:488-566`), the hydro-ocean state types. |
| `mo_operator_ocean_coeff_3d` | `src/ocean/math/mo_operator_ocean_coeff_3d.f90` | ~2728 | many incl. `mo_cf_convention`, `mo_grib2`, `mo_grid_geometry_info` (no ONLY). Re-exports `t_operator_coeff`, plus `no_dual_edges`, `no_primal_edges`. |
| `mo_ocean_tracer_transport_types` | `src/ocean/tracer_transport/mo_ocean_tracer_transport_types.f90` | ~93 (light!) | `mo_kind`, `mo_math_types`, `mo_model_domain`, `mo_impl_constants`. Defines `t_ocean_tracer`, `t_ocean_transport_state`. |
| `mo_ocean_math_operators` | `src/ocean/math/mo_ocean_math_operators.f90` | ~3558 | hosts `rot_vertex_ocean_3d`, `map_edges2vert_3d`, `div_oce_3d`, … → these become `keep_external` for kernels 3/4. |
| `mo_ocean_physics` / `mo_ocean_physics_types` | `src/ocean/physics/mo_ocean_physics.f90` / `…_types.f90` | ~2713 / ~792 | **VERY heavy** — pull in the var_list / CDI / GRIB2 / netCDF I/O stack. Kernel 1's bare `USE mo_ocean_physics` (no ONLY) is the worst offender. |

#### 3.3 Cycles + unqualified USEs (must be handled, §1.4/1.5)

- **Cycle:** `mo_ocean_types` `USE`s `mo_ocean_tracer_transport_types` **without `ONLY:`**
  (`mo_ocean_types.f90:27`), and that module `USE`s `mo_model_domain`; `mo_model_domain` does not
  cycle back, but `t_subset_range` contains a `POINTER :: patch` back to `t_patch`
  (a **definitional** self/mutual reference inside `mo_model_domain.f90`) — the inliner must keep
  that as an opaque POINTER and not attempt to expand it (the bridge keeps it a pointer; flatten
  stops there).
- **Unqualified USEs** (whole-namespace imports — the inliner must still resolve `ONLY`-less imports
  by reachability, only pulling the entities actually referenced):
  - `mo_ocean_tracer_transport_vert.f90:38` — `USE mo_ocean_physics` (no ONLY). **Biggest risk for
    kernel 1.** In practice `upwind_vflux_ppm_onBlock` uses **nothing** from `mo_ocean_physics`
    (it is `USE`d by the module for *other* subroutines). The inliner's reachability (§1.4 step 3),
    being entity-driven from the *kernel body*, should pull **zero** entities from
    `mo_ocean_physics` → the heavy physics/CDI closure is naturally pruned. This is the central
    argument for entity-level extraction over verbatim module splice.
  - `mo_ocean_types.f90:27` `USE mo_ocean_tracer_transport_types` (no ONLY); `mo_model_domain.f90:33`
    `USE mo_kind` (no ONLY); `mo_operator_ocean_coeff_3d.f90` `USE mo_grid_geometry_info` (no ONLY).

### 4. (WP3) Per-kernel extraction specs — ordered cleanest → hardest

For each: target entry symbol (flang mangling `_QM<module>P<name>`, all lower-case — from
`_mangle`/`_resolve_entry` at `build.py:172-173,126`), the entities to inline, the derived types that
must survive, the hazards, and the maturity ladder.

#### 4.1 Kernel 1 — PPM vertical tracer-advection flux (PRIMARY, do first)

- **Target:** `upwind_vflux_ppm_onBlock`,
  `ocean/tracer_transport/mo_ocean_tracer_transport_vert.f90:213-551`.
- **Entry symbol:** `_QMmo_ocean_tracer_transport_vertPupwind_vflux_ppm_onblock`.
- **Built-in oracle:** the sibling `upwind_vflux_ppm_vector` (`:554-902`) computes the *same* result
  with an explicit `MAXVAL` trip-count + `CYCLE` masking idiom (`:617,646-648,802-804,886`). Extract
  **both** into the same TU and use the vector routine as the numerical reference (no f2py of a
  *different* source needed — the two are algebraic equivalents on the same inputs). This is the
  cleanest possible oracle.
- **Args (already struct-light):** `tracer(nproma,n_zlev)`, `w(nproma,n_zlev+1)`, `dtime`,
  `vertical_limiter_type`, `cell_thickeness(nproma,n_zlev)`, `cell_invheight(nproma,n_zlev)`,
  `ppmCoeffs` (`TYPE(t_verticalAdvection_ppm_coefficients)`), `flux_div_vert(nproma,n_zlev)`,
  `startIndex`, `endIndex`, `cells_noOfLevels(nproma)`, `lacc` (`:221-231`). The user's suggestion —
  **pass the 9 ppmCoeffs members as 9 explicit `(nproma,levels)` 2-D args** — is the lowest-risk path:
  it sidesteps struct-flatten entirely for the geometry. The kernel already rebinds them to local
  POINTERs at `:264-272`, so replacing `ppmCoeffs%X` with a dummy `X(:,:)` is mechanical. (Keep the
  struct form as a stretch test once flatten is confirmed working — §6.2.)
- **Inline closure:** `t_verticalAdvection_ppm_coefficients` (only if keeping the struct form),
  `wp` (`mo_kind`), `n_zlev`/`nproma` (consts), `set_acc_host_or_device` (`mo_fortran_tools`),
  `v_ppm_slimiter_mo_onBlock` (`mo_ocean_limiter:53` — **also inline this helper**, it is called at
  `:444`). `max_char_length`/`dbl_eps`/`sea_boundary` are referenced by the module but not by this
  kernel → pruned. `dbg_print`, timers, `finish` → drop/keep_external. `sync_patch_array` not called
  in `_onBlock` (it is in the outer `advect_flux_vertical`) → not needed.
- **Hazards (only ragged columns):** loops to `cells_noOfLevels(jc)-1/-2` with `IF (... ) CYCLE`
  semantics in the scalar form done as nested `DO thisLevel = secondLevel, cells_noOfLevels(jc)-1`
  (`:308,357,460,482,538`); top/bottom special-casing via `IF` (`:389-422`). The bridge expresses
  these as **symbolic loop bounds** `loopbegin/loopend` (README; not masked-tail — that is a DaCe
  vectorizer concern, §6.3). The two-level `n_zlev+1` arrays (`z_face`, `upward_tracer_flux`,
  `w`) are fine. **No gather, no halo sync** — this is why it is the primary.
- **Maturity ladder:** (a) emitted `.f90` parses to HLFIR; (b) builds an SDFG that validates;
  (c) CPU-numerical: SDFG output of `_onBlock` == output of `_vector` (the oracle) within
  `assert_allclose(rtol=1e-12, atol=1e-12)` on random columns with random ragged `cells_noOfLevels`;
  (d) later GPU codegen (out of scope for the first landing — CPU-only first).

#### 4.2 Kernel 2 — implicit vertical-mixing tridiagonal (Thomas) solve

- **Target:** `velocity_diffusion_vertical_implicit_onBlock`,
  `ocean/dynamics/mo_ocean_velocity_diffusion.f90:1078-1272`.
- **Entry symbol:** `_QMmo_ocean_velocity_diffusionPvelocity_diffusion_vertical_implicit_onblock`.
- **Args:** `patch_3d` (`TYPE(t_patch_3d)`), `velocity(:,:)`, `a_v(:,:)`,
  `operators_coefficients` (`TYPE(t_operator_coeff)`), `start_index`, `end_index`, `edge_block`,
  `lacc` (`:1085-1090`). It reads only `patch_3d%p_patch_1d(1)%{dolic_e,inv_prism_thick_e,
  inv_prism_center_dist_e}` (`:1106,1115,1125,1126`) — i.e. a thin slice of `t_patch_vert`. It does
  **not** actually read `operators_coefficients` in the body (declared but the math uses `a_v` and
  the patch arrays) — confirm and, if so, the struct arg can be dropped or kept as a dead flatten
  exercise.
- **Inline closure:** `wp`, `n_zlev`, `nproma`, `dtime` (`mo_run_config`),
  `eliminate_upper_diag` (module var `mo_ocean_velocity_diffusion.f90:58` — a module-level `LOGICAL`,
  must be inlined as a local parameter/var), `min_dolic` (`mo_impl_constants`, if referenced),
  `set_acc_host_or_device`. The `t_patch_3d`/`t_patch_vert` closure must survive **or** (preferred,
  like PPM) the three needed patch arrays are passed as explicit `(nproma,n_zlev)` /
  `(nproma,n_zlev,nblks)` dummies and `dolic_e` as `(nproma,nblks)`.
- **Hazards:** serial vertical recurrence (Thomas forward/back sweep,
  `!$ACC LOOP SEQ` at `:1189,1213,1229,1254`) — a genuine loop-carried dependence that must NOT be
  parallelized over `level`; the bridge must keep it sequential (the SEQ marker is an `!$ACC`
  comment, ignored, so correctness rests on the bridge's dependence analysis — verify in stage c).
  Data-dependent `max_end_level = MAXVAL(dolic_e(start:end,edge_block))` (`:1106`) — a reduction the
  bridge lifts (`hlfir-lift-reduction-operands`). `bottom_level(edge_index) < 2 .OR. level > …` CYCLE
  guards throughout (`:1123,1164,1194,…`). **cpp twin:** `:1273 #else` — the inliner must select the
  `__LVECTOR__` (or default) arm so only one `_onBlock` survives (§1.5). **No gather, no halo sync.**
- **Oracle:** no in-file vector twin (the `#else` arm is the *scalar* version of the same routine —
  it can serve as a cross-check, but they target different `defines`). Use **f2py of the emitted TU**
  as the reference (the `weird_offsets`/`cross_tu` pattern). A known-answer Thomas solve on a small
  SPD tridiagonal is a good unit sanity check too.
- **Maturity ladder:** parse → SDFG-validate → CPU `assert_allclose` vs f2py(same TU) on random SPD
  columns with random `dolic_e` → later GPU.

#### 4.3 Kernel 3 — Zalesak horizontal FCT limiter

- **Target:** `limiter_ocean_zalesak_horizontal_onTriangles`,
  `ocean/tracer_transport/mo_ocean_limiter.f90:587-1059`.
- **Entry symbol:** `_QMmo_ocean_limiterPlimiter_ocean_zalesak_horizontal_ontriangles`.
- **Args (`:600-611`):** `patch_3d`, `vert_velocity`, `tracer`, `p_mass_flx_e`, `flx_tracer_low`,
  `flx_tracer_high`, `flx_tracer_final`, `div_adv_flux_vert`,
  `operators_coefficients` (`TYPE(t_operator_coeff)`), `h_old`, `h_new`, `lacc`. It rebinds many
  POINTERs at `:666-679`: `cellOfEdge_idx/blk` (`edges%cell_idx/blk`),
  `edge_of_cell_idx/blk` (`cells%edge_idx/blk`), `neighbor_cell_idx/blk` (`cells%neighbor_idx/blk`),
  `dolic_e/dolic_c` (`p_patch_1d`), `div_coeff` (`operators_coefficients%div_coeff`),
  `prism_thick_flat_sfc_c`, `del_zlev_m`, `inv_prism_thick_c`,
  `edges_SeaBoundaryLevel` (`operators_coefficients%edges_SeaBoundaryLevel`).
- **Inline closure:** `wp`, `n_zlev`, `nproma`, `p_test_run` (`mo_parallel_config`),
  `dtime`, `dbl_eps` (`mo_math_constants`), `sea_boundary`/`SEA` (`mo_impl_constants`),
  `set_acc_host_or_device`; types `t_patch_3d`, `t_patch`, `t_operator_coeff`, `t_subset_range`
  (for `in_domain`/`get_index_range`). `get_index_range`, `sync_patch_array_mult`,
  `global_mpi_barrier`, `dbg_print`, `message`/`finish` → **`keep_external`** (halo + error + I/O).
- **Hazards (gather + multi-phase halo):**
  - **Unstructured gather:** the 3 edges of each cell (`edge_of_cell_idx/blk(jc,blockNo,1..3)`,
    `:745-750,901-906`), the 3 neighbor cells (`neighbor_cell_idx/blk(jc,blockNo,1..3)`,
    `:894-899`), and the 2 cells of each edge (`cellOfEdge_idx/blk(edge_index,blockNo,1..2)`,
    `:1039-1043`). These are fixed-arity (3 / 3 / 2) indirect reads — closer to the dycore stencil
    gather than the variable-length Coriolis gather, so they map onto the bridge's
    `ExpandVectorSubscriptGather` path. Verify against the gather-lowering work in project memory
    (the multi-dim-tileops gather-index machinery).
  - **Two halo-sync barriers mid-routine:** `sync_patch_array_mult(sync_c1, …, z_tracer_max,
    z_tracer_min)` after the max/min pass (`:869`) and `sync_patch_array_mult(sync_c1, …, r_m, r_p)`
    after the ratio pass (`:1006`). These split the kernel into **three data-parallel phases** with a
    cross-rank dependence between them. Strategy: keep the two `sync_patch_array_mult` as external
    calls (`keep_external`), exactly as the existing velocity integration keeps `sync_patch_array`
    external and lets ICON's real MPI run (`docs/ICON_INTEGRATION.md:46-58`). For a **single-rank**
    unit test the halo exchange is a no-op, so a no-op stub for `sync_patch_array_mult` suffices and
    the three phases compute correctly in sequence.
  - **Ragged dolic** (`dolic_e`/`dolic_c` loop bounds, `:718,802,908,1023`) → symbolic bounds.
  - **`edges_SeaBoundaryLevel > -2` branch** (`:1025`) selecting low-order vs limited flux — a
    data-dependent IF, fine for the bridge.
- **Oracle:** no in-file equivalent twin (`_lvector`/`_general` siblings target different `defines` /
  connectivities). Use **f2py of the emitted TU** with a no-op `sync_patch_array_mult` stub on both
  the SDFG and f2py sides, on a small synthetic triangular-grid connectivity (hand-built
  `edge_of_cell`/`neighbor`/`cellOfEdge` index arrays + random fluxes). Compare `flx_tracer_final`
  with `assert_allclose(rtol=1e-12, atol=1e-12)`.
- **Maturity ladder:** parse → SDFG-validate (likely the first achievable milestone; gather +
  external sync may need bridge work) → CPU-numerical with synthetic connectivity → later GPU.

#### 4.4 Kernel 4 — nonlinear Coriolis / PV flux (hardest, do last)

- **Target:** `nonlinear_coriolis_3d_fast_scalar`, `ocean/math/mo_scalar_product.f90:350-622`.
- **Entry symbol:** `_QMmo_scalar_productPnonlinear_coriolis_3d_fast_scalar`.
- **Args (`:350-358`):** `patch_3d`, `vn`, `p_vn_dual` (`TYPE(t_cartesian_coordinates)` array!),
  `vort_v`, `operators_coefficients` (`TYPE(t_operator_coeff)`), `vort_flux`, `lacc`.
- **Inline closure:** `wp`/`sp` (`mo_kind`), `n_zlev`, `nproma`,
  `l_ANTICIPATED_VORTICITY` (`mo_ocean_nml` — module `LOGICAL`, drives the `:390 IF(.NOT….)` /
  `:495 ELSEIF` split; for the fast-scalar path we only need the `.NOT.` arm, but the inliner must
  bring the symbol in), `no_dual_edges` (`mo_operator_ocean_coeff_3d:34`),
  `sea_boundary`/`sea`/`min_dolic`, `set_acc_host_or_device`; types `t_patch_3d`, `t_patch`,
  `t_operator_coeff`, `t_cartesian_coordinates`, `t_subset_range`. `rot_vertex_ocean_3d`
  (`mo_ocean_math_operators`, called at `:386`), `sync_patch_array` (`:388`),
  `get_index_range` (`:396`) → **`keep_external`**.
- **Hazards (the heaviest):**
  - **Variable-length unstructured gather:** the inner loop runs
    `DO vertex_edge = 1, patch_2d%verts%num_edges(vertexN_idx,vertexN_blk)` (`:421,448`) — a
    **data-dependent trip count** per vertex (5 or 6 edges on an icosahedral grid), with indirect
    `verts%edge_idx/edge_blk(vertexN_idx,vertexN_blk,vertex_edge)` (`:423-424,450-451`) feeding a
    further indirect read `vn(edgeOfVertex_index, level, edgeOfVertex_block)` (`:437,465`). This is a
    **nested indirect gather with a runtime bound** — the structurally hardest of the four, and the
    most likely to need new bridge support (compare to the velocity_nested_indirect test in the
    suite, `tests/velocity_nested_indirect_test.py`).
  - **Data-dependent `MIN(dolic)` level bound:**
    `DO level = startLevel, MIN(dolic_e(je,blockNo), dolic_e(edgeOfVertex_index, edgeOfVertex_block))`
    (`:426-427,453-454`) — the trip count depends on a `MIN` of two indirect reads. The bridge must
    express this as a symbolic bound computed inside the loop nest.
  - **External `rot_vertex_ocean_3d` + halo `sync_patch_array`** (`:386,388`): both `keep_external`;
    no-op sync stub for single-rank tests. But `rot_vertex_ocean_3d` *computes* `vort_v` which the
    kernel then reads — so the unit test must **either** also inline `rot_vertex_ocean_3d` (it lives
    in the very heavy `mo_ocean_math_operators`, ~3558 LoC, itself gather-heavy) **or** treat `vort_v`
    as a *given input* and skip the `rot_vertex_ocean_3d` call (pass a precomputed `vort_v`). The
    latter is strongly preferred for the unit test: extract only the Coriolis flux assembly, feed
    `vort_v`/`p_vn_dual` as inputs, stub the sync. (Inlining `rot_vertex` is a separate, larger
    effort.)
  - `TYPE(t_cartesian_coordinates)` **array argument** `p_vn_dual(nproma,n_zlev,nblks_v)` — an
    array-of-struct dummy (each element has `x(3)`). The bridge flattens AoS-with-array-members
    (`A(i)%x(k)` → `A_x(i,k)`, README support table) — but note the body of the `.NOT.` arm does not
    actually dereference `p_vn_dual` (it is used in the anticipated-vorticity arm). Confirm; if dead
    in the fast-scalar arm, it can be dropped.
- **Oracle:** the sibling `nonlinear_coriolis_3d` (the non-`_fast` version near `:625`) is a
  reference *in spirit* but numerically different (the comment at `:627-628` warns of sensitivity).
  Use **f2py of the emitted TU** on a small synthetic dual-grid connectivity (hand-built
  `verts%num_edges`, `verts%edge_idx/blk`, `edge2edge_viavert_coeff`, random `vn`/`vort_v`/`f_v`),
  no-op sync. Compare `vort_flux` with `assert_allclose` — but expect to need a **looser rtol** given
  the gather summation order; pin it empirically.
- **Maturity ladder:** parse → SDFG-validate is the realistic first milestone; the variable-length
  nested gather is the gating bridge feature. CPU-numerical with synthetic connectivity is the
  stretch goal; GPU is out of scope until the gather lowers.

### 5. (WP3) Unit tests in dace-fortran

#### 5.1 Location + shape

- New dir: `/home/primrose/Work/dace-fortran/tests/icon/ocean/` with `__init__.py` (parallels the
  existing `tests/icon/full/`, `tests/icon/dycore/`). The four extracted single-TU `.f90` artifacts
  live here (checked in, generated by the WP1 CLI, regenerated by a documented command so drift is
  catchable — mirror the `velocity_full.f90` maintenance note `docs/ICON_INTEGRATION.md:114-117`).
- One test module per kernel: `test_ppm_vflux.py`, `test_vmix_tridiag.py`, `test_zalesak.py`,
  `test_coriolis.py`. Each uses `have_flang()` collection-skip (`tests/_util.py:85`).

#### 5.2 How each is driven

- **Primary route** (single-TU): `build_sdfg_from_files([<kernel>.f90], entry="_QM<mod>P<name>",
  name="<name>", out_dir=tmp_path)` (`build.py:525`). Because the `.f90` is already self-contained,
  `merge_used_modules`/the fparser engine is a pass-through at build time; the inliner did its work
  at artifact-generation time. `defines=` for the `__LVECTOR__`/cpp arm selection goes through
  `build_sdfg` → `make_builder(..., defines=…)` (`build.py:263,300-307`; note
  `build_sdfg_from_files` itself does not take `defines=` — if a kernel needs a define, either bake
  the selected arm into the artifact at generation time, or drive via `build_sdfg(text, …,
  defines=…)` on the read artifact text).
- **Fallback route** (parse-only, full ICON): `emit_hlfir_from_codebase(entry_source, out,
  search_dirs=[…/src, externals/*/src], library_stubs=["mpi","netcdf"], defines=<ICON set>,
  include_dirs=[…/src/include])` (`flang_codebase.py:425`) + `clear_external_registry()` +
  `keep_external(sym, stub=True)` (`external.py:309,467`) + `build_sdfg_from_hlfir(hlfir,
  entry="_QM<mod>P<name>")` (`build.py:413`) — mirrors `test_velocity_from_icon_source.py:157-172`.
  Use this when the single-TU artifact is not yet ready or to A/B the inliner against the codebase
  composer. `defines` should include the ICON fallback set used by the existing tests
  (`__ICON__`, `__LOOP_EXCHANGE`, `__NO_ICON_OCEAN__` … — but for ocean kernels we may need the
  ocean-enabling defines instead; resolve empirically) plus `NO_MPI_CHOICE_ARG`.

#### 5.3 What each asserts, by maturity stage

For every kernel, stage up the same ladder; land whatever stage passes and `xfail` the rest with a
reason (the repo's `xfail` helper, `tests/_helpers.py:62`):
1. **Parses → HLFIR:** `compile_to_hlfir(text, tmp)` (`_util.py:182`) returns a `.hlfir`; or
   (codebase route) the mangled symbol appears as a non-private `func.func` in the emitted `.hlfir`
   (the `test_solve_nonhydro_parse.py:97-99` assertion style).
2. **Builds SDFG:** `build_sdfg_from_files([...]).build()` (or `build_sdfg_from_hlfir`) returns an
   SDFG; `sdfg.validate()`; `assert sdfg is not None and sdfg.name` (the
   `test_velocity_from_icon_source.py:219-227` style).
3. **CPU-numerical vs reference:** call the SDFG and compare to the oracle.
   - **PPM:** the oracle is the in-file `upwind_vflux_ppm_vector` (§4.1) — build BOTH into the SDFG
     run / f2py and `assert_allclose(flux_div_vert_scalar, flux_div_vert_vector, rtol=1e-12,
     atol=1e-12)`. This is the cleanest gate and the reason PPM is primary.
   - **Others:** `f2py_compile(<kernel>.f90, …)` (`_util.py:118`) of the same TU as reference; drive
     inputs with the seeded PRNG (`tests/_prng.py:xor64_uniform01`, `tests/_helpers.py`), route
     scalar/len-1-array integer args via `sdfg_call_args` (`_helpers.py:44`) /
     `_util`'s scalar routing; compare with `assert_allclose` (tight `1e-12` where summation order
     matches; looser, pinned empirically, for the Coriolis gather).
   - Inputs: build small synthetic grids (ragged `cells_noOfLevels`/`dolic`, triangular cell-edge /
     dual-vertex connectivity) so the gather indices are valid; zero-fill via a `_allocate`-style
     helper like `tests/icon/full/_harness.py:98`.
4. **GPU (later, explicitly deferred):** same SDFG, GPU codegen. **All four are CPU-only first.**
   `!$ACC` is ignored throughout (`strip_openmp_directives`, `preprocess.py:348`; flang gets
   `-U_OPENMP -U_OPENACC`, `build.py:244` / `flang_codebase.py:475`).

#### 5.4 Reference oracle mechanics (reuse existing)

- f2py compile of the same source: `f2py_compile(src, out_dir, mod_name, only=(…,))`
  (`tests/_util.py:118-179`); the `only=` filter hides inner subroutines whose `TYPE()` dummies
  crackfortran cannot map (`_util.py:140-148`) — relevant if a kernel keeps a struct dummy.
- Deterministic inputs: `xor64_uniform01(n, seed=42)` (`tests/_prng.py:23`) matches the Fortran-side
  RNG; `sdfg_call_args(sdfg, int_values)` (`tests/_helpers.py:44`) routes ints to scalar-vs-len1
  array per descriptor.
- Comparison: `np.testing.assert_allclose(rtol=…, atol=…)` (float) / `assert_array_equal` (exact).
  The FP-conservative flag triple `-O0 -fno-fast-math -ffp-contract=off`
  (`FLANG_PORTABLE_FFLAGS`, `_util.py:94`) keeps SDFG vs reference bit-comparable where the algebra
  is identical (PPM scalar-vs-vector is *not* bit-identical — different evaluation order — so use
  `rtol`, not `array_equal`).

### 6. WP4 — README + Python-frontend function parity, and how this composes

#### 6.1 README (WP4 doc deliverable)

A new section in `/home/primrose/Work/dace-fortran/README.md` (or a `docs/` page) covering:
- The two single-TU engines (`merge_used_modules` regex vs `inline_to_single_tu` fparser), when to
  use which, and that fparser supersedes regex for `ONLY:`-heavy ICON graphs (§0.1, §0.2).
- The CLI for offline kernel extraction; the four ocean artifacts + the exact regeneration command
  (drift-catch).
- The `keep_external` contract for `sync_patch_array(_mult)`, `rot_vertex_ocean_3d`,
  `get_index_range` and the no-op-stub-for-single-rank testing convention.
- The DSL `.inc` macro-expansion requirement (`-I src/include`), and that `!$ACC` is ignored.
- "Python-frontend function parity": the existing d-face Fortran frontend
  (`d-face/dace/frontend/fortran/{fortran_parser,ast_components}.py`) consumes an fparser AST for the
  *pure-Python* SDFG path; the inliner uses the **same** fparser surface
  (`ParserFactory.create(std="f2008")`, `FortranStringReader`, `Fortran2003.*` node classes,
  `walk`). WP4 documents that the inliner does not diverge from that frontend's fparser usage (so a
  future merge of the two does not fight over fparser versions), and confirms the py3.14 import-order
  fix (74110d690) is applied in both.

#### 6.2 Composition with the bridge's derived-type flattening

- The struct-flattening pass is `hlfir-flatten-structs` =
  `/home/primrose/Work/dace-fortran/dace_fortran/passes/FlattenStructs.cpp` (registered
  `dace_fortran/passes/Passes.cpp:40`; in the default pipeline). A `TYPE(t_x)` **dummy argument is
  flattened automatically** into one block arg per member (or per leaf for nested), the function is
  renamed `…_soa`, and a `hlfir.flatten_plan` attribute is written for the bindings emitter
  (`FlattenStructs.cpp:46-54`; recipe in `dace_fortran/bindings/flatten_plan.py:28`). Path-flattened
  names: `base_m1_m2_leaf`; AoS-with-array-members `A(i)%w(j,k)` → `A_w(i,j,k)` (README support
  table). The velocity harness's `_INIT_ARRAY_ORDER` (`tests/icon/full/_harness.py:14-80`) is exactly
  this flattened per-member ABI.
- **Implication for the inliner:** types reaching the kernel must **survive into the TU intact** —
  the inliner must NOT stub/drop them. The flatten pass needs the real `TYPE … END TYPE` with real
  member ranks (post macro-expansion) to compute the plan. The deep nesting
  `t_patch_3d → t_patch → t_grid_cells → t_subset_range` (§3.2) flattens path-wise; the
  `t_subset_range%patch` back-pointer and other bare POINTER members are **out of scope** for flatten
  (loud failure at `extract_vars`, README scope notes) — which is the second reason the kernels'
  per-member-array extraction (PPM's 9 explicit args, kernel 2/3/4 passing patch sub-arrays directly)
  is the pragmatic path: it hands the bridge flat arrays and sidesteps flattening the giant `t_patch`.
  Keep the full-struct form as a later flatten stress test, not a gating requirement.

#### 6.3 Composition with ragged-column / CYCLE / masked-tail machinery

- At the **bridge** level: `EXIT`/`CYCLE` are supported (README control-flow table); variable/ragged
  loop bounds become **symbolic** `loopbegin_<N>`/`loopend_<N>`/`loopstep_<N>` symbols (README); inline
  reductions over slices (`MAXVAL(dolic(…))`, `max(x, MAXVAL(slice))`) are lifted by
  `hlfir-lift-reduction-operands` (`dace_fortran/passes/LiftReductionOperands.cpp`). So the PPM /
  tridiag ragged `dolic` loops and the Coriolis `MIN(dolic)` bound are handled as symbolic bounds at
  the bridge, **not** as masked tiles.
- "**masked-tail**" is a **DaCe-side vectorizer** concept (the multi-dim-tileops work:
  `remainder_strategy=masked_tail`, `MarkTileDims`), per project memory — it lives downstream of the
  bridge and is relevant only when these SDFGs are later **vectorized for CPU SIMD or tiled for
  GPU**. It is NOT part of getting the kernels to parse/build/numerically-validate (stages 1-3). Note
  it in the README as the path the GPU stage (stage 4) will route through.

### 7. Sequencing, risks, and the biggest unknowns

#### 7.1 Phase order (overall)

- **Phase 0:** Locate the windmill inliner branch; pin fparser (0.2.3 recommended) + apply the
  py3.14 import-order fix. (§1.2-1.3)
- **Phase 1 (WP1):** Port `inline_to_single_tu` with `ONLY:`/rename/collision handling + DSL macro
  expansion + cpp-arm selection + `keep_external` pass-through. Selector into
  `preprocess_fortran_source` (default stays regex). (§1)
- **Phase 2 (WP2):** Synthetic fixtures `tests/inliner/` + the regex/fparser equivalence gate. (§2)
- **Phase 3 (WP3, kernel 1 — PPM):** Generate the PPM TU (9-explicit-args form), land stages 1-3
  using the in-file vector oracle. This is the milestone that proves the whole pipeline. (§4.1, §5)
- **Phase 4 (WP3, kernel 2 — tridiag):** clean column kernel; validates the serial-recurrence + cpp-
  arm-selection handling. (§4.2)
- **Phase 5 (WP3, kernel 3 — Zalesak):** fixed-arity gather + external halo sync (no-op stub);
  exercises `ExpandVectorSubscriptGather` + multi-phase. Likely lands stage 2 first, stage 3 with
  synthetic connectivity. (§4.3)
- **Phase 6 (WP3, kernel 4 — Coriolis):** variable-length nested gather; the gating bridge feature.
  Aim for stage 2 (parse+build); stage 3 is a stretch. Feed `vort_v` as input (skip
  `rot_vertex_ocean_3d`). (§4.4)
- **Phase 7 (WP4):** README + parity doc; document GPU as the deferred stage-4 for all four. (§6)

#### 7.2 Biggest risks (and where each is handled)

1. **The DSL `.inc` macros** (`iconfor_dsl_definitions.inc`) — every kernel's types depend on them;
   if not expanded, fparser cannot parse the type defs. **Handled** by macro-expansion in WP1 (§0.5),
   tested by the `dsl_macro/` fixture (§2). Highest-frequency failure mode; do it first.
2. **Coriolis variable-length nested gather** (`verts%num_edges` trip count + double-indirect
   `vn(edge_idx(…),…)`, `:421-437`) — the structurally hardest lowering; **may require new bridge
   work** beyond the fixed-arity gather. Mitigation: feed `vort_v` as input; target parse+build
   first; pin a looser numerical rtol. This is the single biggest unknown.
3. **Zalesak multi-phase halo sync** (two `sync_patch_array_mult`, `:869,1006`) — **handled** by
   `keep_external` + single-rank no-op stub (same pattern as the velocity integration,
   `docs/ICON_INTEGRATION.md`); the three phases compute in sequence on one rank.
4. **Ragged columns** (PPM/tridiag `dolic` bounds + `CYCLE`) — **handled** at the bridge as symbolic
   loop bounds + lifted `MAXVAL` reductions (§6.3); lowest-risk hazard, but the PPM scalar-vs-vector
   equivalence is the explicit gate that it is correct.
5. **`USE … (no ONLY)` over heavy modules** — esp. kernel 1's `USE mo_ocean_physics`
   (`mo_ocean_tracer_transport_vert.f90:38`). **Handled** by entity-driven reachability (§1.4) which
   prunes the unused physics/CDI closure; the `only_clause/`+`helper_proc/` fixtures gate it. If
   reachability is too coarse and pulls the CDI stack, the TU will be huge and flang-slow — a
   correctness-vs-cost signal to tighten the closure, not a wrong answer.
6. **Collision renaming** (`top`, `idt_src`, `str_module`, `init` recurring across modules) — if the
   inliner mis-renames, the TU mis-compiles silently. **Handled** by the `collision/` fixture +
   compile gate; keep the rename deterministic and logged.
7. **fparser version skew** (0.1.4 vendored vs 0.2.3 intended) — node-class names could differ.
   **Handled** by pinning one version in WP1 and verifying the exact `Use_Stmt`/`Only_List`/`Rename`/
   `Derived_Type_Def` class names against it before porting (§1.3).
8. **Struct flatten of the giant `t_patch`** — full-struct dummies would force flattening a 200-member
   nested type with out-of-scope POINTER members. **Avoided** by the per-member-array extraction
   (PPM's 9 args; kernels 2/3/4 passing patch sub-arrays as plain dummies); full-struct flatten is a
   later stress test, not on the critical path (§6.2).

#### 7.3 Definition of done (first landing)

- `dace_fortran/fparser_inliner.py` + fparser pin + selector, `tests/inliner/` green.
- Four single-TU `.f90` artifacts in `tests/icon/ocean/` + their regeneration command.
- `tests/icon/ocean/test_*.py` at: PPM stage 3 (numeric vs in-file vector oracle); tridiag stage 3
  (numeric vs f2py); Zalesak ≥ stage 2 (build+validate), stage 3 `xfail`-or-pass with synthetic
  connectivity; Coriolis ≥ stage 2, stage 3 `xfail`. GPU (stage 4) deferred for all four.
- README/parity doc updated. Nothing pushed until the suite is green locally (repo push discipline).
