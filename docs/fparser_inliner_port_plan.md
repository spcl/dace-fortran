# Plan: fparser-based module inliner as a reusable single-TU tool

Goal: **alternative front path** — an fparser tool that inlines a multi-file Fortran project's USE'd modules/procedures at the SOURCE level into one TU (`.f90`), feedable directly to `build_sdfg` (instead of multi-file + flang HLFIR inlining).

## Grounding (what exists today)
- **Engine to supersede (not duplicate):** `merge_used_modules(source, *, search_dirs=())` at `dace_fortran/preprocess.py:788` — **regex-based/fparser-free**: splices whole `MODULE…END MODULE` blocks verbatim in `USE`-graph dependency order (`_module_blocks` :723, toposort :833, prepend :851). Default-on via `preprocess_fortran_source(..., merge=True)` → `_emit_hlfir` (`build.py:236`). **Limitation:** `_USE_RE` (:650) captures only the module *name* — every `ONLY:`, `=>` rename, cross-module collision punted to flang; whole-module splice pulls vast unrelated code for real ICON graphs.
- **`hlfir-inline-all` is intra-TU only** — so this work is purely a better **source-text single-TU producer**, slotting where `merge_used_modules` sits.
- **Build entry points** (`build.py`): `build_sdfg` :321, `build_sdfg_from_files` :543, `build_sdfg_from_hlfir` :431. **Entry resolution** `_resolve_entry` :126 already accepts mangled `_Q…` / plain `proc` / `module::proc` — REUSE it.
- **fparser:** NOT yet a declared dep (`pyproject.toml:28` only `dace`); installed in py13. Reference surface = **d-face's frontend** (`d-face/dace/frontend/fortran/`): `ParserFactory().create(std="f2008")`, `FortranStringReader`, `from fparser.two import Fortran2003/Fortran2008`, `fparser.two.utils.walk`. **py3.14 circular-import fix (commit 74110d690): import Fortran2003 BEFORE Fortran2008** — replicate in the new module.
- **No `windmill` repo checked out** — plan from first principles using d-face's surface.
- **Prior in-repo doc to refine (don't duplicate):** `docs/icon_ocean_fparser_extraction_plan.md` (Ocean-scoped).
- **Validation kernels present:** 4-module `tests/icon/graupel/aes_graupel/*.f90` (driven via `build_sdfg_from_files(..., entry="graupel_run")`), ICON-Ocean under `tests/icon/full/icon-model/src/ocean`, QE exx_bp under `tests/qe/exx_bp/`.

## Design decisions
1. fparser engine **side-by-side, opt-in**; keep `merge_used_modules` as default (`module_merge_test.py` stays byte-stable). Selector `merge_engine={"regex","fparser"}`.
2. Public API: `inline_to_single_tu(sources, entry, *, search_dirs=(), include_dirs=(), keep_external=(), expand_macros=True, out=None) -> Path|str`; `entry` resolved via the shared `_resolve_entry` convention so it composes with `build_sdfg(..., entry=…)`.
3. Repo rules: no leading-underscore class names; no getattr/hasattr; no hardcoded test paths (`tests.` package imports); yapf col-120 + ruff before commit; reuse `strip_openmp_directives`/`normalize_kind_parameters`/`_balance_cpp`/`_module_blocks`.

## WP1 — fparser engine + dependency
- ADD `dace_fortran/fparser_inliner.py` (import Fortran2003 before Fortran2008, cite 74110d690).
- CHANGE `pyproject.toml:28` add **`fparser==0.2.3`** (verify node-class names: `Use_Stmt`, `Only_List`, `Rename`, `Module_Stmt`, `Derived_Type_Def`, `Type_Declaration_Stmt`, `Entity_Decl`, `Component_Decl`).
- CHANGE `dace_fortran/__init__.py` lazy-export `inline_to_single_tu`.
- CHANGE `preprocess.py:1408` `preprocess_fortran_source` add `merge_engine` param.
- Algorithm: (1) entity-level **index** (name→(module,kind,node)) over search_dirs, finer than module-level; (2) resolve entry → its `USE`/`ONLY:`/`=>`; (3) transitive **reachability closure** (params/module-vars read, derived types recursively, helper procs actually called, stop at `keep_external` + intrinsic modules at preprocess.py:635); (4) emit one `.f90`, params+types before procedures, deps before dependents, strip PRIVATE/PUBLIC, `strip_openmp_directives`.
- CLI: `python -m dace_fortran.fparser_inliner --entry … --src … --search-dir … --include-dir … --keep-external … --out …`; also `preprocess_cli.py --merge-engine {regex,fparser}`.
- **Edge cases (each → a WP2 fixture):** ONLY:/rename; cross-module name collision (deterministic rename `<name>__from_<module>` + rewrite refs); derived types survive intact (FlattenStructs handles later); generic interfaces (keep + specifics); PRIVATE/PUBLIC honored-then-stripped; INCLUDE/DSL `.inc` cpp macros (expand via `flang -E`/`-cpp -I include_dirs -D` BEFORE parsing; fallback macro pre-pass); cpp `#ifdef` twins (select one arm via `defines=`, reuse `_balance_cpp`); `USE`-without-`ONLY` (reachability prunes); USE cycles (drop back-edge).

## WP2 — own test subfolder
- ADD `tests/inliner/{__init__.py,test_*.py,fixtures/<feature>/}` (mirror `_proj()` in `tests/module_merge_test.py:94`, `tests/external_to_modules/` convention).
- Fixtures: only_clause/ (THE differentiator vs regex), rename/, collision/, nested_types/, helper_proc/, keep_external/, dsl_macro/, cycle/, generic_interface/, private_public/.
- Gates: parse (→ HLFIR), numeric (`build_sdfg_from_files` vs `f2py_compile`, rtol/atol 1e-12), **equivalence** (fparser vs regex SDFG numerics equal on the `module_merge_test` synthetic fixtures).

## WP3 — round-trip on real kernels (the headline acceptance)
- **WP3a graupel (do FIRST):** `tests/icon/graupel/test_aes_graupel_inline_roundtrip.py` — `inline_to_single_tu(_GRAUPEL_SOURCES, entry="graupel_run")` → single TU → build → **numerics == the multi-file path** (`test_aes_graupel_build.py:52` baseline, reuse `test_aes_graupel_numerical_correctness.py` harness).
- **WP3b ICON-Ocean** (generalize `docs/icon_ocean_fparser_extraction_plan.md`): PPM `upwind_vflux_ppm_onBlock` (in-file vector oracle, pass 9 ppmCoeffs as 2-D args) → tridiag → Zalesak (keep_external sync_patch_array_mult) → Coriolis (variable-length gather). Maturity ladder: parse→build→CPU-numeric→GPU(deferred); land what passes, xfail the rest; `!$ACC` ignored.
- **WP3c QE exx_bp:** parse-stage test (collisions + EXTERNAL stress); numeric out of scope.

## WP4 — README + parity
- Document the two engines + when to use which + selector default (regex); the `inline_to_single_tu` API/CLI + checked-in Ocean artifacts + regen command; the `keep_external` + single-rank no-op-stub convention; the `.inc` macro requirement.
- **Python-frontend parity:** same fparser surface as d-face's frontend so a future merge doesn't fight over versions; confirm 74110d690 import fix in both.

## Sequencing / DoD
Phase 0 (pin fparser 0.2.3 + py3.14 fix) → WP1 → WP2 → **WP3a graupel round-trip (headline)** → WP3b Ocean → WP3c QE parse → WP4. Top risks: DSL `.inc` macros (cpp pre-expand), collision mis-rename → silent miscompile (compile gate + logged renames), Coriolis nested gather (may need bridge work), fparser version skew (verify 0.2.3 node names). DoD: `fparser_inliner.py` + pin + selector + CLI; `tests/inliner/` green; **graupel round-trip green**; Ocean PPM+tridiag stage-3, Zalesak/Coriolis ≥ stage-2; QE parse green; README updated; nothing pushed until suite green locally.
