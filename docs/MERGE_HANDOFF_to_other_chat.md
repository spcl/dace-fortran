# Merge handoff — `worktree-external-call-policy` → main (for the other chat)

Branch **`worktree-external-call-policy` @ `0d12140`** — 5 commits ahead of merge-base (`31ea011`); main is 6 ahead. Merge produces **7 conflicts**; **1 needs judgement**: `monomorphize_rewrite.py` (my `stack_slots` vs your analyzer integration). `bind_c_shim.py`/`.gitignore` auto-merge (no markers). `block_builders.py` — no conflict (untouched by you).

---

## What I changed and why

### 1. Bindings (your domain — explicitly authorised)
- **`block_builders.py`** — added `_struct_member_symbol_sources(iface)` + last-resort lookup in `_build_symbol_assigns`. Why: a struct member used ONLY symbolically (loop bound `dfftt%ngm`, extent `size(dfftt%nl_d)`) lifts to a free SDFG symbol with NO `FlattenEntry` → plan-driven paths missed it → defaulted 0 → QE **vexx** OOB. Now sourced from the static `struct_types` layout. **Merges clean** (main untouched here).
- **`bind_c_shim.py`** — added `_free_shape_symbols(iface)` + forwarding block in `emit_bind_c_shim`. Why: flat array dummy `tracer(nproma, n_zlev)` left `nproma`/`n_zlev` undeclared in the shim → gfortran `Symbol 'n_zlev' has no IMPLICIT type`. Now forwarded as `integer(c_int), value` C args (prepended, deduped). **Auto-merges** with your `module_symbol_forward` work (different regions) — sanity-check header arg order still looks right.

### 2. ICON-O ocean (overlaps your ocean rework)
- New `tests/icon/ocean/_ocean_e2e.py` + `test_ocean_numerical_e2e.py` — **ppm_vflux full numerical e2e**: auto-generated `bind(c)` binding vs the ORIGINAL kernel (same shim *retargeted* to call the kernel). Each kernel in its own subprocess; DUT/REF each in an `os.fork()` child (avoids a nondeterministic dual-`.so` heap crash). `coriolis_pv`/`veloc_adv_horz` are `xfail` (flatten-structs bug, see bug report).
- New `_monomorphize_solver.py` + `test_monomorphize_solver.py` + committed artifact `dycore_solver_monomorphized.f90` — the real `t_ocean_solve` subsystem run through the monomorphisation engine (0 `fir.dispatch`) + a byte-drift guard.

### 3. dycore CI fix
- **`setup_icon_dycore.sh`** — export `LIBS` dropping `-leccodes`. Why: ICON's config wrapper hardcodes `-leccodes` in `LIBS`; autoconf links its "compiler works" probe with `$LIBS`; GRIB2 disabled + eccodes not installed → `Fortran compiler cannot create executables`. **Likely clean.**

### 4. docs
- `docs/bug_reports/flatten_structs_scalar_read_of_allocatable_member.md` — the bridge pass bug blocking coriolis/veloc.
- `docs/bug_reports/build_fortran_library_name_vs_sdfg_name_divergence.md`.
- This file.

---

## Conflict map (per file)

| file | conflict | how to resolve |
|---|---|---|
| `bindings/bind_c_shim.py` | auto-merged | accept; spot-check `_free_shape_symbols` prepend vs your forwarding |
| `.gitignore` | auto-merged | drop my `!tests/icon/ocean/dycore_solver_monomorphized.f90` line — your `!tests/icon/ocean/**/*.f90` already covers it |
| `bindings/block_builders.py` | none | — |
| `inliner/ast_desugaring/analysis.py` | content (2) | **take yours (main)** — your domain; my side only carries the prior-merge state |
| `inliner/ast_desugaring/desugaring.py` | content (1) | **take yours (main)** |
| `inliner/ast_desugaring/monomorphize_rewrite.py` | **13 — needs you** | take YOUR base, re-apply my `stack_slots` delta (below) |
| `tests/inliner/monomorphize_*` | (yours) | take yours; then add my `monomorphize_sdfg_e2e_test.py` if not present |
| `tests/icon/ocean/_ocean_harness.py` | add/add (6) | **combine**: your `KERNELS` + `ocean_veloc_adv` naming + the `have_icon_ocean` all-present check, PLUS my `OCEAN_EXTERNAL_FUNCTIONS` / `OCEAN_DO_NOT_EMIT` external-policy block + `ExternalFunction` import |
| `tests/icon/ocean/_extract_single_tu.py` | add/add (2) | combine: pass my `external_functions=`/`do_not_emit=` through to `inline_to_single_tu` on top of your version |
| `tests/icon/ocean/test_extract_single_tu.py` | add/add (1) | take yours (drives off `KERNELS`) |
| `tests/icon/ocean/coriolis_pv_single_tu.f90` | add/add (2) | **take yours** (canonical extraction); my ocean e2e just reads it |
| `veloc_adv_horz_single_tu.f90` (mine) vs `ocean_veloc_adv_single_tu.f90` (yours) | rename | keep yours; `git rm` mine; then in `test_ocean_numerical_e2e.py` rename the `veloc_adv_horz` param → `ocean_veloc_adv` + point it at your file/entry |

---

## The one that needs your judgement: `monomorphize_rewrite.py`

Both sides already have `monomorphize(program, spec)` + `AxisSpec`/`MonomorphizationSpec`. Divergence: **I added `stack_slots` (21 refs)**; you added analyzer integration. Take **your** version as base, re-apply my `stack_slots` delta:

- `monomorphize(program, spec, stack_slots=False)` — new 3rd param, forwarded to `monomorphize_local_dispatch(..., stack_slots)` and `monomorphize_component_dispatch(..., stack_slots)`.
- `_expanded_decls(var, plan, stack_slots=False)` / `_expanded_component_decls(name, plan, stack_slots=False)`: `attr = "" if stack_slots else ", allocatable"` on the per-arm slot/component decl.
- `_allocation_rewrite(..., stack_slots)`: when `stack_slots`, factory sets ONLY the tag (`<slot>__tag = k`), no `allocate(...)`.
- component-dispatch factory: `rewrite = set_tag if stack_slots else set_tag + "\nallocate(...)"`.

**Why it must survive the merge:** the bridge cannot lower an *allocatable* derived-type scalar (HLFIR rejects `hlfir.declare` on it). `stack_slots=True` makes each per-arm slot a plain `type(arm) :: slot` (no `allocatable`) → SDFG-lowerable. My SDFG e2e (`tests/inliner/monomorphize_sdfg_e2e_test.py`) calls with `stack_slots=True`; default `False` keeps the faithful/dispatch-count tests unchanged. Dropping `stack_slots` breaks those e2e tests.

---

## Verify after merging
- `tests/bindings/` (135) — my two binding fixes.
- `tests/icon/ocean/test_ocean_numerical_e2e.py::...[ppm_vflux]` + `test_monomorphize_solver.py`.
- `tests/inliner/monomorphize_*` — your analyzer + my engine/e2e together.
- Confirm the `veloc` param rename landed and `veloc_adv_horz_single_tu.f90` is gone.

---

## What I did (done + verified on my branch)
- **block_builders struct-member symbol sourcing** — closes the vexx OOB. 4 unit tests (`tests/bindings/struct_member_symbol_source_test.py`) + full `tests/bindings/` (135) green.
- **bind_c_shim module-var-extent forwarding** — ocean array shapes resolve. 2 unit tests in `bind_c_shim_test.py`; verified the real ppm binding `.so` builds + loads.
- **ppm_vflux full numerical e2e** — SDFG-via-binding == original kernel, bit-close, output non-trivial; passes reliably (fork-isolated). coriolis/veloc are `xfail` (see open items).
- **De-polymorphed solver** `dycore_solver_monomorphized.f90` (0 dispatch) + drift test — 2 green.
- **dycore CI fix** — root-caused (`-leccodes` in autoconf's link probe) + verified the failure mode and that every retained lib is in the CI apt list.
- Two bug reports + this handoff. Committed: `0d12140` (work) + `782125d` (this doc).

## Open items
1. **`hlfir-flatten-structs` bug (bridge/C++)** — blocks lowering coriolis_pv + veloc_adv_horz; their e2e are `xfail(strict=True)`. Root-caused + bisected; see `docs/bug_reports/flatten_structs_scalar_read_of_allocatable_member.md`. Needs a `FlattenStructs.cpp` fix + bridge rebuild + full struct-suite regression. **Not started** (core pass — flag for design discussion before patching). Fixed → drop the two `xfail` marks in `test_ocean_numerical_e2e.py`.
2. **`monomorphize_rewrite.py` `stack_slots` re-apply** — the one merge item needing your hand (section above). Without it my SDFG-e2e monomorphisation tests break.
3. **Post-merge ocean rename** — adopt your `ocean_veloc_adv`; update my e2e `veloc` param + delete `veloc_adv_horz_single_tu.f90`.
4. **`build_fortran_library` name-vs-`sdfg.name` divergence** — silent `undefined symbol` footgun; a guard/auto-derive/doc fix in `build_fortran_library.py` (your domain). See `docs/bug_reports/build_fortran_library_name_vs_sdfg_name_divergence.md`. Worked around by callers for now (use `sdfg.name`).

## Non-issues (resolved / no action)
- "binding should convert any `logical(X)` → `c_bool`": the binding ALREADY uses `c_bool` throughout; the kind-4 coercion is only needed when calling the *raw* kernel (the reference shim), handled in `_ocean_e2e._retarget_shim` via `logical(arg)`.
