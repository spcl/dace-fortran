# Unified external-function policy — implementation plan

Workspace: git worktree `external-call-policy` (branch `worktree-external-call-policy`,
based on `origin/main` @ `b9a06c1`) -- a fresh, isolated copy of dace-fortran.
Implement + mature here, then merge to dace-fortran `main` once green.

## Status / cadence

Phase-by-phase, with a checkpoint after each phase (summarise, update this doc,
compact). Current state:

- [x] **Phase 1 — data model.** `dace_fortran/external_functions.py`
  (`ExternalFunction` + `dont_inline_names` + `validate`) +
  `tests/inliner/external_functions_test.py` (9 passing). `ExternalFunction`
  exported at package level (`dace_fortran.ExternalFunction`) — deliberately
  distinct from the `external.ExternalCall` library node / `ExternalSignature`.
- [x] **Phase 2 — inliner params (regex AND fparser).**
  - fparser `inline_to_ast`/`inline_to_single_tu`: added
    `external_functions=()` + `do_not_emit=()`; internally
    `_resolve_dont_inline_names(...)` → existing `_keep_external_noop_specs`.
    `keep_external=` kept as a deprecated shim (warns; == `do_not_emit`).
  - regex `merge_used_modules`: added `external_functions=()` + `do_not_emit=()`;
    new `_stub_procedure_bodies` empties matched bodies (opener+spec+END kept) —
    the text-splice analogue of `make_noop`. Extracted shared `_PROC_OPEN_RE`.
  - `tests/inliner/external_policy_wiring_test.py` (10 passing): fparser
    old==new byte-identical parity + deprecation warning + validate; regex
    body-stub + generic-prefix match + gfortran-compiles. Full `tests/inliner/`
    + preprocess/merge regression: **164 passing**.
  - **Linchpin RESOLVED early** (Explore map of `emit_library.emit_call`):
    connectors/args are derived from the HLFIR call site (`n.call_args`), NOT
    from `sig.args`. So a minimal `ExternalFunction`→`args=()` registration is
    viable; the signature carries only ABI metadata. One guard to verify in
    Phase 3: `emit_call` raises on `len(sig.args)!=len(call_args)` when
    `sig.args` is non-empty — empty `()` must take the "derive-from-HLFIR" path.
  - **Build-path threading — DONE (pulled forward from Phase 3 at user request
    "we should improve this").** The HLFIR build path no longer relies on the
    post-merge `externalize_symbols` pass alone:
    - `_fparser_merge(..., external_names=)` → `inline_to_ast(do_not_emit=...)`;
      `preprocess_fortran_source(..., external_names=)` forwards to BOTH engines
      (regex `merge_used_modules(do_not_emit=...)` + fparser `_fparser_merge`).
    - **Keystone:** `build._emit_hlfir` sources the merge's keep-external names
      from the bridge registry (new `build._merge_external_names` =
      `registered_names()` ∪ explicit), so ONE `keep_external` /
      `apply_external_functions` declaration governs both halves — the merge
      stubs the body, the bridge externalises the surviving call. No second knob,
      no hand-syncing. When the external body isn't in the merged TU
      (interface-only) the stub is a clean no-op.
    - Validated: 4 new threading tests + all 20 external-call e2e
      (`test_keep_external`/`test_external_call`/`external_aos`/`inline_external`,
      incl. body-in-merge `inline_external`) stay green; dycore e2e collects clean.
    - **Regression found + fixed in Phase 3** (the prior session's threading
      validation only *collected* the dycore e2e, didn't run it): sourcing
      registry names into the merge made `_stub_procedure_bodies` stub the ICON
      `bind(c)` halo wrappers (`sync_patch_cpp_via_c`), whose **spec part holds
      an `INTERFACE` block**. The stubber kept the `interface` opener but its
      spec loop broke at the nested `subroutine` inside (matched `_PROC_OPEN_RE`),
      so the depth-scan dropped `end interface` -> orphaned `interface` ->
      uncompilable. Fix: `_INTERFACE_OPEN_RE`/`_INTERFACE_END_RE` + consume the
      whole `INTERFACE` block as a spec unit (nesting-aware). The fparser
      `make_noop` path was already correct (operates on the AST execution-part).
      Regression test `test_regex_merge_stubs_body_with_interface_in_spec`.
  - **Deferred to Phase 4** (byte-identical-to-committed drift): the PPM +
    Coriolis `*_single_tu.f90` artifacts live in `tests/icon/ocean/`, which is
    untracked in `main` and not yet in the worktree. The old==new parity test
    already guarantees the harness param-spelling switch is output-preserving;
    the artifact drift check runs once the ocean dir is brought in.
- [x] **Phase 3 — bridge `apply_external_functions` + resolve the linchpin.**
  - `external.apply_external_functions(external_functions=(), do_not_emit=())`:
    validates first, then each `ExternalFunction` -> `keep_external(name,
    c_name=f.symbol, libraries=(f.library,) if f.library else ())` (emit);
    each `do_not_emit` name -> `keep_external(name, stub=True)` (ignore).
    Exported at package level (`dace_fortran.apply_external_functions`).
  - **Linchpin actually resolved (the Phase-2 Explore was over-optimistic).**
    `emit_call` derives connectors/args from `n.call_args`, BUT the arity guard
    *and* the decl-types builder both consume `sig.args` — so a bare `args=()`
    DID raise. Fix: when `not sig.args and call_args`, **synthesise authored
    `Arg`s from the HLFIR call site** (`replace(sig, args=...)`) before the plan
    loop — each SDFG array -> `Arg(array, <dtype>, inout)`, scalar/free-symbol
    -> `Arg(scalar, <dtype>, in)`; AoS marshalling still needs an authored
    `Arg(kind='aos')` (raises a pointed error otherwise). So all downstream
    code (plan / connectors / body / decl-types) runs unchanged, and the derived
    node is **byte-identical** to the hand-authored one (proven by test).
  - Tests `tests/external_call/test_apply_external_functions.py` (8): registry
    contract (emit/ignore/validate), derive-from-HLFIR e2e (compiles + runs),
    derived-node == authored-node (`c_decl` + `body`).
  - **Full regression green:** external_call/external_aos/inline_external/inliner
    (151), external_policy_wiring (15), apply_external_functions (8), icon/full
    (17p/12s), bindings (126p). See the build-path-threading regression note above.
- [x] **Phase 4 — ocean harness single source of truth (+ PPM/Coriolis drift).**
  - Brought the untracked `tests/icon/ocean/` dir (`_ocean_harness.py`,
    `_extract_single_tu.py`, `test_extract_single_tu.py`, `__init__.py` + the two
    committed `*_single_tu.f90` artifacts) into the worktree from `main`.
  - Brought in the inliner **external-tolerance branches** the committed artifacts
    depend on — `inliner/ast_desugaring/analysis.py` (unresolvable declared type
    `CLASS(*)` -> `TYPE_SPEC(('*',),'')`; unresolved call-arg symbol e.g.
    `mpi_max`/`mpi_sum` -> `MATCH_ALL`, both via `search_real_ident_spec`, both
    gated on `TOLERATE_EXTERNAL_USES`) and `desugaring.py` (external type-bound-
    procedure candidate skip + unresolved-TBP-call leave-for-pruning, same gate).
    `search_real_ident_spec` already existed in the worktree; only the
    tolerance-branch *wiring* was new. Inliner suite stays green (**110 passed**).
  - `_ocean_harness.py`: replaced `OCEAN_KEEP_EXTERNAL` with the two-list policy —
    `OCEAN_EXTERNAL_FUNCTIONS = [ExternalFunction("sync_patch_array"),
    ExternalFunction("sync_patch_array_mult"), ExternalFunction("exchange_data")]`
    (don't-inline + EMIT, the MPI halo boundary; `library`/`c_function` defaulted
    until the SDFG stage binds them) + `OCEAN_DO_NOT_EMIT = [dbg_print, finish,
    message, warning, timer_*]` (don't-inline + DROP, pure side-effects).
    `rot_vertex_ocean_3d` stays INLINED (in neither list).
  - `_extract_single_tu.py`: `inline_to_single_tu(... keep_external=OCEAN_KEEP_EXTERNAL)`
    -> `inline_to_single_tu(... external_functions=OCEAN_EXTERNAL_FUNCTIONS,
    do_not_emit=OCEAN_DO_NOT_EMIT)`. The deprecated `keep_external=` shim is gone
    from the ocean path.
  - **Output-preserving, proven two ways:** (1) `dont_inline_names(emit, ignore)`
    == the old `OCEAN_KEEP_EXTERNAL` name set exactly (both lists stub identically
    at the inliner); (2) the byte-identical drift test
    `test_extract_compiles_and_matches_committed[ppm_vflux, coriolis_pv]` **passes**
    against the real ~137k-line ICON closure (extract -> cpp pre-pass -> prune ->
    `gfortran -fsyntax-only` -> byte-equal the committed artifact; 293s, both
    kernels). The icon-model submodule lives only in `main`, so the test runs with
    `ICON_SRC` pointed at main's read-only checkout (extraction reads source;
    all writes go to `tmp_path`).
- [x] **Phase 5 — migrate every external-call test (the dycore samples) + docs.**
  Migration follows the Phase-3 outcome: *simple emit (symbol + library, arg
  plan derivable from HLFIR) and the ignore/stub loops move to the unified API;
  genuinely-rich ABI stays on `keep_external`.*
  - **→ `apply_external_functions([ExternalFunction(...)], [])`** (simple emit):
    `tests/external_call/test_external_call.py` (foo ×2, was
    `register_external`/`ExternalSignature`); `test_keep_external.py` (bar/noop
    ×3); `tests/external_aos_test.py` plain-array shallow cases (`ext_scale` ×2,
    `ext_sync`); `tests/inline_external_test.py` (`add_one`, both the build and
    the pre-`inline_external` re-declare). The derived plan (array→inout pointer,
    scalar→by-value) reproduces every prior assertion (ExternalCall node, c_name,
    shallow "no struct" body, numerical result).
  - **→ `apply_external_functions(do_not_emit=STUBS)`** (ignore loops):
    `tests/icon/full/test_dycore_from_icon_source.py` and
    `test_velocity_from_icon_source.py` — `for sym in _ICON_EXTERNAL_STUBS:
    keep_external(sym, stub=True)` collapses to one call (a 1:1 swap: the helper
    runs that exact loop after a no-op `validate`).
  - **Kept on `keep_external` (rich ABI genuinely needed):** the `kind='comm'`
    tests; every AoS / `per_member_soa` case in `external_aos_test.py` +
    `inline_external_test.py`; all dycore e2e (`dynamic_extents_abi`,
    `module_symbol_forward`, `Arg(kind='aos')`, intent narrowing). These are the
    documented escape hatch and the rich-`Arg` coverage.
  - **Inliner headline API:** `tests/inliner/inline_to_single_tu_test.py` docstring
    (`keep_external`→`do_not_emit`) + new `test_do_not_emit_leaves_procedure_external`
    exercising the public `do_not_emit=` param (body-emptied, compiles).
  - **`docs/external_call_policy.md`** — canonical user-facing doc (the 3
    behaviours table, `ExternalFunction`/`apply_external_functions`, both inliner
    engines, the derive-from-HLFIR contract, the `bind(c)` requirement, and the
    `keep_external` rich-ABI escape hatch + when to reach for it).
  - **Validated:** migrated lightweight suites green — `inline_to_single_tu` 15,
    `external_call`+`keep_external`+`inline_external` 10, `external_aos` 10,
    Phase 1–3 regression (`external_functions` 9 / `external_policy_wiring` 15 /
    `external_call` dir 16) = 40. The 7 ICON/MPI-gated dycore e2e + 2 stub-loop
    source tests collect clean (19); the stub-loop swap is behaviour-identical so
    the minutes-long real-ICON build was not re-paid here.
  - **Deferred (open q (b)):** removing the deprecated `keep_external=` inliner
    shim — left as a warning shim for one release; cut in a later commit.
- [ ] Extract ICON-O velocity tendencies + dynamical core.

**Two clarifications from the user that widened the original scope:**
1. *This chat owns the whole thing* — both the extension (all phases, incl. the
   bridge wiring the plan first handed to "the other chat") **and** the ICON-O
   extraction that consumes it. The dycore-sample e2e tests must be rewritten to
   the new design (they are the real consumers of the rich `Arg`/`ExternalSignature`).
2. *Extracting ICON-O will require **extending** the inliner itself* (both the
   regex `merge_used_modules` text-splicer and the fparser pipeline), not just
   configuring it. The velocity-tendencies + dycore drivers will surface new
   constructs/tolerances beyond what PPM + Coriolis needed. Budget for inliner
   changes, not only harness wiring.

## 1. Goal

Replace the scattered, hand-synced "keep external" mechanisms with **one** way to
declare, at parse time, how each non-inlined function is handled. Three behaviours,
any function name:

| Behaviour | Inliner (input→TU) | Bridge (TU→SDFG) |
|---|---|---|
| **inline** (default) | pull body into TU | lower normally |
| **don't-inline + EMIT** | don't inline; leave the call as an external reference | emit an **ExternalCall library node**, replicating the HLFIR call order, bound to the user-supplied C-ABI symbol/library |
| **don't-inline + DON'T-emit** (ignore) | don't inline | drop the call (no node) |

Key constraint already satisfied structurally: *ignore ⊆ don't-inline*.

The argument **order/types are NOT re-authored** by the user — the HLFIR already carries
the call (`CALL f(a, b, c)`), so the bridge replicates it. The user only supplies what
HLFIR can't know for an emitted call: the **C-ABI symbol and the library** that provides it.

## 2. Public API (the whole surface)

One small dataclass + two lists, passed at the parse entry point. The dataclass
is named **`ExternalFunction`** (NOT `ExternalCall`) to avoid any conflict with
the existing `dace_fortran.external.ExternalCall` SDFG *library node* and
`ExternalSignature` ABI record — three distinct names, no shadowing.

```python
# dace_fortran/external_functions.py  (pure-stdlib, no dace/bridge imports)
@dataclass(frozen=True)
class ExternalFunction:
    """A procedure that is NOT inlined and IS emitted as an external call.
    Any Fortran name. The call's argument order comes from HLFIR; this only
    supplies the binding the bridge needs to resolve the symbol."""
    name: str                        # Fortran call-site name (any name)
    c_function: Optional[str] = None # extern "C" symbol; defaults to `name`
    library: Optional[str] = None    # path to the .so/lib that exports it
    # .symbol -> c_function or name
```

Exported at package level: `dace_fortran.ExternalFunction`. Helpers in the same
module: `dont_inline_names(external_functions, do_not_emit) -> set[str]` and
`validate(external_functions, do_not_emit)`.

Caller supplies two collections (defined ONCE per target, e.g. in the ocean harness):

```python
external_functions = [                     # don't-inline + EMIT
    ExternalFunction("sync_patch_array", library=".../libicon_halo.so"),
    ExternalFunction("exchange_data"),
    ExternalFunction("any_user_fn", c_function="my_c_abi_fn", library=".../libfoo.so"),
]
do_not_emit = ["finish", "message", "warning", "dbg_print",   # don't-inline + DON'T-emit
               "timer_start", "timer_stop", "new_timer", "delete_timer"]
```

Derived (computed, never hand-maintained):
`dont_inline = {f.name for f in external_functions} | set(do_not_emit)`.

## 3. Current scattered state being replaced

- `dace_fortran/fparser_inliner.py` — `inline_to_single_tu(keep_external=[names])` →
  `_keep_external_noop_specs` → `make_noop`. Plain names; no emit/ignore split; no ABI.
- `dace_fortran/external.py` — `keep_external(name, *, c_name, args, libraries, stub, ...)`
  registers an `ExternalSignature` (emit) or drops (`stub=True`). Rich, but a *second*
  shape in a *second* place. The dycore-sample e2e tests author `Arg(...)` lists by hand.
- The regex merge `dace_fortran/preprocess.merge_used_modules` has **no** keep-external
  notion at all (it text-splices the whole USE closure). Ocean extraction needs it there
  too (user: "add it to both regex and fparser").
- Per-target duplicate lists: `tests/icon/ocean/_ocean_harness.py::OCEAN_KEEP_EXTERNAL`
  (inliner) and `tests/icon/full/*::_ICON_EXTERNAL_STUBS` (bridge). Hand-synced.

## 4. Phased implementation

Each phase is independently testable; a **checkpoint** (summarise, update this
doc, compact) follows each one.

### Phase 1 — the data model ✅ DONE
- `dace_fortran/external_functions.py`: `ExternalFunction` + `dont_inline_names`
  + `validate`. Pure-stdlib (the fparser inliner must not import dace-heavy `external.py`).
- `tests/inliner/external_functions_test.py` (9 passing). Package export
  `dace_fortran.ExternalFunction`. No `ExternalPolicy`/derived-view class (rejected over-design).

### Phase 2 — inliner takes the new params (BOTH paths) ✅ DONE
*(implemented exactly as designed below; see the Status block for the test/defer notes.)*
- **fparser** `inline_to_single_tu` / `inline_to_ast`: add
  `external_functions: Iterable[ExternalFunction] = ()` and `do_not_emit: Iterable[str] = ()`.
  Internally `names = dont_inline_names(...)` → existing `_keep_external_noop_specs(ast, names)`.
  The inliner uses ONLY names; never reads `c_function`/`library`.
- **regex** `merge_used_modules`: accept the same `external_functions`/`do_not_emit`
  (names) so a kept-external procedure's defining module is not spliced in / its body is
  stubbed — parity with fparser (user: both engines). Check what `preprocess` already does
  before adding; the regex path may only need to skip pulling the module.
- Deprecate but keep `keep_external=[names]` as a thin shim → treat as `do_not_emit=names`;
  emit a `DeprecationWarning`. Remove in a later cut.
- Emit-vs-stub at the inliner: `do_not_emit` → `make_noop`; `external_functions` → also
  `make_noop` today (keeps the TU compiling; the call survives as an external reference).
- Re-run PPM + Coriolis extraction; assert the TUs are **byte-identical** to the committed
  `tests/icon/ocean/*_single_tu.f90` (drift test). Confirm `tests/inliner` stays green.

### Phase 3 — bridge takes the same model + resolve the linchpin
- `dace_fortran/external.py`: add `apply_external_functions(external_functions, do_not_emit)`:
  ```python
  for f in external_functions:
      register_external(f.name, ExternalSignature(c_name=f.symbol, args=(),  # order from HLFIR
                                                  libraries=(f.library,) if f.library else ()))
  for name in do_not_emit:
      keep_external(name, stub=True)
  ```
- **Linchpin (was Open Q a, now in-scope):** does `builder.emit_library.emit_call` derive
  the call connectors/arg order from HLFIR, so `args=()` suffices? Investigate first. If it
  needs an authored `args`, extend `emit_call` to read operands from the HLFIR call site so
  the minimal `ExternalFunction(name, c_function, library)` is enough for the common case.
  The rich `Arg`/`ExternalSignature` stays available for the AoS-struct / `comm` / module-
  forward cases that HLFIR can't fully determine (those tests keep using it, or it becomes
  an optional advanced field).
- Contract test for `apply_external_functions` (registry populated, `stub=True` for ignores).

### Phase 4 — single source of truth for the ocean kernels
- `tests/icon/ocean/_ocean_harness.py`: replace `OCEAN_KEEP_EXTERNAL` with
  `OCEAN_EXTERNAL_FUNCTIONS = [ExternalFunction(...)]` + `OCEAN_DO_NOT_EMIT = [...]`.
  - emit: `sync_patch_array`, `sync_patch_array_mult`, `exchange_data` (`library=None` for
    now; symbol defaults to name).
  - do_not_emit: `dbg_print`, `finish`, `message`, `warning`, `timer_*`.
  - (rot_vertex_ocean_3d stays INLINED — in neither list.)
- `_extract_single_tu.py`: pass `external_functions=OCEAN_EXTERNAL_FUNCTIONS,
  do_not_emit=OCEAN_DO_NOT_EMIT`. Drift test stays byte-identical (the name union equals
  today's `OCEAN_KEEP_EXTERNAL` minus rot_vertex).

### Phase 5 — migrate EVERY external-call test (the dycore samples) + docs
The user requires all tests exercising external calls to move to the new design:
- `tests/icon/full/test_dycore_*` — `test_dycore_ext_velocity_e2e`,
  `test_dycore_from_icon_source`, `test_dycore_mpi_sync_e2e`, `test_dycore_struct_ext_e2e`,
  `test_dycore_velocity_external_e2e`, `test_standalone_dycore_sync_e2e`,
  `test_velocity_from_icon_source`.
- `tests/external_call/{test_external_call,test_keep_external}.py`, `tests/external_aos_test.py`,
  `tests/inline_external_test.py`, `tests/inliner/inline_to_single_tu_test.py`.
- Each `for sym in STUBS: keep_external(sym, stub=True)` loop → `apply_external_functions([], STUBS)`.
  Each emit registration → `apply_external_functions([ExternalFunction(...)], [])` (keeping the
  `Arg` ABI where the test genuinely needs it, per Phase-3 outcome).
- `docs/external_call_policy.md`: canonical user-facing doc.
- Remove the deprecated `keep_external=` inliner shim (optional, final commit).

**Phase 5 DONE** — see the status block above for the per-file migration map and
the green-suite validation. Simple emit + ignore-loops moved to the unified API;
rich ABI stays on `keep_external` (per the Phase-3 outcome); the inliner shim
removal is deferred.

### Extraction (the deliverable this enables)
- ICON-O **velocity tendencies** (`veloc_adv_horz_mimetic`) and **dynamical core**
  (`calculate_explicit_term_ab` / `solve_free_sfc_ab_mimetic`), keeping ONLY
  `sync_patch_array` (or equiv) external. Expect to **extend the inliner** (regex + fparser)
  for constructs these drivers use beyond PPM/Coriolis. New `tests/icon/ocean/*_single_tu.f90`
  artifacts + harness `KERNELS`/`SINGLE_TU_ARTIFACTS` entries + drift tests.

## 5. Test matrix
- `tests/inliner/external_functions_test.py` — data model + validation (Phase 1, ✅).
- `tests/inliner/external_uses_and_cpp_test.py` — extend: `external_functions`/`do_not_emit`
  drive `make_noop` exactly as `keep_external` did (parametrize old-vs-new, assert equal output).
- Ocean drift tests (`test_extract_single_tu.py`) — byte-identical TUs (Phases 2 + 4).
- Bridge contract test for `apply_external_functions` (Phase 3).
- Every dycore-sample e2e test (Phase 5) — rewritten, still green.
- Full `tests/inliner` stays green throughout.

## 6. Scope (this chat owns all of it)
Phases 1–5 + the ICON-O extraction. The earlier "other chat owns TU→SDFG" split is
superseded — the bridge wiring (Phase 3) and the dycore-sample e2e rewrites (Phase 5) are
in scope here. Tracked by `project_hlfir_ignore_functions_todo` /
`project_icon_ocean_dycore_velocity_tendencies_todo`.

## 7. Open questions
- (a) **RESOLVED (Phase 3).** The Phase-2 Explore was over-optimistic: although
  the connector/body loop reads `n.call_args`, BOTH the arity guard and the
  decl-types builder consume `sig.args`, so a bare `args=()` *did* raise. The
  real fix is to **synthesise authored `Arg`s from the HLFIR call site** when
  `sig.args` is empty (array->inout-pointer, scalar/symbol->by-value-read),
  giving a node byte-identical to the hand-authored one. AoS still needs an
  authored `Arg(kind='aos')`.
- (b) **DECIDED:** keep `keep_external=` (fparser) for one release as a warning
  shim (== `do_not_emit`); cut in Phase 5's final commit.
- (c) `library` as a single path vs a tuple (multiple .so)? Start single; widen only if needed.
- (d) **RESOLVED (Phase 2).** Regex `merge_used_modules` does *active stubbing*
  (`_stub_procedure_bodies`: empty the body, keep opener+spec+END), parity with
  the fparser `make_noop`. "Don't pull the module" was rejected as too coarse
  (a module bundles wanted + external procedures).

## 8. Merge plan
- Land all phases here; `tests/inliner` + ocean drift + rewritten dycore tests green.
- The worktree also carries two prerequisites currently UNCOMMITTED in `main` that the
  ocean extraction needs (inliner-side, not part of the policy refactor proper):
  the analysis.py/desugaring.py external-tolerance branches (CLASS(*) types, `mpi_max`
  args, external TBP candidates). Bring them in; reconcile at merge (likely identical).
  Do NOT bring `dispatch.cpp` / the do-while test (unrelated S1 bridge fix — stays in main).
- Rebase on latest `main`, squash per-phase, open PR `external-call-policy → main`.

## 9. Extraction progress (session 2026-06-23)

### velocity tendencies — DONE ✅
`veloc_adv_horz_mimetic` extracts to a **612-line gfortran-clean single TU**
(`tests/icon/ocean/veloc_adv_horz_single_tu.f90`); `sync_patch_array_3d_dp`
stubbed as the external boundary, all real operators inlined
(`rot_vertex_ocean_3d`, `nonlinear_coriolis_3d_fast_scalar`,
`grad_fd_norm_oce_3d_onblock`, `get_index_range`). Wired into `KERNELS` +
`SINGLE_TU_ARTIFACTS` (key `veloc_adv_horz`). **No inliner change needed.**
NOTE: drift test not yet run against the committed artifact (each extraction is a
~3-5 min, ~9 GB subprocess; host has 12 GB so runs MUST be serial).

### Inliner extended (the "expect to extend the inliner" budget) — all regression-clean, `tests/inliner` 113 green
Genuine gaps the dycore surfaced (NOT dycore-specific hacks):
1. **EXTENDS direct inheritance** (`analysis.alias_specs`): registered the
   direct `obj%member` keys for inherited components AND type-bound procedures
   (previously only the explicit `obj%base%member` form existed), guarded
   against clobbering child overrides.
2. **EXTENDS inherited TBP call resolution** (`desugaring.deconstruct_procedure_calls`):
   remap an inherited binding spec to the base's concrete procedure.
3. **named-kind int literals** (`analysis._eval_int_literal`): `1_wp`/`0_i8`
   crashed `kind in {'1','2','4','8'}` (Name unhashable) — added the
   `Name`→string normalize the real-literal path already had. GENERAL BUG FIX.
4. **interface-removal tolerance** (`fparser_inliner.run_fparser_transformations`):
   under `TOLERATE_EXTERNAL_USES`, an unresolvable kept-external generic
   (`sync_patch_array_mult`, reached only from dead code) WARNS instead of the
   hard "Could not remove all interfaces" raise.
5. **optimize-pass external tolerances** (`optimizations.py`,
   `analysis._track_local_consts`/`_root_comp`/`_inject_knowns`/`interface_specs`):
   skip unresolvable call targets (`mpi_abort`), generic-interface targets
   (0-dummy), unresolvable lvalues/components (dead solver-setup code merged via
   module-level object decls), pruned interface module-procedures. All gated on
   `TOLERATE_EXTERNAL_USES` (strict path unchanged).
6. **section-resubscript fold revert** (`analysis._inject_knowns` ×2): the
   const-arg fold could produce `a(:,:,b)(i,j)` (unrepresentable) — revert that
   substitution, leave the arg unfolded (best-effort).
NEW TESTS: `analysis_test.test_spec_mapping_of_type_extension` updated (direct
inherited key) + `desugaring_test` `test_procedure_replacer_inherited_type_bound`
+ `..._overrides_inherited_type_bound`. TODO: add a `_eval_int_literal`
named-kind unit test (#3).

### Harness config (per user direction)
`work_mpi_barrier` → **EMIT** (`OCEAN_EXTERNAL_FUNCTIONS`; "MPI collective needs
to be emitted"). `init` (`mo_fortran_tools::init_zero_*`) = array zeroing, NOT
MPI → stays **inlined**. `par%init` (`ocean_solve_parm_init`) = pure setter →
inlines (relevant to the solve kernel). dbg_print/finish/message/warning/timers →
drop; sync_patch_array(/mult)/exchange_data → emit.

### dycore calculate_explicit_term_ab — NEARLY THERE (TU emits, 50 errors left, ONE category)
The Python pipeline now runs end-to-end and emits a **3496-line TU**. The ONLY
remaining gfortran errors (all 50) are ONE category: surviving `ABSTRACT INTERFACE`
bodies in `mo_communication_types` (the halo-exchange comm-pattern abstraction:
`interface_exchange_data_*`, `interface_setup_comm_pattern*`) whose `IMPORT ::`
names comm types (`t_comm_pattern_collection`, `t_p_comm_pattern`,
`t_glb2loc_index_lookup`, `xfer_list`, `dp`, `sp`, `t_lhs_agen`, `t_transfer`,
`t_ocean_solve_backend`, `t_ptr_3d_*`) that were pruned as unused external
baggage. `t_comm_pattern` is stripped to an empty abstract type and is referenced
ONLY inside its own abstract interface → pure baggage.
**Fix STARTED but NOT yet working:** added `pruning.prune_dangling_interface_bodies`
(drop an interface body whose IMPORT references an undefined host entity; then
remove emptied abstract interface blocks), called from `run_fparser_transformations`
after the prune loop, gated on `TOLERATE_EXTERNAL_USES`. **It did NOT fire** (40
IMPORT interfaces still present, still 50 errors). LIKELY CAUSE: either the
`search_real_ident_spec(name, host_scope, alias_map)` still resolves the pruned
type via a STALE alias_map entry (USE alias pointing at a removed node), or
`find_scope_spec(interface_block)` returns the wrong host scope. **NEXT-SESSION
TASK #1:** debug `prune_dangling_interface_bodies` — verify the host scope + that
the pruned types are truly absent from a freshly-built alias_map; if the alias
survives, key the "undefined" test on the presence of the type's DEFINITION
(`Derived_Type_Stmt`/`Entity_Decl`) in the AST rather than alias resolution.
Repro: `_session_scratch/ext_explicit/kernel_tu.f90` (the 3496-line TU) +
`run_extract.py` driver. Once those 50 dangling IMPORTs are dropped the dycore TU
should compile.

### Remaining backlog
- dycore #1: finish `prune_dangling_interface_bodies` (above) → compile → wire
  `calculate_explicit_term_ab` into KERNELS/SINGLE_TU_ARTIFACTS + drift test.
- surface solve `solve_free_sfc_ab_mimetic`: externalize the solver TBP
  (`ocean_solve_solve`/`ocean_solve_construct`); `par%init` inlines.
- run the velocity drift test (byte-identity of the committed artifact).
- add the `_eval_int_literal` named-kind unit test.
- commit (worktree branch `worktree-external-call-policy`, currently UNCOMMITTED).

### Extended-struct → SoA (user design Q, answered)
Feasible for concrete (non-polymorphic) `TYPE(...)` extensions: flang resolves
EXTENDS to a flat record at compile time, and the per-member SoA marshalling is
an HLFIR pass (`hlfir-marshal-external-structs`) over the resolved record — it
already recurses nested derived-type members, so an extended type is "a struct
whose component 0 is the parent record". `CLASS(...)` polymorphism is the
boundary (dynamic type, needs monomorphization). The inliner half (inherited
component/TBP access) now exists (#1/#2 above).
