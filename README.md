# HLFIR -> DaCe frontend

## Design rationale

Flang already does all Fortran parsing, name binding, type inference,
and elemental-intrinsic lowering. Re-parsing Fortran in Python
duplicates that work and drifts against the standard. This frontend
consumes Flang's already-elaborated HLFIR (MLIR) instead, walks it
into a DaCe SDFG, and regenerates a Fortran wrapper that preserves
the caller's original interface.

The pipeline is five steps, each strengthening an invariant so the
Python walker has to understand only one narrow IR shape.

```
 .f90 --(0)--> .hlfir --(1)--> (2) --(3)--> (4) --(5)--> bindings.f90 + .so
```

- **(0) Pre-process** the Fortran text (flang-friendly rewrites:
  USE-merge, OMP/ACC strip, integer-power expansion, optional
  `IF (intvar)` fix) and run `flang-new-21 -fc1 -emit-hlfir`.
- **(1) Parse** one or more `.hlfir` files into a single MLIR
  `ModuleOp`; snapshot the caller-visible dummy list + derived-type
  layouts as `FortranInterface`.
- **(2) Inline + link.** `hlfir-inline-all` folds the call tree into
  the pinned entry, `symbol-dce` drops dead siblings, and (multi-file)
  `hlfir-verify-no-unresolved-calls` fails loudly if any `fir.call`
  survives outside the runtime / libm / C-stdlib allowlist.
- **(3) Normalise.** HLFIR rewrites in strict order: select-case
  lowering, element-alias folding, vector-subscript gather/scatter
  expansion, polymorphism rejection, AoS->SoA flattening (emits the
  `hlfir.flatten_plan` attribute), shape propagation, intent-less
  default, CF-to-SCF, then `sccp / canonicalize / cse`.
- **(4) Build SDFG.** Walk HLFIR into a small AST (`loop` / `while`
  / `conditional` / `assign` / `copy` / `memset` / `libcall` /
  `reduce` / `break` / `return`), emit the SDFG, run the
  post-generation cleanup (loop-iter SSA, length-1-transient -> scalar,
  integer power exponents), then pin a `FrozenSignature` snapshot of
  the arglist + free symbols.
- **(5) Regen binding.** Read `FortranInterface` + `FrozenSignature`
  + `FlattenPlan` and emit `<entry>_bindings.f90`: a ref-counted
  Fortran module that preserves the caller's original signature,
  populates SDFG symbols via `size` / `lbound`, and per struct member
  picks between a zero-copy `c_f_pointer` alias and a Fortran
  `do`-loop deep copy.

## Quick start

End-to-end: a `.f90` to an SDFG to a Fortran-callable `.so` whose
binding is signature-locked against later DaCe transformations.

```python
import dace_fortran

# 1. Source -> SDFG.  ``entry`` is the mangled flang symbol;
#    ``_QPname`` = free subroutine, ``_QM<mod>Pname`` = module procedure.
sdfg = dace_fortran.build_sdfg(open("velocity_full.f90").read(),
                               entry="_QPvelocity_tendencies")

# 2. Optimise.  Any DaCe transformation is allowed.
sdfg.simplify()

# 3. Emit + link a Fortran-callable library.  ``build_fortran_library``
#    re-checks ``sdfg._frozen_signature`` against the live SDFG before
#    emitting the binding (a transformation that re-ordered arg slots
#    or changed dtypes raises ``SignatureDriftError``); writes the
#    binding to ``<out_dir>/<entry>_bindings.f90``; gfortran-links it
#    with ``extra_sources`` against the SDFG ``.so``.  Build the
#    ``iface`` / ``plan`` arguments as
#    ``tests/icon_full/test_velocity_full_bindings_e2e.py`` does.
from dace_fortran.bindings import build_fortran_library, SignatureDriftError
try:
    lib = build_fortran_library(sdfg, iface, plan, out_dir="build",
                                extra_sources=["caller.f90"],
                                mode="debug")  # or "release", or flags=[...]
except SignatureDriftError as e:
    raise SystemExit(f"binding invalidated by a transformation: {e}")

lib.load()                                      # -> ctypes.CDLL
print("binding:", lib.bindings_f90, "/ library:", lib.so_path)
```

If the SDFG needs an OpenMP runtime, provide it via
``LD_PRELOAD=<libgomp/libomp> python ...``; the library never hard-codes
one.  The lower-level explicit sequence (``emit_bindings`` →
``frozen.verify_against(sdfg)`` → gfortran link) is still supported
and is what ``build_fortran_library`` consolidates.

### Worked examples

**Single Fortran file → SDFG.**  No external deps, no cmake.
Reference: most of `tests/*_test.py` (e.g. `baseline_arithmetic_test.py`,
`do_loop_exit_test.py`).

```python
src = """
subroutine add_array(a, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  integer :: i
  do i = 1, n
    a(i) = a(i) + 1.0d0
  end do
end subroutine add_array
"""
sdfg = dace_fortran.build_sdfg(src, entry="_QPadd_array")
a = np.asfortranarray(np.zeros(8, dtype=np.float64))
sdfg(a=a, n=8)
assert (a == 1.0).all()
```

**A real project (CMake or Autotools).**  See *Building an SDFG
from a real project* below for the full walkthrough; the short
version is: get a `compile_commands.json` from the build, then one
call.

```python
sdfg = dace_fortran.build_sdfg_from_project(
    "build/compile_commands.json",
    entry="_QMmod_jacobiPjacobi2d_update",
    stubs=["mpi_stub.f90", "netcdf_stub.f90"])
```

**External `bind(c)` library.**  The kernel calls a separately
compiled Fortran function; register its signature out-of-band and
the bridge lowers the `CALL` to an `ExternalCall` library node.
Reference: `tests/external_call/test_external_call.py`,
`tests/external_call/test_keep_external.py`.

```fortran
! foo.f90 -- compiled standalone into libfoo.so
subroutine foo(a, n) bind(c, name="foo")
  use iso_c_binding
  integer(c_int), value :: n
  real(c_double), intent(inout) :: a(n)
  a = a + 1.0d0
end subroutine foo
```

```python
import subprocess
from dace_fortran import Arg, build_sdfg, register_external, ExternalSignature
subprocess.check_call(["gfortran", "-shared", "-fPIC", "-o", "libfoo.so", "foo.f90"])

register_external("foo", ExternalSignature(
    c_name="foo",
    args=[Arg("array", "float64", "inout"),    # inout default; missed write = silent bug
          Arg("scalar", "int32", "in")],
    libraries=["/abs/path/libfoo.so"]))        # linked into the SDFG .so with rpath

src = """
subroutine run(a, n)
  use iso_c_binding
  integer(c_int), intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  interface
    subroutine foo(a, n) bind(c, name="foo")
      use iso_c_binding
      real(c_double), intent(inout) :: a(*)
      integer(c_int), value :: n
    end subroutine foo
  end interface
  call foo(a, n)
end subroutine run
"""
sdfg = build_sdfg(src, entry="_QPrun")
```

## Pipeline detail

### Pre-process rewrites — step (0)

`dace_fortran.preprocess` holds the text rewrites that must run
before flang (they change what flang accepts, or what arithmetic
each backend is free to pick).  All are SED-style regex transforms,
not a Fortran parser, so each is deliberately narrow; comment- and
string-awareness is shared via `_scan_line` so a `!` or `**` inside
a character literal is never touched.

- `merge_used_modules` -- inlines every externally-`USE`-d module's
  real source so flang sees one self-contained translation unit.
  Pass-through for self-contained input (the entire inline-source
  test suite); only genuine multi-file projects activate it.
- `strip_openmp_directives` -- drops `!$OMP` / `!$ACC` / `!$`
  sentinel lines, the ICON `#include "omp_definitions.inc"` cpp
  include, and `#ifdef _OPENMP` / `#ifdef _OPENACC` conditional
  blocks (taking `#else` when present); unrelated cpp passes
  through. Runs **unconditionally** so legacy codebases that ship
  accelerator sentinels are consumable without `-fopenmp`.
- `rewrite_integer_powers` -- expands an integer-valued REAL-literal
  power (`x**2.0` -> `(x*x)`, `(p-q)**3.0` -> `((p-q)*(p-q)*(p-q))`).
  Runs **unconditionally**: algebraically exact and removes a
  backend-dependent `pow(x, 2.0)` vs `x*x` rounding difference
  against the gfortran reference. A bare-integer exponent (`x**2`)
  is left for flang's own (bit-identical) integer-power lowering;
  genuine fractional powers (`**0.5`) stay as `pow()`; a base
  containing a call/array reference is left alone (duplicating it
  would invoke it twice).
- `promote_real_literals_to_double` -- single/default REAL literals to
  an explicit double form (`2.0` -> `2.0D0`). A standalone utility
  applied directly to kernel source on disk when a codebase must be
  globally double; **not** wired into the build path.
- `preprocess_fortran` -- `IF (intvar)` -> `IF (intvar /= 0)` for
  INTEGER scalars. flang-new-21 rejects bare INTEGER as an IF
  condition (only LOGICAL is legal); ECRAD / CloudSC (`IF
  (laericeauto)`) / ICON scaffolding ship this. **Opt-in** per call
  site (pass `preprocess=True` to `build_sdfg`); off by default so
  we don't paper over real issues in clean source.

### HLFIR normalisation passes — steps (2) + (3)

These are the `DEFAULT_PIPELINE` entries (see
`builder/__init__.py`).  Step (2)'s `hlfir-inline-all` is the
second entry in this same table; it is broken out in the top-level
pipeline only because inlining is the conceptual hinge between
"raw HLFIR" and "normalised single-TU IR".

| Pass | Purpose |
| --- | --- |
| `lower-fir-select-case` | Lower `fir.select_case` to `cf.cond_br` BEFORE inlining (the inliner's block-operand remap segfaults on a callee containing a select-case) |
| `hlfir-inline-all` | Splice every callee body into the pinned entry |
| `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elemental / scalar-arg procedures |
| `hlfir-expand-vector-subscript-gather` | Replace `hlfir.associate` of an `hlfir.elemental` (Flang's gather temp for noncontiguous slice arguments) with an explicit `fir.alloca` + gather DO loop |
| `hlfir-expand-vector-subscript-scatter` | Replace `hlfir.region_assign` with an `hlfir.elemental_addr` destination (vector-subscripted scatter `d(cols) = source`) by an explicit DO loop |
| `symbol-dce` | Drop private callee bodies once `hlfir-inline-all` has folded them in |
| `fir-polymorphic-op` | Statically devirtualise resolvable `fir.dispatch` / `fir.select_type`; lowers the rest to an indirect-call shape that the next pass catches |
| `hlfir-reject-polymorphism` | Loud-fail on any surviving `fir.dispatch`, `fir.select_type`, or `fir.box_tdesc` (residual indirect dispatch from `fir-polymorphic-op`)  --  the bridge supports CLASS-as-monomorphic-box only |
| `hlfir-flatten-structs` | AoS -> SoA; emits `hlfir.flatten_plan` module attribute. Peels `fir.class<T>` via `BaseBoxType` so monomorphic CLASS receivers flatten through the same path as TYPE |
| `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols |
| `hlfir-default-intent` | Intent-less dummies default to `intent_inout` |
| `lift-cf-to-scf` | Raw-CFG loops (`DO WHILE`, `DO...EXIT`) -> `scf.while` + `scf.if` |
| `sccp` -> `canonicalize` -> `cse` | Fold + simplify + dedupe after HLFIR exposed its constants |

Multi-file builds additionally run
`hlfir-verify-no-unresolved-calls` so any `fir.call` that survives
the inliner outside the Flang-runtime / libm / C-stdlib allowlist
fails the build loudly.  Inlining requires the
`DialectInlinerInterface` extensions for `fir` / `func` / `LLVM`
to be attached to the context; the bridge constructor does this
once.

### SDFG emission — step (4)

`bridge/extract_vars.cpp` classifies every `hlfir.declare` as
`array`, `symbol`, or `scalar` (rules in *Mechanisms* below).
`bridge/extract_ast.cpp` is the dispatcher; the AST extraction is
split into focused translation units under `bridge/ast/`  --
`expressions.cpp` (RHS rendering), `assigns.cpp` (assignment-shape
builders), `elementals.cpp` (reductions + libcall + select-case),
`control_flow.cpp` (cmp / boolean / scf-while / merge),
`dispatch.cpp` (top-level walker).  Cross-TU API + thread-local
state lives in `ast/ast_helpers.h`; internal cross-TU helpers in
`ast/ast_internal.h`.  The walker produces a recursive `ASTNode`
tree covering `loop` / `while` / `conditional` / `assign` / `copy`
/ `memset` / `libcall` / `reduce` / `break` / `return`.

**Loop bounds + IF conditions are hoisted to symbols.** Every
non-trivial loop bound (anything beyond a bare identifier or integer
literal) and every non-trivial IF condition is materialised as an SDFG
symbol on a state-change before the block. Names follow a global
counter: `loopbegin_<N>` / `loopend_<N>` for loop bounds and
`if_cond_<N>` for branch guards. The `LoopRegion` / `ConditionalBlock`
itself then references **only** the symbol  --  no expression rewriting
in the bound or condition. This:
  * keeps the bridge's emitters small (no iter-rename plumbing in
    bound expressions, no ad-hoc `[0]` subscripting in IF conditions);
  * funnels indirect-array reads through the existing symbol-staging
    machinery (a bound containing `row_ptr[i+1] - 1` becomes one
    interstate-edge assignment that the C++ codegen renders correctly
    via the array-aware sympy printer);
  * gives the SSA loop-iter pass a uniform input shape.

`builder/SDFGBuilder` emits the SDFG from that tree, then runs the
post-generation cleanup below, then snapshots `sdfg.arglist()` +
free symbols into a `FrozenSignature` and pins it on the SDFG.

### Post-generation cleanup — between (4) and (5)

Run over the freshly-built SDFG, in order:

1. **`SSALoopIterators`** (`dace.transformation.passes.ssa_loop_iterators`).
   Renames every `LoopRegion.loop_variable` to a globally-unique
   `_it_<N>` symbol and propagates the rename through the body
   (memlet subsets, tasklet bodies, interstate-edge assignments,
   nested SDFG symbol mappings).  Adds a reconstruction state after
   each loop that re-asserts `<original_var> = <loop_end>` so
   downstream code reading the un-renamed name sees the correct
   post-loop value.  Skips while-shape loops (no induction variable).
   Renders the reconstruction RHS via `dace.symbolic.symstr(arrayexprs=...)`
   so an array-subscripted bound like `row_ptr[i+1] - 1` renders with
   `[]` (not `()`, which sympy would print and the C++ codegen would
   reject).  The bridge consequently emits each `LoopRegion` using the
   source-Fortran iter name (`jk`, `je`, ...) verbatim and lets this
   pass handle the uniquification  --  no `iter_map` plumbing in the
   emitters.

2. **`replace_length_one_arrays_with_scalars`**
   (`dace.sdfg.construction_utils`).  Walks every length-1 ``Array``
   on the SDFG and rewrites the descriptor to a true ``Scalar``,
   stripping leftover ``[0]`` subscripts from interstate-edge
   assignments, conditional-block guards, and loop-region condition
   expressions.  Runs with **`transient_only=True`** at the top
   level: only LOCAL 1-element transients (loop accumulators left as
   length-1 arrays) get folded.  Signature scalars follow the bridge's
   I/O convention  --  `intent(in)` / `VALUE` are emitted directly as
   `Scalar` by `descriptors.py`, while `intent(out)` / `intent(inout)`
   stay as length-1 ``Array`` so callers can pass a numpy 1-element
   buffer to receive the value.  Recurses into nested SDFGs (their
   transient-only sub-cleanup follows the same rule).

3. **`IntegerizePowerExponents`**
   (`dace_fortran.integer_power_exponents`).  Retypes
   every integer-valued floating-point ``**`` exponent in a Python
   tasklet (``base**2.0``, ``base**-3.0``) to the corresponding
   ``int``.  C++ codegen routes a float exponent through libm
   ``dace::math::pow`` but an integer exponent through
   ``dace::math::ipow`` (plain left-to-right repeated multiply,
   ``base*base``)  --  the latter is bit-identical to what a
   Fortran/C reference compiler emits for a small integer power, so
   retyping removes a trailing-bit divergence on long real(8) chains.
   Only the exponent literal changes (no base duplication, no
   connector renumbering), which is why it is safe as a post-split
   cleanup.  Genuinely fractional exponents (``0.5``, ``0.333``) are
   left untouched.

The cleanup runs **before** the `FrozenSignature` snapshot is taken
so the bindings emitter sees the post-cleanup signature.

**Loop iterator validation.** SDFG validation rejects writing to a
`LoopRegion.loop_variable` from an interstate-edge assignment inside
its own region.  The `LoopRegion` already owns the iterator update
via `init_expr` / `update_expr`; mutating it elsewhere races with
that machinery and breaks the SSA invariant the iter pass relies on.

## Data artefacts

These are the structured records that flow between the pipeline
steps.  They are
the frontend's stable contract surface  --  new features extend them,
they do not invent parallel channels.

| Artefact | Produced at | Consumed at | Role |
| --- | --- | --- | --- |
| `FortranInterface` | (1) snapshot | (5) emit | Caller-facing dummy list + derived-type layouts |
| `FlattenPlan` (MLIR attr) | (3) `hlfir-flatten-structs` | (5) emit | Per-dummy AoS->SoA unpack recipe (`flat_names`, `read_exprs`, `shape_exprs`, `aliasable`, `aos_alloc`+`cap_symbol` for padding-to-max) |
| `VarInfo[]` | (4) `extract_vars` | (4) `SDFGBuilder` | Classification + shape + intent per variable |
| `ASTNode` tree | (4) `extract_ast` | (4) `SDFGBuilder` | Normalised CFG + assigns + library-op references |
| `FrozenSignature` | (4) end of `build()` | codegen, (5) emit | SDFG arglist snapshot  --  drift check at codegen time |

## Mechanisms

**Symbol vs scalar classification.** A Fortran integer is a *symbol*
iff it's a DO induction variable, an array shape extent, a DO bound,
an `hlfir.designate` index, or feeds a control-flow condition --
everything else integer is a *scalar*.  Writes to symbols become
interstate-edge assignments; writes to scalars become tasklets.
Only symbols can appear as array indices.

**Fortran lbound handling.** Every array descriptor carries
`shape_symbols` + `lower_bounds`.  `access.build_memlet_index` folds
the lbound offset once at subset-build time; DaCe's descriptor
`offset` field stays at zero so downstream transformations only
reason about one convention.

**Assumed-shape alias re-basing.** When `hlfir-inline-all` splices
an `arr(:)` callee into a caller whose actual has custom bounds
(`x(-2:2)`), flang emits a second `hlfir.declare` aliasing the
outer storage.  The bridge skips the alias in `extract_vars`,
follows it in `traceToDecl`, and rewrites each access index by
`outer_lbound - inner_lbound` so the lbound fold fires uniformly.

**ELEMENTAL inlining.** Flang lowers each elemental call to
`fir.do_loop { hlfir.designate per-arg; fir.call scalar_body }`.
After `hlfir-inline-all` the per-element alias declares are folded
by `hlfir-fold-element-aliases`; the SDFG builder sees the same
shape as a hand-written per-element loop.

**OPTIONAL dummies.** `fir.is_present %x` becomes a scalar
`<name>_present` (i32) on the SDFG signature.  The existing if/else
lowering reads it like any other condition -- no new AST kind.
Intent-less optionals default to `intent_in` so they don't
misclassify as transients.

**AoS↔SoA flattening.** See *Pipeline detail* above -- `hlfir-flatten-structs`
hoists every struct member as its own top-level dummy and stamps the
`hlfir.flatten_plan` recipe that step (5)'s binding consumes to
restore the caller-side AoS view.  Three supported layouts (plain /
nested members, array-of-struct with array members, AoS + allocatable
via padding-to-max) and their boundaries are listed in *Supported /
not supported* below.

**Section reductions.** Whole-array `SUM` / `PRODUCT` / `ANY` /
`ALL` lower to DaCe's `standard.Reduce`.  Section reductions
(`ANY(mask(lo:hi, jk))`) synthesise an init + `kind="loop"` AST
because DaCe's Reduce can't express a dynamic-section input
directly.

**Sibling-assign RAW hazards in loop bodies.** Multiple assigns in
one `fir.do_loop` body that target the same non-transient storage
race when all wired into one state.  The emitter detects read-write
name overlap across siblings and serialises them into a chain of
states (one tasklet per state).  Non-overlapping siblings still
share a state -- the check is per-loop-body.

**Exponentiation.** `math.fpowi` / `math.powf` / `math.powi` /
`math.ipowi` all surface as `(a ** b)` in the tasklet; the
post-generation `IntegerizePowerExponents` pass retypes
integer-valued float exponents to `int` so DaCe codegen routes
them through `dace::math::ipow` (bit-identical to gfortran's
small-integer power lowering).

**Signature freezing.** `codegen.generate_code` verifies
`sdfg._frozen_signature` before emitting the C++ header; drift
raises `SignatureDriftError`.  Transformations mutate SDFGs freely
-- a header that disagrees with the emitted Fortran binding cannot
ship.

**Defensive walk budgets.** `trace_utils::limits::*` constants
(`kBuildExprDepth=128`, `kConvertChainDepth=32`,
`kTraceToDeclMax=1024`) cap SSA-tracing depth on pathological IR.
Bumping them never changes semantics on well-formed HLFIR.

## Components

```
dace_fortran/
|--- bridge/            C++  --  HLFIR parser + classifier + walker (nanobind)
|   |--- bridge.cpp              MLIRContext, pass pipeline, Python exports
|   |--- extract_vars.cpp        hlfir.declare -> VarInfo[]
|   |--- extract_ast.cpp         entry point; calls into ast/dispatch.cpp
|   |--- trace_utils.cpp         SSA tracing + alias helpers + depth limits
|   \--- ast/                    AST extraction split per-responsibility (real TUs)
|       |--- ast_helpers.h         cross-TU public API + inline thread_local state
|       |--- ast_internal.h        cross-TU internal helper declarations
|       |--- expressions.cpp       buildExpr, buildIndexExpr, lowerIsPresent
|       |--- assigns.cpp           buildAssignNode, copy/memset/libcall, sections
|       |--- elementals.cpp        reductions, elemental walks, select-case chains
|       |--- control_flow.cpp      cmp predicates, buildBoolExpr, scf.while/if walkers
|       \--- dispatch.cpp          top-level walker; calls into the others
|--- passes/            C++  --  HLFIR -> HLFIR rewrites
|   |--- LowerFirSelectCase.cpp  fir.select_case -> cf.cond_br (pre-inline)
|   |--- InlineAll.cpp
|   |--- FoldElementAliases.cpp  erase elemental-body alias declares
|   |--- ExpandVectorSubscriptGather.cpp  hlfir.associate(elemental) -> alloca + gather loop
|   |--- ExpandVectorSubscriptScatter.cpp  hlfir.region_assign(elemental_addr) -> scatter loop
|   |--- RejectPolymorphism.cpp  loud-fail on residual virtual dispatch / SELECT TYPE
|   |--- FlattenStructs.cpp      stamps hlfir.flatten_plan
|   |--- PropagateShapes.cpp
|   |--- DefaultIntent.cpp
|   |--- VerifyNoUnresolvedCalls.cpp
|   \--- Passes.cpp              registerAllBridgePasses()
|--- builder/           Python  --  SDFG emission (step 4)
|   |--- __init__.py             SDFGBuilder, _emit dispatch, pipelines
|   |--- context.py              _Ctx (state, pending assigns, iter_map)
|   |--- descriptors.py          add_descriptors, DTYPE mapping
|   |--- access.py               build_memlet_index, indirect-symbol lifting
|   |--- emit_tasklet.py         per-occurrence tasklet + emit_scalar_assign
|   |--- emit_cfg.py             assign / loop / while / conditional
|   \--- emit_library.py         copy / memset / libcall / reduce / break / return
|--- intrinsics/        Python  --  Fortran intrinsic registry (consumed by bindings)
|--- bindings/          Python  --  Fortran wrapper emitter (step 5)
|   |--- frozen_signature.py     FrozenArg + FrozenSignature + drift check
|   |--- fortran_interface.py    OriginalInterface (outer surface)
|   |--- flatten_plan.py         FlattenPlan + to_dict / from_dict
|   |--- block_builders.py       per-Fortran-section emitters
|   |--- loop_copy.py            alias vs deep-copy renderers
|   \--- emit_bindings.py        -> <entry>_bindings.f90
|--- build.py           public entry: build_sdfg / build_sdfg_from_files /
|                                     build_sdfg_from_hlfir / _from_project (tier 3)
|--- emit_hlfir.py      tier-3 helper: ``python -m dace_fortran.emit_hlfir
|                       <build>/compile_commands.json --out <build>/hlfir [--stub ...]``
|--- external.py        register_external / keep_external (ExternalCall libnode +
|                       link-flag injection for separately-compiled bind(c) callees)
|--- preprocess.py      Fortran-text rewrites: USE merge, OMP/ACC strip,
|                       integer-power expansion, IF (intvar) -> /=0
|--- build_bridge.py    one-time CMake build of the C++ HLFIR bridge
|--- hlfir_to_sdfg.py   back-compat shim re-exporting SDFGBuilder / generate_sdfg from builder/
\--- integer_power_exponents.py  post-generation pass: float ``**int_value`` -> int exponent
```

## Entry point

Three tiers ordered by how much the bridge does on the caller's
behalf.  All return a validated `dace.SDFG` with
`sdfg._frozen_signature` attached; `import dace_fortran` is lazy
(the C++ bridge builds on first use).

`entry` is the mangled flang symbol -- `_QPname` for a free
subroutine, `_QM<mod>Pname` for a module procedure -- and is
required (an SDFG targets one specific procedure).

```python
import dace_fortran

# (1) Inline source.
sdfg = dace_fortran.build_sdfg(src, entry="_QPcompute_tendencies")

# (2) Multi-file project (driver + the modules it USEs, any order).
#     The file defining ``entry`` is the root; the rest are merged
#     into one TU via ``merge_used_modules``.
sdfg = dace_fortran.build_sdfg_from_files(
    ["driver.f90", "math_utils.f90"], entry="_QPcompute_tendencies")

# (3) A real CMake / Autotools project -- tier 3.  One call from
#     the build's compile_commands.json; see section below.
sdfg = dace_fortran.build_sdfg_from_project(
    "build/compile_commands.json", entry="_QMmod_jacobiPjacobi2d_update",
    stubs=["mpi_stub.f90", "netcdf_stub.f90"])
```

For tiers (1) and (2), a kernel that ``CALL``s a separately-compiled
function declares its signature out-of-band -- the callee must
present a stable `bind(c, name=...)` symbol (Fortran name mangling
is compiler-specific):

```python
dace_fortran.register_external("foo", dace_fortran.ExternalSignature(
    c_name="foo",
    args=[dace_fortran.Arg("array", "float64"),    # inout by default
          dace_fortran.Arg("scalar", "int32")],
    libraries=["/abs/path/libfoo.so"]))
```

`intent` defaults to `inout` (a missed write into an opaque external
is a silent correctness bug; a missed read is just an optimisation
miss).  Listed `libraries` are linked into the SDFG `.so` with an
rpath, so it stays self-contained -- no `LD_PRELOAD`.
`keep_external(name, ...)` is the shorter form when defaults are fine.
MPI (`MPI_Send/Recv/Isend/Irecv/Wait`, including non-default
communicator) is recognised automatically and lowered to
`dace.libraries.mpi` nodes; no registration needed.

### Building an SDFG from a real project (CMake / Autotools) — tier 3

Tiers (1) and (2) drive flang internally -- fine for self-contained
kernels and small multi-file projects, but they don't scale to
codebases with hundreds of modules and real `netcdf` / `hdf5` /
`yaxt` externals plus custom cpp gates.  Those projects already have
a working build system that knows the right include paths,
intrinsic-module path, and cpp defines; tier 3 reuses it.

The whole contract is: **(a) get a `compile_commands.json` from your
build, (b) one Python call.**  Step (b) is identical for every build
system; only step (a) differs.

**Step (a) — get the compilation database.**

The database records each Fortran TU's compiler invocation (file +
`-I`/`-D` flags) in build order.  `emit_hlfir` reads it, so it never
has to guess flags or dependency order.

```bash
# --- CMake / Ninja: one configure flag drops it into the build dir ---
cmake -S src/ -B build/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/

# --- Autotools / plain Make (ICON's shape: autoconf + a hand-written
#     Makefile.in -- NOT automake): wrap the build in `bear`, which
#     intercepts compiler exec() calls.  Agnostic to whether the
#     Makefile came from automake or was hand-written. ---
./configure ...
bear -- make                  # writes ./compile_commands.json
```

**Step (b) — one call.**

```python
import dace_fortran
sdfg = dace_fortran.build_sdfg_from_project(
    "build/compile_commands.json",
    entry="_QMmod_jacobiPjacobi2d_update",     # mangled flang symbol
    stubs=["mpi_stub.f90", "netcdf_stub.f90"]) # see below
```

`stubs` are flang-buildable stand-ins for modules flang ships no
`.mod` for (`mpi` / `netcdf` / `hdf5` / ...): a small module that
declares the names the project `USE`s.  Compiled before the project
TUs so the `USE` lines resolve.  Omit when the project has no such
externals.

To emit once and lower several entries (or to inspect the
intermediate `.hlfir` files), use the two explicit steps that
`build_sdfg_from_project` wraps:

```bash
python -m dace_fortran.emit_hlfir build/compile_commands.json \
    --out build/hlfir --stub mpi_stub.f90 --stub netcdf_stub.f90
```
```python
sdfg = dace_fortran.build_sdfg_from_hlfir(
    "build/hlfir", entry="_QMmod_jacobiPjacobi2d_update")
```

`tests/prebuilt_hlfir/` ships one worked project per DB-capture
route, both with a plain build that knows nothing about HLFIR or
the bridge:

| Project | Build | Files | Externals | Demonstrates |
|---|---|---|---|---|
| `jacobi/` | autotools + `bear -- make` | 4 | MPI + netCDF (2 stubs) | ICON-shape build; entry stays MPI-free even though sibling `halo_exchange` USEs MPI |
| `csr_spmv/` | cmake export flag | 2 | none | minimal happy path |

**Inlining scope**: inlining is intra-TU -- flang emits one `.hlfir`
per translation unit and the bridge consumes one of them.  A
procedure `USE`d from a different TU stays as an external symbol
reference in the SDFG (which is the right contract for things like
halo exchanges or I/O routines you *want* left external).

## Extending the frontend

| If you're adding... | Change here | Then cover in |
| --- | --- | --- |
| a new `math.*` intrinsic | `extract_ast.cpp` `unary_math` / `binary_math` tables | `tests/hlfir/elemwise_intrinsics_test.py` |
| a new reducer | `extract_ast.cpp::kRedTable` (+ `buildSectionReduceAssign` for section form) | `tests/hlfir/reduce_intrinsics_test.py` |
| a new CFG op | `extract_ast.cpp::buildAST` dispatch + `builder/__init__.py::_EMIT_DISPATCH` + emitter in `builder/emit_cfg.py` | ports from `baseline_*_test.py` |
| a new binding layout rule | `bindings/loop_copy.py` + new `FlattenRecipe` field | `tests/hlfir/bindings/emit_bindings_test.py` |
| a new HLFIR pass | file in `passes/`, register in `Passes.cpp`, slot into `DEFAULT_PIPELINE` | `tests/hlfir/<pass>_test.py` |

## Supported / not supported

[OK] supported, [!] planned (tracked in xfails), [X] never (out
of scope).

### Types

| Feature | Status | Notes |
|---|---|---|
| `INTEGER(1/2/4/8)` | [OK] | mapped to `int8/16/32/64` |
| `REAL(4/8)` | [OK] | mapped to `float32/64` |
| `LOGICAL(1/4/8)` | [OK] | surfaced as `uint8/int32/int64` for f2py ABI |
| `COMPLEX(4/8)` | [OK] | arrays only -- scalar by-value is a DaCe-core gap |
| `CHARACTER(*)` | [X] | string handling out of scope |
| Derived type, flat members | [OK] | `hlfir-flatten-structs` |
| Derived type, nested | [OK] | recursive leaf collection, path-flattened name `base_m1_m2_leaf` |
| Array-of-struct with array members (`A(N)%w(M,M)`) | [OK] | shape concatenation; `A(i)%w(j,k)` -> `A_w(i,j,k)` |
| Whole-member access on AoS (`A(i)%w = ...`) | [OK] | triplet section `A_w(i, 1:M:1, ...)` |
| Cross-subroutine struct args (incl. AoS) | [OK] | per-member block args + `hlfir.flatten_plan` recipe |
| Derived type, allocatable members | [OK] | flat top-level allocatable companion + per-allocate-site rename |
| AoS + allocatable, uniform constant inner size | [OK] | static 2D companion `A_w(N, M)`; allocate/freemem chain erased |
| AoS + allocatable as SDFG-boundary dummy | [OK] | padding-to-max: binding computes `cap = max_i(size(A(i)%w))`, packs/unpacks live regions; runtime cap symbol on signature |
| AoS + allocatable, kernel-internal first allocation (`intent(out)`) | [!] | needs an HLFIR shape-discovery pre-pass + caller stub interface |
| Jagged AoS, two allocatable members of differing per-instance lengths | [!] | padding-to-max works mechanically; two-cap-symbol shape not yet exercised |
| Derived type with parametric array dim from struct field | [!] | 1 xfail |
| Circular type definitions (recursion through pointer chain) | [X] | out of scope |

### Control flow

| Feature | Status | Notes |
|---|---|---|
| `DO`, `DO WHILE`, `DO CONCURRENT` | [OK] | LoopRegion + scf.while |
| `IF` / `ELSE IF` / `ELSE` | [OK] | scf.if |
| `SELECT CASE` | [OK] | `lower-fir-select-case` lifts to cf.cond_br |
| `EXIT`, `CYCLE` | [OK] | |
| Statement functions (`f(x) = ...`) | [!] | 1 xfail |
| `GOTO` | [X] | unstructured GOTO doesn't lift to scf |
| `SELECT TYPE` | [X] | requires runtime type discrimination |

### Subprograms / linkage

| Feature | Status | Notes |
|---|---|---|
| Module-contained `SUBROUTINE` / `FUNCTION` | [OK] | inlined by `hlfir-inline-all` |
| Internal subprograms (`CONTAINS` inside subroutine) | [OK] | |
| `INTERFACE` blocks | [OK] | resolved at flang time |
| `USE`, `USE ... ONLY:` | [OK] | flang resolves at lowering |
| `OPTIONAL` dummy + `PRESENT` | [OK] | folded statically post-inline |
| `ALLOCATABLE`, `ALLOCATE`, `DEALLOCATE` | [OK] | local + dummy |
| Separately-compiled `bind(c)` external | [OK] | `register_external` / `keep_external` |
| `EXTERNAL` statements | [X] | use modules instead |
| `POINTER` | [X] | requires SSA-breaking aliasing |
| BLAS/LAPACK via `EXTERNAL` | [X] | use module-contained or DaCe libnodes |

### Polymorphism

| Feature | Status | Notes |
|---|---|---|
| `CLASS(t)` as a monomorphic box (no virtual dispatch) | [OK] | `FlattenStructs` peels `fir.class<T>` via `BaseBoxType` |
| Type-bound procedure with statically-known receiver | [OK] | `fir-polymorphic-op` devirtualises before flatten |
| Truly virtual dispatch (caller-set receiver type) | [X] | `hlfir-reject-polymorphism` fails loudly on residual `fir.box_tdesc` |
| `SELECT TYPE` / runtime type discrimination | [X] | same |

### Slicing / array ops

| Feature | Status | Notes |
|---|---|---|
| Contiguous slice `a(i:j, k:l)` | [OK] | |
| Whole-array assign `a = b` | [OK] | `hlfir.elemental` + emit_library |
| Elementwise intrinsics (sin/cos/exp/sqrt/...) on real / complex | [OK] | |
| Reductions (sum/product/min/max/all/any/count/minval/maxval) | [OK] | |
| BLAS/LAPACK (matmul, transpose) | [OK] | dense -> libnode, strided -> explicit DO loop |
| Noncontiguous gather `a(idx, :)` -- rank-1, constant extent | [OK] | `hlfir-expand-vector-subscript-gather` |
| Noncontiguous gather -- rank-2+ (`d(cols2, cols)`) | [!] | 5 xfails -- pass currently bails with a clear error |
| Noncontiguous slice -- symbolic extent | [X] | DaCe can't express runtime-sized symbol arrays |
| Noncontiguous scatter -- `intent(out)` write-back | [!] | not yet modelled |
| `ASSOCIATE` block | [!] | relative indexing only |

### Codegen targets

| Feature | Status | Notes |
|---|---|---|
| CPU C++ tasklets | [OK] | |
| GPU CUDA | [X] | would need OpenACC-style shim emission |
| Native `!$OMP` directives | [X] | DaCe handles parallelism itself |
| COARRAY | [X] | |

## Non-goals

- Re-parsing Fortran in Python. Flang is authoritative.
- GPU target bindings (would need OpenACC shim emission).
- Fortran SIMD / COARRAY semantics.
- Cross-kernel fusion across translation-unit boundaries  --  inline-all
  handles intra-TU fusion; cross-TU is the binding emitter's problem.

## Testing

Every supported construct has a seeded numerical test against
gfortran / f2py under `tests/hlfir/`. Binding-specific tests live
in `tests/hlfir/bindings/`. The six E6 velocity-advection
representative loopnests have SDFG-vs-f2py comparisons in
`tests/hlfir/icon_loopnests/`. All executable-Fortran tests
compile with `gfortran` (Ubuntu's `flang-new-21` ships without
`libflang_rt` so it's emit-HLFIR-only).

```bash
# Main xdist sweep -- excludes the multi-rank mpi-marked tests (they
# need mpirun, not a thread worker).
python3 -m pytest -n auto -m "not mpi" tests/

# Multi-rank MPI tests (Send/Recv + Isend/Irecv + non-default
# communicator).  ``--oversubscribe`` lets the 4-rank tests run on
# laptops with fewer than 4 physical cores, the same flag CI uses.
mpirun --oversubscribe -n 4 python3 -m pytest -m mpi \
    -p no:cacheprovider tests/

# Dump built SDFGs for inspection.
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/hlfir/
```
