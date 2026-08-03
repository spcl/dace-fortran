# Value-projection SSA

DaCe rule: each name is a **symbol XOR data**. A Fortran integer value can be
needed as a *symbol* (array extent, subset bound, loop range, interstate
condition) while its source is *data* that gets written. `sz(i)` sizes an
allocation **and** `sz` is `intent(inout)` and mutated. Dual use — one name
cannot be both, and the whole array cannot become a scalar symbol.

Value-projection SSA promotes the **symbolic projection of a value** — the
scalar `sz(i)` at one program point — to a symbol, per reaching definition.

## The pass (three parts)

1. **Symbol-need lattice** `{data-only ⊑ symbol-needed}`. Seed at every
   symbolic-context use, propagate backward through arithmetic
   (`nh(nt)+1` ⇒ `nh(nt)`, `nt` symbol-needed). Fixpoint over use-def.
2. **Reaching-def versioning.** Each symbol-needed value gets one SSA version
   per reaching def. Snapshot `V = arr[idx]` on the dominating interstate edge.
   A write to `arr` (or redefinition of `idx`) **kills** the version.
3. **Soundness.** Inside `[snapshot edge, next kill)` the value is invariant, so
   the frozen symbol equals the live value. The per-site snapshot **removes the
   global constancy requirement** — mutations before the snapshot are read,
   mutations after are irrelevant.

## Why per-site matters (the loop case)

```fortran
do i = 1, n
  allocate(tmp(sz(i)))     ! extent = sz(i), read HERE
  ...
  sz(i) = sz(i) + 100      ! same array mutated in the same iteration
  deallocate(tmp)
end do
```

The extent must snapshot `sz(i)` **at the allocate**, before the write, **every
iteration**. An entry-time snapshot (the old rule) is wrong twice: `i` is a loop
variable, not a constant index; and `sz` is not loop-invariant. The extent
symbol is **non-free** (defined inside the loop), so DaCe Scope-allocates `tmp`
inside the loop body with a per-iteration size — no over-allocation, no
1-past-end write. This is the heap bug fixed in `0fbe410`;
`tests/value_projection_ssa_test.py` is the regression (glibc aborts on the
corrupt free; ASan pinpoints the write — see `scripts/lint_generated_kernel.py`).

## Existing bridge mechanisms are points on the design space

| mechanism | where it lives | version granularity |
|---|---|---|
| `internPosSymbol __sym_arr_7` | assigns.cpp / expressions.cpp | const index, **immutable** array, single entry snapshot |
| `<arr>_at<gid>` per-read-site | assigns.cpp (b) + access.py | **mutable** array / runtime index — one version **per read site** |
| `value_symbols __sym_x` (shape) | extract_vars.cpp + `_seed_value_symbols` | one **entry** snapshot; `_check_value_symbols_constant` refuses on write |
| `VersionShapeScalars <name>_ext<k>` | MLIR pass | per-ALLOCATE extent version / freeze |
| `2cbd1fe`+`0fbe410` element-extent | dispatch.cpp | per-alloc element extent → synthetic, bound in scope |

**Coverage today.** Data-access *indices* are already per-site: `internPosSymbol`
gates to immutable arrays (assigns.cpp:209); a **mutable** or runtime-indexed read
falls through to `<arr>_at<gid>`, which re-reads the element at each use — reaching-def
VPS in all but name. `z(tab(sel))` with `tab` written between two uses builds and is
correct (`tests/array_value_as_symbol_test.py::test_value_symbol_reaching_def_resnapshot`).
Allocate *extents* are per-site via `VersionShapeScalars` / `0fbe410`.

**The automatic-extent refusal (closed).** `__sym_<arr>_<idx>` used as an
automatic-array **extent** (`work(sizes(sel))`) is a single entry snapshot. It was
guarded by a constancy check that refused *any* write to the source — **spurious**,
because Fortran evaluates an automatic-array bound once at entry and freezes it, so a
later write cannot change it (verified: `size(work)` stays the entry value). No
freeze-point flag is needed to fix it: `value_symbols_` is minted at exactly one site
(`resolveShapeSyms`, extract_vars.cpp:184) and only from a declare shape operand, so
**every** entry in the list is an entry-frozen automatic extent. Allocatable ALLOCATE
extents never enter it — they take `internPosSymbol` / `<arr>_at<gid>`. The check is
now narrowed to a pure soundness guard: refuse only if such a symbol *leaked into a
data-access subset* (where the entry snapshot could go stale) — unreachable via the
current mint paths, kept as defense. `work(sizes(sel))` with the source written now
builds and is correct (`tests/array_value_as_symbol_test.py::test_value_symbol_automatic_extent_source_write_allowed`).

## Coverage — all four symbolic-use contexts

| context | example | mechanism | verified |
|---|---|---|---|
| **extent** | `work(sizes(sel))`, `work(mat(i,j))`, `allocate(buf(sz(i)))` | `__sym` entry seed (single/multi-index automatic), `VersionShapeScalars`/`0fbe410` (scalar/allocatable) | size_expr 15/15 |
| **data-access index** | `z(tab(sel))`, `idx1(idx2(i))` | `<arr>_at<gid>` per read site | array_value_as_symbol 3/3 |
| **loop range** | `do k = 1, tab(sel)` (trip count frozen at entry) | per-site read | probe P2 |
| **interstate condition** | `if (tab(sel) > 0)` | per-site read | probe P3 |

Multi-index element extents (`work(mat(i,j))`, both indices runtime) were the last hard
failure — they collapsed `mat` to the bare array name and collided it with a symbol;
`arrayElementExtent` now emits a comma-joined multi-index value-symbol (`__sym_mat_i_j`,
seeded `mat[(i)-1, (j)-1]`). VPS is functionally complete across the four contexts.

The unifying frame — symbol-need lattice + reaching-def versioning + dominating-edge
snapshot — describes all of the above as one pass; that write-up is the paper's frontend
contribution (the implementations still live in three layers: C++ mint, MLIR
`VersionShapeScalars`, Python seed / `access.py`).
