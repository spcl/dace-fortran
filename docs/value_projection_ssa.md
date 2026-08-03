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

## Existing bridge mechanisms are degenerate cases

| mechanism | restriction it adds to the general pass |
|---|---|
| `internPosSymbol __sym_arr_7` | constant index, single version |
| `value_symbols __sym_x` | one **entry** snapshot + `_check_value_symbols_constant` |
| `VersionShapeScalars <name>_ext<k>` | per-ALLOCATE extent version (already the idea) |
| `2cbd1fe` element-extent | per-alloc element extent → synthetic, bound in scope |

The full pass — lattice + reaching-def versioning + dominating-edge snapshot with
mutation-between-uses kill/split — subsumes all four. That generalization (all
symbolic-use contexts, not just allocation extents) is the frontend contribution
of the paper.
