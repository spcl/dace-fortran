# External-function policy

How to tell dace-fortran that a `CALL` in your Fortran kernel should **not**
be inlined — and, when it isn't, whether the bridge should **emit** it as an
external call or **drop** it entirely.

One declaration drives **both** halves of the toolchain:

* the **inliner** (input Fortran → one self-contained translation unit) leaves
  the named procedure external — its body is stubbed so its unlowerable
  internals (MPI, polymorphic dispatch, string scans, …) never enter the TU;
* the **bridge** (TU → SDFG) either lowers the surviving `CALL` to an
  [`ExternalCall`](../dace_fortran/external.py) library node bound to a C-ABI
  symbol, or drops the call.

## The three behaviours

| Behaviour | Inliner (input → TU) | Bridge (TU → SDFG) |
|---|---|---|
| **inline** (default) | pull the body into the TU | lower it normally |
| **don't-inline + EMIT** | stub the body; leave the `CALL` | emit an `ExternalCall` node, replicating the HLFIR call order, bound to your C-ABI symbol / library |
| **don't-inline + DON'T-emit** (ignore) | stub the body; leave the `CALL` | drop the call (no node) |

Structural invariant: *ignore ⊆ don't-inline*. The argument order/types are
**not** re-authored by you — HLFIR already carries `CALL f(a, b, c)`, so the
bridge replicates it. For an emitted call you supply only what HLFIR cannot
know: the `extern "C"` symbol and the library that exports it.

## Public API

Declare the policy **once per target** with two collections, then apply it to
each half.

```python
from dace_fortran import ExternalFunction, apply_external_functions

EXTERNAL = [                                   # don't-inline + EMIT
    ExternalFunction("sync_patch_array", library=".../libicon_halo.so"),
    ExternalFunction("exchange_data"),         # symbol defaults to the name
    ExternalFunction("my_fn", c_function="my_c_abi_fn", library=".../libfoo.so"),
]
IGNORE = ["finish", "message", "warning", "dbg_print",   # don't-inline + DON'T-emit
          "timer_start", "timer_stop"]
```

* [`ExternalFunction`](../dace_fortran/external_functions.py) — a frozen,
  pure-stdlib dataclass: `name` (the Fortran call-site name), optional
  `c_function` (the `extern "C"` symbol; defaults to `name` via `.symbol`),
  optional `library` (the `.so` that exports it; `None` leaves it unresolved,
  fine for an extract / compile-check flow).
* The derived don't-inline set is computed, never hand-maintained:
  `dont_inline = {f.name for f in external_functions} | set(do_not_emit)`.

### Bridge half

```python
apply_external_functions(EXTERNAL, IGNORE)
```

Validates first (no duplicate emit names, no name in both lists), then registers
each `ExternalFunction` as an emitted external bound to `f.symbol` (with
`f.library` linked in), and each `do_not_emit` name as a dropped call. The
argument plan for an emitted call is **derived from the HLFIR call site**
(array → `inout` pointer, scalar / free-symbol → by-value read), so a minimal
`ExternalFunction(name, c_function, library)` is enough.

### Inliner half

Both inliner engines take the same two collections:

```python
from dace_fortran import inline_to_single_tu, merge_used_modules

inline_to_single_tu(sources, entry=..., out_dir=...,        # fparser pipeline
                    external_functions=EXTERNAL, do_not_emit=IGNORE)

merge_used_modules(text, search_dirs=...,                   # regex text-splicer
                   external_functions=EXTERNAL, do_not_emit=IGNORE)
```

The inliner consumes only the **name union** — both lists are stubbed
identically (opener + spec + `END` kept, body emptied). Only the bridge
distinguishes emit from drop. When the HLFIR build path is used end-to-end
(`build_sdfg_from_*`), the merge sources its keep-external names from the bridge
registry, so the single `apply_external_functions(...)` declaration governs both
halves automatically.

## Worked example: a separately-compiled `bind(c)` function

```fortran
! foo.f90 — compiled to libfoo.so on its own
subroutine foo(a, n) bind(c, name="foo")
  use iso_c_binding
  integer(c_int), value :: n
  real(c_double), intent(inout) :: a(n)
  a = a + 1.0d0
end subroutine foo
```

```python
apply_external_functions([ExternalFunction("foo", library="libfoo.so")])
sdfg = build_sdfg(kernel_src, out_dir, name="run", entry="run_mod::run").build()
# CALL foo(a, n) lowered to an ExternalCall("foo"); the SDFG .so links libfoo.so
# (rpath) so it resolves at load with no LD_PRELOAD.
sdfg(a=a, n=n)
```

See [`tests/external_call/test_external_call.py`](../tests/external_call/test_external_call.py)
and [`test_apply_external_functions.py`](../tests/external_call/test_apply_external_functions.py).

## Contract: the emitted target must be `bind(c)`

Fortran name mangling is compiler-specific and a `.mod` is not C-consumable, so
the only portable way to call a Fortran routine from the generated C++ is a
stable `bind(c, name="…")` symbol — native, or via a thin shim that `USE`s the
module and forwards. Full rationale in
[`tests/external_call/DESIGN.md`](../tests/external_call/DESIGN.md).

## The rich-ABI escape hatch: `keep_external`

`apply_external_functions` covers the common case where the bridge can derive
the argument plan. When the C ABI carries facts HLFIR **cannot** infer, register
an authored [`ExternalSignature`](../dace_fortran/external.py) directly with
`keep_external(name, *, c_name, args=(Arg(...), ...), libraries, ...)`:

* **AoS structs** — `Arg(kind="aos")` (a whole derived-type dummy crossing as a
  packed struct pointer) or `Arg(kind="aos", c_abi="per_member_soa")` (the SoA
  leaves forwarded verbatim to a sibling SDFG / shim).
* **`MPI_Comm`** — `Arg(kind="comm")`: an opaque handle, not a data pointer.
* **`dynamic_extents_abi=True`** — the callee was built with a `bind_c_shim`
  that needs one `int` runtime extent per dynamic dim ahead of each leaf.
* **`module_symbol_forward`** — forward Fortran module globals across a library
  boundary (each library has its own BSS copy under default ELF linking).
* **intent narrowing** — declare an array `intent="in"` / `"out"` when the safe
  derived `inout` over-models the dataflow.

`apply_external_functions` calls `keep_external` under the hood, so the two are
the same registry; reach for the authored form only when one of the above
applies. Examples: the dycore externalisation e2e tests under
[`tests/icon/full/`](../tests/icon/full/) and the layout cases in
[`tests/external_aos_test.py`](../tests/external_aos_test.py).

## Deprecations

The inliner's `keep_external=[names]` parameter is a backward-compatible shim
for `do_not_emit=[names]` and emits a `DeprecationWarning`. Use `do_not_emit`
(or `external_functions`) instead.
