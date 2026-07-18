# ISSUE: `build_fortran_library(name=...)` diverging from `sdfg.name` silently breaks the loaded `.so` (undefined `__dace_init_*`)

**Severity:** sharp footgun; the `.so` links + builds fine but fails at `dlopen` / first call with `undefined symbol: __dace_init_<name>`.
**Component:** `dace_fortran/bindings/build_fortran_library.py`.
**Status:** worked-around by callers (use `sdfg.name`); a guard or auto-derivation would prevent the trap.

## What happens

`build_sdfg(src, entry=..., name="ppm_vflux")` does NOT rename the SDFG — `sdfg.name` stays the entry procedure name (`upwind_vflux_ppm_onblock`, from `.dacecache`). DaCe exports init/program symbols as `__dace_init_<sdfg.name>` / `__program_<sdfg.name>`.

`build_fortran_library(sdfg, name="ppm_vflux", bind_c_shim=True)` builds the binding's bind(c) symbols from `name` (`build_auto_interface(raw, name)` → `iface.entry = "ppm_vflux"`), so the binding declares/calls `bind(c, name='__dace_init_ppm_vflux')` — a symbol that doesn't exist (the SDFG `.so` exports `__dace_init_upwind_vflux_ppm_onblock`). Link succeeds (rpath-links the SDFG `.so`, but the *symbol name* is wrong); failure only surfaces at load/call:

```
OSError: .../libppm_vflux.so: undefined symbol: __dace_init_ppm_vflux
```

## Why it's easy to hit

`build_fortran_library` documents `name` as "library/base name; defaults to `sdfg.name`" and uses it BOTH for (a) the output `.so` filename and (b) the bind(c) entry symbols. (a) is cosmetic; (b) must equal `sdfg.name`. Any `name` differing from `sdfg.name` — e.g. a friendly kernel key — silently desyncs (b).

## Suggested fix (pick one)

1. **Guard:** if `name` is given and `name != sdfg.name`, raise a clear error (or warn) — "bind(c) symbols key off sdfg.name=`<x>`; pass name=`<x>` or omit it".
2. **Split the param:** keep `name` for the `.so` filename only; always derive the bind(c) entry from `sdfg.name`. This makes a friendly filename safe.
3. **Doc + assert** at minimum: the docstring should state that `name` must equal `sdfg.name` when `bind_c_shim=True`/the binding is emitted.

## Repro / acceptance

```python
sdfg = build_sdfg(src, entry="mod::proc", name="friendly")   # sdfg.name == "proc"
lib  = build_fortran_library(sdfg, name="friendly", bind_c_shim=True, ...)
ctypes.CDLL(lib.so_path)   # -> undefined symbol: __dace_init_friendly
```

Fix makes this either error early (option 1) or work (option 2). The ocean e2e harness (`tests/icon/ocean/_ocean_e2e.py`) sidesteps it today by NOT passing `name=`, using `sdfg.name` everywhere downstream.
