# External (`bind(c)`) function calls — design rationale

## Goal

Let a Fortran kernel `CALL` a function that is **compiled separately**
(not in the translation unit the bridge sees) and still produce a
runnable SDFG, by **registering the function's signature**.

## Contract: the target must be `bind(c)`

This is a hard contract, not a limitation we chose arbitrarily:

- Fortran name mangling is **compiler-specific** (gfortran
  `__<mod>_MOD_<name>` / free `name_`; ifort `<mod>_mp_<name>`; flang
  differs).
- A `.mod` file is **compiler-version binary metadata**, *not*
  C-consumable — from generated C++ only the compiled object's
  symbols are callable.

So the only portable, safe way to call a Fortran routine from the
generated C++ is a stable `ISO_C_BINDING` `bind(c, name="…")` symbol
(plus C-interoperable args: by-ref / `value` / `type(c_ptr)`, no
hidden character-length or array-descriptor args).

If the target is **not** `bind(c)` and only a `.mod` + object exist,
write a thin `bind(c)` **shim** that `USE`s the module and forwards
(`subroutine foo_c(...) bind(c,name="foo_c"); use m; call foo(...)`),
compile it against that `.mod` (must match the producing compiler /
version — `.mod`s are not cross-version compatible), link it
alongside, and register the shim's name. No bridge change is needed
to allow that — it's just "register the shim's `c_name`".

## How it lowers

1. **Registry** (`dace_fortran.external`): `register_external(name,
   ExternalSignature(c_name, args=[Arg(kind, dtype, intent), …]))`.
   The signature is supplied out-of-band; the bridge does not parse
   the interface.
2. **Bridge** (`dispatch.cpp`): a non-MPI `fir.call` already yields a
   `kind="call"` node; it now also resolves each operand to a decl
   name (`traceToDecl`) into `call_args`. Harmless for unregistered
   callees.
3. **Builder** (`emit_call`): unregistered callee → no-op (prior
   behaviour preserved); registered → an
   `ExternalCall` **library node** (one expansion to a side-effecting
   CPP tasklet; validates its own connector names). The node carries
   the `extern "C" void <c_name>(…);` declaration and the call
   statement; array args are pointer connectors (read / written per
   `intent`, paired `_aN` / `_aN_o` aliasing the same array),
   scalars by-value, shape-only free symbols referenced inline in
   the call body.
4. **Linking**: `register_external(..., libraries=[libfoo.so])`
   merges, for each library, `-Wl,--no-as-needed <abs .so path>
   -Wl,-rpath,<dir>` into `compiler.linker.args` (the verbatim
   `CMAKE_SHARED_LINKER_FLAGS`, not the CMake-list `DACE_LIBS`).  So
   the SDFG `.so` is **linked against the library with an rpath** —
   self-contained, no `LD_PRELOAD` / load-order dance.
   `-Wl,--no-as-needed` is required because the shared-linker flags
   land *before* the SDFG objects and the default `--as-needed`
   would drop a not-yet-referenced library. `clear_external_registry`
   restores `compiler.linker.args` (the global mutation is
   register/clear-scoped — no leak). (DaCe's own `dtypes.callback`
   is for *Python* callbacks, not native separately-compiled
   functions, so it does not apply here.)

## This test

Compiles a stand-alone `bind(c)` Fortran `foo` (`a(:) = a(:) + 1`)
into its own `libfoo.so`, registers it (with `libraries=[libfoo]`),
builds + runs an SDFG for a kernel that only declares `foo`'s
interface and calls it — the SDFG `.so` links libfoo with an rpath,
so it just runs — and asserts the array was incremented.
