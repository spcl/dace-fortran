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

## `keep_external` — same registry, intent-first API

`keep_external(name, args=…, libraries=…)` is a thin wrapper around
`register_external`.  Functionally identical (same registry, same
lookup, same link-flag merge); the distinct name surfaces the intent
("leave this procedure external; do not lower it into a kernel")
without forcing the caller to spell out an `ExternalSignature` object.

`c_name` defaults to the Fortran call-site name (the common case;
override only when the `bind(c)` symbol uses a different label).  The
`bar`-flavoured `test_keep_external.py` covers the same end-to-end
flow as the `foo` test (compile separately, register, build, run,
assert the mutation).

## Real-world target: ICON `velocity_tendencies`

The intent for `keep_external` is calls like ICON's
`velocity_tendencies` (`mo_velocity_advection` in the real source,
`fake_mo_velocity_advection` in the carved fake DyCore).  Its 14-arg
signature is:

```fortran
SUBROUTINE velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag,
                               z_w_concorr_me, z_kin_hor_e, z_vt_ie,
                               ntnd, istep, lvn_only, dtime,
                               dt_linintp_ubc, ldeepatmo)
  TYPE(t_nh_prog),    INTENT(INOUT) :: p_prog
  TYPE(t_patch),      TARGET, INTENT(IN) :: p_patch
  TYPE(t_int_state),  TARGET, INTENT(IN) :: p_int
  TYPE(t_nh_metrics), INTENT(INOUT) :: p_metrics
  TYPE(t_nh_diag),    INTENT(INOUT) :: p_diag
  REAL(8), DIMENSION(:,:,:), INTENT(INOUT) :: z_w_concorr_me, z_kin_hor_e, z_vt_ie
  INTEGER, INTENT(IN) :: ntnd, istep
  LOGICAL, INTENT(IN) :: lvn_only, ldeepatmo
  REAL(8), INTENT(IN) :: dtime, dt_linintp_ubc
END SUBROUTINE
```

The first 5 args are Fortran derived types: not C-interoperable as-is,
so a direct `register_external`/`keep_external` registration cannot
describe them.  The portable path is a **hand-written `bind(c)`
shim** that takes the derived-type *leaves* (the inner arrays / scalars
the kernel actually reads / writes) as flat C-interoperable pointers
and forwards to the original procedure:

```fortran
subroutine velocity_tendencies_c(                                          &
    ! t_nh_prog leaves the kernel reads/writes ...
    p_prog_w_ptr, p_prog_vn_ptr,                                            &
    ! t_patch leaves ...                                                    &
    p_patch_nblks_c, p_patch_nblks_e, p_patch_nblks_v,                      &
    p_patch_nlev, p_patch_nlevp1, p_patch_nshift,                           &
    ! ... continue for t_int_state, t_nh_metrics, t_nh_diag ...
    z_w_concorr_me_ptr, z_kin_hor_e_ptr, z_vt_ie_ptr,                       &
    ntnd, istep, lvn_only_i8, dtime, dt_linintp_ubc, ldeepatmo_i8)          &
  bind(c, name="velocity_tendencies_c")
  use iso_c_binding
  use mo_nonhydro_types,    only: t_nh_prog, t_nh_diag, t_nh_metrics
  use mo_model_domain,      only: t_patch
  use mo_intp_data_strc,    only: t_int_state
  use mo_velocity_advection, only: velocity_tendencies
  ! Reconstruct the derived types from the flat leaves and forward.
  ...
  call velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag,      &
                           z_w_concorr_me, z_kin_hor_e, z_vt_ie,           &
                           ntnd, istep, lvn_only, dtime,                   &
                           dt_linintp_ubc, ldeepatmo)
end subroutine
```

Register it:

```python
keep_external(
    "velocity_tendencies",                # the Fortran call-site name
    c_name="velocity_tendencies_c",       # the bind(c) shim
    args=[Arg("array", "float64", "inout"),  # p_prog_w_ptr
          Arg("array", "float64", "inout"),  # p_prog_vn_ptr
          Arg("scalar", "int32", "in"),      # p_patch_nblks_c
          ...],
    libraries=["/abs/path/libvelocity_tendencies_shim.so"])
```

The shim is the only Fortran-side work; the registration is mechanical
once the leaf list is fixed.  The bridge then emits one
`ExternalCall` library node per `CALL velocity_tendencies(...)` in
the kernel, exactly as for `foo` / `bar` -- the SDFG `.so` links the
shim's library and the call resolves at run time.
