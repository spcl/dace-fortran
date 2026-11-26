# Changelog

All notable changes to dace-fortran 1.x, the line on DaCe's FaCe branch.

## 1.4.0

Pins DaCe FaCe 29359ee6cd. Replaces the planned 1.3.1.

### Moved from DaCe FaCe into dace-fortran
- `dace_fortran.specialize`: `specialize_scalars`, `specialize_symbols`, `bake_symbols`, `scope_iterators` and
  `nested_bindings`, which only the Fortran frontend uses. FaCe keeps main's single `specialize_scalar`.
- `dace_fortran.config_propagation`: `ConfigPropagation`, which bakes namelist configurations into external SDFG
  variants behind a dispatcher. Its unit tests and the ICON velocity e2e test come with it.

### Fixes
- `ALLOCATE(x, SOURCE=y)` lowers to an allocation plus an assignment and keeps the source's lower bounds; the bridge
  refuses the forms the pass leaves.
- A module global the kernel allocates and frees within one call becomes a kernel-owned transient; no symbol is read
  off it.
- An inlined optional array dummy named like an entry variable stays its own variable.
- A free symbol with no source is a build error; a sourceless symbol is refused only if it sizes an array.
- Bindings: a member record that points back to its container is walked once (was a RecursionError).
- QE FFT calls transform only the array section they name.
- flang's extent clamp is peeled when a remap view's extent is rendered.
- Builder: two FaCe API differences.

### DaCe FaCe fixes picked up through the pin
- An array that each arm of a conditional sizes is allocated in every arm, and a transient is allocated after the
  branch that sizes it; the late allocation's early exit returns a list (a codegen race fix).
- Complex parts follow DaCe's fft-stockham approach (`.real`/`.imag`).

### Tests
- `vexx_bp_k_gpu` is checked against gfortran on the real and USPP exchange paths.
