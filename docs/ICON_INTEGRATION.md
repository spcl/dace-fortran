# Full ICON CPU integration with a DaCe-generated velocity_tendencies

ICON is built with its own `configure`. Two things change: `mo_velocity_advection.f90`'s `velocity_tendencies`
body forwards to the DaCe-generated `libvelocity_inner_wrap.so`, and the final ICON link rule gains that library.
Every other source, the build system, the run scripts and the experiments are untouched.

`tests/icon/full/run_icon_e2e.sh` runs the whole flow below and diffs the result against stock ICON; the steps are
listed here so each can be run or debugged on its own.

## Shape

```
ICON mo_solve_nonhydro                        (unchanged)
  └─ CALL velocity_tendencies(...)
       └─ mo_velocity_advection::velocity_tendencies   (body replaced by
            scripts/icon_patches/icon_velocity_dace_dispatch.patch)
            └─ libvelocity_inner_wrap.so      (generated: bindings + bind(c) shim + SDFG)
                 └─ CALL sync_patch_array     (external call back into ICON's mo_sync: the real MPI halo exchange)
```

The SDFG keeps `sync_patch_array` external (`keep_external` / `ExternalFunction`), so at link time it resolves to
ICON's own objects.

## Steps

1. **Build stock ICON first.** Configure an out-of-source build with `scripts/configure_icon_dace_cpu.sh` and
   `DACE_LIBS_DIR` unset, then `make`. The library build needs this configuration's `-D` defines (they decide the
   conditional type layouts) and its `.mod` files.

   ```bash
   cd "$ICON_SRC" && mkdir -p build/dace_cpu && cd build/dace_cpu
   "$DACE_FORTRAN/scripts/configure_icon_dace_cpu.sh" && make -j 8
   ```

   The script pins the Ubuntu apt locations of NetCDF, HDF5, libxml2, eccodes, fyaml and LAPACK (adapt the path
   block at its top for another site), builds fp64 only (no `-D__MIXED_PRECISION`) and uses
   `-O0 -g -fno-fast-math -ffp-contract=off -fPIC`, the flags that keep the stock-vs-DaCe comparison bit-exact.

2. **Build the library from ICON's real source, against that configuration.**

   ```bash
   python -m scripts.build_icon_dace_libs --icon-src "$ICON_SRC" --icon-build "$ICON_SRC/build/dace_cpu" \
       --out-dir "$DACE_LIBS"
   ```

   Without `--icon-src` / `--icon-build` the script lowers the stub-typed `tests/icon/full/velocity_full.f90`,
   which is what the e2e tests use; a library built from it crashes inside ICON because its derived-type layouts
   are not ICON's. `--release` builds every layer with `-O3 -fno-fast-math -ffp-contract=off` (results within one
   ULP). `--with-dycore` also builds `libdycore_wrapper.so`, which ICON does not link; it exercises the
   SDFG-to-SDFG C-ABI chain.

3. **Patch and relink.** Apply `scripts/icon_patches/icon_velocity_dace_dispatch.patch` to the ICON tree, rerun
   the configure script with `DACE_LIBS_DIR="$DACE_LIBS"` (it appends the library to the final link rule in
   `icon.mk`, after ICON's own objects, with `--no-as-needed` and an rpath) and `make` again.

4. **Run** any experiment as for stock ICON, e.g. `mpirun -n 4 ./exp.<experiment>.run`.

5. **Compare** against the stock binary on the same experiment; `run_icon_e2e.sh` does it variable by variable with
   `compare_icon_runs.py`. Under the flags of step 1 the outputs are identical.

## Tests

- `tests/icon/full/test_dycore_velocity_external_e2e.py`: `velocity_tendencies` through the `per_member_soa` C ABI
  with Fortran and C++ sync helpers, bit-exact at `-O0`.
- `tests/icon/full/test_standalone_dycore_sync_e2e.py`: a minimal standalone dycore with a no-op sync.
- `tests/icon/full/test_dycore_mpi_sync_e2e.py`: two ranks with a real `MPI_Sendrecv` halo exchange.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `undefined reference to velocity_tendencies_dace_` | The configure script did not patch `icon.mk` | Rerun it with `DACE_LIBS_DIR` set; `icon.mk` must contain `-l:libvelocity_inner_wrap.so` |
| `libvelocity_inner_wrap.so: cannot open shared object` at run time | The library moved after linking | Rebuild it in its final location, or set `LD_LIBRARY_PATH` |
| Crash in the first `velocity_tendencies` call | The library was built from the stub types, or against another ICON configuration | Step 2 with `--icon-src` / `--icon-build` of the build being linked |
| Outputs differ from stock | A layer was built without `-fno-fast-math -ffp-contract=off` | `build_icon_dace_libs.py` pins all three layers (DaCe C++, the bind(c) shim, the bindings wrapper) |

## Scope

CPU only, fp64 only, one `p_patch` per call. Replacing `solve_nh` itself is the separate
`scripts/build_icon_solve_nh_libs.py` / `scripts/run_icon_solve_nh_atmosphere_e2e.py` flow (the manual
`icon-atmosphere` workflow).
