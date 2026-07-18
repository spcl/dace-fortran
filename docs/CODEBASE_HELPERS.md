# `dace_fortran.flang_codebase` -- driving flang on real-world Fortran codebases

> **Status**: validated against ICON release `icon-2026.04-public` (pinned by the `tests/icon/full/icon-model` submodule); worked examples: [test_velocity_from_icon_source.py](../tests/icon/full/test_velocity_from_icon_source.py), [test_dycore_from_icon_source.py](../tests/icon/full/test_dycore_from_icon_source.py). Companion to tier-3 (`build_sdfg_from_project`), [README](../README.md#tier-3----a-real-cmake--autotools-project).

## Why this exists

Tiers 1+2 ([README](../README.md#the-three-input-tiers)) drive flang internally and merge a hand-curated source list. Tier 3 expects a `compile_commands.json` (`cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` or `bear -- make`). Some projects fit neither:

  * ICON: autotools build, so `cmake --export` is out; `bear -- make` adds a dependency to the inner CI loop.
  * netcdf-fortran: ships only a gfortran-binary `.mod` on every Linux distro (flang can't consume it) -- but can ingest netcdf-fortran's *source* from github.
  * OpenMPI: same problem -- gfortran `.mod` plus a portable `mpif-config.h` / `mpif-constants.h` / ... include set.
  * flang-21: known ICEs / false-positives a real codebase trips into immediately (`WhyNotInteroperableDerivedType` segfault on `BIND(C) INTERFACE` with an unresolved `IMPORT` target; `MPI_SIZEOF` generic-resolution false positive on OpenMPI's `mpif-sizeof.h`).

`dace_fortran.flang_codebase` turns those four into per-codebase configuration rather than per-codebase hand-rolling.

## What it ships

```
dace_fortran/flang_codebase.py
├── 1. Library stubs registry        (LIBRARY_STUBS: dict[str, LibraryStub])
│      * mpi  -- wraps OpenMPI's mpif-*.h
│      * netcdf -- vendors netcdf-fortran's release tarball
├── 2. flang-21 bug-patch registry   (FLANG_BUG_PATCHES: dict[str, callable])
│      * mpi_sizeof -- static byte count for the MPI_SIZEOF call
├── 3. Compile-arg extractor         (extract_make_compile_args)
│      Lifts -D / -I from one `make -n` recipe of any GNU-make build
└── 4. Composer                      (prepare_flang_translation_unit, emit_hlfir_from_codebase)
       Stitches all three together + runs merge_used_modules
```

All four independently usable; the composer is the one-call convenience -- opt out of any layer by dropping it from the kwargs.

## The five-ingredient recipe (for ICON)

Bisected through ICON's USE closure to find the minimum adjustment set that gets flang-21 to lower the real `mo_velocity_advection.f90` (and `mo_solve_nonhydro.f90`) cleanly:

| # | Ingredient | What it does |
|---|---|---|
| 1 | 8 search dirs for `merge_used_modules` | Resolve every USE'd module ICON itself bundles (`src/`, `externals/fortran-support/src`, `externals/mtime/src`, `externals/iconmath/src`, `externals/cdi/src`, `externals/memman/src/bindings/fortran`, `support/`) |
| 2 | `library_stubs=["mpi", "netcdf"]` | mpi wraps `mpif-*.h` (skips `mpif-sizeof.h` -- flang rejects its `COMPLEX(KIND=-1)`); netcdf vendors netcdf-fortran v4.6.2 source |
| 3 | ICON's 15 `-D__NO_*__` defines + `-DNO_MPI_CHOICE_ARG` | Replay the same `#ifdef` gates ICON's own configure-driven build skips |
| 4 | `patches=["mpi_sizeof"]` | Substitute the 6 `CALL MPI_SIZEOF(arg, sz, err)` lines in `mo_mpi` with a static byte count -- flang-21 generic-resolution false positive |
| 5 | `-I src/include` + the OpenMPI include dir | Resolve the `#include "handle_mpi_error.inc"` / `"omp_definitions.inc"` etc. that ICON's source uses, and the `mpif-*.h` family |

All five are inputs to `emit_hlfir_from_codebase`; the only project-specific bits are (1) and (3) -- (2), (4), (5) are universal or auto-detected.

Complete bisect record (which module triggered each step, in order) is in this session's git history. Notable findings:

* `WhyNotInteroperableDerivedType` ICE localises to `mo_util_mtime.f90:304` -- ICON's `BIND(C) INTERFACE` for `julianDeltaToTimeDelta` `IMPORT`s `juliandelta`, a type defined only inside `mtime`. Fix: add `externals/mtime/src` to the merge search dirs.
* `MPI_SIZEOF` false positive lives in `mo_mpi`, called exactly 6 times, all on built-in REAL/INTEGER kinds whose byte size is statically derivable from the arg name. Patch substitutes `sz = 8; err = 0` (or `4` for `i4`/`sp`).
* ICON's `t_PackedMessage` macro-expanded type-bound procedures (LEN_TRIM string scans, `CLASS(*)` polymorphism) survive the flang stage but block the HLFIR-to-SDFG bridge -- marked external stubs in the *test*-side `keep_external` list (never called numerically).

## The end-to-end pattern (for ICON's velocity)

```python
import dace_fortran
from pathlib import Path

ICON_SRC   = Path("tests/icon/full/icon-model")          # the submodule
ICON_BUILD = ICON_SRC / "build" / "stock_cpu"             # ICON's own make

# 1. Extract ICON's -D / -I from its build dir's Makefile.
args = dace_fortran.extract_make_compile_args(
    makefile_dir=ICON_BUILD,
    target="src/atm_dyn_iconam/mo_velocity_advection.o")

# 2. Compose + run flang.
hlfir = dace_fortran.emit_hlfir_from_codebase(
    entry_source=(ICON_SRC / "src/atm_dyn_iconam/mo_velocity_advection.f90").read_text(),
    out_path=tmp_path / "velocity.hlfir",
    search_dirs=[
        ICON_SRC / "src",
        ICON_SRC / "externals/fortran-support/src",
        ICON_SRC / "externals/mtime/src",
        ICON_SRC / "externals/iconmath/src",
        ICON_SRC / "externals/cdi/src",
        ICON_SRC / "externals/memman/src/bindings/fortran",
    ],
    library_stubs=["mpi", "netcdf"],
    patches=["mpi_sizeof"],
    defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
    include_dirs=args["include_dirs"],
    cache_dir=Path("~/.cache/dace-fortran").expanduser(),
)

# 3. Lower to SDFG.
sdfg = dace_fortran.build_sdfg_from_hlfir(
    hlfir, entry="_QMmo_velocity_advectionPvelocity_tendencies")
sdfg.validate()
```

## The dycore case (with `velocity_tendencies` + `sync_patch_array` external)

`mo_solve_nonhydro.f90` (3166 LoC) calls into `mo_sync`'s polymorphic halo-exchange. Same recipe as velocity, plus:

* Register the procedures `solve_nh` *directly* calls as external -- not their downstream callees. `keep_external("sync_patch_array_3d_dp", stub=True)` strips the body before `hlfir-inline-all` runs, so everything it transitively reaches (the polymorphic `mo_communication.exchange_data_*` dispatch chain, the `CLASS(*)` communication-pattern receiver) goes with it -- no need to enumerate downstream callees; the bridge's `hlfir-reject-polymorphism` pass never sees them.
* Register the generic-interface specialisations one by one -- `INTERFACE sync_patch_array` resolves at compile time to `sync_patch_array_3d_dp` / `_2d_int` / ... before HLFIR is emitted. Full list at [test_dycore_from_icon_source.py:107](../tests/icon/full/test_dycore_from_icon_source.py#L107).
* The iso-C wrapper bridging the SDFG's bind-C call site to ICON's polymorphic `sync_patch_array` is at [icon_sync_iso_c.f90](../tests/icon/full/icon_sync_iso_c.f90) -- four bind-C entries; the `c_name` field on the `keep_external` registration resolves to those at runtime.

## Pluggable surface

* `LIBRARY_STUBS`: register a new upstream library via `LibraryStub(name, source, flags)`. `source(**ctx)` returns Fortran source; `flags(**ctx)` returns the `-I` paths flang needs for any `#include`. Both receive the composer's full context (`cache_dir`, `openmpi_include`, ...) as kwargs.
* `FLANG_BUG_PATCHES`: register a `(name, transform)` pair; `transform(source: str) -> str` is a pure source-to-source rewrite. Opt-in via the composer's `patches=` arg (no implicit per-codebase magic).
* `extract_make_compile_args(makefile_dir, target)`: project-agnostic parser for any GNU-make-driven Fortran build whose recipe invokes one compiler line per source (cmake's makefiles fit -- one `mpifort ... -c source.f90` per object).

## Bridge fixes that landed alongside

Real ICON source hit HLFIR constructs the bridge hadn't seen before. Three fixes landed in this session:

1. `buildExpr` fall-through handlers for 10 unhandled ops: `fir.embox` / `hlfir.as_expr` / `hlfir.declare` / `fir.emboxchar` / `fir.box_addr` / `fir.zero_bits` / `hlfir.concat` / `fir.address_of` / `fir.alloca` / `fir.unboxchar` ([expressions.cpp](../dace_fortran/bridge/ast/expressions.cpp)). Each is a pass-through to the underlying value, except `fir.zero_bits` -> `"0"`, `hlfir.concat` -> Python `+`.
2. ICON utility procedures (`finish` / `message` / `timer_start` / ...) marked as external stubs in the [test setup](../tests/icon/full/test_velocity_from_icon_source.py) so their unlowerable bodies (LEN_TRIM scans, polymorphic dispatch) don't reach the bridge.
3. Whole-array fast path in `emit_scalar_assign` for pointer rebinds (`icidx => p_patch%edges%cell_idx`) that `RewritePointerAssigns` doesn't yet collapse ([emit_tasklet.py](../dace_fortran/builder/emit_tasklet.py)) -- when target + source are both multi-dim arrays of the same rank, emit an `AccessNode -> AccessNode` whole-array copy memlet instead of a tasklet with the wrong scalar subset.

A regression-mode diagnostic for the bridge's `?` sentinel is gated on `DACE_FORTRAN_DEBUG_BUILDEXPR=1`; flipping it on prints which op the bridge couldn't decode, so the next codebase's missing case surfaces immediately.
