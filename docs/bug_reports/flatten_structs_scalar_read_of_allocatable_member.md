# BUG: `hlfir-flatten-structs` mis-types a scalar that reads one element of an allocatable struct-member array

**Severity:** blocks SDFG lowering of two ICON-O ocean kernels (coriolis_pv, veloc_adv_horz).
**Component:** `dace_fortran/passes/FlattenStructs.cpp` (bridge pass `hlfir-flatten-structs`).
**Status:** root-caused + bisected; reproducer below; FIX NOT applied (core pass — needs a bridge rebuild + full struct-suite regression).

## Symptom

`build_sdfg(...)` raises during `run_passes`:

```
loc("...coriolis_pv.hlfir":3446:13): error: 'hlfir.declare' op first result type
  is inconsistent with variable properties: expected '!fir.ref<i32>'
RuntimeError: run_passes: pipeline failed
```

Source loc 3446 is the `hlfir.declare` of the scalar `edgeOfVertex_index`.

## Minimal source pattern

A plain integer scalar assigned ONE element of an **allocatable** array member of a derived type, then used as an index:

```fortran
INTEGER :: edgeofvertex_index
! patch_2d%verts%edge_idx is  INTEGER, ALLOCATABLE :: edge_idx(:,:,:)
! i.e. fir.box<fir.heap<fir.array<?x?x?xi32>>>  inside t_grid_vertices
edgeofvertex_index = patch_2d % verts % edge_idx(vertex1_idx, vertex1_blk, vertex_edge)
...
this_vort_flux(level,1) = ... vn(edgeofvertex_index, level, edgeofvertex_block) ...
```

(`tests/icon/ocean/coriolis_pv_single_tu.f90:248`; the same routine is embedded in `veloc_adv_horz_single_tu.f90`, so both kernels fail identically.)

## Bisection — the failing pass

Running `DEFAULT_PIPELINE` one pass at a time over the emitted `.hlfir`:

| # | pass | result |
|---|------|--------|
| 21 | `hlfir-marshal-external-structs` | OK |
| **22** | **`hlfir-flatten-structs`** | **FAIL (verifier)** |

So `hlfir-flatten-structs` is the culprit; everything before it verifies.

## What the IR shows

`hlfir-flatten-structs` correctly creates the flat companions for the allocatable members:

```mlir
%74 = hlfir.declare(%73) {uniq_name = "...Epatch_2d_verts_edge_idx"}
     : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> (... , ...)   // box descriptor, OK
```

The post-failure `dump()` shows `edgeOfVertex_index`'s own declare as *consistent* (`!fir.ref<i32>` → `!fir.ref<i32>`) — the invalid state is **transient inside the pass**; the verifier maps the diagnostic back to that scalar's source loc.

## Suspected location

`rewriteDesignate` (FlattenStructs.cpp ~line 1444). The element-read `patch_2d%verts%edge_idx(i,j,k)` is a designate chain whose member companion is a `box<heap<array>>` (allocatable). The whole-member designate (`%verts{edge_idx}`, empty indices) is replaced by the companion via `replaceAllUsesWith(newBase)` (line 1593); a subsequent `fir.load` of the box + element designate then reads the scalar. Mis-typing is in how the **allocatable** companion (`box<heap>`, needs a `load` before the element designate) threads vs the plain `ref<array>` member case the other branches handle. Non-allocatable struct-member element-read works (exercised by the passing struct tests) — the gap is specific to `box<heap<array>>` / `box<ptr<array>>` element-read feeding a scalar.

## Reproducer (no rebuild needed to observe)

```python
import dace_fortran.builder as B
hb = B.hb
from dace_fortran.hlfir_to_sdfg import DEFAULT_PIPELINE
passes = [p.strip() for p in DEFAULT_PIPELINE.split(',')]
upto = passes[:passes.index('hlfir-flatten-structs')]
m = hb.HLFIRModule(); m.parse_file('<coriolis_pv.hlfir>')
m.set_entry_symbol('_QMmo_scalar_productPnonlinear_coriolis_3d_fast_scalar')
m.run_passes(','.join(upto))        # OK
m.run_passes('hlfir-flatten-structs')  # raises the verifier error
print(m.dump())                     # inspect the (transiently) invalid IR
```

Emit the `.hlfir` once with:
```python
from dace_fortran.build import build_sdfg
build_sdfg(open('tests/icon/ocean/coriolis_pv_single_tu.f90').read(),
           entry='mo_scalar_product::nonlinear_coriolis_3d_fast_scalar',
           name='coriolis_pv', out_dir='<dir>')   # writes <dir>/coriolis_pv.hlfir then fails
```

## Acceptance / how to verify a fix

1. `tests/icon/ocean/coriolis_pv_single_tu.f90` and `veloc_adv_horz_single_tu.f90` lower to an SDFG without the verifier error.
2. The two `xfail(strict=True)` params in `tests/icon/ocean/test_ocean_numerical_e2e.py` flip to PASS (remove the marks) — the SDFG binding output then matches the gfortran reference bit-closely.
3. Re-run the full struct-flattening suite (`tests/bindings/`, `tests/icon/`) — no regressions (this is a core pass; that is the real risk).

## Notes

- This is a **bridge pass bug, NOT a binding gap**. The bind(c) binding + shim build fine once the SDFG lowers (proven for ppm_vflux, which has the same struct-flatten machinery but no allocatable-member *scalar element* read).
- Bridge rebuild: `python dace_fortran/build_bridge.py` (CMake cache present under `dace_fortran/build`).
