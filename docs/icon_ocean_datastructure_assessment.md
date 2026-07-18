# ICON-O kernel data-structure assessment: numerics vs. I/O / bookkeeping

Scope: four ICON ocean compute kernels planned for extraction into self-contained Fortran + DaCe SDFG lowering. Classification per derived type/member touched:

| Code | Meaning | Disposition for DaCe extraction |
|------|---------|---------------------------------|
| **N** | Numeric floating-point data the kernel reads/writes | Flatten → plain `REAL(wp)` array argument(s) |
| **C** | Connectivity / mesh-topology **integer index** array (gather) | Pass as plain `INTEGER` array argument(s). NOT a list — fully supportable. |
| **P** | Parallel / halo bookkeeping (subset ranges, `sync_patch_array`, owned/in_domain) | Reduce to plain loop-bound scalars / drop / external-stub |
| **I** | I/O / metadata / var_list / diagnostics / timers / `dbg_print` | Drop or make external call |

Global scalars used by all kernels (module-level, not struct members — pass as scalar args): `nproma` (`mo_parallel_config`), `n_zlev` (`mo_ocean_nml`), `dtime` (`mo_run_config`), `no_dual_edges` (= `verts%max_connectivity`, `mo_operator_ocean_coeff_3d`).

DSL macro expansion (`src/include/iconfor_dsl_definitions.inc`), needed to read the type defs:
- `onCellsBlock` → `REAL(wp), POINTER, DIMENSION(:,:)` (a per-block **2-D** array)
- `mapEdgesToCells` → `REAL(wp), POINTER, DIMENSION(:,:,:,:)`
- `mapEdgesToEdges` → `REAL(wp), POINTER, DIMENSION(:,:,:,:)`
- `onEdges_3D_Int` → `INTEGER, POINTER, DIMENSION(:,:,:)`

**Headline result — no kernel needs any list/registry structure for its numerics.** `grep` for `add_var | add_ref | t_var_list | new_var_list` returns **0** in all four kernel modules. `t_var_list`, `t_hydro_ocean_state/prog/diag/aux`, the `*_Pointer_3d_wp` linked pointer lists, and `t_ocean_monitor` are **never referenced** inside these kernel bodies.

---

## Kernel 1 — `upwind_vflux_ppm_onBlock`
`tracer_transport/mo_ocean_tracer_transport_vert.f90:213-551`

Signature args lines 213-231. Body member-accesses confirmed by grep: only `ppmCoeffs%<9 members>` (264-272); **no `patch%` access inside the onBlock body**. Already receives everything as flat arrays *except* the PPM coefficients.

| Structure | Member (file:line) | Class | Concrete flat shape | Disposition |
|-----------|--------------------|-------|---------------------|-------------|
| (scalar args) | `tracer` 221, `w` 222, `cell_thickeness` 224, `cell_invheight` 225, `flux_div_vert` 228 | N | `(nproma,n_zlev)` (`w`: `(nproma,n_zlev+1)`) | already plain arrays — keep |
| (scalar args) | `dtime` 223, `vertical_limiter_type` 227, `startIndex`/`endIndex` 229 | N/P | scalars | keep as scalars |
| (int array arg) | `cells_noOfLevels(nproma)` 230 | C | `(nproma)` | already plain int array — keep (per-column level count = local dolic_c slice) |
| `t_verticalAdvection_ppm_coefficients` (def `dynamics/mo_ocean_types.f90:56-71`) | `cellHeightRatio_This_toBelow` 264 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_This_toThisBelow` 265 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeight_2xBelow_x_RatioThis_toThisBelow` 266 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_This_toThisAboveBelow` 267 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_2xAboveplusThis_toThisBelow` 268 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_2xBelowplusThis_toThisAbove` 269 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_ThisAbove_to2xThisplusBelow` 270 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeightRatio_ThisBelow_to2xThisplusAbove` 271 | N | `(nproma,n_zlev)` | flatten → array arg |
| ″ | `cellHeight_inv_ThisAboveBelow2Below` 272 | N | `(nproma,n_zlev)` | flatten → array arg |

`t_verticalAdvection_ppm_coefficients` = nine `onCellsBlock` = nine `(nproma,n_zlev)` real arrays (def comment line 60: "all dimensions are (nproma, levels)"). Pure container; trivially flattened.

Calls to stub/external: **none** — kernel calls only `set_acc_host_or_device` (ACC host flag — drop) and `v_ppm_slimiter_mo_onBlock` (line 444, in-module pure-array helper — inline/extract alongside, don't stub; takes the same flat arrays).

Proposed flat interface:
```
upwind_vflux_ppm_onBlock_flat(
  tracer(nproma,n_zlev), w(nproma,n_zlev+1), dtime,
  vertical_limiter_type, cell_thickeness(nproma,n_zlev), cell_invheight(nproma,n_zlev),
  ppm_c1(nproma,n_zlev), ... ppm_c9(nproma,n_zlev),       ! the 9 coeff arrays
  flux_div_vert(nproma,n_zlev),                            ! INOUT result
  startIndex, endIndex, cells_noOfLevels(nproma))
```
- Numeric arrays needed: **9 (ppm coeffs) + 5 (tracer,w,cell_thickeness,cell_invheight,flux_div_vert) = 14 real arrays**, plus `cells_noOfLevels` int array.
- Needs any list/var_list? **NO.**
- External/stub calls: **none** (drop `set_acc_host_or_device`; inline `v_ppm_slimiter_mo_onBlock`).

---

## Kernel 2 — `limiter_ocean_zalesak_horizontal_onTriangles`
`tracer_transport/mo_ocean_limiter.f90:587-1059`

Pointers bound at lines 656-679, used in the body. Connectivity is a triangle stencil (each cell has exactly 3 edges / 3 neighbour cells; edge has 2 cells).

| Structure → member (file:line) | Class | Concrete flat shape | Disposition |
|--------------------------------|-------|---------------------|-------------|
| arg `vert_velocity` 601 | N | `(nproma,n_zlev+1,nblks_c)` | keep (note: declared INOUT but body does not read it — candidate to drop; see note) |
| arg `tracer` 602 | N | `(nproma,n_zlev,nblks_c)` | keep |
| arg `p_mass_flx_e` 603 | N | `(nproma,n_zlev,nblks_e)` | keep (declared INOUT; body does not use — droppable) |
| arg `flx_tracer_low` 604 | N | `(nproma,n_zlev,nblks_e)` | keep (read) |
| arg `flx_tracer_high` 605 | N | `(nproma,n_zlev,nblks_e)` | keep (read) |
| arg `flx_tracer_final` 606 | N | `(nproma,n_zlev,nblks_e)` | keep (write, result) |
| arg `div_adv_flux_vert` 607 | N | `(nproma,n_zlev,nblks_c)` | keep (read) |
| arg `h_old` 609 / `h_new` 610 | N | `(nproma,nblks_c)` | keep (read) |
| `patch_3d%p_patch_2d(1)` (alloc_cell_blocks, nblks_e/_c) | P | scalars | become array-extent scalars `nblks_c`, `nblks_e` |
| `patch_2d%edges%in_domain` 657 → `%start_block`/`%end_block` 661-662 | P | scalars | loop bounds `edges_start_block/end_block` |
| `patch_2d%cells%in_domain` 658 → `%start_block`/`%end_block` 663-664 | P | scalars | loop bounds `cells_start_block/end_block` |
| `get_index_range(... )` 709,733,886,1013 → `start_index,end_index` | P | scalars | per-block bounds; precompute as `(nblks)` int arrays or pass start/end arrays |
| `patch_2d%edges%cell_idx` / `cell_blk` 666-667 (`cellOfEdge_*`) | C | `(nproma,nblks_e,2)` | int gather arrays (edge→its 2 cells) |
| `patch_2d%cells%edge_idx` / `edge_blk` 668-669 (`edge_of_cell_*`) | C | `(nproma,nblks_c,3)` | int gather arrays (cell→its 3 edges) |
| `patch_2d%cells%neighbor_idx` / `neighbor_blk` 670-671 | C | `(nproma,nblks_c,3)` | int gather arrays (cell→3 neighbour cells) |
| `patch_3d%p_patch_1d(1)%dolic_e` 673 | C | `(nproma,nblks_e)` | int per-edge bottom-level array |
| `patch_3d%p_patch_1d(1)%dolic_c` 674 | C | `(nproma,nblks_c)` | int per-cell bottom-level array |
| `operators_coefficients%div_coeff` 675 | N | `(nproma,n_zlev,nblks_c,3)` | real divergence coeff (4th dim = 3 edges) — flatten to array |
| `patch_3d%p_patch_1D(1)%prism_thick_flat_sfc_c` 676 | N | `(nproma,n_zlev,nblks_c)` | real array |
| `patch_3d%p_patch_1d(1)%del_zlev_m` 677 | N | `(n_zlev)` | real 1-D array |
| `patch_3D%p_patch_1d(1)%inv_prism_thick_c` 678 | N | `(nproma,n_zlev,nblks_c)` | real array |
| `operators_coefficients%edges_SeaBoundaryLevel` 679 | C | `(nproma,n_zlev,nblks_e)` (`onEdges_3D_Int`) | int boundary-level mask (read at line 1025) |
| `sync_patch_array_mult(sync_c1,...)` 869, 1006 | P | — | **external-stub / drop** (halo exchange; serial-extract = no-op) |
| `p_test_run` 696 (logical global) | I/P | scalar | drop test-only zeroing branch |
| `dbl_eps` 979, 984 (global tiny) | N | scalar constant | keep as parameter |

Notes: `vert_velocity`/`p_mass_flx_e` are in the signature but `onTriangles` body never reads them (grep) — interface-compat carryovers, **droppable**. All connectivity is fixed-arity triangle topology (3 edges/3 neighbours per cell, 2 cells per edge) — `(...,3)`/`(...,2)` index dims are compile-time constants, clean gather.

Proposed flat interface (after stripping):
```
limiter_zalesak_onTriangles_flat(
  ! numeric fields
  tracer(nproma,n_zlev,nblks_c), flx_tracer_low(nproma,n_zlev,nblks_e),
  flx_tracer_high(nproma,n_zlev,nblks_e), flx_tracer_final(nproma,n_zlev,nblks_e),  ! OUT
  div_adv_flux_vert(nproma,n_zlev,nblks_c), h_old(nproma,nblks_c), h_new(nproma,nblks_c),
  div_coeff(nproma,n_zlev,nblks_c,3), prism_thick_flat_sfc_c(nproma,n_zlev,nblks_c),
  del_zlev_m(n_zlev), inv_prism_thick_c(nproma,n_zlev,nblks_c),
  ! integer connectivity / topology
  cellOfEdge_idx(nproma,nblks_e,2), cellOfEdge_blk(nproma,nblks_e,2),
  edge_of_cell_idx(nproma,nblks_c,3), edge_of_cell_blk(nproma,nblks_c,3),
  neighbor_cell_idx(nproma,nblks_c,3), neighbor_cell_blk(nproma,nblks_c,3),
  dolic_e(nproma,nblks_e), dolic_c(nproma,nblks_c),
  edges_SeaBoundaryLevel(nproma,n_zlev,nblks_e),
  ! scalars / bounds
  dtime, dbl_eps, nblks_c, nblks_e,
  edges_start_block, edges_end_block, cells_start_block, cells_end_block,
  edges_start_index(nblks_e), edges_end_index(nblks_e),     ! materialise get_index_range
  cells_start_index(nblks_c), cells_end_index(nblks_c))
```
- Numeric arrays needed: **~11 real** (`tracer, flx_tracer_low/high/final, div_adv_flux_vert, h_old, h_new, div_coeff, prism_thick_flat_sfc_c, del_zlev_m, inv_prism_thick_c`) + **9 integer** connectivity/topology arrays (`cellOfEdge_idx/blk, edge_of_cell_idx/blk, neighbor_cell_idx/blk, dolic_e, dolic_c, edges_SeaBoundaryLevel`).
- Needs any list/var_list? **NO.**
- External/stub calls: **`sync_patch_array_mult` ×2 (drop / external-stub)**; drop the `p_test_run` zeroing branch and `set_acc_host_or_device`; replace `get_index_range` with precomputed start/end index arrays (or plain `1..nproma` bounds in single-block extract).

---

## Kernel 3 — `velocity_diffusion_vertical_implicit_onBlock`
`dynamics/mo_ocean_velocity_diffusion.f90:1078-1272`

Column tridiagonal solve, one edge-block. `velocity`/`a_v` already arrive as plain `(:,:)` arrays; only struct touched is `patch_3d%p_patch_1d(1)`. `operators_coefficients` is in the signature but the body **never accesses any of its members** (grep).

| Structure → member (file:line) | Class | Concrete flat shape | Disposition |
|--------------------------------|-------|---------------------|-------------|
| arg `velocity(:,:)` 1086 | N | `(nproma,n_zlev)` | keep (INOUT, the solve result) |
| arg `a_v(:,:)` 1087 | N | `(nproma,n_zlev)` | keep (read, vertical viscosity) |
| arg `operators_coefficients` 1088 | — | — | **unused in body → drop entirely** |
| arg `start_index,end_index,edge_block` 1089 | P | scalars | keep as scalars |
| `patch_3d%p_patch_1d(1)%dolic_e` 1106,1115 | C | `(nproma,nblks_e)` | int per-edge bottom-level array (only `(:,edge_block)` slice used) |
| `patch_3d%p_patch_1d(1)%inv_prism_thick_e` 1125 | N | `(nproma,n_zlev,nblks_e)` | real array (only `(:,:,edge_block)` slice) |
| `patch_3d%p_patch_1d(1)%inv_prism_center_dist_e` 1126 | N | `(nproma,n_zlev,nblks_e)` | real array (only `(:,:,edge_block)` slice) |
| `dtime` (global `mo_run_config`) 1151,1165,1166,1179 | N | scalar | pass as scalar |
| `eliminate_upper_diag` (global logical) 1186 | P/config | scalar | pass as scalar (selects which elimination branch) |

Proposed flat interface (single edge-block; can pass full 3-D arrays + `edge_block`, or pre-sliced 2-D):
```
velocity_diffusion_vert_implicit_onBlock_flat(
  velocity(nproma,n_zlev),                     ! INOUT
  a_v(nproma,n_zlev),
  dolic_e(nproma,nblks_e), inv_prism_thick_e(nproma,n_zlev,nblks_e),
  inv_prism_center_dist_e(nproma,n_zlev,nblks_e),
  dtime, eliminate_upper_diag,
  start_index, end_index, edge_block, nblks_e)
```
(Or, since only the `edge_block` slice is ever touched: pass `dolic_e_blk(nproma)`, `inv_prism_thick_e_blk(nproma,n_zlev)`, `inv_prism_center_dist_e_blk(nproma,n_zlev)`.)
- Numeric arrays needed: **3 real** (`velocity, a_v` + the two `inv_prism_*` geometry arrays = effectively `velocity, a_v, inv_prism_thick_e, inv_prism_center_dist_e` → 4) + **1 integer** (`dolic_e`).
- Needs any list/var_list? **NO.** (And `operators_coefficients` is dead-passed — drop it.)
- External/stub calls: **none** (drop `set_acc_host_or_device`; the `routine` string + any `finish` error path is non-numeric → drop).

---

## Kernel 4 — `nonlinear_coriolis_3d_fast_scalar`
`math/mo_scalar_product.f90:350-622`

Indirect edge-of-vertex gather (variable arity `verts%num_edges`). Two halves: `.NOT.l_ANTICIPATED_VORTICITY` branch (350-493, default/primary), `l_ANTICIPATED_VORTICITY` branch (495-619). Members below cover both.

| Structure → member (file:line) | Class | Concrete flat shape | Disposition |
|--------------------------------|-------|---------------------|-------------|
| arg `vn` 353 | N | `(nproma,n_zlev,nblks_e)` | keep (read, normal velocity on edges) |
| arg `p_vn_dual` 354 (`t_cartesian_coordinates`) | N (pass-through only) | `(3,nproma,n_zlev,nblks_v)` if flattened | **body never reads `p_vn_dual%x`**; only forwarded to `rot_vertex_ocean_3d` (386). If that call is stubbed/precomputed, `p_vn_dual` can be **dropped** from this kernel. |
| arg `vort_v` 355 | N | `(nproma,n_zlev,nblks_v)` | keep (vorticity at vertices; written by `rot_vertex_ocean_3d`, then read at 480-484) |
| arg `operators_coefficients` 356 | (container) | — | only `%edge2edge_viavert_coeff` used |
| arg `vort_flux` 357 | N | `(nproma,n_zlev,nblks_e)` | keep (OUT result) |
| `operators_coefficients%edge2edge_viavert_coeff` 438,466 (`mapEdgesToEdges`) | N | `(nproma,n_zlev,nblks_e,2*no_dual_edges)` | real coeff; 4th index `vertex_edge` / `no_dual_edges+vertex_edge` → flatten to array |
| `patch_2d%edges%vertex_idx` / `vertex_blk` 402-405 | C | `(nproma,nblks_e,4)` (uses idx 1,2) | int gather (edge→its 2 endpoint vertices) |
| `patch_2d%verts%num_edges` 421,448 | C | `(nproma,nblks_v)` | int arity per vertex (loop trip count — data-dependent gather length) |
| `patch_2d%verts%edge_idx` / `edge_blk` 423-424,450-451 | C | `(nproma,nblks_v,6)` | int gather (vertex→its ≤6 edges) |
| `patch_3d%p_patch_1d(1)%dolic_e` 426-427,453-454,474,582 | C | `(nproma,nblks_e)` | int per-edge bottom-level |
| `patch_2d%verts%f_v` 480,484,588,592 | N | `(nproma,nblks_v)` | real Coriolis param at vertices |
| `patch_3d%p_patch_1d(1)%prism_thick_e` 539,567 | N | `(nproma,n_zlev,nblks_e)` | real (ANTICIPATED branch only) |
| `patch_2d%edges%primal_edge_length` 600 | N | `(nproma,nblks_e)` | real (ANTICIPATED branch only) |
| `patch_2d%edges%in_domain` 379 → start/end_block 395,504 | P | scalars | loop bounds |
| `get_index_range(...)` 396,505 → start/end_edge_index | P | scalars | per-block bounds → precompute / plain `1..nproma` |
| `rot_vertex_ocean_3d(...)` 386 | — | — | **external call** — computes `vort_v` from `vn,p_vn_dual,opcoeff`. Either (a) precompute `vort_v` upstream and pass it in (then drop `p_vn_dual` + the call), or (b) extract `rot_vertex_ocean_3d` as a second kernel. It is itself an indirect vertex gather (`mo_ocean_math_operators.f90`). |
| `sync_patch_array(SYNC_V, patch_2D, vort_v)` 388 | P | — | **external-stub / drop** (halo exchange) |
| `no_dual_edges` 466,555 (global, = `verts%max_connectivity`) | scalar | scalar | pass as scalar (offset into 4th coeff index) |
| `l_ANTICIPATED_VORTICITY` 390,495 (global logical) | config | scalar | compile-time / pass scalar; default `.FALSE.` selects the simple branch |
| `vort_flux_old` 376,498,594 | N (scratch) | `(nproma,n_zlev,nblks_e)` | debug-only local in ANTICIPATED branch — drop |

Proposed flat interface (default `.NOT.l_ANTICIPATED_VORTICITY` branch, with `vort_v` precomputed upstream so `rot_vertex_ocean_3d` and `p_vn_dual` are out):
```
nonlinear_coriolis_3d_fast_scalar_flat(
  vn(nproma,n_zlev,nblks_e),
  vort_v(nproma,n_zlev,nblks_v),                 ! IN (precomputed by rot_vertex_ocean_3d)
  edge2edge_viavert_coeff(nproma,n_zlev,nblks_e,2*no_dual_edges),
  f_v(nproma,nblks_v),
  edge_vertex_idx(nproma,nblks_e,4), edge_vertex_blk(nproma,nblks_e,4),
  verts_num_edges(nproma,nblks_v),
  verts_edge_idx(nproma,nblks_v,6), verts_edge_blk(nproma,nblks_v,6),
  dolic_e(nproma,nblks_e),
  vort_flux(nproma,n_zlev,nblks_e),              ! OUT
  no_dual_edges, n_zlev, nproma,
  edges_start_block, edges_end_block,
  edges_start_index(nblks_e), edges_end_index(nblks_e))
```
- Numeric arrays needed: **4 real** in the default branch (`vn, vort_v, edge2edge_viavert_coeff, f_v` + `vort_flux` OUT = 5) — the ANTICIPATED branch adds `prism_thick_e` and `primal_edge_length` (2 more). **5 (default) / 7 (with ANTICIPATED)** real arrays + **6 integer** connectivity arrays (`edge_vertex_idx/blk, verts_num_edges, verts_edge_idx/blk, dolic_e`).
- Needs any list/var_list? **NO.** `p_vn_dual` is `t_cartesian_coordinates` but is pure pass-through (never `%x`-accessed here) — not needed if `vort_v` is precomputed.
- External/stub calls: **`rot_vertex_ocean_3d`** (precompute `vort_v` upstream and pass it, OR extract as a companion kernel) and **`sync_patch_array(SYNC_V,...)`** (drop / external-stub); drop `set_acc_host_or_device` and the ANTICIPATED-branch debug `vort_flux_old`.

---

## Cross-kernel summary

| # | Kernel | Numeric (real) arrays | Integer connectivity arrays | Needs list/var_list? | External / stubbed calls |
|---|--------|----------------------|-----------------------------|----------------------|--------------------------|
| 1 | `upwind_vflux_ppm_onBlock` | **14** (9 PPM coeffs + tracer,w,cell_thickeness,cell_invheight,flux_div_vert) | 1 (`cells_noOfLevels`) | **NO** | none (inline `v_ppm_slimiter_mo_onBlock`; drop ACC host flag) |
| 2 | `limiter_ocean_zalesak_horizontal_onTriangles` | **~11** (tracer, flx_low/high/final, div_adv_flux_vert, h_old, h_new, div_coeff, prism_thick_flat_sfc_c, del_zlev_m, inv_prism_thick_c) | **9** (cellOfEdge_idx/blk, edge_of_cell_idx/blk, neighbor_cell_idx/blk, dolic_e, dolic_c, edges_SeaBoundaryLevel) | **NO** | `sync_patch_array_mult` ×2 (drop); `get_index_range`→precomputed bounds; drop `p_test_run` branch |
| 3 | `velocity_diffusion_vertical_implicit_onBlock` | **4** (velocity, a_v, inv_prism_thick_e, inv_prism_center_dist_e) | 1 (`dolic_e`) | **NO** | none (`operators_coefficients` is dead-passed → drop; drop ACC host flag) |
| 4 | `nonlinear_coriolis_3d_fast_scalar` | **5** default (vn, vort_v, edge2edge_viavert_coeff, f_v, vort_flux) / **7** with ANTICIPATED (+prism_thick_e, +primal_edge_length) | **6** (edge_vertex_idx/blk, verts_num_edges, verts_edge_idx/blk, dolic_e) | **NO** | `rot_vertex_ocean_3d` (precompute `vort_v` upstream or extract as companion); `sync_patch_array(SYNC_V)` (drop); `p_vn_dual` droppable |

### Confirmations requested by the brief
- **`add_var`/`add_ref`/`t_var_list` NOT used in any of the four kernel modules** — `grep -c` returns 0 in all four files. Var-list/state container types (`t_hydro_ocean_state/prog/diag/aux`, the `*_Pointer_3d_wp` linked pointer lists, `t_ocean_monitor`) are field-allocation + I/O-metadata registries, never appear in compute bodies. Safe to exclude from extraction. **Confirmed.**
- **`t_cartesian_coordinates`** (`externals/iconmath/src/support/mo_math_types.f90:31-33`) — `BIND(C)` type, single member `REAL(wp) :: x(3)`, trivially flattenable to a leading `(3,...)` dim. Only kernel 4 has it (`p_vn_dual`); body never dereferences `%x` (pure pass-through to `rot_vertex_ocean_3d`).
- **`t_subset_range`** (`grid/mo_model_domain.f90:92-113`) — only members read are scalar `start_block`/`end_block` (+ via `get_index_range`, `start_index`/`end_index`). All P-class → plain loop-bound scalars. `TYPE(t_patch), POINTER :: patch` + `name`/`decomp_info` never touched.
- **Connectivity members** in `t_grid_cells/edges/vertices` (`neighbor_idx/blk`, `edge_idx/blk`, `vertex_idx/blk`, `cell_idx/blk`, `num_edges`) — all `INTEGER, ALLOCATABLE (:,:[,:])` mesh-topology arrays, compile-time-constant last dims (2/3/4/6) — clean integer gather args, **not** lists.
- **`t_patch_vert` (`p_patch_1d`)** members used (`dolic_c/e`, `del_zlev_m`, `prism_thick_e`, `prism_thick_flat_sfc_c`, `inv_prism_thick_c/e`, `inv_prism_center_dist_e`) — all plain `REAL(wp)`/`INTEGER` POINTER arrays `(nproma,[n_zlev,]nblks_*)` — N/C, flatten to plain args.
- **`t_operator_coeff`** (`dynamics/mo_ocean_types.f90:488-566`) — only 3 members touched across all four kernels: `div_coeff` (`mapEdgesToCells`=`(nproma,n_zlev,nblks_c,3)`, kernel 2), `edges_SeaBoundaryLevel` (`onEdges_3D_Int`=`(nproma,n_zlev,nblks_e)`, kernel 2), `edge2edge_viavert_coeff` (`mapEdgesToEdges`=`(nproma,n_zlev,nblks_e,2*no_dual_edges)`, kernel 4). `verticalAdvectionPPMcoeffs` (line 556, a `blockList` of kernel-1's coeff type) is the *only* list-shaped member of `t_operator_coeff`, and **none of the four kernels access it** (kernel 1 gets `ppmCoeffs` directly, not via the operator-coeff list). Rest of `t_operator_coeff` untouched.

**Overall: zero of the four kernels require any list/registry/var_list structure for their numerics.** Every structure touched is flat floating-point data (N), an integer mesh-topology gather array (C), a scalar loop bound/halo-exchange (P → drop/stub), or I/O/diagnostics/timers (I → drop). Two genuinely external calls: `sync_patch_array*` (no-op in single-rank extraction), `rot_vertex_ocean_3d` (kernel 4 only — precompute `vort_v` upstream or extract as a companion kernel).
