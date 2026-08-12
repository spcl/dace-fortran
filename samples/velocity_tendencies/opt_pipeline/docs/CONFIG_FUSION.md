# Maps fused by config propagation

Freezing `{lvn_only, istep}` (+ fixed `lextra_diffu`, `ldeepatmo`) removes the
`if` barriers between same-shape maps, so `LiftTrivialIf` + map fusion merge them.

| Site (velocity_modified.f90) | Barrier removed | Fused result |
|---|---|---|
| L444-461 `istep==1` edge block | `IF(.NOT.lvn_only)` around `z_vt_ie` (L450) | `vn_ie`/`z_kin_hor_e` + `z_vt_ie` + `z_w_concorr_me` → 1 edge map |
| L432-478 whole `istep==1` block | `IF(istep==1)` | block lifted/dropped per variant; inner maps fuse straight-line |
| L483-492 vs L493-502 | `IF(.NOT.lvn_only)` / `…ldeepatmo` | `z_v_grad_w` map kept once, deepatmo arm pruned |
| L573 `IF(lvn_only) CYCLE` | dynamic cycle | cell loop tail fuses into the block map |
| L608-633 edge tendencies | `IF(.NOT.ldeepatmo)` | `ddt_vn_apc_pc` + `ddt_vn_cor_pc` → 1 map, deep-atmo `ELSE` pruned |

Per-variant: `lvn_only=1` prunes the `z_vt_ie`/`z_v_grad_w` maps entirely;
`istep=2` drops the whole L432-478 block.

## Always-fixed configs

`_FIXED = {lextra_diffu: 1, ldeepatmo: 0}`:

- `lextra_diffu` extra-diffusion blocks (L586-597, L635).
- `ldeepatmo` deep-atmosphere `ELSE` arms (L493-502, L608-633).

## int32 -> uint16/uint8 neighbor lists


(Possible because we can detect `*_idx` and `*_blk` arrays are only used for array accesses and thus their range is within the range of the arrays they are used to access to)
- `*_idx` neighbor lists -> **uint16** under `nproma*nblks_* < 65536`.
- multi-valued `*_blk` -> **uint8** under `nblks_* < 256`.
- single-valued `*_blk` -> folded to literal `1`, array dropped.

Each emits a compressed clone + runtime dispatch; halves/quarters index traffic
in the fused maps above.

## Monomorphization (vtable -> if-else)

In ocean all solvers provide implementas to the abstract `%solve` function.
We first monomorphize the functions:
`if (solver_type == 1) %solve_type_1`

Then use the configuration to remove dead/unused solvers.


