#ifndef __VELOCITY_TENDENCIES_SPLIT_DISPATCH_H__
#define __VELOCITY_TENDENCIES_SPLIT_DISPATCH_H__

// Hides the 4-way (lvn_only, istep) split behind one caller-facing interface shaped like the
// original Fortran binding: init once, call many, finalize once. Callers -- the standalone runner
// here, ICON later -- never name a variant, so the split stays an implementation detail of the
// DaCe pipeline rather than something every call site has to branch on.
//
// The 4 variants share a byte-identical signature (unify_variant_signatures enforces it), so one
// argument list drives all of them and init can bring up all four from the same inputs.

#include "velocity_tendencies_no_nproma.h"

// Same parameter order as velocity_acc_setup, so the two engines stay swappable at the call site.
#define VELOCITY_ENGINE_PARAMS                                                                     \
  global_data_type &global_data, t_nh_diag &p_diag, t_int_state &p_int, t_nh_metrics &p_metrics,   \
      t_patch &p_patch, t_nh_prog &p_prog, double *z_kin_hor_e, double *z_vt_ie,                   \
      double *z_w_concorr_me, double dt_linintp_ubc, double dtime, int istep, int ldeepatmo,       \
      int lvn_only, int ntnd

#define VELOCITY_ENGINE_ARGS                                                                       \
  global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e, z_vt_ie, z_w_concorr_me,    \
      dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only, ntnd

// Brings up all 4 variants. Idempotent: a second call without an intervening finalize is a no-op.
void velocity_dace_init(VELOCITY_ENGINE_PARAMS);

// Dispatches on (lvn_only, istep); throws on any other combination.
void velocity_dace_run(VELOCITY_ENGINE_PARAMS);

// Tears down all 4 variants.
void velocity_dace_finalize();

#endif // __VELOCITY_TENDENCIES_SPLIT_DISPATCH_H__
