#include "velocity_split_dispatch.h"

#include <stdexcept>

#include "call_velocity.h"

namespace {

// One handle per variant, all brought up by velocity_dace_init. DaCe's __dace_init_* allocates the
// variant's persistent GPU transients, so doing it once here keeps that cost out of every call --
// which is the whole point of separating init from run.
struct Handles {
  velocity_no_nproma_if_prop_lvn_only_0_istep_1Handle_t lvn0_istep1 = nullptr;
  velocity_no_nproma_if_prop_lvn_only_0_istep_2Handle_t lvn0_istep2 = nullptr;
  velocity_no_nproma_if_prop_lvn_only_1_istep_1Handle_t lvn1_istep1 = nullptr;
  velocity_no_nproma_if_prop_lvn_only_1_istep_2Handle_t lvn1_istep2 = nullptr;
  bool up = false;
};

Handles g_handles;

} // namespace

void velocity_dace_init(VELOCITY_ENGINE_PARAMS) {
  if (g_handles.up) {
    return;
  }
  g_handles.lvn0_istep1 = __dace_init_velocity_no_nproma_if_prop_lvn_only_0_istep_1(
      VELOCITY_INIT_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                         z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only,
                         ntnd));
  g_handles.lvn0_istep2 = __dace_init_velocity_no_nproma_if_prop_lvn_only_0_istep_2(
      VELOCITY_INIT_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                         z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only,
                         ntnd));
  g_handles.lvn1_istep1 = __dace_init_velocity_no_nproma_if_prop_lvn_only_1_istep_1(
      VELOCITY_INIT_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                         z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only,
                         ntnd));
  g_handles.lvn1_istep2 = __dace_init_velocity_no_nproma_if_prop_lvn_only_1_istep_2(
      VELOCITY_INIT_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                         z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only,
                         ntnd));
  g_handles.up = true;
}

void velocity_dace_run(VELOCITY_ENGINE_PARAMS) {
  if (!g_handles.up) {
    throw std::runtime_error("velocity_dace_run before velocity_dace_init");
  }
  if (lvn_only == 0 && istep == 1) {
    __program_velocity_no_nproma_if_prop_lvn_only_0_istep_1(
        g_handles.lvn0_istep1,
        VELOCITY_CALL_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                           z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo,
                           lvn_only, ntnd));
  } else if (lvn_only == 0 && istep == 2) {
    __program_velocity_no_nproma_if_prop_lvn_only_0_istep_2(
        g_handles.lvn0_istep2,
        VELOCITY_CALL_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                           z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo,
                           lvn_only, ntnd));
  } else if (lvn_only == 1 && istep == 1) {
    __program_velocity_no_nproma_if_prop_lvn_only_1_istep_1(
        g_handles.lvn1_istep1,
        VELOCITY_CALL_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                           z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo,
                           lvn_only, ntnd));
  } else if (lvn_only == 1 && istep == 2) {
    __program_velocity_no_nproma_if_prop_lvn_only_1_istep_2(
        g_handles.lvn1_istep2,
        VELOCITY_CALL_ARGS(global_data, p_diag, p_int, p_metrics, p_patch, p_prog, z_kin_hor_e,
                           z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo,
                           lvn_only, ntnd));
  } else {
    throw std::runtime_error("velocity_dace_run: (lvn_only, istep) outside {0,1} x {1,2}");
  }
}

void velocity_dace_finalize() {
  if (!g_handles.up) {
    return;
  }
  __dace_exit_velocity_no_nproma_if_prop_lvn_only_0_istep_1(g_handles.lvn0_istep1);
  __dace_exit_velocity_no_nproma_if_prop_lvn_only_0_istep_2(g_handles.lvn0_istep2);
  __dace_exit_velocity_no_nproma_if_prop_lvn_only_1_istep_1(g_handles.lvn1_istep1);
  __dace_exit_velocity_no_nproma_if_prop_lvn_only_1_istep_2(g_handles.lvn1_istep2);
  g_handles = Handles{};
}
