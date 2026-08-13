#ifndef __VELOCITY_TENDENCIES_ACC_BRIDGE_H__
#define __VELOCITY_TENDENCIES_ACC_BRIDGE_H__

// Entry points of src/velocity_acc_bridge.f90 -- the original single-TU OpenACC Fortran
// velocity_tendencies, driven from the same serde-deserialised structs as the DaCe variants so the
// two engines run on byte-identical inputs and their per-timestep runtimes are comparable.

#include <cstddef>

#include "velocity_tendencies_no_nproma.h"

extern "C" {

// Aborts unless every C++ sizeof matches the generated Fortran mirror; call once before setup.
void velocity_acc_check_sizes(int n, const size_t *sizes);

// Binds the Fortran derived types onto these buffers (POINTER components) or copies them
// (ALLOCATABLE components) and stages everything on the device. zdims/zlbs are the shapes and
// Fortran lower bounds of z_kin_hor_e, z_vt_ie, z_w_concorr_me, in that order, 3 entries each.
void velocity_acc_setup(const global_data_type *gd, const t_nh_diag *diag, const t_int_state *intp,
                        const t_nh_metrics *metrics, const t_patch *patch, const t_nh_prog *prog,
                        const double *z_kin_hor_e, const double *z_vt_ie, const double *z_w_concorr_me,
                        const int *zdims, const int *zlbs, int ntnd, int istep, int lvn_only,
                        int ldeepatmo, double dtime, double dt_linintp_ubc);

// One call of velocity_tendencies, ACC-synchronised on both sides; returns milliseconds.
double velocity_acc_run(void);

double velocity_acc_max_vcfl(void);

// Copies the write set back off the device into the buffers passed to setup.
void velocity_acc_teardown(void);
}

// Order must match STRUCT_ORDER in tools/gen_acc_mirror.py.
inline void velocity_acc_assert_layout() {
  const size_t sizes[] = {
      sizeof(t_grid_edges), sizeof(t_int_state),  sizeof(t_nh_prog),       sizeof(global_data_type),
      sizeof(t_grid_domain_decomp_info), sizeof(t_grid_cells), sizeof(t_nh_metrics),
      sizeof(t_grid_vertices), sizeof(t_patch),   sizeof(t_nh_diag),
  };
  velocity_acc_check_sizes(static_cast<int>(sizeof(sizes) / sizeof(sizes[0])), sizes);
}

#endif // __VELOCITY_TENDENCIES_ACC_BRIDGE_H__
