#ifndef __DACE_CODEGEN_VELOCITY_TENDENCIES__
#define __DACE_CODEGEN_VELOCITY_TENDENCIES__

#include <dace/dace.h>


struct t_grid_edges {
int __f2dace_SA_area_edge_d_0_s_188 = {};
int __f2dace_SA_area_edge_d_1_s_189 = {};
int __f2dace_SA_cell_blk_d_0_s_167 = {};
int __f2dace_SA_cell_blk_d_1_s_168 = {};
int __f2dace_SA_cell_blk_d_2_s_169 = {};
int __f2dace_SA_cell_idx_d_0_s_164 = {};
int __f2dace_SA_cell_idx_d_1_s_165 = {};
int __f2dace_SA_cell_idx_d_2_s_166 = {};
int __f2dace_SA_end_block_d_0_s_199 = {};
int __f2dace_SA_end_index_d_0_s_197 = {};
int __f2dace_SA_f_e_d_0_s_190 = {};
int __f2dace_SA_f_e_d_1_s_191 = {};
int __f2dace_SA_fn_e_d_0_s_192 = {};
int __f2dace_SA_fn_e_d_1_s_193 = {};
int __f2dace_SA_ft_e_d_0_s_194 = {};
int __f2dace_SA_ft_e_d_1_s_195 = {};
int __f2dace_SA_inv_dual_edge_length_d_0_s_186 = {};
int __f2dace_SA_inv_dual_edge_length_d_1_s_187 = {};
int __f2dace_SA_inv_primal_edge_length_d_0_s_184 = {};
int __f2dace_SA_inv_primal_edge_length_d_1_s_185 = {};
int __f2dace_SA_quad_blk_d_0_s_181 = {};
int __f2dace_SA_quad_blk_d_1_s_182 = {};
int __f2dace_SA_quad_blk_d_2_s_183 = {};
int __f2dace_SA_quad_idx_d_0_s_178 = {};
int __f2dace_SA_quad_idx_d_1_s_179 = {};
int __f2dace_SA_quad_idx_d_2_s_180 = {};
int __f2dace_SA_start_block_d_0_s_198 = {};
int __f2dace_SA_start_index_d_0_s_196 = {};
int __f2dace_SA_tangent_orientation_d_0_s_176 = {};
int __f2dace_SA_tangent_orientation_d_1_s_177 = {};
int __f2dace_SA_vertex_blk_d_0_s_173 = {};
int __f2dace_SA_vertex_blk_d_1_s_174 = {};
int __f2dace_SA_vertex_blk_d_2_s_175 = {};
int __f2dace_SA_vertex_idx_d_0_s_170 = {};
int __f2dace_SA_vertex_idx_d_1_s_171 = {};
int __f2dace_SA_vertex_idx_d_2_s_172 = {};
int __f2dace_SOA_area_edge_d_0_s_188 = {};
int __f2dace_SOA_area_edge_d_1_s_189 = {};
int __f2dace_SOA_cell_blk_d_0_s_167 = {};
int __f2dace_SOA_cell_blk_d_1_s_168 = {};
int __f2dace_SOA_cell_blk_d_2_s_169 = {};
int __f2dace_SOA_cell_idx_d_0_s_164 = {};
int __f2dace_SOA_cell_idx_d_1_s_165 = {};
int __f2dace_SOA_cell_idx_d_2_s_166 = {};
int __f2dace_SOA_end_block_d_0_s_199 = {};
int __f2dace_SOA_end_index_d_0_s_197 = {};
int __f2dace_SOA_f_e_d_0_s_190 = {};
int __f2dace_SOA_f_e_d_1_s_191 = {};
int __f2dace_SOA_fn_e_d_0_s_192 = {};
int __f2dace_SOA_fn_e_d_1_s_193 = {};
int __f2dace_SOA_ft_e_d_0_s_194 = {};
int __f2dace_SOA_ft_e_d_1_s_195 = {};
int __f2dace_SOA_inv_dual_edge_length_d_0_s_186 = {};
int __f2dace_SOA_inv_dual_edge_length_d_1_s_187 = {};
int __f2dace_SOA_inv_primal_edge_length_d_0_s_184 = {};
int __f2dace_SOA_inv_primal_edge_length_d_1_s_185 = {};
int __f2dace_SOA_quad_blk_d_0_s_181 = {};
int __f2dace_SOA_quad_blk_d_1_s_182 = {};
int __f2dace_SOA_quad_blk_d_2_s_183 = {};
int __f2dace_SOA_quad_idx_d_0_s_178 = {};
int __f2dace_SOA_quad_idx_d_1_s_179 = {};
int __f2dace_SOA_quad_idx_d_2_s_180 = {};
int __f2dace_SOA_start_block_d_0_s_198 = {};
int __f2dace_SOA_start_index_d_0_s_196 = {};
int __f2dace_SOA_tangent_orientation_d_0_s_176 = {};
int __f2dace_SOA_tangent_orientation_d_1_s_177 = {};
int __f2dace_SOA_vertex_blk_d_0_s_173 = {};
int __f2dace_SOA_vertex_blk_d_1_s_174 = {};
int __f2dace_SOA_vertex_blk_d_2_s_175 = {};
int __f2dace_SOA_vertex_idx_d_0_s_170 = {};
int __f2dace_SOA_vertex_idx_d_1_s_171 = {};
int __f2dace_SOA_vertex_idx_d_2_s_172 = {};
double* area_edge = {};
int* cell_blk = {};
int* cell_idx = {};
int* end_block = {};
int* end_index = {};
double* f_e = {};
double* fn_e = {};
double* ft_e = {};
double* inv_dual_edge_length = {};
double* inv_primal_edge_length = {};
int* quad_blk = {};
int* quad_idx = {};
int* start_block = {};
int* start_index = {};
double* tangent_orientation = {};
int* vertex_blk = {};
int* vertex_idx = {};
};


struct t_int_state {
int __f2dace_SA_c_lin_e_d_0_s_25 = {};
int __f2dace_SA_c_lin_e_d_1_s_26 = {};
int __f2dace_SA_c_lin_e_d_2_s_27 = {};
int __f2dace_SA_cells_aw_verts_d_0_s_31 = {};
int __f2dace_SA_cells_aw_verts_d_1_s_32 = {};
int __f2dace_SA_cells_aw_verts_d_2_s_33 = {};
int __f2dace_SA_e_bln_c_s_d_0_s_28 = {};
int __f2dace_SA_e_bln_c_s_d_1_s_29 = {};
int __f2dace_SA_e_bln_c_s_d_2_s_30 = {};
int __f2dace_SA_geofac_grdiv_d_0_s_37 = {};
int __f2dace_SA_geofac_grdiv_d_1_s_38 = {};
int __f2dace_SA_geofac_grdiv_d_2_s_39 = {};
int __f2dace_SA_geofac_n2s_d_0_s_43 = {};
int __f2dace_SA_geofac_n2s_d_1_s_44 = {};
int __f2dace_SA_geofac_n2s_d_2_s_45 = {};
int __f2dace_SA_geofac_rot_d_0_s_40 = {};
int __f2dace_SA_geofac_rot_d_1_s_41 = {};
int __f2dace_SA_geofac_rot_d_2_s_42 = {};
int __f2dace_SA_rbf_vec_coeff_e_d_0_s_34 = {};
int __f2dace_SA_rbf_vec_coeff_e_d_1_s_35 = {};
int __f2dace_SA_rbf_vec_coeff_e_d_2_s_36 = {};
int __f2dace_SOA_c_lin_e_d_0_s_25 = {};
int __f2dace_SOA_c_lin_e_d_1_s_26 = {};
int __f2dace_SOA_c_lin_e_d_2_s_27 = {};
int __f2dace_SOA_cells_aw_verts_d_0_s_31 = {};
int __f2dace_SOA_cells_aw_verts_d_1_s_32 = {};
int __f2dace_SOA_cells_aw_verts_d_2_s_33 = {};
int __f2dace_SOA_e_bln_c_s_d_0_s_28 = {};
int __f2dace_SOA_e_bln_c_s_d_1_s_29 = {};
int __f2dace_SOA_e_bln_c_s_d_2_s_30 = {};
int __f2dace_SOA_geofac_grdiv_d_0_s_37 = {};
int __f2dace_SOA_geofac_grdiv_d_1_s_38 = {};
int __f2dace_SOA_geofac_grdiv_d_2_s_39 = {};
int __f2dace_SOA_geofac_n2s_d_0_s_43 = {};
int __f2dace_SOA_geofac_n2s_d_1_s_44 = {};
int __f2dace_SOA_geofac_n2s_d_2_s_45 = {};
int __f2dace_SOA_geofac_rot_d_0_s_40 = {};
int __f2dace_SOA_geofac_rot_d_1_s_41 = {};
int __f2dace_SOA_geofac_rot_d_2_s_42 = {};
int __f2dace_SOA_rbf_vec_coeff_e_d_0_s_34 = {};
int __f2dace_SOA_rbf_vec_coeff_e_d_1_s_35 = {};
int __f2dace_SOA_rbf_vec_coeff_e_d_2_s_36 = {};
double* c_lin_e = {};
double* cells_aw_verts = {};
double* e_bln_c_s = {};
double* geofac_grdiv = {};
double* geofac_n2s = {};
double* geofac_rot = {};
double* rbf_vec_coeff_e = {};
};


struct t_nh_prog {
int __f2dace_SA_vn_d_0_s_288 = {};
int __f2dace_SA_vn_d_1_s_289 = {};
int __f2dace_SA_vn_d_2_s_290 = {};
int __f2dace_SA_w_d_0_s_285 = {};
int __f2dace_SA_w_d_1_s_286 = {};
int __f2dace_SA_w_d_2_s_287 = {};
int __f2dace_SOA_vn_d_0_s_288 = {};
int __f2dace_SOA_vn_d_1_s_289 = {};
int __f2dace_SOA_vn_d_2_s_290 = {};
int __f2dace_SOA_w_d_0_s_285 = {};
int __f2dace_SOA_w_d_1_s_286 = {};
int __f2dace_SOA_w_d_2_s_287 = {};
double* vn = {};
double* w = {};
};


struct global_data_type {
int i_am_accel_node = {};
int lextra_diffu = {};
int* nflatlev = {};
int nproma = {};
int* nrdmax = {};
int timer_intp = {};
int timer_solve_nh_veltend = {};
int timers_level = {};
};


struct t_grid_domain_decomp_info {
int __f2dace_SA_owner_mask_d_0_s_2 = {};
int __f2dace_SA_owner_mask_d_1_s_3 = {};
int __f2dace_SOA_owner_mask_d_0_s_2 = {};
int __f2dace_SOA_owner_mask_d_1_s_3 = {};
int* owner_mask = {};
};


struct t_grid_cells {
int __f2dace_SA_area_d_0_s_158 = {};
int __f2dace_SA_area_d_1_s_159 = {};
int __f2dace_SA_edge_blk_d_0_s_155 = {};
int __f2dace_SA_edge_blk_d_1_s_156 = {};
int __f2dace_SA_edge_blk_d_2_s_157 = {};
int __f2dace_SA_edge_idx_d_0_s_152 = {};
int __f2dace_SA_edge_idx_d_1_s_153 = {};
int __f2dace_SA_edge_idx_d_2_s_154 = {};
int __f2dace_SA_end_block_d_0_s_163 = {};
int __f2dace_SA_end_index_d_0_s_161 = {};
int __f2dace_SA_neighbor_blk_d_0_s_149 = {};
int __f2dace_SA_neighbor_blk_d_1_s_150 = {};
int __f2dace_SA_neighbor_blk_d_2_s_151 = {};
int __f2dace_SA_neighbor_idx_d_0_s_146 = {};
int __f2dace_SA_neighbor_idx_d_1_s_147 = {};
int __f2dace_SA_neighbor_idx_d_2_s_148 = {};
int __f2dace_SA_start_block_d_0_s_162 = {};
int __f2dace_SA_start_index_d_0_s_160 = {};
int __f2dace_SOA_area_d_0_s_158 = {};
int __f2dace_SOA_area_d_1_s_159 = {};
int __f2dace_SOA_edge_blk_d_0_s_155 = {};
int __f2dace_SOA_edge_blk_d_1_s_156 = {};
int __f2dace_SOA_edge_blk_d_2_s_157 = {};
int __f2dace_SOA_edge_idx_d_0_s_152 = {};
int __f2dace_SOA_edge_idx_d_1_s_153 = {};
int __f2dace_SOA_edge_idx_d_2_s_154 = {};
int __f2dace_SOA_end_block_d_0_s_163 = {};
int __f2dace_SOA_end_index_d_0_s_161 = {};
int __f2dace_SOA_neighbor_blk_d_0_s_149 = {};
int __f2dace_SOA_neighbor_blk_d_1_s_150 = {};
int __f2dace_SOA_neighbor_blk_d_2_s_151 = {};
int __f2dace_SOA_neighbor_idx_d_0_s_146 = {};
int __f2dace_SOA_neighbor_idx_d_1_s_147 = {};
int __f2dace_SOA_neighbor_idx_d_2_s_148 = {};
int __f2dace_SOA_start_block_d_0_s_162 = {};
int __f2dace_SOA_start_index_d_0_s_160 = {};
double* area = {};
t_grid_domain_decomp_info* decomp_info = {};
int* edge_blk = {};
int* edge_idx = {};
int* end_block = {};
int* end_index = {};
int* neighbor_blk = {};
int* neighbor_idx = {};
int* start_block = {};
int* start_index = {};
};


struct t_nh_metrics {
int __f2dace_SA_coeff1_dwdz_d_0_s_332 = {};
int __f2dace_SA_coeff1_dwdz_d_1_s_333 = {};
int __f2dace_SA_coeff1_dwdz_d_2_s_334 = {};
int __f2dace_SA_coeff2_dwdz_d_0_s_335 = {};
int __f2dace_SA_coeff2_dwdz_d_1_s_336 = {};
int __f2dace_SA_coeff2_dwdz_d_2_s_337 = {};
int __f2dace_SA_coeff_gradekin_d_0_s_329 = {};
int __f2dace_SA_coeff_gradekin_d_1_s_330 = {};
int __f2dace_SA_coeff_gradekin_d_2_s_331 = {};
int __f2dace_SA_ddqz_z_full_e_d_0_s_314 = {};
int __f2dace_SA_ddqz_z_full_e_d_1_s_315 = {};
int __f2dace_SA_ddqz_z_full_e_d_2_s_316 = {};
int __f2dace_SA_ddqz_z_half_d_0_s_317 = {};
int __f2dace_SA_ddqz_z_half_d_1_s_318 = {};
int __f2dace_SA_ddqz_z_half_d_2_s_319 = {};
int __f2dace_SA_ddxn_z_full_d_0_s_308 = {};
int __f2dace_SA_ddxn_z_full_d_1_s_309 = {};
int __f2dace_SA_ddxn_z_full_d_2_s_310 = {};
int __f2dace_SA_ddxt_z_full_d_0_s_311 = {};
int __f2dace_SA_ddxt_z_full_d_1_s_312 = {};
int __f2dace_SA_ddxt_z_full_d_2_s_313 = {};
int __f2dace_SA_deepatmo_gradh_ifc_d_0_s_340 = {};
int __f2dace_SA_deepatmo_gradh_mc_d_0_s_338 = {};
int __f2dace_SA_deepatmo_invr_ifc_d_0_s_341 = {};
int __f2dace_SA_deepatmo_invr_mc_d_0_s_339 = {};
int __f2dace_SA_wgtfac_c_d_0_s_320 = {};
int __f2dace_SA_wgtfac_c_d_1_s_321 = {};
int __f2dace_SA_wgtfac_c_d_2_s_322 = {};
int __f2dace_SA_wgtfac_e_d_0_s_323 = {};
int __f2dace_SA_wgtfac_e_d_1_s_324 = {};
int __f2dace_SA_wgtfac_e_d_2_s_325 = {};
int __f2dace_SA_wgtfacq_e_d_0_s_326 = {};
int __f2dace_SA_wgtfacq_e_d_1_s_327 = {};
int __f2dace_SA_wgtfacq_e_d_2_s_328 = {};
int __f2dace_SOA_coeff1_dwdz_d_0_s_332 = {};
int __f2dace_SOA_coeff1_dwdz_d_1_s_333 = {};
int __f2dace_SOA_coeff1_dwdz_d_2_s_334 = {};
int __f2dace_SOA_coeff2_dwdz_d_0_s_335 = {};
int __f2dace_SOA_coeff2_dwdz_d_1_s_336 = {};
int __f2dace_SOA_coeff2_dwdz_d_2_s_337 = {};
int __f2dace_SOA_coeff_gradekin_d_0_s_329 = {};
int __f2dace_SOA_coeff_gradekin_d_1_s_330 = {};
int __f2dace_SOA_coeff_gradekin_d_2_s_331 = {};
int __f2dace_SOA_ddqz_z_full_e_d_0_s_314 = {};
int __f2dace_SOA_ddqz_z_full_e_d_1_s_315 = {};
int __f2dace_SOA_ddqz_z_full_e_d_2_s_316 = {};
int __f2dace_SOA_ddqz_z_half_d_0_s_317 = {};
int __f2dace_SOA_ddqz_z_half_d_1_s_318 = {};
int __f2dace_SOA_ddqz_z_half_d_2_s_319 = {};
int __f2dace_SOA_ddxn_z_full_d_0_s_308 = {};
int __f2dace_SOA_ddxn_z_full_d_1_s_309 = {};
int __f2dace_SOA_ddxn_z_full_d_2_s_310 = {};
int __f2dace_SOA_ddxt_z_full_d_0_s_311 = {};
int __f2dace_SOA_ddxt_z_full_d_1_s_312 = {};
int __f2dace_SOA_ddxt_z_full_d_2_s_313 = {};
int __f2dace_SOA_deepatmo_gradh_ifc_d_0_s_340 = {};
int __f2dace_SOA_deepatmo_gradh_mc_d_0_s_338 = {};
int __f2dace_SOA_deepatmo_invr_ifc_d_0_s_341 = {};
int __f2dace_SOA_deepatmo_invr_mc_d_0_s_339 = {};
int __f2dace_SOA_wgtfac_c_d_0_s_320 = {};
int __f2dace_SOA_wgtfac_c_d_1_s_321 = {};
int __f2dace_SOA_wgtfac_c_d_2_s_322 = {};
int __f2dace_SOA_wgtfac_e_d_0_s_323 = {};
int __f2dace_SOA_wgtfac_e_d_1_s_324 = {};
int __f2dace_SOA_wgtfac_e_d_2_s_325 = {};
int __f2dace_SOA_wgtfacq_e_d_0_s_326 = {};
int __f2dace_SOA_wgtfacq_e_d_1_s_327 = {};
int __f2dace_SOA_wgtfacq_e_d_2_s_328 = {};
double* coeff1_dwdz = {};
double* coeff2_dwdz = {};
double* coeff_gradekin = {};
double* ddqz_z_full_e = {};
double* ddqz_z_half = {};
double* ddxn_z_full = {};
double* ddxt_z_full = {};
double* deepatmo_gradh_ifc = {};
double* deepatmo_gradh_mc = {};
double* deepatmo_invr_ifc = {};
double* deepatmo_invr_mc = {};
double* wgtfac_c = {};
double* wgtfac_e = {};
double* wgtfacq_e = {};
};


struct t_grid_vertices {
int __f2dace_SA_cell_blk_d_0_s_203 = {};
int __f2dace_SA_cell_blk_d_1_s_204 = {};
int __f2dace_SA_cell_blk_d_2_s_205 = {};
int __f2dace_SA_cell_idx_d_0_s_200 = {};
int __f2dace_SA_cell_idx_d_1_s_201 = {};
int __f2dace_SA_cell_idx_d_2_s_202 = {};
int __f2dace_SA_edge_blk_d_0_s_209 = {};
int __f2dace_SA_edge_blk_d_1_s_210 = {};
int __f2dace_SA_edge_blk_d_2_s_211 = {};
int __f2dace_SA_edge_idx_d_0_s_206 = {};
int __f2dace_SA_edge_idx_d_1_s_207 = {};
int __f2dace_SA_edge_idx_d_2_s_208 = {};
int __f2dace_SA_end_block_d_0_s_215 = {};
int __f2dace_SA_end_index_d_0_s_213 = {};
int __f2dace_SA_start_block_d_0_s_214 = {};
int __f2dace_SA_start_index_d_0_s_212 = {};
int __f2dace_SOA_cell_blk_d_0_s_203 = {};
int __f2dace_SOA_cell_blk_d_1_s_204 = {};
int __f2dace_SOA_cell_blk_d_2_s_205 = {};
int __f2dace_SOA_cell_idx_d_0_s_200 = {};
int __f2dace_SOA_cell_idx_d_1_s_201 = {};
int __f2dace_SOA_cell_idx_d_2_s_202 = {};
int __f2dace_SOA_edge_blk_d_0_s_209 = {};
int __f2dace_SOA_edge_blk_d_1_s_210 = {};
int __f2dace_SOA_edge_blk_d_2_s_211 = {};
int __f2dace_SOA_edge_idx_d_0_s_206 = {};
int __f2dace_SOA_edge_idx_d_1_s_207 = {};
int __f2dace_SOA_edge_idx_d_2_s_208 = {};
int __f2dace_SOA_end_block_d_0_s_215 = {};
int __f2dace_SOA_end_index_d_0_s_213 = {};
int __f2dace_SOA_start_block_d_0_s_214 = {};
int __f2dace_SOA_start_index_d_0_s_212 = {};
int* cell_blk = {};
int* cell_idx = {};
int* edge_blk = {};
int* edge_idx = {};
int* end_block = {};
int* end_index = {};
int* start_block = {};
int* start_index = {};
};


struct t_patch {
t_grid_cells* cells = {};
t_grid_edges* edges = {};
int nblks_c = {};
int nblks_e = {};
int nblks_v = {};
t_grid_vertices* verts = {};
};


struct t_nh_diag {
int __f2dace_SA_ddt_vn_apc_pc_d_0_s_300 = {};
int __f2dace_SA_ddt_vn_apc_pc_d_1_s_301 = {};
int __f2dace_SA_ddt_vn_apc_pc_d_2_s_302 = {};
int __f2dace_SA_ddt_vn_apc_pc_d_3_s_303 = {};
int __f2dace_SA_ddt_w_adv_pc_d_0_s_304 = {};
int __f2dace_SA_ddt_w_adv_pc_d_1_s_305 = {};
int __f2dace_SA_ddt_w_adv_pc_d_2_s_306 = {};
int __f2dace_SA_ddt_w_adv_pc_d_3_s_307 = {};
int __f2dace_SA_vn_ie_d_0_s_294 = {};
int __f2dace_SA_vn_ie_d_1_s_295 = {};
int __f2dace_SA_vn_ie_d_2_s_296 = {};
int __f2dace_SA_vt_d_0_s_291 = {};
int __f2dace_SA_vt_d_1_s_292 = {};
int __f2dace_SA_vt_d_2_s_293 = {};
int __f2dace_SA_w_concorr_c_d_0_s_297 = {};
int __f2dace_SA_w_concorr_c_d_1_s_298 = {};
int __f2dace_SA_w_concorr_c_d_2_s_299 = {};
int __f2dace_SOA_ddt_vn_apc_pc_d_0_s_300 = {};
int __f2dace_SOA_ddt_vn_apc_pc_d_1_s_301 = {};
int __f2dace_SOA_ddt_vn_apc_pc_d_2_s_302 = {};
int __f2dace_SOA_ddt_vn_apc_pc_d_3_s_303 = {};
int __f2dace_SOA_ddt_w_adv_pc_d_0_s_304 = {};
int __f2dace_SOA_ddt_w_adv_pc_d_1_s_305 = {};
int __f2dace_SOA_ddt_w_adv_pc_d_2_s_306 = {};
int __f2dace_SOA_ddt_w_adv_pc_d_3_s_307 = {};
int __f2dace_SOA_vn_ie_d_0_s_294 = {};
int __f2dace_SOA_vn_ie_d_1_s_295 = {};
int __f2dace_SOA_vn_ie_d_2_s_296 = {};
int __f2dace_SOA_vt_d_0_s_291 = {};
int __f2dace_SOA_vt_d_1_s_292 = {};
int __f2dace_SOA_vt_d_2_s_293 = {};
int __f2dace_SOA_w_concorr_c_d_0_s_297 = {};
int __f2dace_SOA_w_concorr_c_d_1_s_298 = {};
int __f2dace_SOA_w_concorr_c_d_2_s_299 = {};
double* ddt_vn_apc_pc = {};
double* ddt_w_adv_pc = {};
double max_vcfl_dyn = {};
double* vn_ie = {};
double* vt = {};
double* w_concorr_c = {};
};
// DaCe-generated entry-point declarations for the 4 compiled variants.
// These files are produced by generate_baselines.py; main_per.cu dispatches to
// one of them at runtime based on (lvn_only, istep).
#include "velocity_no_nproma_if_prop_lvn_only_0_istep_1.h"
#include "velocity_no_nproma_if_prop_lvn_only_0_istep_2.h"
#include "velocity_no_nproma_if_prop_lvn_only_1_istep_1.h"
#include "velocity_no_nproma_if_prop_lvn_only_1_istep_2.h"

#endif // __DACE_CODEGEN_VELOCITY_TENDENCIES__