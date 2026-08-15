/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
constexpr int Cdp = 8;
constexpr int Cblocksize = 256;
constexpr double Ctpi = 6.283185307179586;

struct addusxx_g_gpu_state_t {
    dace::cuda::Context *gpu_context;
    dace::complex128 * __restrict__ __0_gpu_becphi_c;
    dace::complex128 * __restrict__ __0_gpu_becpsi_c;
    int * __restrict__ __0_gpu_dfftt_nl;
    dace::complex128 * __restrict__ __0_gpu_eigts1;
    dace::complex128 * __restrict__ __0_gpu_eigts2;
    dace::complex128 * __restrict__ __0_gpu_eigts3;
    int * __restrict__ __0_gpu_ijtoh;
    int * __restrict__ __0_gpu_mill;
    dace::complex128 * __restrict__ __0_gpu_qgm;
    dace::complex128 * __restrict__ __0_gpu_rhoc;
    double * __restrict__ __0_gpu_tau;
    double * __restrict__ __0_gpu_xk;
    double * __restrict__ __0_gpu_xkq;
    dace::complex128 * __restrict__ __0_gpu_aux1;
    dace::complex128 * __restrict__ __0_gpu_aux2;
    dace::complex128 * __restrict__ __0_gpu_eigqts;
};

DACE_EXPORTED void __dace_runkernel_single_state_body_map_7_1_7(addusxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_eigqts, const double * __restrict__ gpu_tau, const double * __restrict__ gpu_xk, const double * __restrict__ gpu_xkq, int nat, int64_t tau_d0, int64_t tau_d1);
DACE_EXPORTED void __dace_runkernel_single_state_body_map_13_1_15(addusxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_aux2, int dfftt_nl_d0, int scal_dfftt_ngm);
DACE_EXPORTED void __dace_runkernel_single_state_body_map_13_1_16(addusxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_aux1, dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_becphi_c, const dace::complex128 * __restrict__ gpu_becpsi_c, const int * __restrict__ gpu_ijtoh, const dace::complex128 * __restrict__ gpu_qgm, int64_t _loop_it_2, int dfftt_nl_d0, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int loopend_58, int loopend_65, int nij, int nkb, int64_t qgm_d0, int64_t qgm_d1, int scal_dfftt_ngm);
DACE_EXPORTED void __dace_runkernel_single_state_body_0_map_13_2_16(addusxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_eigqts, const dace::complex128 * __restrict__ gpu_eigts1, const dace::complex128 * __restrict__ gpu_eigts2, const dace::complex128 * __restrict__ gpu_eigts3, const int * __restrict__ gpu_mill, int64_t _loop_it_4, int dfftt_nl_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t mill_d0, int64_t mill_d1, int nat, int scal_dfftt_ngm);
DACE_EXPORTED void __dace_runkernel_scatter_map_13_2_18(addusxx_g_gpu_state_t *__state, const dace::complex128 * __restrict__ gpu_aux2, const int * __restrict__ gpu_dfftt_nl, dace::complex128 * __restrict__ gpu_rhoc, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t offset_dfftt_nl_d0, int scal_dfftt_ngm);
void __program_addusxx_g_gpu_internal(addusxx_g_gpu_state_t*__state, dace::complex128 * __restrict__ becphi_c, double * __restrict__ becphi_r, dace::complex128 * __restrict__ becpsi_c, double * __restrict__ becpsi_r, dace::complex128 * __restrict__ dfftt_aux, int * __restrict__ dfftt_comm, int * __restrict__ dfftt_comm2, int * __restrict__ dfftt_comm3, int * __restrict__ dfftt_grid_id, bool * __restrict__ dfftt_has_task_groups, int * __restrict__ dfftt_i0r2p, int * __restrict__ dfftt_i0r3p, int * __restrict__ dfftt_indp, int * __restrict__ dfftt_indw, int * __restrict__ dfftt_indw_tg, int * __restrict__ dfftt_iplp, int * __restrict__ dfftt_iplw, int * __restrict__ dfftt_iproc, int * __restrict__ dfftt_iproc2, int * __restrict__ dfftt_iproc3, int * __restrict__ dfftt_ir1p, int * __restrict__ dfftt_ir1w, int * __restrict__ dfftt_ir1w_tg, int * __restrict__ dfftt_isind, int * __restrict__ dfftt_ismap, int * __restrict__ dfftt_iss, bool * __restrict__ dfftt_lgamma, bool * __restrict__ dfftt_lpara, int * __restrict__ dfftt_my_i0r2p, int * __restrict__ dfftt_my_i0r3p, int * __restrict__ dfftt_my_nr2p, int * __restrict__ dfftt_my_nr3p, int * __restrict__ dfftt_mype, int * __restrict__ dfftt_mype2, int * __restrict__ dfftt_mype3, int * __restrict__ dfftt_ngl, int * __restrict__ dfftt_ngm, int * __restrict__ dfftt_ngw, int * __restrict__ dfftt_nl, int * __restrict__ dfftt_nlm, int * __restrict__ dfftt_nnp, int * __restrict__ dfftt_nnr_tg, int * __restrict__ dfftt_nproc, int * __restrict__ dfftt_nproc2, int * __restrict__ dfftt_nproc3, int * __restrict__ dfftt_nr1, int * __restrict__ dfftt_nr1p, int * __restrict__ dfftt_nr1w, int * __restrict__ dfftt_nr1w_tg, int * __restrict__ dfftt_nr1x, int * __restrict__ dfftt_nr2, int * __restrict__ dfftt_nr2p, int * __restrict__ dfftt_nr2p_offset, int * __restrict__ dfftt_nr2x, int * __restrict__ dfftt_nr3, int * __restrict__ dfftt_nr3p, int * __restrict__ dfftt_nr3p_offset, int * __restrict__ dfftt_nr3x, int * __restrict__ dfftt_nsp, int * __restrict__ dfftt_nsp_offset, int * __restrict__ dfftt_nst, int * __restrict__ dfftt_nsw, int * __restrict__ dfftt_nsw_offset, int * __restrict__ dfftt_nsw_tg, int * __restrict__ dfftt_nwl, int * __restrict__ dfftt_root, int * __restrict__ dfftt_tg_rcv, int * __restrict__ dfftt_tg_rdsp, int * __restrict__ dfftt_tg_sdsp, int * __restrict__ dfftt_tg_snd, bool * __restrict__ dfftt_use_pencil_decomposition, dace::complex128 * __restrict__ eigts1, dace::complex128 * __restrict__ eigts2, dace::complex128 * __restrict__ eigts3, double * __restrict__ g, bool * __restrict__ gamma_only, int * __restrict__ ijtoh, int * __restrict__ ityp, int * __restrict__ lmaxq, int * __restrict__ mill, int * __restrict__ nh, int * __restrict__ nhm, int * __restrict__ nij_type, int * __restrict__ ofsbeta, bool * __restrict__ okvan, dace::complex128 * __restrict__ qgm, dace::complex128 * __restrict__ rhoc, double * __restrict__ tau, bool * __restrict__ upf_tvanp, dace::complex128 * __restrict__ vkb, double * __restrict__ xk, double * __restrict__ xkq, int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int nkb, int nsp, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0)
{
    bool scal_okvan_0;
    int scal_dfftt_ngm;
    int64_t _loop_it_2;
    int64_t if_cond_41;
    int nij;
    int64_t _loop_it_4;
    int64_t if_cond_49;
    int ijkb0;
    int64_t loopend_58;
    int64_t loopend_65;



    if ((qe_copy_in != 0)) {

        {

            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_becphi_c, becphi_c, nkb * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_becpsi_c, becpsi_c, nkb * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_dfftt_nl, dfftt_nl, dfftt_nl_d0 * sizeof(int), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_eigts1, eigts1, (eigts1_d0 * eigts1_d1) * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_eigts2, eigts2, (eigts2_d0 * eigts2_d1) * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_eigts3, eigts3, (eigts3_d0 * eigts3_d1) * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_ijtoh, ijtoh, ((ijtoh_d0 * ijtoh_d1) * ijtoh_d2) * sizeof(int), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_mill, mill, (mill_d0 * mill_d1) * sizeof(int), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_qgm, qgm, (qgm_d0 * qgm_d1) * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_rhoc, rhoc, dfftt_nnr * sizeof(dace::complex128), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_tau, tau, (tau_d0 * tau_d1) * sizeof(double), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_xk, xk, 3 * sizeof(double), cudaMemcpyHostToDevice, nullptr));
            DACE_GPU_CHECK(cudaMemcpyAsync(__state->__0_gpu_xkq, xkq, 3 * sizeof(double), cudaMemcpyHostToDevice, nullptr));

        }
    } else {

    }



    if ((qe_compute != 0)) {


        scal_okvan_0 = okvan[0];
        scal_dfftt_ngm = dfftt_ngm[0];

        if (scal_okvan_0) {
            {

                __dace_runkernel_single_state_body_map_7_1_7(__state, __state->__0_gpu_eigqts, __state->__0_gpu_tau, __state->__0_gpu_xk, __state->__0_gpu_xkq, nat, tau_d0, tau_d1);

            }

            for (_loop_it_2 = 1; (_loop_it_2 < (nsp + 1)); _loop_it_2 = (_loop_it_2 + 1)) {

                if_cond_41 = upf_tvanp[(_loop_it_2 - offset_upf_tvanp_d0)];

                if (if_cond_41) {

                    nij = nij_type[(_loop_it_2 - 1)];

                    for (_loop_it_4 = 1; (_loop_it_4 < (nat + 1)); _loop_it_4 = (_loop_it_4 + 1)) {

                        if_cond_49 = (ityp[(_loop_it_4 - 1)] == _loop_it_2);

                        if (if_cond_49) {

                            ijkb0 = ofsbeta[(_loop_it_4 - 1)];
                            loopend_58 = nh[(_loop_it_2 - 1)];
                            loopend_65 = nh[(_loop_it_2 - 1)];
                            {

                                __dace_runkernel_single_state_body_map_13_1_15(__state, __state->__0_gpu_aux2, dfftt_nl_d0, scal_dfftt_ngm);
                                __dace_runkernel_single_state_body_map_13_1_16(__state, __state->__0_gpu_aux1, __state->__0_gpu_aux2, __state->__0_gpu_becphi_c, __state->__0_gpu_becpsi_c, __state->__0_gpu_ijtoh, __state->__0_gpu_qgm, _loop_it_2, dfftt_nl_d0, ijkb0, ijtoh_d0, ijtoh_d1, ijtoh_d2, loopend_58, loopend_65, nij, nkb, qgm_d0, qgm_d1, scal_dfftt_ngm);

                            }
                            {

                                __dace_runkernel_single_state_body_0_map_13_2_16(__state, __state->__0_gpu_aux2, __state->__0_gpu_eigqts, __state->__0_gpu_eigts1, __state->__0_gpu_eigts2, __state->__0_gpu_eigts3, __state->__0_gpu_mill, _loop_it_4, dfftt_nl_d0, eigts1_d0, eigts1_d1, eigts2_d0, eigts2_d1, eigts3_d0, eigts3_d1, mill_d0, mill_d1, nat, scal_dfftt_ngm);
                                __dace_runkernel_scatter_map_13_2_18(__state, __state->__0_gpu_aux2, __state->__0_gpu_dfftt_nl, __state->__0_gpu_rhoc, dfftt_nl_d0, dfftt_nnr, offset_dfftt_nl_d0, scal_dfftt_ngm);

                            }
                        }


                    }

                }


            }

        }

    } else {

    }



    if ((qe_copy_out != 0)) {

        {

            DACE_GPU_CHECK(cudaMemcpyAsync(rhoc, __state->__0_gpu_rhoc, dfftt_nnr * sizeof(dace::complex128), cudaMemcpyDeviceToHost, nullptr));

        }
    } else {

    }

    {

        {

            ///////////////////
            DACE_GPU_CHECK(gpuStreamSynchronize(nullptr));
            ///////////////////

        }

    }
}

DACE_EXPORTED void __dace_gpu_drain_error(addusxx_g_gpu_state_t *__state);
DACE_EXPORTED void __program_addusxx_g_gpu(addusxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ becphi_c, double * __restrict__ becphi_r, dace::complex128 * __restrict__ becpsi_c, double * __restrict__ becpsi_r, dace::complex128 * __restrict__ dfftt_aux, int * __restrict__ dfftt_comm, int * __restrict__ dfftt_comm2, int * __restrict__ dfftt_comm3, int * __restrict__ dfftt_grid_id, bool * __restrict__ dfftt_has_task_groups, int * __restrict__ dfftt_i0r2p, int * __restrict__ dfftt_i0r3p, int * __restrict__ dfftt_indp, int * __restrict__ dfftt_indw, int * __restrict__ dfftt_indw_tg, int * __restrict__ dfftt_iplp, int * __restrict__ dfftt_iplw, int * __restrict__ dfftt_iproc, int * __restrict__ dfftt_iproc2, int * __restrict__ dfftt_iproc3, int * __restrict__ dfftt_ir1p, int * __restrict__ dfftt_ir1w, int * __restrict__ dfftt_ir1w_tg, int * __restrict__ dfftt_isind, int * __restrict__ dfftt_ismap, int * __restrict__ dfftt_iss, bool * __restrict__ dfftt_lgamma, bool * __restrict__ dfftt_lpara, int * __restrict__ dfftt_my_i0r2p, int * __restrict__ dfftt_my_i0r3p, int * __restrict__ dfftt_my_nr2p, int * __restrict__ dfftt_my_nr3p, int * __restrict__ dfftt_mype, int * __restrict__ dfftt_mype2, int * __restrict__ dfftt_mype3, int * __restrict__ dfftt_ngl, int * __restrict__ dfftt_ngm, int * __restrict__ dfftt_ngw, int * __restrict__ dfftt_nl, int * __restrict__ dfftt_nlm, int * __restrict__ dfftt_nnp, int * __restrict__ dfftt_nnr_tg, int * __restrict__ dfftt_nproc, int * __restrict__ dfftt_nproc2, int * __restrict__ dfftt_nproc3, int * __restrict__ dfftt_nr1, int * __restrict__ dfftt_nr1p, int * __restrict__ dfftt_nr1w, int * __restrict__ dfftt_nr1w_tg, int * __restrict__ dfftt_nr1x, int * __restrict__ dfftt_nr2, int * __restrict__ dfftt_nr2p, int * __restrict__ dfftt_nr2p_offset, int * __restrict__ dfftt_nr2x, int * __restrict__ dfftt_nr3, int * __restrict__ dfftt_nr3p, int * __restrict__ dfftt_nr3p_offset, int * __restrict__ dfftt_nr3x, int * __restrict__ dfftt_nsp, int * __restrict__ dfftt_nsp_offset, int * __restrict__ dfftt_nst, int * __restrict__ dfftt_nsw, int * __restrict__ dfftt_nsw_offset, int * __restrict__ dfftt_nsw_tg, int * __restrict__ dfftt_nwl, int * __restrict__ dfftt_root, int * __restrict__ dfftt_tg_rcv, int * __restrict__ dfftt_tg_rdsp, int * __restrict__ dfftt_tg_sdsp, int * __restrict__ dfftt_tg_snd, bool * __restrict__ dfftt_use_pencil_decomposition, dace::complex128 * __restrict__ eigts1, dace::complex128 * __restrict__ eigts2, dace::complex128 * __restrict__ eigts3, double * __restrict__ g, bool * __restrict__ gamma_only, int * __restrict__ ijtoh, int * __restrict__ ityp, int * __restrict__ lmaxq, int * __restrict__ mill, int * __restrict__ nh, int * __restrict__ nhm, int * __restrict__ nij_type, int * __restrict__ ofsbeta, bool * __restrict__ okvan, dace::complex128 * __restrict__ qgm, dace::complex128 * __restrict__ rhoc, double * __restrict__ tau, bool * __restrict__ upf_tvanp, dace::complex128 * __restrict__ vkb, double * __restrict__ xk, double * __restrict__ xkq, int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int nkb, int nsp, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0)
{
    __dace_gpu_drain_error(__state);
    __program_addusxx_g_gpu_internal(__state, becphi_c, becphi_r, becpsi_c, becpsi_r, dfftt_aux, dfftt_comm, dfftt_comm2, dfftt_comm3, dfftt_grid_id, dfftt_has_task_groups, dfftt_i0r2p, dfftt_i0r3p, dfftt_indp, dfftt_indw, dfftt_indw_tg, dfftt_iplp, dfftt_iplw, dfftt_iproc, dfftt_iproc2, dfftt_iproc3, dfftt_ir1p, dfftt_ir1w, dfftt_ir1w_tg, dfftt_isind, dfftt_ismap, dfftt_iss, dfftt_lgamma, dfftt_lpara, dfftt_my_i0r2p, dfftt_my_i0r3p, dfftt_my_nr2p, dfftt_my_nr3p, dfftt_mype, dfftt_mype2, dfftt_mype3, dfftt_ngl, dfftt_ngm, dfftt_ngw, dfftt_nl, dfftt_nlm, dfftt_nnp, dfftt_nnr_tg, dfftt_nproc, dfftt_nproc2, dfftt_nproc3, dfftt_nr1, dfftt_nr1p, dfftt_nr1w, dfftt_nr1w_tg, dfftt_nr1x, dfftt_nr2, dfftt_nr2p, dfftt_nr2p_offset, dfftt_nr2x, dfftt_nr3, dfftt_nr3p, dfftt_nr3p_offset, dfftt_nr3x, dfftt_nsp, dfftt_nsp_offset, dfftt_nst, dfftt_nsw, dfftt_nsw_offset, dfftt_nsw_tg, dfftt_nwl, dfftt_root, dfftt_tg_rcv, dfftt_tg_rdsp, dfftt_tg_sdsp, dfftt_tg_snd, dfftt_use_pencil_decomposition, eigts1, eigts2, eigts3, g, gamma_only, ijtoh, ityp, lmaxq, mill, nh, nhm, nij_type, ofsbeta, okvan, qgm, rhoc, tau, upf_tvanp, vkb, xk, xkq, dfftt_indp_d0, dfftt_indw_d0, dfftt_iproc_d0, dfftt_nl_d0, dfftt_nnr, dfftt_nsp_offset_d0, dfftt_nsw_offset_d0, eigts1_d0, eigts1_d1, eigts2_d0, eigts2_d1, eigts3_d0, eigts3_d1, g_d0, ijtoh_d0, ijtoh_d1, ijtoh_d2, mill_d0, mill_d1, nat, nkb, nsp, offset_dfftt_nl_d0, offset_upf_tvanp_d0, qe_compute, qe_copy_in, qe_copy_out, qgm_d0, qgm_d1, tau_d0, tau_d1, vkb_d0);
}
DACE_EXPORTED int __dace_init_cuda(addusxx_g_gpu_state_t *__state, int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int nkb, int nsp, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0);
DACE_EXPORTED int __dace_exit_cuda(addusxx_g_gpu_state_t *__state);

DACE_EXPORTED addusxx_g_gpu_state_t *__dace_init_addusxx_g_gpu(int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int nkb, int nsp, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0)
{

    int __result = 0;
    addusxx_g_gpu_state_t *__state = new addusxx_g_gpu_state_t();
    __result |= __dace_init_cuda(__state, dfftt_indp_d0, dfftt_indw_d0, dfftt_iproc_d0, dfftt_nl_d0, dfftt_nnr, dfftt_nsp_offset_d0, dfftt_nsw_offset_d0, eigts1_d0, eigts1_d1, eigts2_d0, eigts2_d1, eigts3_d0, eigts3_d1, g_d0, ijtoh_d0, ijtoh_d1, ijtoh_d2, mill_d0, mill_d1, nat, nkb, nsp, offset_dfftt_nl_d0, offset_upf_tvanp_d0, qe_compute, qe_copy_in, qe_copy_out, qgm_d0, qgm_d1, tau_d0, tau_d1, vkb_d0);
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_becphi_c, nkb * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_becpsi_c, nkb * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_dfftt_nl, dfftt_nl_d0 * sizeof(int)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_eigts1, ((eigts1_d0 * (eigts1_d1 - 1)) + eigts1_d0) * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_eigts2, ((eigts2_d0 * (eigts2_d1 - 1)) + eigts2_d0) * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_eigts3, ((eigts3_d0 * (eigts3_d1 - 1)) + eigts3_d0) * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_ijtoh, ((((ijtoh_d0 * ijtoh_d1) * (ijtoh_d2 - 1)) + (ijtoh_d0 * (ijtoh_d1 - 1))) + ijtoh_d0) * sizeof(int)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_mill, ((mill_d0 * (mill_d1 - 1)) + mill_d0) * sizeof(int)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_qgm, ((qgm_d0 * (qgm_d1 - 1)) + qgm_d0) * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_rhoc, dfftt_nnr * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_tau, ((tau_d0 * (tau_d1 - 1)) + tau_d0) * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_xk, 3 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_xkq, 3 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_aux1, dfftt_nl_d0 * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_aux2, dfftt_nl_d0 * sizeof(dace::complex128)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gpu_eigqts, nat * sizeof(dace::complex128)));

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_addusxx_g_gpu(addusxx_g_gpu_state_t *__state)
{

    int __err = 0;
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_becphi_c));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_becpsi_c));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_dfftt_nl));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_eigts1));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_eigts2));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_eigts3));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_ijtoh));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_mill));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_qgm));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_rhoc));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_tau));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_xk));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_xkq));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_aux1));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_aux2));
    DACE_GPU_CHECK(cudaFree(__state->__0_gpu_eigqts));

    int __err_cuda = __dace_exit_cuda(__state);
    if (__err_cuda) {
        __err = __err_cuda;
    }
    delete __state;
    return __err;
}
