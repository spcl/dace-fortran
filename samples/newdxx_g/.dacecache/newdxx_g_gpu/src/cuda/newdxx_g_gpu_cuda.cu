
#include <cuda_runtime.h>
#include <dace/dace.h>

constexpr int Cdp = 8;
constexpr int Cblocksize = 256;
constexpr double Ctpi = 6.283185307179586;

struct newdxx_g_gpu_state_t {
    dace::cuda::Context *gpu_context;
    dace::complex128 * __restrict__ __0_gpu_becphi_c;
    double * __restrict__ __0_gpu_becphi_r;
    dace::complex128 * __restrict__ __0_gpu_deexx;
    int * __restrict__ __0_gpu_dfftt_nl;
    dace::complex128 * __restrict__ __0_gpu_eigts1;
    dace::complex128 * __restrict__ __0_gpu_eigts2;
    dace::complex128 * __restrict__ __0_gpu_eigts3;
    int * __restrict__ __0_gpu_ijtoh;
    int * __restrict__ __0_gpu_mill;
    int * __restrict__ __0_gpu_nh;
    dace::complex128 * __restrict__ __0_gpu_qgm;
    double * __restrict__ __0_gpu_tau;
    dace::complex128 * __restrict__ __0_gpu_vc;
    double * __restrict__ __0_gpu_xk;
    double * __restrict__ __0_gpu_xkq;
    dace::complex128 * __restrict__ __0_gpu_aux2;
    dace::complex128 * __restrict__ __0_gpu_auxvc;
    dace::complex128 * __restrict__ __0_gpu_eigqts;
};



DACE_EXPORTED int __dace_init_cuda(newdxx_g_gpu_state_t *__state, int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int gstart, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int64_t nh_d0, int nkb, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0);
DACE_EXPORTED int __dace_exit_cuda(newdxx_g_gpu_state_t *__state);
DACE_EXPORTED int __dace_gpu_last_error(newdxx_g_gpu_state_t *__state);
DACE_EXPORTED void __dace_gpu_drain_error(newdxx_g_gpu_state_t *__state);
DACE_EXPORTED bool __dace_gpu_set_stream(newdxx_g_gpu_state_t *__state, int streamid, gpuStream_t stream);
DACE_EXPORTED void __dace_gpu_set_all_streams(newdxx_g_gpu_state_t *__state, gpuStream_t stream);

DACE_DFI void reduce_24_0_12(double* __restrict__ _in, double* __restrict__ _out) {

    {

        {
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {
                double acc[1]  DACE_ALIGN(64);
                {
                    double __o;

                    ///////////////////
                    // Tasklet code (reset)
                    __o = 0;
                    ///////////////////

                    acc[0] = __o;
                }
                {
                    #pragma omp simd
                    for (auto _i0 = 0; _i0 < 3; _i0 += 1) {
                        {
                            double __b = _in[_i0];
                            double __a = acc[0];
                            double __o;

                            ///////////////////
                            // Tasklet code (identity)
                            __o = __b;
                            ///////////////////

                            dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(acc, __o);
                        }
                    }
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                acc, _out + _o0, 1);
            }
        }

    }
}

DACE_DFI void loop_body_7_1_0(const double * __restrict__ gpu_tau, const double * __restrict__ gpu_xk, const double * __restrict__ gpu_xkq, dace::complex128 * __restrict__ gpu_eigqts, int _loop_it_0, int64_t tau_d0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;

    {
        double _mask_0[3]  DACE_ALIGN(64);
        double arg;
        double _QQred_lift_0;

        {
            double _in_tau_0 = gpu_tau[(tau_d0 * (_loop_it_0 - 1))];
            double _in_xk_0 = gpu_xk[0];
            double _in_xkq_0 = gpu_xkq[0];
            double _out__mask_0;

            ///////////////////
            // Tasklet code (t_0)
            _out__mask_0 = ((_in_xk_0 - _in_xkq_0) * _in_tau_0);
            ///////////////////

            _mask_0[0] = _out__mask_0;
        }
        {
            double _in_tau_0 = gpu_tau[((tau_d0 * (_loop_it_0 - 1)) + 1)];
            double _in_xk_0 = gpu_xk[1];
            double _in_xkq_0 = gpu_xkq[1];
            double _out__mask_0;

            ///////////////////
            // Tasklet code (t_0)
            _out__mask_0 = ((_in_xk_0 - _in_xkq_0) * _in_tau_0);
            ///////////////////

            _mask_0[1] = _out__mask_0;
        }
        {
            double _in_tau_0 = gpu_tau[((tau_d0 * (_loop_it_0 - 1)) + 2)];
            double _in_xk_0 = gpu_xk[2];
            double _in_xkq_0 = gpu_xkq[2];
            double _out__mask_0;

            ///////////////////
            // Tasklet code (t_0)
            _out__mask_0 = ((_in_xk_0 - _in_xkq_0) * _in_tau_0);
            ///////////////////

            _mask_0[2] = _out__mask_0;
        }
        reduce_24_0_12(&_mask_0[0], &_QQred_lift_0);
        {
            double _in__QQred_lift_0 = _QQred_lift_0;
            double _out;

            ///////////////////
            // Tasklet code (set_arg)
            _out = (_in__QQred_lift_0 * 6.283185307179586);
            ///////////////////

            arg = _out;
        }
        {
            double _in_arg = arg;
            dace::complex128 _out_eigqts;

            ///////////////////
            // Tasklet code (t_37)
            _out_eigqts = (cos(_in_arg) + (dace::complex128(0.0, 1.0) * (- sin(_in_arg))));
            ///////////////////

            gpu_eigqts[(_loop_it_0 - 1)] = _out_eigqts;
        }

    }
}

DACE_DFI void loop_body_7_1_7(const int * __restrict__ gpu_dfftt_nl, const dace::complex128 * __restrict__ gpu_vc, dace::complex128 * __restrict__ gpu_auxvc, int _loop_it_2, int64_t offset_dfftt_nl_d0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    int64_t dfftt_nl_at0;


    dfftt_nl_at0 = gpu_dfftt_nl[(_loop_it_2 - offset_dfftt_nl_d0)];
    {

        {
            dace::complex128 _in_vc_0 = gpu_vc[(dfftt_nl_at0 - 1)];
            dace::complex128 _out_auxvc;

            ///////////////////
            // Tasklet code (t_0)
            _out_auxvc = _in_vc_0;
            ///////////////////

            gpu_auxvc[(_loop_it_2 - 1)] = _out_auxvc;
        }

    }
}

DACE_DFI void loop_body_11_1_0(const dace::complex128 * __restrict__ gpu_auxvc, const dace::complex128 * __restrict__ gpu_eigqts, const dace::complex128 * __restrict__ gpu_eigts1, const dace::complex128 * __restrict__ gpu_eigts2, const dace::complex128 * __restrict__ gpu_eigts3, const int * __restrict__ gpu_mill, dace::complex128 * __restrict__ gpu_aux2, int _loop_it_5, int _loop_it_6, int _loop_it_7, int64_t eigts1_d0, int64_t eigts2_d0, int64_t eigts3_d0, int64_t mill_d0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    int64_t mill_at9;
    int64_t mill_at10;
    int64_t mill_at11;


    mill_at9 = gpu_mill[(mill_d0 * (((256 * _loop_it_5) + _loop_it_7) - 257))];
    mill_at10 = gpu_mill[((mill_d0 * (((256 * _loop_it_5) + _loop_it_7) - 257)) + 1)];
    mill_at11 = gpu_mill[((mill_d0 * (((256 * _loop_it_5) + _loop_it_7) - 257)) + 2)];
    {

        {
            dace::complex128 _in_auxvc_0 = gpu_auxvc[(((256 * _loop_it_5) + _loop_it_7) - 257)];
            dace::complex128 _in_eigqts_0 = gpu_eigqts[(_loop_it_6 - 1)];
            dace::complex128 _in_eigts1_0 = gpu_eigts1[((((eigts1_d0 * (_loop_it_6 - 1)) + mill_at9) + ((eigts1_d0 + 1) / 2)) - 1)];
            dace::complex128 _in_eigts2_0 = gpu_eigts2[((((eigts2_d0 * (_loop_it_6 - 1)) + mill_at10) + ((eigts2_d0 + 1) / 2)) - 1)];
            dace::complex128 _in_eigts3_0 = gpu_eigts3[((((eigts3_d0 * (_loop_it_6 - 1)) + mill_at11) + ((eigts3_d0 + 1) / 2)) - 1)];
            dace::complex128 _out_aux2;

            ///////////////////
            // Tasklet code (t_0)
            _out_aux2 = ((((conj(_in_auxvc_0) * _in_eigqts_0) * _in_eigts1_0) * _in_eigts2_0) * _in_eigts3_0);
            ///////////////////

            gpu_aux2[(_loop_it_7 - 1)] = _out_aux2;
        }

    }
}

DACE_DFI void loop_body_13_3_0(dace::complex128* __restrict__ aux1, int _loop_it_9) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;

    {

        {
            dace::complex128 _out_aux1;

            ///////////////////
            // Tasklet code (t_0)
            _out_aux1 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
            ///////////////////

            aux1[(_loop_it_9 - 1)] = _out_aux1;
        }

    }
}

DACE_DFI void loop_body_16_0_0(const double * __restrict__ gpu_becphi_r, const int * __restrict__ gpu_ijtoh, const dace::complex128 * __restrict__ gpu_qgm, dace::complex128* __restrict__ aux1, int _loop_it_10, int _loop_it_11, int _loop_it_5, int _loop_it_8, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int nij, int nt, int64_t qgm_d0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    int ijtoh_at12;


    ijtoh_at12 = gpu_ijtoh[(((_loop_it_8 + ((ijtoh_d0 * ijtoh_d1) * (nt - 1))) + (ijtoh_d0 * (_loop_it_10 - 1))) - 1)];
    {

        {
            dace::complex128 _in_aux1_0 = aux1[(_loop_it_11 - 1)];
            double _in_becphi_r_0 = gpu_becphi_r[((_loop_it_10 + ijkb0) - 1)];
            dace::complex128 _in_qgm_0 = gpu_qgm[(((_loop_it_11 + (256 * _loop_it_5)) + (qgm_d0 * ((ijtoh_at12 + nij) - 1))) - 257)];
            dace::complex128 _out_aux1;

            ///////////////////
            // Tasklet code (t_0)
            _out_aux1 = (_in_aux1_0 + ((_in_becphi_r_0 + (dace::complex128(0.0, 1.0) * 0.0)) * conj(_in_qgm_0)));
            ///////////////////

            aux1[(_loop_it_11 - 1)] = _out_aux1;
        }

    }
}

DACE_DFI void loop_body_18_0_0(const dace::complex128 * __restrict__ gpu_becphi_c, const int * __restrict__ gpu_ijtoh, const dace::complex128 * __restrict__ gpu_qgm, dace::complex128* __restrict__ aux1, int _loop_it_10, int _loop_it_12, int _loop_it_5, int _loop_it_8, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int nij, int nt, int64_t qgm_d0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    int ijtoh_at13;


    ijtoh_at13 = gpu_ijtoh[(((_loop_it_8 + ((ijtoh_d0 * ijtoh_d1) * (nt - 1))) + (ijtoh_d0 * (_loop_it_10 - 1))) - 1)];
    {

        {
            dace::complex128 _in_aux1_0 = aux1[(_loop_it_12 - 1)];
            dace::complex128 _in_becphi_c_0 = gpu_becphi_c[((_loop_it_10 + ijkb0) - 1)];
            dace::complex128 _in_qgm_0 = gpu_qgm[(((_loop_it_12 + (256 * _loop_it_5)) + (qgm_d0 * ((ijtoh_at13 + nij) - 1))) - 257)];
            dace::complex128 _out_aux1;

            ///////////////////
            // Tasklet code (t_0)
            _out_aux1 = (_in_aux1_0 + (_in_becphi_c_0 * conj(_in_qgm_0)));
            ///////////////////

            aux1[(_loop_it_12 - 1)] = _out_aux1;
        }

    }
}

DACE_DFI void dot_product__QQred_lift_1_114_sdfg_13_1_7(const dace::complex128 * __restrict__ _x, dace::complex128* __restrict__ _y, dace::complex128&  _result, int _loop_it_5, int scal_dfftt_ngm_0) {

    {

        {
            dace::complex128 _out;

            ///////////////////
            // Tasklet code (_dot_init)
            _out = 0;
            ///////////////////

            _result = _out;
        }

    }
    {

        {
            #pragma omp simd
            for (auto __i = 0; __i < Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)); __i += 1) {
                {
                    dace::complex128 __x = _x[__i];
                    dace::complex128 __y = _y[__i];
                    dace::complex128 __out;

                    ///////////////////
                    // Tasklet code (dot)
                    __out = (conj(__x) * __y);
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce(&_result, __out);
                }
            }
        }

    }
}

DACE_DFI void loop_body_11_1_10(const double&  fact, const dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_becphi_c, const double * __restrict__ gpu_becphi_r, const int * __restrict__ gpu_ijtoh, const int * __restrict__ gpu_nh, const dace::complex128 * __restrict__ gpu_qgm, const double&  scal_omega, dace::complex128 * __restrict__ gpu_deexx, int _loop_it_5, int _loop_it_8, int gstart, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int nij, int nt, int64_t qgm_d0, int scal_dfftt_ngm_0, bool scal_gamma_only_0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    dace::complex128 aux1[256]  DACE_ALIGN(64);
    int64_t loopend_100;
    int64_t _loop_it_10;

    {

        {
            #pragma omp simd
            for (auto _loop_it_9 = 1; _loop_it_9 < 257; _loop_it_9 += 1) {
                loop_body_13_3_0(&aux1[0], _loop_it_9);
            }
        }

    }
    loopend_100 = gpu_nh[(nt - 1)];
    for (_loop_it_10 = 1; (_loop_it_10 < (loopend_100 + 1)); _loop_it_10 = (_loop_it_10 + 1)) {
        if (scal_gamma_only_0) {
            {

                {
                    #pragma omp simd
                    for (auto _loop_it_11 = 1; _loop_it_11 < (Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)) + 1); _loop_it_11 += 1) {
                        loop_body_16_0_0(&gpu_becphi_r[0], &gpu_ijtoh[0], &gpu_qgm[0], &aux1[0], _loop_it_10, _loop_it_11, _loop_it_5, _loop_it_8, ijkb0, ijtoh_d0, ijtoh_d1, nij, nt, qgm_d0);
                    }
                }

            }
        } else {
            {

                {
                    #pragma omp simd
                    for (auto _loop_it_12 = 1; _loop_it_12 < (Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)) + 1); _loop_it_12 += 1) {
                        loop_body_18_0_0(&gpu_becphi_c[0], &gpu_ijtoh[0], &gpu_qgm[0], &aux1[0], _loop_it_10, _loop_it_12, _loop_it_5, _loop_it_8, ijkb0, ijtoh_d0, ijtoh_d1, nij, nt, qgm_d0);
                    }
                }

            }
        }

    }
    {
        dace::complex128 _QQred_lift_1;

        dot_product__QQred_lift_1_114_sdfg_13_1_7(&gpu_aux2[0], &aux1[0], _QQred_lift_1, _loop_it_5, scal_dfftt_ngm_0);
        {
            dace::complex128 _in__QQred_lift_1 = _QQred_lift_1;
            dace::complex128 _in_deexx_0 = gpu_deexx[((_loop_it_8 + ijkb0) - 1)];
            double _in_fact = fact;
            dace::complex128 _out_deexx;

            ///////////////////
            // Tasklet code (t_115)
            _out_deexx = (_in_deexx_0 + ((_in_fact + (dace::complex128(0.0, 1.0) * 0.0)) * _in__QQred_lift_1));
            ///////////////////

            gpu_deexx[((_loop_it_8 + ijkb0) - 1)] = _out_deexx;
        }

    }
    if (((scal_gamma_only_0 && (gstart == 2)) && (_loop_it_5 == 1))) {
        {

            {
                dace::complex128 _in_aux1_0 = aux1[0];
                dace::complex128 _in_aux2_0 = gpu_aux2[0];
                dace::complex128 _in_deexx_0 = gpu_deexx[((_loop_it_8 + ijkb0) - 1)];
                double _in_omega = scal_omega;
                dace::complex128 _out_deexx;

                ///////////////////
                // Tasklet code (t_119)
                _out_deexx = (_in_deexx_0 - (((_in_omega + (dace::complex128(0.0, 1.0) * 0.0)) * conj(_in_aux2_0)) * _in_aux1_0));
                ///////////////////

                gpu_deexx[((_loop_it_8 + ijkb0) - 1)] = _out_deexx;
            }

        }
    }
}



int __dace_init_cuda(newdxx_g_gpu_state_t *__state, int64_t dfftt_indp_d0, int64_t dfftt_indw_d0, int64_t dfftt_iproc_d0, int64_t dfftt_nl_d0, int dfftt_nnr, int64_t dfftt_nsp_offset_d0, int64_t dfftt_nsw_offset_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t g_d0, int gstart, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int64_t mill_d0, int64_t mill_d1, int nat, int64_t nh_d0, int nkb, int64_t offset_dfftt_nl_d0, int64_t offset_upf_tvanp_d0, int qe_compute, int qe_copy_in, int qe_copy_out, int64_t qgm_d0, int64_t qgm_d1, int64_t tau_d0, int64_t tau_d1, int64_t vkb_d0) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // One GPU per process, selected here and never changed, so the memory pool, every kernel and
    // every library handle share it. Which physical GPU is the process's business: the visible-
    // devices variable renumbers what it exposes, so a rank's own GPU is device 0. An ordinal
    // fixed at codegen time cannot do that -- every rank shares one build.
    const int __dace_device = 0;
    if (cudaSetDevice(__dace_device) != cudaSuccess)
    {
        printf("ERROR: could not select cuda device 0 out of %d visible\n", count);
        return 4;
    }

    __dace_gpu_drain_error(__state);

    // Initialize cuda before we run the application
    float *dev_X;
    DACE_GPU_CHECK(cudaMalloc((void **) &dev_X, 1));
    DACE_GPU_CHECK(cudaFree(dev_X));

    __state->gpu_context = new dace::cuda::Context(1, 1);

    // After the context exists: DACE_GPU_CHECK records into it.
    

    // Create cuda streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaStreamCreateWithFlags(&__state->gpu_context->internal_streams[i], cudaStreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming));
    }

    

    return 0;
}

int __dace_exit_cuda(newdxx_g_gpu_state_t *__state) {
    

    // Check for CUDA errors; synchronization suppressed by compiler.cuda.emit_synchronization
    int __err = static_cast<int>(__state->gpu_context->lasterror);

    // Destroy cuda streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaStreamDestroy(__state->gpu_context->internal_streams[i]));
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaEventDestroy(__state->gpu_context->events[i]));
    }

    delete __state->gpu_context;
    return __err;
}

// Discard a pending error left by another GPU user in this process, so the next checked call does
// not report it as its own. Sticky errors survive this and are reported normally.
// Must not touch __state->gpu_context: init calls this before the context exists.
void __dace_gpu_drain_error(newdxx_g_gpu_state_t *__state) {
    (void)__state;
    gpuError_t __pre_existing = cudaGetLastError();
    if (__pre_existing != (gpuError_t)0) {
        printf("WARNING: a GPU error was already pending on entry to a DaCe program and has been "
               "discarded: %s (%d). It was not caused by this SDFG.\n",
               gpuGetErrorString(__pre_existing), __pre_existing);
    }
}

// Returns what the generated code recorded, not the runtime's shared slot, and clears it.
int __dace_gpu_last_error(newdxx_g_gpu_state_t *__state) {
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    __state->gpu_context->lasterror = (gpuError_t)0;
    return __err;
}

bool __dace_gpu_set_stream(newdxx_g_gpu_state_t *__state, int streamid, gpuStream_t stream)
{
    if (streamid < 0 || streamid >= 1)
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}

void __dace_gpu_set_all_streams(newdxx_g_gpu_state_t *__state, gpuStream_t stream)
{
    for (int i = 0; i < 1; ++i)
        __state->gpu_context->streams[i] = stream;
}

__global__ void  __launch_bounds__(32) single_state_body_map_7_1_16(dace::complex128 * __restrict__ gpu_eigqts, const double * __restrict__ gpu_tau, const double * __restrict__ gpu_xk, const double * __restrict__ gpu_xkq, int nat, int64_t tau_d0, int64_t tau_d1) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    {
        int b__loop_it_0 = ((32 * blockIdx.x) + 1);
        {
            {
                int _loop_it_0 = (threadIdx.x + b__loop_it_0);
                if (_loop_it_0 >= b__loop_it_0 && _loop_it_0 < (Min(nat, (b__loop_it_0 + 31)) + 1)) {
                    loop_body_7_1_0(&gpu_tau[0], &gpu_xk[0], &gpu_xkq[0], &gpu_eigqts[0], _loop_it_0, tau_d0);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_single_state_body_map_7_1_16(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_eigqts, const double * __restrict__ gpu_tau, const double * __restrict__ gpu_xk, const double * __restrict__ gpu_xkq, int nat, int64_t tau_d0, int64_t tau_d1);
void __dace_runkernel_single_state_body_map_7_1_16(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_eigqts, const double * __restrict__ gpu_tau, const double * __restrict__ gpu_xk, const double * __restrict__ gpu_xkq, int nat, int64_t tau_d0, int64_t tau_d1)
{

    if (((int_ceil(nat, 32)) <= 0)) {

        return;
    }

    void  *single_state_body_map_7_1_16_args[] = { (void *)&gpu_eigqts, (void *)&gpu_tau, (void *)&gpu_xk, (void *)&gpu_xkq, (void *)&nat, (void *)&tau_d0, (void *)&tau_d1 };
    gpuError_t __err = cudaLaunchKernel((void*)single_state_body_map_7_1_16, dim3(int_ceil(nat, 32), 1, 1), dim3(32, 1, 1), single_state_body_map_7_1_16_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "single_state_body_map_7_1_16", int_ceil(nat, 32), 1, 1, 32, 1, 1);
}
__global__ void  __launch_bounds__(32) single_state_body_0_map_7_1_18(dace::complex128 * __restrict__ gpu_auxvc, const int * __restrict__ gpu_dfftt_nl, const dace::complex128 * __restrict__ gpu_vc, int dfftt_nl_d0, int dfftt_nnr, int64_t offset_dfftt_nl_d0, int scal_dfftt_ngm_0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    {
        int b__loop_it_2 = ((32 * blockIdx.x) + 1);
        {
            {
                int _loop_it_2 = (threadIdx.x + b__loop_it_2);
                if (_loop_it_2 >= b__loop_it_2 && _loop_it_2 < (Min(scal_dfftt_ngm_0, (b__loop_it_2 + 31)) + 1)) {
                    loop_body_7_1_7(&gpu_dfftt_nl[0], &gpu_vc[0], &gpu_auxvc[0], _loop_it_2, offset_dfftt_nl_d0);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_single_state_body_0_map_7_1_18(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_auxvc, const int * __restrict__ gpu_dfftt_nl, const dace::complex128 * __restrict__ gpu_vc, int dfftt_nl_d0, int dfftt_nnr, int64_t offset_dfftt_nl_d0, int scal_dfftt_ngm_0);
void __dace_runkernel_single_state_body_0_map_7_1_18(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_auxvc, const int * __restrict__ gpu_dfftt_nl, const dace::complex128 * __restrict__ gpu_vc, int dfftt_nl_d0, int dfftt_nnr, int64_t offset_dfftt_nl_d0, int scal_dfftt_ngm_0)
{

    if (((int_ceil(scal_dfftt_ngm_0, 32)) <= 0)) {

        return;
    }

    void  *single_state_body_0_map_7_1_18_args[] = { (void *)&gpu_auxvc, (void *)&gpu_dfftt_nl, (void *)&gpu_vc, (void *)&dfftt_nl_d0, (void *)&dfftt_nnr, (void *)&offset_dfftt_nl_d0, (void *)&scal_dfftt_ngm_0 };
    gpuError_t __err = cudaLaunchKernel((void*)single_state_body_0_map_7_1_18, dim3(int_ceil(scal_dfftt_ngm_0, 32), 1, 1), dim3(32, 1, 1), single_state_body_0_map_7_1_18_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "single_state_body_0_map_7_1_18", int_ceil(scal_dfftt_ngm_0, 32), 1, 1, 32, 1, 1);
}
__global__ void  __launch_bounds__(32) single_state_body_map_11_1_22(dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_auxvc, const dace::complex128 * __restrict__ gpu_eigqts, const dace::complex128 * __restrict__ gpu_eigts1, const dace::complex128 * __restrict__ gpu_eigts2, const dace::complex128 * __restrict__ gpu_eigts3, const int * __restrict__ gpu_mill, int64_t _loop_it_5, int64_t _loop_it_6, int dfftt_nl_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t mill_d0, int64_t mill_d1, int nat, int scal_dfftt_ngm_0) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    {
        int b__loop_it_7 = ((32 * blockIdx.x) + 1);
        {
            {
                int _loop_it_7 = (threadIdx.x + b__loop_it_7);
                if (_loop_it_7 >= b__loop_it_7 && _loop_it_7 < (Min(256, (b__loop_it_7 + 31), (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)) + 1)) {
                    loop_body_11_1_0(&gpu_auxvc[0], &gpu_eigqts[0], &gpu_eigts1[0], &gpu_eigts2[0], &gpu_eigts3[0], &gpu_mill[0], &gpu_aux2[0], _loop_it_5, _loop_it_6, _loop_it_7, eigts1_d0, eigts2_d0, eigts3_d0, mill_d0);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_single_state_body_map_11_1_22(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_auxvc, const dace::complex128 * __restrict__ gpu_eigqts, const dace::complex128 * __restrict__ gpu_eigts1, const dace::complex128 * __restrict__ gpu_eigts2, const dace::complex128 * __restrict__ gpu_eigts3, const int * __restrict__ gpu_mill, int64_t _loop_it_5, int64_t _loop_it_6, int dfftt_nl_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t mill_d0, int64_t mill_d1, int nat, int scal_dfftt_ngm_0);
void __dace_runkernel_single_state_body_map_11_1_22(newdxx_g_gpu_state_t *__state, dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_auxvc, const dace::complex128 * __restrict__ gpu_eigqts, const dace::complex128 * __restrict__ gpu_eigts1, const dace::complex128 * __restrict__ gpu_eigts2, const dace::complex128 * __restrict__ gpu_eigts3, const int * __restrict__ gpu_mill, int64_t _loop_it_5, int64_t _loop_it_6, int dfftt_nl_d0, int64_t eigts1_d0, int64_t eigts1_d1, int64_t eigts2_d0, int64_t eigts2_d1, int64_t eigts3_d0, int64_t eigts3_d1, int64_t mill_d0, int64_t mill_d1, int nat, int scal_dfftt_ngm_0)
{

    if (((int_ceil(Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)), 32)) <= 0)) {

        return;
    }

    void  *single_state_body_map_11_1_22_args[] = { (void *)&gpu_aux2, (void *)&gpu_auxvc, (void *)&gpu_eigqts, (void *)&gpu_eigts1, (void *)&gpu_eigts2, (void *)&gpu_eigts3, (void *)&gpu_mill, (void *)&_loop_it_5, (void *)&_loop_it_6, (void *)&dfftt_nl_d0, (void *)&eigts1_d0, (void *)&eigts1_d1, (void *)&eigts2_d0, (void *)&eigts2_d1, (void *)&eigts3_d0, (void *)&eigts3_d1, (void *)&mill_d0, (void *)&mill_d1, (void *)&nat, (void *)&scal_dfftt_ngm_0 };
    gpuError_t __err = cudaLaunchKernel((void*)single_state_body_map_11_1_22, dim3(int_ceil(Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)), 32), 1, 1), dim3(32, 1, 1), single_state_body_map_11_1_22_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "single_state_body_map_11_1_22", int_ceil(Min(256, (((-256 * _loop_it_5) + scal_dfftt_ngm_0) + 256)), 32), 1, 1, 32, 1, 1);
}
__global__ void  __launch_bounds__(32) single_state_body_0_map_11_1_24(const dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_becphi_c, const double * __restrict__ gpu_becphi_r, dace::complex128 * __restrict__ gpu_deexx, const int * __restrict__ gpu_ijtoh, const int * __restrict__ gpu_nh, const dace::complex128 * __restrict__ gpu_qgm, int64_t _loop_it_5, const double fact, int gstart, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int loopend_93, int64_t nh_d0, int nij, int nkb, int nt, int64_t qgm_d0, int64_t qgm_d1, int scal_dfftt_ngm_0, bool scal_gamma_only_0, const double scal_omega) {
    constexpr int Cdp = 8;
    constexpr int Cblocksize = 256;
    constexpr double Ctpi = 6.283185307179586;
    {
        int b__loop_it_8 = ((32 * blockIdx.x) + 1);
        {
            {
                int _loop_it_8 = (threadIdx.x + b__loop_it_8);
                if (_loop_it_8 >= b__loop_it_8 && _loop_it_8 < (Min(loopend_93, (b__loop_it_8 + 31)) + 1)) {
                    loop_body_11_1_10(fact, &gpu_aux2[0], &gpu_becphi_c[0], &gpu_becphi_r[0], &gpu_ijtoh[0], &gpu_nh[0], &gpu_qgm[0], scal_omega, &gpu_deexx[0], _loop_it_5, _loop_it_8, gstart, ijkb0, ijtoh_d0, ijtoh_d1, nij, nt, qgm_d0, scal_dfftt_ngm_0, scal_gamma_only_0);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_single_state_body_0_map_11_1_24(newdxx_g_gpu_state_t *__state, const dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_becphi_c, const double * __restrict__ gpu_becphi_r, dace::complex128 * __restrict__ gpu_deexx, const int * __restrict__ gpu_ijtoh, const int * __restrict__ gpu_nh, const dace::complex128 * __restrict__ gpu_qgm, int64_t _loop_it_5, const double fact, int gstart, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int loopend_93, int64_t nh_d0, int nij, int nkb, int nt, int64_t qgm_d0, int64_t qgm_d1, int scal_dfftt_ngm_0, bool scal_gamma_only_0, const double scal_omega);
void __dace_runkernel_single_state_body_0_map_11_1_24(newdxx_g_gpu_state_t *__state, const dace::complex128 * __restrict__ gpu_aux2, const dace::complex128 * __restrict__ gpu_becphi_c, const double * __restrict__ gpu_becphi_r, dace::complex128 * __restrict__ gpu_deexx, const int * __restrict__ gpu_ijtoh, const int * __restrict__ gpu_nh, const dace::complex128 * __restrict__ gpu_qgm, int64_t _loop_it_5, const double fact, int gstart, int ijkb0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t ijtoh_d2, int loopend_93, int64_t nh_d0, int nij, int nkb, int nt, int64_t qgm_d0, int64_t qgm_d1, int scal_dfftt_ngm_0, bool scal_gamma_only_0, const double scal_omega)
{

    if (((int_ceil(loopend_93, 32)) <= 0)) {

        return;
    }

    void  *single_state_body_0_map_11_1_24_args[] = { (void *)&gpu_aux2, (void *)&gpu_becphi_c, (void *)&gpu_becphi_r, (void *)&gpu_deexx, (void *)&gpu_ijtoh, (void *)&gpu_nh, (void *)&gpu_qgm, (void *)&_loop_it_5, (void *)&fact, (void *)&gstart, (void *)&ijkb0, (void *)&ijtoh_d0, (void *)&ijtoh_d1, (void *)&ijtoh_d2, (void *)&loopend_93, (void *)&nh_d0, (void *)&nij, (void *)&nkb, (void *)&nt, (void *)&qgm_d0, (void *)&qgm_d1, (void *)&scal_dfftt_ngm_0, (void *)&scal_gamma_only_0, (void *)&scal_omega };
    gpuError_t __err = cudaLaunchKernel((void*)single_state_body_0_map_11_1_24, dim3(int_ceil(loopend_93, 32), 1, 1), dim3(32, 1, 1), single_state_body_0_map_11_1_24_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "single_state_body_0_map_11_1_24", int_ceil(loopend_93, 32), 1, 1, 32, 1, 1);
}

