// Device-inline reductions for use from inside CUDA/HIP kernels.
// Same preamble shape as ``reductions_kernel.cuh`` so the ``.cu`` and
// any kernel TU see one consistent cross-backend surface.
#pragma once
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(HIP_PLATFORM_AMD)
#include <hip/hip_runtime.h>
#endif

// Max of a ``double`` array. Identity ``0.0`` (matches the velocity
// reference expectations).
__device__ __inline__
double reduce_max_device(const double *__restrict__ d_in, int size) {
    double m = 0.0;
    for (int i = 0; i < size; i++) m = (d_in[i] > m) ? d_in[i] : m;
    return m;
}

__device__ __inline__
void reduce_max_store_device(const double *__restrict__ d_in,
                             double *__restrict__ d_out, int size) {
    d_out[0] = reduce_max_device(d_in, size);
}

// Sum of an ``int`` array.
__device__ __inline__
int reduce_sum_device(const int *__restrict__ d_in, int size) {
    int s = 0;
    for (int i = 0; i < size; i++) s += d_in[i];
    return s;
}

__device__ __inline__
void reduce_sum_store_device(const int *__restrict__ d_in,
                             int *__restrict__ d_out, int size) {
    d_out[0] = reduce_sum_device(d_in, size);
}

// 1 iff any element is strictly positive. Bit-exact equivalent of
// the velocity reference's any-nonzero-clip check.
__device__ __inline__
int reduce_any_device(const int *__restrict__ d_in, int size) {
    for (int i = 0; i < size; i++) if (d_in[i] > 0) return 1;
    return 0;
}
