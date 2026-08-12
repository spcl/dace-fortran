// Host-launched GPU reductions for the velocity pipeline. Single TU,
// compiles under both nvcc (CUDA) and hipcc (ROCm/HIP) via the same
// preprocessor shim used by ``reductions_kernel.cuh``.
//
// Implementation notes
// --------------------
// * Scalar returns use CUB's ``DeviceReduce`` to write the result into
//   a small device-resident scratch, then ``Memcpy`` it to the host
//   and synchronise. That round-trip is cheap relative to the
//   reduction itself and avoids pulling thrust into the build --
//   which keeps the thrust version-namespace warnings off the log
//   and shaves compile time.
// * Store variants drop the device-to-host copy entirely; the caller
//   sees the result on the stream like any other async kernel.
// * Temp storage (both CUB scratch and scalar outputs) is lazily
//   allocated and grows monotonically. ``reduce_gpu_finalize`` -- the
//   pass inserts a call into ``__dace_exit_cuda_*`` -- releases it.
#if defined(__HIP_PLATFORM_AMD__) || defined(HIP_PLATFORM_AMD)
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
typedef hipStream_t reduce_stream_t;
#define gpuMalloc         hipMalloc
#define gpuFree           hipFree
#define gpuMemcpyAsync    hipMemcpyAsync
#define gpuStreamSync     hipStreamSynchronize
#define gpuMemcpyD2H      hipMemcpyDeviceToHost
#define DEVICE_REDUCE     hipcub::DeviceReduce
#else
#include <cuda_runtime.h>
#include <cub/cub.cuh>
typedef cudaStream_t reduce_stream_t;
#define gpuMalloc         cudaMalloc
#define gpuFree           cudaFree
#define gpuMemcpyAsync    cudaMemcpyAsync
#define gpuStreamSync     cudaStreamSynchronize
#define gpuMemcpyD2H      cudaMemcpyDeviceToHost
#define DEVICE_REDUCE     cub::DeviceReduce
#endif

#include "reductions_kernel.cuh"

// --- Lazy scratch buffers ----------------------------------------------
//
// One CUB temp storage per primitive (max / sum) -- size depends on
// the input length, so we query CUB for the requirement and grow
// monotonically. One device-resident scalar slot per numeric type
// used by the scalar-return entry points so the D2H copy doesn't need
// a fresh allocation every call.

static void*   g_cub_tmp_max = nullptr;  static size_t g_cub_tmp_max_bytes = 0;
static void*   g_cub_tmp_sum = nullptr;  static size_t g_cub_tmp_sum_bytes = 0;

static double* g_scalar_out_double = nullptr;
static int*    g_scalar_out_int    = nullptr;

static inline void ensure_cub_tmp(void **buf, size_t *have, size_t need) {
    if (need > *have) {
        if (*buf) gpuFree(*buf);
        gpuMalloc(buf, need);
        *have = need;
    }
}

static inline double* ensure_scalar_out_double(void) {
    if (!g_scalar_out_double) gpuMalloc(&g_scalar_out_double, sizeof(double));
    return g_scalar_out_double;
}

static inline int* ensure_scalar_out_int(void) {
    if (!g_scalar_out_int) gpuMalloc(&g_scalar_out_int, sizeof(int));
    return g_scalar_out_int;
}

// --- Store variants (async on stream) ----------------------------------

extern "C"
void reduce_max_store_gpu(const double *d_in, double *d_out, int size,
                          reduce_stream_t stream) {
    size_t bytes = 0;
    DEVICE_REDUCE::Max(nullptr, bytes, d_in, d_out, size, stream);
    ensure_cub_tmp(&g_cub_tmp_max, &g_cub_tmp_max_bytes, bytes);
    DEVICE_REDUCE::Max(g_cub_tmp_max, bytes, d_in, d_out, size, stream);
}

extern "C"
void reduce_sum_store_gpu(const int *d_in, int *d_out, int size,
                          reduce_stream_t stream) {
    size_t bytes = 0;
    DEVICE_REDUCE::Sum(nullptr, bytes, d_in, d_out, size, stream);
    ensure_cub_tmp(&g_cub_tmp_sum, &g_cub_tmp_sum_bytes, bytes);
    DEVICE_REDUCE::Sum(g_cub_tmp_sum, bytes, d_in, d_out, size, stream);
}

// --- Scalar-return variants (sync on completion) -----------------------

extern "C"
double reduce_max_gpu(const double *d_in, int size, reduce_stream_t stream) {
    double *d_out = ensure_scalar_out_double();
    reduce_max_store_gpu(d_in, d_out, size, stream);
    double h_out;
    gpuMemcpyAsync(&h_out, d_out, sizeof(double), gpuMemcpyD2H, stream);
    gpuStreamSync(stream);
    return h_out;
}

extern "C"
int reduce_sum_gpu(const int *d_in, int size, reduce_stream_t stream) {
    int *d_out = ensure_scalar_out_int();
    reduce_sum_store_gpu(d_in, d_out, size, stream);
    int h_out;
    gpuMemcpyAsync(&h_out, d_out, sizeof(int), gpuMemcpyD2H, stream);
    gpuStreamSync(stream);
    return h_out;
}

extern "C"
int reduce_any_gpu(const int *d_in, int size, reduce_stream_t stream) {
    // Reuse sum + threshold. Any positive input pushes the sum > 0;
    // negative inputs don't occur in the velocity call sites (the
    // scan is over a 0/1 mask array), so ``>0`` is exact for our
    // callers. Keeps the scratch surface to one buffer.
    return (reduce_sum_gpu(d_in, size, stream) > 0) ? 1 : 0;
}

// --- Async scalar-return variants -------------------------------------
//
// These are the overlap-friendly versions of the scalar returns: the
// reduction and the D2H copy both enqueue on the stream and return
// immediately. The host-side result slot (``h_out``) is populated
// when the stream drains, not when the function returns. A deferred
// sync tasklet (emitted by the pass) flushes before the first
// consumer; any GPU work enqueued between the async call and the
// deferred sync runs concurrently with the D2H on the same stream.

extern "C"
void reduce_max_async_host_gpu(const double *d_in, double *h_out,
                               int size, reduce_stream_t stream) {
    double *d_out = ensure_scalar_out_double();
    reduce_max_store_gpu(d_in, d_out, size, stream);
    gpuMemcpyAsync(h_out, d_out, sizeof(double), gpuMemcpyD2H, stream);
    // No sync.
}

extern "C"
void reduce_sum_async_host_gpu(const int *d_in, int *h_out,
                               int size, reduce_stream_t stream) {
    int *d_out = ensure_scalar_out_int();
    reduce_sum_store_gpu(d_in, d_out, size, stream);
    gpuMemcpyAsync(h_out, d_out, sizeof(int), gpuMemcpyD2H, stream);
}

extern "C"
void reduce_any_async_host_gpu(const int *d_in, int *h_out,
                               int size, reduce_stream_t stream) {
    int *d_out = ensure_scalar_out_int();
    reduce_sum_store_gpu(d_in, d_out, size, stream);
    // Clamp the host-side value to {0,1} after the stream drains.
    // The async variant doesn't synchronise, so we can only write the
    // raw sum into ``h_out``; the tasklet site emits a ``(sum > 0)``
    // guard post-sync for callers that need the 0/1 form.
    gpuMemcpyAsync(h_out, d_out, sizeof(int), gpuMemcpyD2H, stream);
}

// --- Lifecycle ---------------------------------------------------------

extern "C" void reduce_gpu_init(void) {
    // Intentionally empty; scratch allocs are lazy so first-use
    // absorbs the cost. Kept as a symmetry hook for future eager
    // allocation if we care to preheat.
}

extern "C" void reduce_gpu_finalize(void) {
    if (g_cub_tmp_max)       { gpuFree(g_cub_tmp_max);       g_cub_tmp_max = nullptr;       g_cub_tmp_max_bytes = 0; }
    if (g_cub_tmp_sum)       { gpuFree(g_cub_tmp_sum);       g_cub_tmp_sum = nullptr;       g_cub_tmp_sum_bytes = 0; }
    if (g_scalar_out_double) { gpuFree(g_scalar_out_double); g_scalar_out_double = nullptr; }
    if (g_scalar_out_int)    { gpuFree(g_scalar_out_int);    g_scalar_out_int    = nullptr; }
}
