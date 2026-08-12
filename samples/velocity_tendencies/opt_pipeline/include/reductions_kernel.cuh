// Host-launched GPU reductions used by the velocity pipeline.
// Implementations in ``src/reductions_kernel.cu``; all entry points
// have C linkage so DaCe-generated tasklets can call them without
// name mangling.
//
// Temporary scratch buffers are allocated lazily on first use.
// ``reduce_gpu_finalize`` (emitted by the pass into
// ``__dace_exit_cuda_*``) frees them at SDFG teardown.
#pragma once
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(HIP_PLATFORM_AMD)
#include <hip/hip_runtime.h>
typedef hipStream_t reduce_stream_t;
#else
#include <cuda_runtime.h>
typedef cudaStream_t reduce_stream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Scalar-return reductions. Synchronous w.r.t. the stream -- the
// returned value is ready for immediate use on the host.
double reduce_max_gpu(const double *d_in, int size, reduce_stream_t stream);
int    reduce_sum_gpu(const int    *d_in, int size, reduce_stream_t stream);
int    reduce_any_gpu(const int    *d_in, int size, reduce_stream_t stream);

// Async scalar-return: reduction + D2H copy are enqueued on the
// stream; the call returns before the write to ``h_out`` has landed.
// Callers MUST synchronise the stream before reading ``*h_out``.
// The ``side_effects=True`` tasklet wrapper the pass emits around
// these calls keeps DaCe from reordering the write against later
// kernels; a follow-up sync tasklet (inserted by
// ``replace_reductions_with_tasklets``) guards the first consumer.
void reduce_max_async_host_gpu(const double *d_in, double *h_out,
                               int size, reduce_stream_t stream);
void reduce_sum_async_host_gpu(const int    *d_in, int    *h_out,
                               int size, reduce_stream_t stream);
void reduce_any_async_host_gpu(const int    *d_in, int    *h_out,
                               int size, reduce_stream_t stream);

// Store-into-pointer variants. ``d_out`` is device memory; the call
// enqueues the reduction on ``stream`` and returns. Caller syncs
// before reading host-side.
void reduce_max_store_gpu(const double *d_in, double *d_out, int size,
                          reduce_stream_t stream);
void reduce_sum_store_gpu(const int    *d_in, int    *d_out, int size,
                          reduce_stream_t stream);

// Free every lazily-allocated scratch buffer. Idempotent.
void reduce_gpu_finalize(void);
// Symmetry hook; currently a no-op because scratch storage is lazy.
void reduce_gpu_init(void);

#ifdef __cplusplus
}
#endif
