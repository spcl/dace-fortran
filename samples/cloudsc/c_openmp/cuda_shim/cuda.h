/* CUDA-neutralising shim for the CPU c-openmp lane.
 *
 * The vendored C rewrite of CLOUDSC lives in the dwarf's cloudsc_cuda tree
 * (src/cloudsc_cuda/cloudsc/cloudsc_c.cu): the kernel body is plain C, and the
 * only CUDA in it is `__global__` on the signature plus the two index reads
 *     jl = threadIdx.x;  ibl = blockIdx.z;
 * near the top.  Putting THIS header first on the include path turns that file
 * into an ordinary host function whose per-column/per-block coordinates come
 * from two thread-local variables the OpenMP driver sets before each call, so
 * the CPU lane runs the same arithmetic as the CUDA lane with no edit to the
 * vendored source (which is why the lane is honest about being "the C rewrite").
 *
 * Only cloudsc_c.h reaches for <cuda.h>; the vendored load_state/validate/mycpu
 * translation units are already pure host code.
 */
#ifndef CLOUDSC_CPU_CUDA_SHIM_H
#define CLOUDSC_CPU_CUDA_SHIM_H

/* nvcc injects the math declarations implicitly; a host compiler does not, and
 * dtype.h's MYMAX/MYMIN/MYEXP/MYPOW/MYABS macros expand to them unqualified. */
#include <math.h>

#define __global__

struct cloudsc_cpu_idx3 {
    int x;
    int y;
    int z;
};

/* thread_local: every OpenMP thread owns the block/column it is working on. */
extern thread_local struct cloudsc_cpu_idx3 threadIdx;
extern thread_local struct cloudsc_cpu_idx3 blockIdx;

#endif /* CLOUDSC_CPU_CUDA_SHIM_H */
