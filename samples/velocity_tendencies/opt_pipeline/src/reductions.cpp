// Host-side (OpenMP) reductions matching the ``reductions_cpu.h``
// surface. Every symbol is C-linkage; callers may be CUDA TUs going
// through nvcc or pure g++ C++.
#include <omp.h>
#include "reductions_cpu.h"

// Tuned at a coarse grain: below this element count the parallel
// overhead (thread team dispatch, reduction tree) exceeds the serial
// cost.
static constexpr int PARALLEL_THRESHOLD = 10000;

extern "C" double reduce_max_cpu(const double *d_in, int size) {
    double m = 0.0;
    if (size > PARALLEL_THRESHOLD) {
#pragma omp parallel for reduction(max : m)
        for (int i = 0; i < size; i++) m = (d_in[i] > m) ? d_in[i] : m;
    } else {
        for (int i = 0; i < size; i++) m = (d_in[i] > m) ? d_in[i] : m;
    }
    return m;
}

extern "C" void reduce_max_store_cpu(const double *d_in, double *d_out,
                                     int size) {
    d_out[0] = reduce_max_cpu(d_in, size);
}

extern "C" int reduce_sum_cpu(const int *d_in, int size) {
    int s = 0;
    if (size > PARALLEL_THRESHOLD) {
#pragma omp parallel for reduction(+ : s)
        for (int i = 0; i < size; i++) s += d_in[i];
    } else {
        for (int i = 0; i < size; i++) s += d_in[i];
    }
    return s;
}

extern "C" void reduce_sum_store_cpu(const int *d_in, int *d_out, int size) {
    d_out[0] = reduce_sum_cpu(d_in, size);
}

extern "C" int reduce_any_cpu(const int *d_in, int size) {
    // Early-exit serial scan; OpenMP doesn't help here because any
    // hit is enough and the overhead outweighs the parallel gain.
    for (int i = 0; i < size; i++) if (d_in[i] > 0) return 1;
    return 0;
}
