// Host (OpenMP-parallelised) reductions used by the velocity pipeline.
// All entry points are declared extern "C" so the DaCe-generated code
// can call them without name mangling.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Max of a ``double`` array. Behaves as ``max(0.0, max(d_in[0..size]))``
// (matches the velocity reference). ``_store`` variant writes the
// result to ``d_out[0]``.
double reduce_max_cpu(const double *d_in, int size);
void   reduce_max_store_cpu(const double *d_in, double *d_out, int size);

// Sum of an ``int`` array.
int    reduce_sum_cpu(const int *d_in, int size);
void   reduce_sum_store_cpu(const int *d_in, int *d_out, int size);

// Any-nonzero over an ``int`` array -- returns 1 iff any element
// is strictly greater than zero, else 0. Replaces what the old
// pipeline called ``reduce_scan``; the new name describes the
// semantics rather than the implementation.
int    reduce_any_cpu(const int *d_in, int size);

#ifdef __cplusplus
}
#endif
