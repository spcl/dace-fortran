/* Placeholder so a stray #include "cuda_runtime.h" resolves in the CPU lane.
 * Deliberately empty: the CPU driver never calls the CUDA runtime API, and the
 * one vendored header that pulls this in (cloudsc_driver.h) is not compiled
 * here -- cloudsc_driver_omp.cpp replaces it.
 */
#ifndef CLOUDSC_CPU_CUDA_RUNTIME_SHIM_H
#define CLOUDSC_CPU_CUDA_RUNTIME_SHIM_H
#endif
