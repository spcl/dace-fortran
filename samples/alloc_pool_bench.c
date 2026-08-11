// Allocator microbenchmark for the DaCe-Fortran measurement harness.
//
// Reproduces the two heap-traffic shapes the warm dacecaches actually exhibit:
//   mixed  -- the brief's 256 B / 512 KiB alternating churn
//   cloudsc-- one `loop_body_0_0_129` block iteration: 82 aligned news, then 82
//             deletes, sized klon / 5*klon / 25*klon / klev*klon doubles
// Both run inside an OpenMP parallel region because the generated code allocates
// from inside `#pragma omp parallel for` over blocks -- contention is the point.
//
// Build:  gcc -O2 -fopenmp -o alloc_pool_bench alloc_pool_bench.c
// Run:    ./alloc_pool_bench [threads] [iters]

#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 64, CLOUDSC_NALLOC = 82 };

// Sizes in bytes for klon=32, klev=137 (the nproma=32 sweep point).
static size_t cloudsc_sizes[CLOUDSC_NALLOC];

static void build_cloudsc_sizes(int klon, int klev) {
    int i = 0;
    for (int k = 0; k < 46; k++) cloudsc_sizes[i++] = (size_t)klon * 8;
    for (int k = 0; k < 14; k++) cloudsc_sizes[i++] = (size_t)5 * klon * 8;
    for (int k = 0; k < 4; k++) cloudsc_sizes[i++] = (size_t)25 * klon * 8;
    for (int k = 0; k < 18; k++) cloudsc_sizes[i++] = (size_t)klev * klon * 8;
}

// Touch one byte per 4 KiB page so the page is really faulted in -- an allocator
// that hands back never-committed memory must not look free.
static void touch(void *p, size_t n) {
    volatile char *c = p;
    for (size_t off = 0; off < n; off += 4096) c[off] = 1;
    c[n - 1] = 1;
}

static double bench_mixed(int iters) {
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int it = 0; it < iters; it++) {
            void *small = aligned_alloc(ALIGN, 256);
            void *big = aligned_alloc(ALIGN, 512 * 1024);
            touch(small, 256);
            touch(big, 512 * 1024);
            free(small);
            free(big);
        }
    }
    return omp_get_wtime() - t0;
}

static double bench_cloudsc(int iters) {
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
        void *buf[CLOUDSC_NALLOC];
#pragma omp for schedule(static)
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < CLOUDSC_NALLOC; i++) {
                buf[i] = aligned_alloc(ALIGN, (cloudsc_sizes[i] + ALIGN - 1) / ALIGN * ALIGN);
                touch(buf[i], cloudsc_sizes[i]);
            }
            for (int i = 0; i < CLOUDSC_NALLOC; i++) free(buf[i]);
        }
    }
    return omp_get_wtime() - t0;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    int iters = argc > 2 ? atoi(argv[2]) : 20000;
    omp_set_num_threads(threads);
    build_cloudsc_sizes(32, 137);

    bench_mixed(iters / 10);  // warm the pool / the OS arena alike
    bench_cloudsc(iters / 10);

    double tm = bench_mixed(iters);
    double tc = bench_cloudsc(iters);

    printf("threads=%d iters=%d\n", threads, iters);
    printf("  mixed   256B+512KiB : %8.3f ms  (%7.1f ns/alloc-pair)\n", tm * 1e3,
           tm * 1e9 / iters);
    printf("  cloudsc 82-alloc blk: %8.3f ms  (%7.1f ns/alloc, %7.2f us/block)\n", tc * 1e3,
           tc * 1e9 / ((double)iters * CLOUDSC_NALLOC), tc * 1e6 / iters);
    return 0;
}
