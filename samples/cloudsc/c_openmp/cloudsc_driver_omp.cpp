/*
 * CPU driver for the vendored C rewrite of CLOUDSC -- the `c-openmp` baseline lane.
 *
 * Derived from the dwarf's src/cloudsc_cuda/cloudsc/cloudsc_driver.cu with every
 * cudaMalloc/cudaMemcpy/launch removed: the host arrays the CUDA driver already
 * allocates and fills via load_state() are handed straight to cloudsc_c(), and the
 * CUDA launch geometry (blockDim.x = NPROMA columns, gridDim.z = NGPBLKS blocks)
 * becomes an OpenMP loop over blocks with an inner column loop.  cuda_shim/cuda.h
 * supplies the two thread-local index structs the kernel reads, so cloudsc_c.cu is
 * compiled verbatim -- this lane measures the same C the CUDA lane measures.
 *
 * Timing contract (host timers, per rep, as the CPU lanes require):
 *   CLOUDSC_REPS / CLOUDSC_WARMUP pick the rep count; one ` REP <i> <ms>` line per
 *   timed rep, then the dwarf's own NUMOMP/NGPTOT/TOTAL block so the existing
 *   total_ms() parser in samples/cloudsc/baselines.sh also reads this binary.
 *   Only the block loop is inside the timer: no I/O, no allocation, no validation.
 *
 * argv is the dwarf contract, unchanged:  dwarf-cloudsc-c-omp NUMOMP NGPTOT NPROMA
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h" /* the shim: defines __global__ + the thread-local index structs */

#include "cloudsc_c.h"
#include "cloudsc_validate.h"
#include "dtype.h"
#include "load_state.h"
#include "mycpu.h"
#include "yoecldp_c.h"

/* The storage behind the shim's extern declarations; one instance per OpenMP thread. */
thread_local struct cloudsc_cpu_idx3 threadIdx = {0, 0, 0};
thread_local struct cloudsc_cpu_idx3 blockIdx = {0, 0, 0};

#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

static int env_int(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (raw == NULL || *raw == '\0') {
        return fallback;
    }
    return atoi(raw);
}

static dtype *alloc_d(size_t n) {
    dtype *p = (dtype *)malloc(sizeof(dtype) * n);
    if (p == NULL) {
        fprintf(stderr, "FATAL: out of memory for %zu doubles\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}

void cloudsc_driver_omp(int numthreads, int numcols, int nproma) {
    const int nclv = 5; /* number of microphysics variables */
    int klon, nlev;
    query_state(&klon, &nlev);

    const int nblocks = (numcols / nproma) + MIN2(numcols % nproma, 1);
    const size_t n_lev = (size_t)nblocks * nlev * nproma;
    const size_t n_levp1 = (size_t)nblocks * (nlev + 1) * nproma;
    const size_t n_surf = (size_t)nblocks * nproma;
    const size_t n_cld = (size_t)nblocks * nlev * nproma * nclv;

    struct TECLDP *yrecldp = (struct TECLDP *)malloc(sizeof(struct TECLDP));

    dtype *tend_loc_t = alloc_d(n_lev), *tend_loc_q = alloc_d(n_lev);
    dtype *tend_loc_a = alloc_d(n_lev), *tend_loc_cld = alloc_d(n_cld);
    dtype *tend_cml_t = alloc_d(n_lev), *tend_cml_q = alloc_d(n_lev);
    dtype *tend_cml_a = alloc_d(n_lev), *tend_cml_cld = alloc_d(n_cld);
    dtype *tend_tmp_t = alloc_d(n_lev), *tend_tmp_q = alloc_d(n_lev);
    dtype *tend_tmp_a = alloc_d(n_lev), *tend_tmp_cld = alloc_d(n_cld);

    dtype *plcrit_aer = alloc_d(n_lev), *picrit_aer = alloc_d(n_lev);
    dtype *pre_ice = alloc_d(n_lev), *pccn = alloc_d(n_lev), *pnice = alloc_d(n_lev);
    dtype *pt = alloc_d(n_lev), *pq = alloc_d(n_lev);
    dtype *pvfa = alloc_d(n_lev), *pvfl = alloc_d(n_lev), *pvfi = alloc_d(n_lev);
    dtype *pdyna = alloc_d(n_lev), *pdynl = alloc_d(n_lev), *pdyni = alloc_d(n_lev);
    dtype *phrsw = alloc_d(n_lev), *phrlw = alloc_d(n_lev), *pvervel = alloc_d(n_lev);
    dtype *pap = alloc_d(n_lev), *paph = alloc_d(n_levp1), *plsm = alloc_d(n_surf);
    int *ldcum = (int *)malloc(sizeof(int) * n_surf);
    int *ktype = (int *)malloc(sizeof(int) * n_surf);
    dtype *plu = alloc_d(n_lev), *plude = alloc_d(n_lev), *psnde = alloc_d(n_lev);
    dtype *pmfu = alloc_d(n_lev), *pmfd = alloc_d(n_lev), *pa = alloc_d(n_lev);
    dtype *pclv = alloc_d(n_cld), *psupsat = alloc_d(n_lev);
    dtype *pcovptot = alloc_d(n_lev), *prainfrac_toprfz = alloc_d(n_surf);
    dtype *pfsqlf = alloc_d(n_levp1), *pfsqif = alloc_d(n_levp1);
    dtype *pfcqnng = alloc_d(n_levp1), *pfcqlng = alloc_d(n_levp1);
    dtype *pfsqrf = alloc_d(n_levp1), *pfsqsf = alloc_d(n_levp1);
    dtype *pfcqrng = alloc_d(n_levp1), *pfcqsng = alloc_d(n_levp1);
    dtype *pfsqltur = alloc_d(n_levp1), *pfsqitur = alloc_d(n_levp1);
    dtype *pfplsl = alloc_d(n_levp1), *pfplsn = alloc_d(n_levp1);
    dtype *pfhpsl = alloc_d(n_levp1), *pfhpsn = alloc_d(n_levp1);

    dtype ptsphy;
    dtype rg, rd, rcpd, retv, rlvtt, rlstt, rlmlt, rtt, rv, r2es, r3les, r3ies;
    dtype r4les, r4ies, r5les, r5ies, r5alvcp, r5alscp, ralvdcp, ralsdcp, ralfdcp;
    dtype rtwat, rtice, rticecu, rtwat_rtice_r, rtwat_rticecu_r, rkoop1, rkoop2;

    load_state(klon, nlev, nclv, numcols, nproma, &ptsphy, plcrit_aer, picrit_aer, pre_ice, pccn, pnice, pt, pq,
               tend_cml_t, tend_cml_q, tend_cml_a, tend_cml_cld, tend_tmp_t, tend_tmp_q, tend_tmp_a, tend_tmp_cld,
               pvfa, pvfl, pvfi, pdyna, pdynl, pdyni, phrsw, phrlw, pvervel, pap, paph, plsm, ktype, plu, plude, psnde,
               pmfu, pmfd, pa, pclv, psupsat, yrecldp, &rg, &rd, &rcpd, &retv, &rlvtt, &rlstt, &rlmlt, &rtt, &rv,
               &r2es, &r3les, &r3ies, &r4les, &r4ies, &r5les, &r5ies, &r5alvcp, &r5alscp, &ralvdcp, &ralsdcp, &ralfdcp,
               &rtwat, &rtice, &rticecu, &rtwat_rtice_r, &rtwat_rticecu_r, &rkoop1, &rkoop2);

    /* plude is the kernel's only in-out field: restage it so every rep does identical work. */
    dtype *plude_pristine = alloc_d(n_lev);
    memcpy(plude_pristine, plude, sizeof(dtype) * n_lev);

    omp_set_num_threads(numthreads);
    const int nreps = env_int("CLOUDSC_REPS", 1);
    const int nwarmup = env_int("CLOUDSC_WARMUP", 0);
    double last = 0.0;

    for (int rep = -nwarmup; rep < nreps; rep++) {
        memcpy(plude, plude_pristine, sizeof(dtype) * n_lev);

        const double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; b++) {
            /* Column count of THIS block; the CUDA lane can only pass block 0's count to
             * every block, which agrees whenever nproma divides numcols (it does for every
             * size in the sweep) and is correct rather than merely equal otherwise. */
            const int icend = MIN2(nproma, numcols - b * nproma);
            blockIdx.z = b;
            for (int l = 0; l < icend; l++) {
                threadIdx.x = l;
                cloudsc_c(1, icend, nproma, ptsphy, pt, pq, tend_tmp_t, tend_tmp_q, tend_tmp_a, tend_tmp_cld,
                          tend_loc_t, tend_loc_q, tend_loc_a, tend_loc_cld, pvfa, pvfl, pvfi, pdyna, pdynl, pdyni,
                          phrsw, phrlw, pvervel, pap, paph, plsm, ktype, plu, plude, psnde, pmfu, pmfd, pa, pclv,
                          psupsat, plcrit_aer, picrit_aer, pre_ice, pccn, pnice, pcovptot, prainfrac_toprfz, pfsqlf,
                          pfsqif, pfcqnng, pfcqlng, pfsqrf, pfsqsf, pfcqrng, pfcqsng, pfsqltur, pfsqitur, pfplsl,
                          pfplsn, pfhpsl, pfhpsn, yrecldp, nblocks, rg, rd, rcpd, retv, rlvtt, rlstt, rlmlt, rtt, rv,
                          r2es, r3les, r3ies, r4les, r4ies, r5les, r5ies, r5alvcp, r5alscp, ralvdcp, ralsdcp, ralfdcp,
                          rtwat, rtice, rticecu, rtwat_rtice_r, rtwat_rticecu_r, rkoop1, rkoop2);
            }
        }
        last = omp_get_wtime() - start;

        if (rep >= 0) {
            printf(" REP %d %.6f\n", rep, 1000.0 * last);
        }
    }

    /* Same block the Fortran/CUDA drivers print, so total_ms() parses this binary too.
     * zhpm: IBM P7 HPM flop count for 100 columns at L137 (dwarf convention). */
    const double zhpm = 12482329.0;
    const double zmflops = last > 0.0 ? 1.0e-06 * zhpm * ((double)numcols / 100.0) / last : 0.0;
    const double zthrput = last > 0.0 ? (double)numcols / last : 0.0;
    printf("     NUMOMP=%d, NGPTOT=%d, NPROMA=%d, NGPBLKS=%d\n", numthreads, numcols, nproma, nblocks);
    printf(" %10s%10s%10s%10s%10s %+4s : %10s%10s%10s\n", "NUMOMP", "NGPTOT", "#GP-cols", "#BLKS", "NPROMA", "tid#",
           "Time(msec)", "MFlops/s", "col/s");
    printf(" %10d%10d%10d%10d%10d %4d : %10d%10d%10d TOTAL\n", numthreads, numcols, numcols, nblocks, nproma, -1,
           (int)(last * 1000.0), (int)zmflops, (int)zthrput);

    cloudsc_validate(klon, nlev, nclv, numcols, nproma, plude, pcovptot, prainfrac_toprfz, pfsqlf, pfsqif, pfcqlng,
                     pfcqnng, pfsqrf, pfsqsf, pfcqrng, pfcqsng, pfsqltur, pfsqitur, pfplsl, pfplsn, pfhpsl, pfhpsn,
                     tend_loc_a, tend_loc_q, tend_loc_t, tend_loc_cld);

    free(plude_pristine);
    free(yrecldp);
    free(tend_loc_t); free(tend_loc_q); free(tend_loc_a); free(tend_loc_cld);
    free(tend_cml_t); free(tend_cml_q); free(tend_cml_a); free(tend_cml_cld);
    free(tend_tmp_t); free(tend_tmp_q); free(tend_tmp_a); free(tend_tmp_cld);
    free(plcrit_aer); free(picrit_aer); free(pre_ice); free(pccn); free(pnice);
    free(pt); free(pq); free(pvfa); free(pvfl); free(pvfi);
    free(pdyna); free(pdynl); free(pdyni); free(phrsw); free(phrlw);
    free(pvervel); free(pap); free(paph); free(plsm); free(ldcum); free(ktype);
    free(plu); free(plude); free(psnde); free(pmfu); free(pmfd); free(pa);
    free(pclv); free(psupsat); free(pcovptot); free(prainfrac_toprfz);
    free(pfsqlf); free(pfsqif); free(pfcqnng); free(pfcqlng);
    free(pfsqrf); free(pfsqsf); free(pfcqrng); free(pfcqsng);
    free(pfsqltur); free(pfsqitur); free(pfplsl); free(pfplsn);
    free(pfhpsl); free(pfhpsn);
}

int main(int argc, char *argv[]) {
    int numthreads = 1;
    int ngptot = 100;
    int nproma = 4;

    if (argc == 4) {
        numthreads = atoi(argv[1]);
        ngptot = atoi(argv[2]);
        nproma = atoi(argv[3]);
        if (numthreads <= 0) {
            numthreads = omp_get_max_threads();
        }
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s NUMOMP NGPTOT NPROMA\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (ngptot <= 0 || nproma <= 0) {
        fprintf(stderr, "FATAL: NGPTOT and NPROMA must be positive (got %d, %d)\n", ngptot, nproma);
        return EXIT_FAILURE;
    }

    cloudsc_driver_omp(numthreads, ngptot, nproma);
    return EXIT_SUCCESS;
}
