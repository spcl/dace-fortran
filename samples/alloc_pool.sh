#!/usr/bin/env bash
# Pooled-allocator selection for the measurement harness. SOURCE this, do not execute:
#   source "$REPO/samples/alloc_pool.sh"
#
# WHY: the warm dacecaches allocate on the hot path from inside `#pragma omp parallel for`
# over blocks -- cloudscouter does 82 aligned news+deletes per block iteration (46 of them
# klon-sized, i.e. 256 B at nproma=32) and velocity_tendencies does 3. glibc malloc serves
# those from per-thread arenas but still returns spans to the OS, so every block iteration
# re-faults pages. A pooling allocator with purging switched off keeps the spans hot and
# turns the steady-state malloc into a thread-local free-list pop.
#
# MEAS_ALLOC selects the allocator; mimalloc is the DEFAULT for every native lane
# (dace-gcc, dace-llvm, gfortran-serial/autopar, original-openmp, flang-serial, nvfortran).
#   mimalloc  (default) -- spack mimalloc, LD_PRELOAD, purging disabled
#   jemalloc            -- spack jemalloc, LD_PRELOAD, decay disabled
#   system              -- explicit escape hatch: no preload, plain glibc malloc
# If the requested .so is missing or fails to preload, this falls back to system and prints
# a loud warning rather than failing the job -- but MEAS_ALLOC_ACTIVE then reads `system`,
# so the CSV records what actually ran.
#
# FAIRNESS: every lane inside one comparison must use the same allocator. Stamp
# MEAS_ALLOC_ACTIVE into an `alloc` CSV column on every row so a mimalloc run is never
# silently compared against an older system-malloc run.
#
# Exports: MEAS_ALLOC_ACTIVE (system|mimalloc|jemalloc), MEAS_ALLOC_SO (path or empty),
#          LD_PRELOAD, and the allocator tuning vars.

# Idempotent: sourced once per job, srun steps inherit the exports. Re-sourcing would
# otherwise append the same .so to LD_PRELOAD again.
if [ -n "${MEAS_ALLOC_LOADED:-}" ]; then
    return 0 2> /dev/null || true
fi

MEAS_ALLOC="${MEAS_ALLOC:-mimalloc}"
MEAS_ALLOC_ACTIVE=system
MEAS_ALLOC_SO=""

# Resolve an allocator .so: explicit MEAS_ALLOC_SO override, else spack, else a
# LIBRARY_PATH-style sweep. spack first per house rule.
_alloc_pool_find() {
    local pkg="$1" soname="$2" prefix d
    if [ -n "${MEAS_ALLOC_SO_OVERRIDE:-}" ]; then
        printf '%s' "$MEAS_ALLOC_SO_OVERRIDE"
        return 0
    fi
    if command -v spack > /dev/null 2>&1; then
        prefix="$(spack location -i "$pkg" 2> /dev/null || true)"
        if [ -n "$prefix" ]; then
            for d in lib64 lib; do
                if [ -f "$prefix/$d/$soname" ]; then
                    printf '%s' "$prefix/$d/$soname"
                    return 0
                fi
            done
        fi
    fi
    local search="${LD_LIBRARY_PATH:-}"
    for d in ${search//:/ } /usr/lib64 /usr/lib; do
        [ -n "$d" ] && [ -f "$d/$soname" ] && {
            printf '%s' "$d/$soname"
            return 0
        }
    done
    return 1
}

# A preload that aborts every process (wrong arch, missing dep, glibc symbol clash) would
# silently zero the whole lane, so prove it on /bin/true before exporting it.
_alloc_pool_verify() {
    LD_PRELOAD="$1" /bin/true > /dev/null 2>&1
}

case "$MEAS_ALLOC" in
    system)
        echo "alloc_pool: MEAS_ALLOC=system -- no preload, glibc malloc"
        ;;
    mimalloc | jemalloc)
        if [ "$MEAS_ALLOC" = mimalloc ]; then
            _alloc_so="$(_alloc_pool_find mimalloc libmimalloc.so || true)"
        else
            _alloc_so="$(_alloc_pool_find jemalloc libjemalloc.so.2 || true)"
            [ -n "$_alloc_so" ] || _alloc_so="$(_alloc_pool_find jemalloc libjemalloc.so || true)"
        fi
        if [ -z "$_alloc_so" ]; then
            echo "alloc_pool: WARNING: MEAS_ALLOC=$MEAS_ALLOC but no shared library found" >&2
            echo "alloc_pool: WARNING: falling back to system malloc; install with 'spack install $MEAS_ALLOC'" >&2
        elif ! _alloc_pool_verify "$_alloc_so"; then
            echo "alloc_pool: WARNING: preloading $_alloc_so aborts a trivial process" >&2
            echo "alloc_pool: WARNING: falling back to system malloc" >&2
        else
            MEAS_ALLOC_ACTIVE="$MEAS_ALLOC"
            MEAS_ALLOC_SO="$_alloc_so"
            export LD_PRELOAD="$_alloc_so${LD_PRELOAD:+:$LD_PRELOAD}"
            echo "alloc_pool: MEAS_ALLOC=$MEAS_ALLOC_ACTIVE preload=$MEAS_ALLOC_SO"
        fi
        unset _alloc_so
        ;;
    *)
        echo "alloc_pool: WARNING: unknown MEAS_ALLOC='$MEAS_ALLOC' (want system|mimalloc|jemalloc)" >&2
        echo "alloc_pool: WARNING: falling back to system malloc" >&2
        ;;
esac

# --- tuning ------------------------------------------------------------------
# The single knob that matters is retention: stop the allocator handing freed spans back
# to the OS, so the next block iteration's malloc is a free-list pop instead of a page fault.
if [ "$MEAS_ALLOC_ACTIVE" = mimalloc ]; then
    # purge_delay=-1 disables purging outright. Default 1000 (ms) means mimalloc madvise()s
    # idle spans away, which is exactly the re-fault we are trying to remove.
    export MIMALLOC_PURGE_DELAY="${MIMALLOC_PURGE_DELAY:--1}"
    # Commit arena memory up front rather than page-by-page on first touch (default 2 = eager
    # only on large arenas); the harness measures steady state, not first-touch.
    export MIMALLOC_ARENA_EAGER_COMMIT="${MIMALLOC_ARENA_EAGER_COMMIT:-1}"
    export MIMALLOC_EAGER_COMMIT_DELAY="${MIMALLOC_EAGER_COMMIT_DELAY:-0}"
    # Keep more full pages in the thread-local heap instead of abandoning them (default 2).
    # 82 live buffers per block across 4 size classes overflows the default retain budget.
    export MIMALLOC_PAGE_FULL_RETAIN="${MIMALLOC_PAGE_FULL_RETAIN:-8}"
    # 72 threads x per-thread heaps: give the process one big arena rather than growing in
    # 1 GiB steps under contention.
    export MIMALLOC_ARENA_RESERVE="${MIMALLOC_ARENA_RESERVE:-2097152}" # KiB = 2 GiB
    # Opt-in only: 2 MiB pages cut TLB misses on the klev*klon buffers but inflate RSS and
    # can fail on a busy node, which would perturb the very measurement we want.
    if [ -n "${MEAS_ALLOC_HUGE:-}" ]; then
        export MIMALLOC_ALLOW_LARGE_OS_PAGES=1
    fi
    # `if`, not `[ ... ] && ...`: common.sh sources this under `set -e`, where a bare AND-list
    # whose left side is false returns nonzero and would kill the sourcing shell.
    if [ -n "${MEAS_ALLOC_VERBOSE:-}" ]; then
        export MIMALLOC_VERBOSE=1 MIMALLOC_SHOW_STATS=1
    fi
elif [ "$MEAS_ALLOC_ACTIVE" = jemalloc ]; then
    # dirty/muzzy decay -1 = never return pages to the OS: jemalloc's equivalent of
    # purge_delay=-1. background_thread:false keeps a purger thread off the pinned cores.
    export MALLOC_CONF="${MALLOC_CONF:-dirty_decay_ms:-1,muzzy_decay_ms:-1,background_thread:false,tcache:true}"
fi

MEAS_ALLOC_LOADED=1
export MEAS_ALLOC MEAS_ALLOC_ACTIVE MEAS_ALLOC_SO MEAS_ALLOC_LOADED
