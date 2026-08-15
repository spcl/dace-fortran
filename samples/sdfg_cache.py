"""Crash-safe .sdfgz cache IO shared by the sample runners (phase A artifacts).

A job killed mid-``sdfg.save`` leaves a TRUNCATED gzip file.  ``SDFG.from_file`` then fails its
gzip attempt with OSError, silently retries the file as a raw pickle and dies with
``UnpicklingError: invalid load key, '\\x1f'`` -- the gzip magic, not the real cause -- which
poisons every later run that finds the cache "hit".  So: publish through ``<name>.partial`` +
``os.replace`` (no reader ever sees a half file), and treat a cache that will not load as a MISS.
"""
from __future__ import annotations

import os
from pathlib import Path


def save_sdfg_atomic(sdfg, cache) -> None:
    """Write ``sdfg`` to ``cache`` (gzipped) so the final name only ever appears complete.

    The staging name carries the pid: the cache key is grid-independent (nproma/nblks stay
    symbolic), so two jobs measuring different grids build the SAME key concurrently, and a
    shared ``.partial`` means one job's ``os.replace`` consumes the other's staging file --
    ``FileNotFoundError`` mid-run (job 4480423). ``os.replace`` onto the final name stays
    atomic and the payloads are identical, so the last writer simply wins.
    """
    cache = Path(cache)
    partial = cache.with_name(f"{cache.name}.{os.getpid()}.partial")
    cache.parent.mkdir(parents=True, exist_ok=True)
    sdfg.save(str(partial), compress=True)
    os.replace(partial, cache)


def load_sdfg_cached(cache, label: str = "phase A"):
    """Return the cached SDFG, or None when there is nothing usable at ``cache``.

    A corrupt artifact is deleted and reported as a miss: rebuilding costs minutes, a poisoned
    cache costs the whole job (and every job that depends on it).
    """
    from dace import SDFG
    cache = Path(cache)
    if not cache.is_file():
        return None
    try:
        sdfg = SDFG.from_file(str(cache))
    except Exception as exc:  # noqa: BLE001 -- a truncated pickle raises anything; all of it is a miss
        print(f"{label}: cache corrupt, rebuilding {cache} ({type(exc).__name__}: {exc})", flush=True)
        try:
            cache.unlink()
        except OSError as unlink_exc:
            print(f"{label}: could not remove {cache}: {unlink_exc}", flush=True)
        return None
    print(f"{label}: cache hit {cache}", flush=True)
    return sdfg
