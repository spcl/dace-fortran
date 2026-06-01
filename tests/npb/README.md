# NAS Parallel Benchmarks (NPB) -- multi-file Fortran corpus

Each subdirectory under ``tests/npb/`` carries one NPB benchmark from
the upstream LLNL fork
(https://github.com/llnl/NPB/tree/master/NPB3.4/NPB3.4-OMP) in its
Modern-Fortran multi-file form, plus a pytest module that drives the
bridge over the whole project at once.

Status:

| Benchmark | Folder            | Test                              | State |
|-----------|-------------------|-----------------------------------|-------|
| LU        | ``tests/npb/lu``  | ``test_lu_multi_file_build.py``   | PASS  |
| BT        | -                 | -                                 | TODO  |
| CG        | -                 | -                                 | TODO  |
| EP        | -                 | -                                 | TODO  |
| FT        | -                 | -                                 | TODO  |
| MG        | -                 | -                                 | TODO  |
| SP        | -                 | -                                 | TODO  |

## Layout per benchmark

```
tests/npb/<bench>/
├── __init__.py                    # empty (pytest package marker)
├── <bench>.F90                    # the benchmark proper, ported to Modern Fortran
├── use<bench>.F90                 # 10-line driver: USE <bench>, ONLY: do<bench> ; CONTAINS ...
└── test_<bench>_multi_file_build.py
```

The driver module exists so the SDFG entry is a stable, public
symbol (``use<bench>::call_do<bench>``) regardless of how the
benchmark itself decorates its top-level procedure.  See ``lu/``
for the canonical example.

## Adding a new benchmark

1. Port (or copy) the benchmark's NPB3.4-OMP source to a single
   ``MODULE <bench>`` with public entry ``do<bench>``.
2. Drop a 10-line ``use<bench>.F90`` driver alongside it
   (verbatim shape from ``lu/useapplu.F90``).
3. Copy ``lu/test_lu_multi_file_build.py`` -- swap the entry symbol
   (``_QMuse<bench>Pcall_do<bench>``) and the ``_LU_KERNELS`` tuple
   for compute kernels actually present in ``<bench>.F90``.
4. Run the test once.  If it passes, land it with just the ``long``
   marker (LU did, on first contact -- the bridge handled SSOR,
   block triangular solves, and module-level state out of the box).
   If it surfaces a bridge gap, land it as ``xfail(strict=False) +
   long`` so the test flips to PASS automatically when the gap closes.

The ``long`` marker keeps the per-benchmark test out of the
default CI sweep (select with ``-m "long"`` to run them).

## Why multi-file

NPB benchmarks split their compute kernels across multiple modules
(geometry / coefficients / flux / Jacobian / solver) the same way
real climate models do.  Driving the bridge over them as
``build_sdfg_from_files([...])`` exercises ``merge_used_modules``
on a benchmark with a known reference -- a faithful precursor to
the ICON dynamical core which has the same shape at larger scale.
