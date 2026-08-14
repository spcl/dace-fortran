#!/usr/bin/env python3
"""Verify + benchmark the newdxx_g GPU SDFG against QE-dumped ground truth.

Clone of ``samples/addusxx_g/bench_addusxx_gpu.py``: calls the compiled SDFG
DIRECTLY from Python (no Fortran bindings) -- the GPU offload keeps the host
array ABI, copy-in/copy-out states inside the SDFG move the data -- with numpy
arrays built from the dump decks (``data/<MAT>/adxndx_static_*`` +
``ndx_<set>_*``, same decks as verify_newdxx.f90).  Gate on the inout deexx:
max|out - ref| <= tol * max(1, max|ref|).

Timing: warmup/reps per set come from ``--warmup``/``--reps`` (defaults in the
argparse block below).  Each timed call spans the whole SDFG -- the copy states
are part of it -- so the copy states are Timer-instrumented and the report
gives BOTH numbers: the full call, and execution-only = full minus the H2D
copy-in state (qgm is the big upload).  The D2H copy-out state (deexx only,
<1KB) is queued behind the kernels on the same stream, so its timer reads as
kernel drain + download and is reported but NOT subtracted.

    cd /workspace/dace-fortran/samples/newdxx_g && PYTHONHASHSEED=0 \
        python3 bench_newdxx_gpu.py --deck data/BaO_nat002 --reps 20
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")
sys.path.insert(0, str(HERE))

import dace  # noqa: E402
from dace import data, dtypes  # noqa: E402
from dace import symbolic  # noqa: E402
from dace.config import Config  # noqa: E402

#: root-level copy states of the GPU SDFG; Timer-instrumented so the timed
#: reps can be reported both as full calls and as execution-only (H2D excluded)
_COPY_STATES = ("_cpu_to_gpu_copy_in", "_gpu_to_cpu_copy_out")


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        k, v = parts
        try:
            meta[k] = int(v)
        except ValueError:
            meta[k] = float(v)
    return meta


def load_deck(deck: Path, iset: int):
    """Returns (arrays, symbols, deexx_in, ref) for one dump set of a deck."""
    sm = read_meta(deck / "adxndx_static_meta.txt")
    pm = read_meta(deck / f"ndx_{iset}_meta.txt")
    nat, nsp, nkb, nnr, nhm = sm["nat"], sm["nsp"], sm["nkb"], sm["nnr"], sm["nhm"]

    def rd(name, dtype):
        return np.fromfile(deck / name, dtype=dtype)

    def f2d(flat, d0):
        assert flat.size % d0 == 0, f"{flat.size} % {d0}"
        # copy: reshape returns a VIEW and DaCe rejects view arguments
        return flat.reshape((d0, flat.size // d0), order="F").copy(order="F")

    arrays = {}
    symbols = {"nat": nat, "nkb": nkb, "dfftt_nnr": nnr, "gstart": sm["gstart"]}

    for i in (1, 2, 3):
        e = rd(f"adxndx_static_eigts{i}.bin", np.complex128)
        # eigts is on the DENSE grid (-nr_dense:nr_dense, nat); derive d0 from
        # the dump and only require the (2*nr+1)-odd shape.
        assert e.size % nat == 0, f"eigts{i}: {e.size} % nat={nat}"
        d0 = e.size // nat
        assert d0 % 2 == 1, f"eigts{i}: d0={d0} not of 2*nr+1 form"
        arrays[f"eigts{i}"] = f2d(e, d0)
        symbols[f"eigts{i}_d0"], symbols[f"eigts{i}_d1"] = d0, nat

    mill = f2d(rd("adxndx_static_mill.bin", np.int32), 3)
    arrays["mill"] = mill
    symbols["mill_d0"], symbols["mill_d1"] = 3, mill.shape[1]

    arrays["dfftt_nl"] = rd("adxndx_static_dfftt_nl.bin", np.int32)
    symbols["dfftt_nl_d0"] = arrays["dfftt_nl"].size

    arrays["ityp"] = rd("adxndx_static_ityp.bin", np.int32)
    assert arrays["ityp"].size == nat
    symbols["ityp_d0"] = nat

    arrays["tau"] = f2d(rd("adxndx_static_tau.bin", np.float64), 3)
    assert arrays["tau"].shape == (3, nat)
    symbols["tau_d0"], symbols["tau_d1"] = 3, nat

    arrays["ofsbeta"] = rd("adxndx_static_ofsbeta.bin", np.int32)
    assert arrays["ofsbeta"].size == nat
    symbols["ofsbeta_d0"] = nat

    ijtoh = rd("adxndx_static_ijtoh.bin", np.int32)
    assert ijtoh.size == nhm * nhm * nsp
    arrays["ijtoh"] = ijtoh.reshape((nhm, nhm, nsp), order="F").copy(order="F")
    symbols["ijtoh_d0"], symbols["ijtoh_d1"], symbols["ijtoh_d2"] = nhm, nhm, nsp

    g = f2d(rd("adxndx_static_g.bin", np.float64), 3)
    arrays["g"] = g
    symbols["g_d0"], symbols["g_d1"] = 3, g.shape[1]

    gg = rd("adxndx_static_gg.bin", np.float64)
    arrays["gg"] = gg
    symbols["gg_d0"] = gg.size

    vkb = rd("adxndx_static_vkb.bin", np.complex128)
    arrays["vkb"] = f2d(vkb, vkb.size // nkb)
    symbols["vkb_d0"], symbols["vkb_d1"] = vkb.size // nkb, nkb

    arrays["nh"] = np.array([sm[f"nh_{i}"] for i in range(1, nsp + 1)], dtype=np.int32)
    symbols["nh_d0"] = nsp
    arrays["upf_tvanp"] = np.array([sm[f"tvanp_{i}"] == 1 for i in range(1, nsp + 1)],
                                   dtype=np.bool_)
    symbols["upf_tvanp_d0"] = nsp
    symbols["nij_type_d0"] = nsp

    arrays["okvan"] = np.array([True], dtype=np.bool_)
    arrays["gamma_only"] = np.array([False], dtype=np.bool_)
    arrays["dfftt_ngm"] = np.array([sm["ngmt"]], dtype=np.int32)
    arrays["nhm"] = np.array([nhm], dtype=np.int32)
    arrays["lmaxq"] = np.array([sm["lmaxq"]], dtype=np.int32)
    arrays["nsp"] = np.array([nsp], dtype=np.int32)   # ABI array here, not a symbol
    arrays["omega"] = np.array([sm["omega"]], dtype=np.float64)

    # ---- per-set state -------------------------------------------------------
    qd0, qd1 = pm["qgm_d1"], pm["qgm_d2"]
    qgm = rd(f"ndx_{iset}_qgm.bin", np.complex128)
    assert qgm.size == qd0 * qd1
    arrays["qgm"] = f2d(qgm, qd0)
    symbols["qgm_d0"], symbols["qgm_d1"] = qd0, qd1

    arrays["nij_type"] = rd(f"ndx_{iset}_nij_type.bin", np.int32)
    assert arrays["nij_type"].size == nsp

    arrays["becphi_c"] = rd(f"ndx_{iset}_becphi_c.bin", np.complex128)
    assert arrays["becphi_c"].size == nkb
    arrays["becphi_r"] = np.zeros(nkb, dtype=np.float64)  # gamma arm not taken; ABI dummy
    arrays["vc"] = rd(f"ndx_{iset}_vc_in.bin", np.complex128)
    assert arrays["vc"].size == nnr
    arrays["xkq"] = rd(f"ndx_{iset}_xkq.bin", np.float64)
    arrays["xk"] = rd(f"ndx_{iset}_xk.bin", np.float64)

    deexx_in = rd(f"ndx_{iset}_deexx_in.bin", np.complex128)
    ref = rd(f"ndx_{iset}_deexx_out.bin", np.complex128)
    assert deexx_in.size == nkb
    return arrays, symbols, deexx_in, ref


def build_kwargs(sdfg: dace.SDFG, arrays: dict, symbols: dict, deexx: np.ndarray):
    """arglist-driven assembly; unknown arrays become zero dummies with all their
    dimension symbols forced to 1 (dead ABI plumbing -- reported loudly once)."""
    symvals = dict(symbols)
    for s in sorted(map(str, sdfg.free_symbols)):
        if s not in symvals:
            symvals[s] = 1  # offset_* lbound offsets, *_present flags, dead dfftt_* dims
    kwargs = {}
    dummied = []
    for name, desc in sdfg.arglist().items():
        if not isinstance(desc, data.Array):
            continue
        if name == "deexx":
            kwargs[name] = deexx
            continue
        if name in arrays:
            kwargs[name] = arrays[name]
            continue
        shape = tuple(int(symbolic.evaluate(sh, symvals)) for sh in desc.shape)
        kwargs[name] = np.zeros(shape, dtype=desc.dtype.as_numpy_dtype(), order="F")
        dummied.append(f"{name}{shape}")
    for name, arr in kwargs.items():
        desc = sdfg.arrays[name]
        want = tuple(int(symbolic.evaluate(sh, symvals)) for sh in desc.shape)
        got = arr.shape if arr.ndim > 0 else (1,)
        assert got == want, f"{name}: shape {got} != descriptor {want}"
        assert arr.dtype == desc.dtype.as_numpy_dtype(), \
            f"{name}: dtype {arr.dtype} != {desc.dtype}"
        if arr.ndim >= 2:
            assert arr.flags["F_CONTIGUOUS"], f"{name} not Fortran-contiguous"
    for s, v in symvals.items():
        kwargs[s] = v
    return kwargs, dummied


def compile_with_cuda_const_fix(sdfg: dace.SDFG):
    """DaCe 2.0.0a5 workaround (generated-file fix, same class as the repo's
    established .cpp patches): arrays whose descriptor size is an
    interstate-ASSIGNED symbol (auxvc(ngms), ngms = dfftt_ngm[0]) take the
    ``declared_arrays`` path in ``emit_memlet_reference`` and lose the
    ``const`` qualifier on the nested device-function parameter, while the
    enclosing kernel's parameter is (correctly) const -- nvcc rejects the
    call.  Stripping pointee-const from pointer parameters in the generated
    .cu makes every call compatible; no semantics change (no writes are
    introduced, ``__restrict__`` stays)."""
    try:
        return sdfg.compile()
    except Exception as exc:  # noqa: BLE001 -- retry only for the known nvcc const clash
        build = Path(sdfg.build_folder)
        cu = build / "src" / "cuda" / f"{sdfg.name}_cuda.cu"
        if not cu.is_file():
            raise
        txt = cu.read_text()
        patched, n = re.subn(r"\bconst\b\s+(\w[\w:]*\s*\*\s*__restrict__)", r"\1", txt)
        if n == 0:
            raise
        cu.write_text("// QE_CONST_STRIPPED: nested-arg const clash workaround "
                      "(bench_newdxx_gpu.py)\n" + patched)
        print(f"cuda const fix: {n} pointer params de-consted in {cu.name}; "
              f"rebuilding ({type(exc).__name__} was: nested-arg const clash)", flush=True)
        subprocess.check_call(["ninja"], cwd=build / "build")
        from dace.codegen.compiler import load_precompiled_sdfg
        return load_precompiled_sdfg(str(build))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdfg", type=Path, default=HERE / "outputs" / "newdxx_g_gpu.sdfg")
    ap.add_argument("--deck", type=Path, default=HERE / "data" / "BaO_nat002")
    ap.add_argument("--sets", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-11)
    args = ap.parse_args()

    if not args.sdfg.is_file():
        sys.exit(f"SDFG not found: {args.sdfg} (run offload_newdxx.py first)")
    if not (args.deck / "adxndx_static_meta.txt").is_file():
        sys.exit(f"deck not found: {args.deck} (run ./download_data.sh)")

    Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    t0 = time.time()
    sdfg = dace.SDFG.from_file(str(args.sdfg))
    print(f"loaded {args.sdfg.name} in {time.time() - t0:.0f}s", flush=True)

    instrumented = []
    for st in sdfg.all_states():
        if st.label in _COPY_STATES:
            st.instrument = dtypes.InstrumentationType.Timer
            instrumented.append(st.label)

    t0 = time.time()
    csdfg = compile_with_cuda_const_fix(sdfg)
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    print(f"bench config: warmup={args.warmup} reps={args.reps} per set "
          f"(--warmup/--reps; defaults in the argparse block of {Path(__file__).name})",
          flush=True)

    nfail = 0
    timings = []  # (iset, [per-rep full-call ms]) -- printed after the report flush
    for iset in args.sets:
        if not (args.deck / f"ndx_{iset}_meta.txt").is_file():
            print(f"set {iset}: no dump, skip", flush=True)
            continue
        arrays, symbols, deexx_in, ref = load_deck(args.deck, iset)
        deexx = np.array(deexx_in, copy=True)
        kwargs, dummied = build_kwargs(sdfg, arrays, symbols, deexx)
        if iset == args.sets[0] and dummied:
            print(f"dead-ABI dummies ({len(dummied)}): {', '.join(dummied[:8])}"
                  f"{' ...' if len(dummied) > 8 else ''}", flush=True)

        # ---- verify ----------------------------------------------------------
        deexx[:] = deexx_in
        csdfg(**kwargs)
        maxref = max(1.0, np.abs(ref).max())
        maxdiff = np.abs(deexx - ref).max()
        rel = maxdiff / maxref
        verdict = "PASS" if rel <= args.tol else "FAIL"
        print(f"set {iset}: max|diff| = {maxdiff:.4e}  max|ref| = {np.abs(ref).max():.4e}  "
              f"rel = {rel:.4e}  VERIFY: {verdict}", flush=True)
        if verdict == "FAIL":
            nfail += 1
            continue

        # ---- bench (every timed call re-verified against the same gate) ------
        for _ in range(args.warmup):
            deexx[:] = deexx_in
            csdfg(**kwargs)
        times = []
        for _ in range(args.reps):
            deexx[:] = deexx_in
            t0 = time.perf_counter()
            csdfg(**kwargs)
            times.append((time.perf_counter() - t0) * 1e3)
            if np.abs(deexx - ref).max() / maxref > args.tol:
                sys.exit(f"set {iset}: timed call FAILED the correctness gate")
        timings.append((iset, times))

    # ---- copy-state timings: finalize flushes the instrumentation report ----
    copy_ms = {}
    if instrumented and timings:
        csdfg.finalize()
        report = sdfg.get_latest_report()
        if report is not None:
            for label in instrumented:
                vals = [ev.duration / 1000.0 for ev in report.events
                        if ev.name == f"State {label}"]  # µs -> ms
                if vals:
                    copy_ms[label] = statistics.median(vals)
    ci = copy_ms.get("_cpu_to_gpu_copy_in")
    co = copy_ms.get("_gpu_to_cpu_copy_out")
    if ci is not None:
        # copy-out is queued after the kernels on the same stream, so its timer
        # includes the kernel drain: execution = full - copy-in only.
        print(f"copy states (median over all calls): H2D copy-in {ci:8.2f} ms | "
              f"D2H copy-out (incl. kernel drain) {co:8.2f} ms" if co is not None else
              f"copy states (median over all calls): H2D copy-in {ci:8.2f} ms", flush=True)
    for iset, times in timings:
        med = statistics.median(times)
        line = (f"set {iset}: {args.reps} reps  full-call median {med:8.2f} ms  "
                f"min {min(times):8.2f} ms  max {max(times):8.2f} ms")
        if ci is not None:
            line += f"  |  execution (full - H2D copy-in) {med - ci:8.2f} ms"
        print(line, flush=True)

    print("RESULT:", "FAIL" if nfail else "PASS", flush=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
