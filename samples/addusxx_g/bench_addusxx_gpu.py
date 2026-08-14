#!/usr/bin/env python3
"""Verify + benchmark the addusxx_g GPU SDFG against QE-dumped ground truth.

Calls the compiled SDFG DIRECTLY from Python (no Fortran bindings): the GPU
offload keeps the host array ABI -- copy-in/copy-out states inside the SDFG move
the data -- so numpy arrays built from the dump decks (same decks as
verify_addusxx.f90: ``data/<MAT>/adxndx_static_*`` + ``adx_<set>_*``) can be
passed straight in, with the shape/offset symbols reconstructed from the meta
files.  Gate: max|out - ref| <= tol * max(1, max|ref|), same as the Fortran
verify driver.

Timing: warmup/reps per set come from ``--warmup``/``--reps`` (defaults in the
argparse block below).  The SDFG owns its copy states, so ``qe_gpu_timing``
gates them behind symbols and the rep loop calls the three phases separately:
copy-in (untimed) -> COMPUTE (timed) -> copy-out (untimed) -> verify.  The
headline number is compute only; the copy phases are host-timed too and
reported as a separate line, never folded in.  Under ``__DACE_NO_SYNC=1`` the
graph's one sync is the trailing ``_qe_sync`` state, so a phase call returns
when that phase's GPU work has drained -- including the copies the verify
reads.

    cd /workspace/dace-fortran/samples/addusxx_g && PYTHONHASHSEED=0 \
        python3 bench_addusxx_gpu.py --deck data/BaO_nat002 --reps 20

``--sdfg`` accepts any SDFG with the same ABI (e.g. the CPU
``outputs/addusxx_g_original_opt.sdfg`` with ``--fix-eigts`` applied in-memory,
as a harness cross-check without any GPU involvement).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from qe_deck_replicate import replicate_addusxx  # noqa: E402
# Before dace: qe_gpu_timing pins __DACE_NO_SYNC=1, which codegen reads at import time.
from qe_gpu_timing import gate_phases, run_all_phases, run_phase  # noqa: E402

import dace  # noqa: E402
from dace import data  # noqa: E402
from dace import symbolic  # noqa: E402
from dace.config import Config  # noqa: E402


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
    """Returns (arrays, symbols) dicts for one dump set of a deck."""
    sm = read_meta(deck / "adxndx_static_meta.txt")
    pm = read_meta(deck / f"adx_{iset}_meta.txt")
    nat, nsp, nkb, nnr, nhm = sm["nat"], sm["nsp"], sm["nkb"], sm["nnr"], sm["nhm"]

    def rd(name, dtype):
        return np.fromfile(deck / name, dtype=dtype)

    def f2d(flat, d0):
        assert flat.size % d0 == 0, f"{flat.size} % {d0}"
        # copy: reshape returns a VIEW and DaCe rejects view arguments
        return flat.reshape((d0, flat.size // d0), order="F").copy(order="F")

    arrays = {}
    symbols = {"nat": nat, "nsp": nsp, "nkb": nkb, "dfftt_nnr": nnr}

    for i in (1, 2, 3):
        e = rd(f"adxndx_static_eigts{i}.bin", np.complex128)
        # eigts is on the DENSE grid (-nr_dense:nr_dense, nat), not the FFT-t grid;
        # derive d0 from the dump and only require the (2*nr+1)-odd shape.
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

    vkb = rd("adxndx_static_vkb.bin", np.complex128)
    arrays["vkb"] = f2d(vkb, vkb.size // nkb)
    symbols["vkb_d0"], symbols["vkb_d1"] = vkb.size // nkb, nkb

    arrays["nh"] = np.array([sm[f"nh_{i}"] for i in range(1, nsp + 1)], dtype=np.int32)
    symbols["nh_d0"] = nsp
    arrays["upf_tvanp"] = np.array([sm[f"tvanp_{i}"] == 1 for i in range(1, nsp + 1)],
                                   dtype=np.bool_)
    symbols["upf_tvanp_d0"] = nsp

    arrays["okvan"] = np.array([True], dtype=np.bool_)
    arrays["gamma_only"] = np.array([False], dtype=np.bool_)
    arrays["dfftt_ngm"] = np.array([sm["ngmt"]], dtype=np.int32)
    arrays["nhm"] = np.array([nhm], dtype=np.int32)
    arrays["lmaxq"] = np.array([sm["lmaxq"]], dtype=np.int32)

    # ---- per-set state -------------------------------------------------------
    qd0, qd1 = pm["qgm_d1"], pm["qgm_d2"]
    qgm = rd(f"adx_{iset}_qgm.bin", np.complex128)
    assert qgm.size == qd0 * qd1
    arrays["qgm"] = f2d(qgm, qd0)
    symbols["qgm_d0"], symbols["qgm_d1"] = qd0, qd1

    arrays["nij_type"] = rd(f"adx_{iset}_nij_type.bin", np.int32)
    assert arrays["nij_type"].size == nsp
    symbols["nij_type_d0"] = nsp

    arrays["becphi_c"] = rd(f"adx_{iset}_becphi_c.bin", np.complex128)
    arrays["becpsi_c"] = rd(f"adx_{iset}_becpsi_c.bin", np.complex128)
    assert arrays["becphi_c"].size == nkb
    arrays["becphi_r"] = np.zeros(nkb, dtype=np.float64)  # r-arm not baked; ABI dummies
    arrays["becpsi_r"] = np.zeros(nkb, dtype=np.float64)
    arrays["xkq"] = rd(f"adx_{iset}_xkq.bin", np.float64)
    arrays["xk"] = rd(f"adx_{iset}_xk.bin", np.float64)

    rhoc_in = rd(f"adx_{iset}_rhoc_in.bin", np.complex128)
    ref = rd(f"adx_{iset}_rhoc_out.bin", np.complex128)
    assert rhoc_in.size == nnr
    return arrays, symbols, rhoc_in, ref


def build_kwargs(sdfg: dace.SDFG, arrays: dict, symbols: dict, rhoc: np.ndarray):
    """arglist-driven assembly; unknown arrays become zero dummies with all their
    dimension symbols forced to 1 (dead ABI plumbing -- reported loudly once)."""
    symvals = dict(symbols)
    for s in sorted(map(str, sdfg.free_symbols)):
        if s not in symvals:
            symvals[s] = 1  # offset_* lbound offsets and dead dfftt_* dims
    kwargs = {}
    dummied = []
    for name, desc in sdfg.arglist().items():
        if not isinstance(desc, data.Array):
            continue
        if name == "rhoc":
            kwargs[name] = rhoc
            continue
        if name in arrays:
            kwargs[name] = arrays[name]
            continue
        shape = tuple(int(symbolic.evaluate(sh, symvals)) for sh in desc.shape)
        kwargs[name] = np.zeros(shape, dtype=desc.dtype.as_numpy_dtype(), order="F")
        dummied.append(f"{name}{shape}")
    # dtype/shape sanity for the real arrays
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdfg", type=Path, default=HERE / "outputs" / "addusxx_g_gpu.sdfg")
    ap.add_argument("--deck", type=Path, default=HERE / "data" / "BaO_nat002")
    ap.add_argument("--sets", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--rep", type=int, default=1, help="deck replication factor (1 = the raw deck)")
    ap.add_argument("--no-offset", action="store_true",
                    help="replicate without the indirection offsets: negative control, must FAIL")
    ap.add_argument("--fix-eigts", action="store_true",
                    help="apply the SDFG-level eigts lbound rebase before compiling "
                         "(harness cross-check on the un-offloaded CPU SDFG)")
    args = ap.parse_args()

    if not args.sdfg.is_file():
        sys.exit(f"SDFG not found: {args.sdfg} (run offload_addusxx.py first)")
    if not (args.deck / "adxndx_static_meta.txt").is_file():
        sys.exit(f"deck not found: {args.deck} (run ./download_data.sh)")

    Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    t0 = time.time()
    sdfg = dace.SDFG.from_file(str(args.sdfg))
    print(f"loaded {args.sdfg.name} in {time.time() - t0:.0f}s", flush=True)
    if args.fix_eigts:
        from offload_addusxx import _rebase_eigts_lbounds
        _rebase_eigts_lbounds(sdfg)
        sdfg.name = sdfg.name + "_eigtsfix"

    gate_phases(sdfg)

    t0 = time.time()
    csdfg = sdfg.compile()
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    print(f"bench config: warmup={args.warmup} reps={args.reps} per set "
          f"(--warmup/--reps; defaults in the argparse block of {Path(__file__).name})",
          flush=True)

    nfail = 0
    rows = []  # one per set: (iset, compute ms list, h2d ms list, d2h ms list)
    for iset in args.sets:
        if not (args.deck / f"adx_{iset}_meta.txt").is_file():
            print(f"set {iset}: no dump, skip", flush=True)
            continue
        arrays, symbols, rhoc_in, ref = load_deck(args.deck, iset)
        arrays, symbols, rhoc_in, ref = replicate_addusxx(arrays, symbols, rhoc_in, ref,
                                                          args.rep, not args.no_offset)
        rhoc = np.array(rhoc_in, copy=True)
        kwargs, dummied = build_kwargs(sdfg, arrays, symbols, rhoc)
        if iset == args.sets[0] and dummied:
            print(f"dead-ABI dummies ({len(dummied)}): {', '.join(dummied[:8])}"
                  f"{' ...' if len(dummied) > 8 else ''}", flush=True)

        # ---- verify (one un-gated whole call, the shipped SDFG's behaviour) --
        rhoc[:] = rhoc_in
        run_all_phases(csdfg, kwargs)
        maxref = max(1.0, np.abs(ref).max())
        maxdiff = np.abs(rhoc - ref).max()
        rel = maxdiff / maxref
        verdict = "PASS" if rel <= args.tol else "FAIL"
        print(f"set {iset}: max|diff| = {maxdiff:.4e}  max|ref| = {np.abs(ref).max():.4e}  "
              f"rel = {rel:.4e}  VERIFY: {verdict}", flush=True)
        if verdict == "FAIL":
            nfail += 1
            continue

        # ---- bench: compute phase only inside the bracket, every rep verified
        h2d, comp, d2h = [], [], []
        for i in range(args.warmup + args.reps):
            rhoc[:] = rhoc_in
            t_in = run_phase(csdfg, kwargs, qe_copy_in=1)
            t_comp = run_phase(csdfg, kwargs, qe_compute=1)
            t_out = run_phase(csdfg, kwargs, qe_copy_out=1)
            if np.abs(rhoc - ref).max() / maxref > args.tol:
                sys.exit(f"set {iset}: timed call FAILED the correctness gate")
            if i >= args.warmup:
                h2d.append(t_in)
                comp.append(t_comp)
                d2h.append(t_out)
        rows.append((iset, comp, h2d, d2h))

    for iset, comp, h2d, d2h in rows:
        med = statistics.median(comp)
        print(f"set {iset}: {args.reps} reps  COMPUTE median {med:8.2f} ms  "
              f"min {min(comp):8.2f} ms  max {max(comp):8.2f} ms  |  copies (not in the "
              f"bracket): H2D {statistics.median(h2d):8.2f} ms  D2H {statistics.median(d2h):8.2f} ms",
              flush=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as fh:
            fh.write("kernel,deck,rep,set,reps,compute_ms,compute_min_ms,compute_max_ms,h2d_ms,d2h_ms,verify\n")
            for iset, comp, h2d, d2h in rows:
                fh.write(f"addusxx_g,{args.deck.name},{args.rep},{iset},{args.reps},{statistics.median(comp):.3f},"
                         f"{min(comp):.3f},{max(comp):.3f},{statistics.median(h2d):.3f},"
                         f"{statistics.median(d2h):.3f},PASS\n")

    print("RESULT:", "FAIL" if nfail else "PASS", flush=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
