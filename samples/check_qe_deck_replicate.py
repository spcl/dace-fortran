#!/usr/bin/env python3
"""End-to-end check of qe_deck_replicate against QE-dumped ground truth.

Deck-side counterpart to the Fortran drivers: it reads a deck exactly the way
``bench_{addusxx,newdxx}_gpu.py:load_deck`` does (same keys, same F-order, same
symbols), replicates it, and runs a small numpy model of each kernel's flag='c'
arm.  The model is validated by the SAME run: rep=1 gates it against the pw.x
reference, so a rep=R pass is evidence about the replicator, not about the
model.  Gate: max|out-ref| <= 1e-11 * max(1, max|ref|), the drivers' bound.

Every rep>1 case is run twice -- with the indirection shift and without it (the
NEGATIVE CONTROL, which must FAIL).  A run where the control passes is reported
as a failure of this script.

    PYTHONHASHSEED=0 python3 check_qe_deck_replicate.py \
        --deck addusxx_g/data/BaO_nat002 --reps 1 2 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qe_deck_replicate import read_meta, replicate_addusxx, replicate_newdxx

TOL = 1e-11


def load(deck: Path, pfx: str, iset: int) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Only the keys the two kernels' flag='c' arms read (see the drivers)."""
    sm = read_meta(deck / "adxndx_static_meta.txt")
    nat, nsp, nkb, nnr, nhm = (int(sm[k]) for k in ("nat", "nsp", "nkb", "nnr", "nhm"))

    def rd(name: str, dtype: type) -> np.ndarray:
        return np.fromfile(deck / name, dtype=dtype)

    def f2d(flat: np.ndarray, d0: int) -> np.ndarray:
        return flat.reshape((d0, flat.size // d0), order="F").copy(order="F")

    a: dict[str, np.ndarray] = {}
    s: dict[str, int] = {"nat": nat, "nkb": nkb, "dfftt_nnr": nnr}
    for i in (1, 2, 3):
        e = rd(f"adxndx_static_eigts{i}.bin", np.complex128)
        a[f"eigts{i}"] = f2d(e, e.size // nat)
        s[f"eigts{i}_d1"] = nat
    a["mill"] = f2d(rd("adxndx_static_mill.bin", np.int32), 3)
    s["mill_d1"] = a["mill"].shape[1]
    a["dfftt_nl"] = rd("adxndx_static_dfftt_nl.bin", np.int32)
    s["dfftt_nl_d0"] = a["dfftt_nl"].size
    a["ityp"] = rd("adxndx_static_ityp.bin", np.int32)
    s["ityp_d0"] = nat
    a["tau"] = f2d(rd("adxndx_static_tau.bin", np.float64), 3)
    s["tau_d1"] = nat
    a["ofsbeta"] = rd("adxndx_static_ofsbeta.bin", np.int32)
    s["ofsbeta_d0"] = nat
    a["ijtoh"] = rd("adxndx_static_ijtoh.bin", np.int32).reshape((nhm, nhm, nsp), order="F").copy(order="F")
    vkb = rd("adxndx_static_vkb.bin", np.complex128)
    a["vkb"] = f2d(vkb, vkb.size // nkb)
    s["vkb_d1"] = nkb
    a["nh"] = np.array([sm[f"nh_{i}"] for i in range(1, nsp + 1)], dtype=np.int32)
    a["upf_tvanp"] = np.array([sm[f"tvanp_{i}"] == 1 for i in range(1, nsp + 1)], dtype=np.bool_)
    a["dfftt_ngm"] = np.array([int(sm["ngmt"])], dtype=np.int32)
    a["omega"] = np.array([float(sm["omega"])], dtype=np.float64)

    pm = read_meta(deck / f"{pfx}_{iset}_meta.txt")
    qd0 = int(pm["qgm_d1"])
    a["qgm"] = f2d(rd(f"{pfx}_{iset}_qgm.bin", np.complex128), qd0)
    s["qgm_d0"], s["qgm_d1"] = qd0, int(pm["qgm_d2"])
    a["nij_type"] = rd(f"{pfx}_{iset}_nij_type.bin", np.int32)
    a["xkq"] = rd(f"{pfx}_{iset}_xkq.bin", np.float64)
    a["xk"] = rd(f"{pfx}_{iset}_xk.bin", np.float64)
    a["becphi_c"] = rd(f"{pfx}_{iset}_becphi_c.bin", np.complex128)
    a["becphi_r"] = np.zeros(nkb, dtype=np.float64)
    if pfx == "adx":
        a["becpsi_c"] = rd(f"{pfx}_{iset}_becpsi_c.bin", np.complex128)
        return a, s, rd(f"{pfx}_{iset}_rhoc_in.bin", np.complex128), rd(f"{pfx}_{iset}_rhoc_out.bin", np.complex128)
    a["vc"] = rd(f"{pfx}_{iset}_vc_in.bin", np.complex128)
    return a, s, rd(f"{pfx}_{iset}_deexx_in.bin", np.complex128), rd(f"{pfx}_{iset}_deexx_out.bin", np.complex128)


def structure_factor(a: dict, na: int, ngms: int) -> np.ndarray:
    """eigqts(na) * eigts1*eigts2*eigts3 at the Miller indices of the first ngms G."""
    arg = 2.0 * np.pi * float(((a["xk"] - a["xkq"]) * a["tau"][:, na]).sum())
    sf = np.cos(arg) - 1j * np.sin(arg)
    for i in (1, 2, 3):
        e = a[f"eigts{i}"]
        sf = sf * e[a["mill"][i - 1, :ngms] + (e.shape[0] - 1) // 2, na]
    return sf


def addusxx_model(a: dict, rhoc_in: np.ndarray) -> np.ndarray:
    ngms = int(a["dfftt_ngm"][0])
    rhoc = rhoc_in.copy()
    for na in range(a["ityp"].size):
        nt = int(a["ityp"][na]) - 1
        if not a["upf_tvanp"][nt]:
            continue
        nij, ijkb0, nh = int(a["nij_type"][nt]), int(a["ofsbeta"][na]), int(a["nh"][nt])
        aux2 = np.zeros(ngms, dtype=np.complex128)
        for ih in range(nh):
            aux1 = np.zeros(ngms, dtype=np.complex128)
            for jh in range(nh):
                aux1 += a["qgm"][:ngms, nij + int(a["ijtoh"][ih, jh, nt]) - 1] * a["becpsi_c"][ijkb0 + jh]
            aux2 += aux1 * np.conj(a["becphi_c"][ijkb0 + ih])
        # add.at, not fancy-index +=: the aliasing control must fail on the
        # aliasing, not on numpy silently dropping duplicate scatter targets.
        np.add.at(rhoc, a["dfftt_nl"][:ngms].astype(np.int64) - 1, aux2 * structure_factor(a, na, ngms))
    return rhoc


def newdxx_model(a: dict, deexx_in: np.ndarray) -> np.ndarray:
    ngms = int(a["dfftt_ngm"][0])
    auxvc = a["vc"][a["dfftt_nl"][:ngms].astype(np.int64) - 1]
    fact = float(a["omega"][0])
    deexx = deexx_in.copy()
    for na in range(a["ityp"].size):
        nt = int(a["ityp"][na]) - 1
        if not a["upf_tvanp"][nt]:
            continue
        nij, ijkb0, nh = int(a["nij_type"][nt]), int(a["ofsbeta"][na]), int(a["nh"][nt])
        aux2 = np.conj(auxvc) * structure_factor(a, na, ngms)
        for ih in range(nh):
            aux1 = np.zeros(ngms, dtype=np.complex128)
            for jh in range(nh):
                aux1 += a["becphi_c"][ijkb0 + jh] * np.conj(a["qgm"][:ngms, nij + int(a["ijtoh"][ih, jh, nt]) - 1])
            deexx[ijkb0 + ih] += fact * np.vdot(aux2, aux1)
    return deexx


def gate(out: np.ndarray, ref: np.ndarray) -> tuple[float, bool]:
    rel = float(np.abs(out - ref).max() / max(1.0, float(np.abs(ref).max())))
    return rel, rel <= TOL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True, help="samples/<kernel>/data/<MAT>")
    ap.add_argument("--sets", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--reps", type=int, nargs="*", default=[1, 2, 8])
    args = ap.parse_args()

    pfx = "adx" if (args.deck / "adx_0_meta.txt").is_file() else "ndx"
    model = addusxx_model if pfx == "adx" else newdxx_model
    replicate = replicate_addusxx if pfx == "adx" else replicate_newdxx
    print(f"deck {args.deck} kernel={'addusxx_g' if pfx == 'adx' else 'newdxx_g'} tol={TOL:g}")

    nfail = 0
    for iset in args.sets:
        if not (args.deck / f"{pfx}_{iset}_meta.txt").is_file():
            continue
        base = load(args.deck, pfx, iset)
        for rep in args.reps:
            for offset in (True, False):
                if rep == 1 and not offset:
                    continue  # rep=1 has no replica to shift
                a, _, io_in, ref = replicate(*base, rep, offset)
                rel, ok = gate(model(a, io_in), ref)
                want = offset  # the control MUST fail
                tag = "PASS" if ok else "FAIL"
                print(f"  set {iset} rep={rep:<4} offset={offset!s:<5} rel={rel:.4e}  {tag}"
                      f"{'' if ok is want else '   <-- UNEXPECTED'}")
                if ok is not want:
                    nfail += 1
    print("RESULT:", "FAIL" if nfail else "PASS")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
