#!/usr/bin/env python3
"""Load-time deck replication for the QE us_exx kernels addusxx_g / newdxx_g.

The shipped decks are too small to measure (addusxx_g BaO is 0.30 ms at 72
threads).  This module grows one dump set R-fold IN MEMORY -- no giant files on
disk -- by tiling the replicable payload and SHIFTING the kernel's output
indirection array so replica k addresses its own output slice:

  addusxx_g  G-vector axis.  Output is a SCATTER rhoc(dfftt%nl(ig)) += ...
             qgm/mill/dfftt_nl tiled R times, dfftt_ngm -> R*ngmt,
             dfftt_nnr -> R*nnr, rhoc tiled; nl of replica k shifted by k*nnr.
             dfftt%nl is an injective G -> FFT-grid map and the shift is what
             keeps the union injective -- which is also what lets the GPU arm
             do the scatter with no atomics/WCR.

  newdxx_g   ATOM axis.  Output is a REDUCTION deexx(ofsbeta(na)+ih) += ...,
             so the output indirection is ofsbeta, not nl.  nat -> R*nat,
             nkb -> R*nkb, ityp/tau/eigts/vkb/becphi/deexx tiled; ofsbeta of
             replica k shifted by k*nkb.  qgm/mill/dfftt/vc are species- or
             G-indexed and stay shared.  This axis is also newdxx_g's OpenMP
             `do` loop (nat = 2..5 in the shipped decks).

Both take the (arrays, symbols, io_in, ref) tuple that the bench scripts'
``load_deck`` returns and give back the same tuple, replicated.  ``ref`` comes
back tiled, so the EXISTING gate max|out-ref| <= 1e-11*max(1,max|ref|) over the
whole array is a per-replica check: every replica must reproduce the pw.x
ground truth on its own slice.

``offset=False`` is the NEGATIVE CONTROL: the shift is dropped, replicas alias
replica 0, and the gate must fail.  Run this file directly to see the index
algebra checked on a real deck, with and without the shift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

Arrays = dict[str, np.ndarray]
Symbols = dict[str, int]
Deck = tuple[Arrays, Symbols, np.ndarray, np.ndarray]


def read_meta(path: Path) -> dict[str, int | float]:
    meta: dict[str, int | float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            meta[parts[0]] = int(parts[1])
        except ValueError:
            meta[parts[0]] = float(parts[1])
    return meta


def tile_1d(a: np.ndarray, rep: int, n: int | None = None) -> np.ndarray:
    """Tile the first ``n`` entries of a 1-D array ``rep`` times.

    ``np.copy``, not ``ascontiguousarray``: np.tile ends in a reshape, so its result is a VIEW of
    the repeat temporary (``base is not None``) and is already contiguous -- which makes
    ascontiguousarray a no-op that leaves the view marker in place, and DaCe refuses view
    arguments outright.  The 2-D tilers escape this only because asfortranarray has to re-copy a
    C-ordered tile.
    """
    return np.copy(np.tile(a[:n], rep))


def tile_axis1(a: np.ndarray, rep: int, ncols: int | None = None) -> np.ndarray:
    """Tile the first ``ncols`` COLUMNS of a 2-D array ``rep`` times along axis 1.

    ``a[:, :ncols]`` keeps the axis; ``a[:, ncols]`` would drop it.
    """
    return np.asfortranarray(np.tile(a[:, :ncols], (1, rep)))


def tile_axis0(a: np.ndarray, rep: int, nrows: int | None = None) -> np.ndarray:
    """Tile the first ``nrows`` ROWS of a 2-D array ``rep`` times along axis 0."""
    return np.asfortranarray(np.tile(a[:nrows, :], (rep, 1)))


def replicate_nl(nl: np.ndarray, nnr: int, rep: int, offset: bool = True) -> np.ndarray:
    """addusxx_g scatter map: replica k gets nl + k*nnr (or nl, as the control)."""
    step = nnr if offset else 0
    ngmt = nl.size
    out = np.empty(rep * ngmt, dtype=nl.dtype)
    for k in range(rep):
        out[k * ngmt:(k + 1) * ngmt] = nl + k * step
    return out


def replicate_ofsbeta(ofsbeta: np.ndarray, nkb: int, rep: int, offset: bool = True) -> np.ndarray:
    """newdxx_g deexx map: replica k gets ofsbeta + k*nkb (or ofsbeta, control)."""
    step = nkb if offset else 0
    nat = ofsbeta.size
    out = np.empty(rep * nat, dtype=ofsbeta.dtype)
    for k in range(rep):
        out[k * nat:(k + 1) * nat] = ofsbeta + k * step
    return out


def nl_is_injective(nl: np.ndarray, nnr_total: int) -> bool:
    """A scatter with no atomics is only correct while this holds."""
    idx = nl.astype(np.int64)
    return bool(idx.min() >= 1 and idx.max() <= nnr_total and np.unique(idx).size == idx.size)


def deexx_targets_disjoint(ofsbeta: np.ndarray, ityp: np.ndarray, nh: np.ndarray, nkb_total: int) -> bool:
    """Same question for newdxx_g: the per-atom beta manifolds must not overlap."""
    hit = np.concatenate([np.arange(o + 1, o + 1 + int(nh[int(t) - 1])) for o, t in zip(ofsbeta, ityp, strict=True)])
    return bool(hit.min() >= 1 and hit.max() <= nkb_total and np.unique(hit).size == hit.size)


def replicate_addusxx(arrays: Arrays,
                      symbols: Symbols,
                      rhoc_in: np.ndarray,
                      ref: np.ndarray,
                      rep: int,
                      offset: bool = True) -> Deck:
    if rep < 1:
        raise ValueError(f"rep must be >= 1, got {rep}")
    if rep == 1:
        return arrays, symbols, rhoc_in, ref
    a, s = dict(arrays), dict(symbols)
    nnr, ngmt = int(s["dfftt_nnr"]), int(s["dfftt_nl_d0"])
    if int(s["qgm_d0"]) != ngmt:
        raise ValueError(f"qgm_d0={s['qgm_d0']} != ngmt={ngmt}: G-axis replication assumes they match")

    a["qgm"] = tile_axis0(a["qgm"], rep)
    s["qgm_d0"] = rep * ngmt
    a["mill"] = tile_axis1(a["mill"], rep, ngmt)  # Miller indices index the SHARED eigts tables
    s["mill_d1"] = rep * ngmt
    a["dfftt_nl"] = replicate_nl(a["dfftt_nl"], nnr, rep, offset)
    s["dfftt_nl_d0"] = rep * ngmt
    a["dfftt_ngm"] = np.array([rep * ngmt], dtype=a["dfftt_ngm"].dtype)
    s["dfftt_nnr"] = rep * nnr

    if offset and not nl_is_injective(a["dfftt_nl"], rep * nnr):
        raise AssertionError("replicated dfftt_nl is not injective")
    return a, s, tile_1d(rhoc_in, rep), tile_1d(ref, rep)


def replicate_newdxx(arrays: Arrays,
                     symbols: Symbols,
                     deexx_in: np.ndarray,
                     ref: np.ndarray,
                     rep: int,
                     offset: bool = True) -> Deck:
    if rep < 1:
        raise ValueError(f"rep must be >= 1, got {rep}")
    if rep == 1:
        return arrays, symbols, deexx_in, ref
    a, s = dict(arrays), dict(symbols)
    nat, nkb = int(s["nat"]), int(s["nkb"])

    for i in (1, 2, 3):
        a[f"eigts{i}"] = tile_axis1(a[f"eigts{i}"], rep, nat)
        s[f"eigts{i}_d1"] = rep * nat
    a["tau"] = tile_axis1(a["tau"], rep, nat)
    s["tau_d1"] = rep * nat
    a["ityp"] = tile_1d(a["ityp"], rep, nat)
    s["ityp_d0"] = rep * nat
    a["ofsbeta"] = replicate_ofsbeta(a["ofsbeta"], nkb, rep, offset)
    s["ofsbeta_d0"] = rep * nat
    a["vkb"] = tile_axis1(a["vkb"], rep, nkb)
    s["vkb_d1"] = rep * nkb
    a["becphi_c"] = tile_1d(a["becphi_c"], rep, nkb)
    a["becphi_r"] = np.zeros(rep * nkb, dtype=a["becphi_r"].dtype)
    s["nat"], s["nkb"] = rep * nat, rep * nkb

    if offset and not deexx_targets_disjoint(a["ofsbeta"], a["ityp"], a["nh"], rep * nkb):
        raise AssertionError("replicated deexx target ranges overlap")
    return a, s, tile_1d(deexx_in, rep), tile_1d(ref, rep)


def main() -> int:
    ap = argparse.ArgumentParser(description="index-algebra self-check on a real deck")
    ap.add_argument("--deck", type=Path, required=True, help="samples/<kernel>/data/<MAT>")
    ap.add_argument("--rep", type=int, default=4)
    args = ap.parse_args()

    sm = read_meta(args.deck / "adxndx_static_meta.txt")
    nnr, nat, nkb, nsp = int(sm["nnr"]), int(sm["nat"]), int(sm["nkb"]), int(sm["nsp"])
    nl = np.fromfile(args.deck / "adxndx_static_dfftt_nl.bin", dtype=np.int32)
    ofsbeta = np.fromfile(args.deck / "adxndx_static_ofsbeta.bin", dtype=np.int32)
    ityp = np.fromfile(args.deck / "adxndx_static_ityp.bin", dtype=np.int32)
    nh = np.array([sm[f"nh_{i}"] for i in range(1, nsp + 1)], dtype=np.int32)

    print(f"deck {args.deck.name}: nnr={nnr} ngmt={nl.size} nat={nat} nkb={nkb} rep={args.rep}")
    print(f"  base nl injective ......... {nl_is_injective(nl, nnr)}")
    print(f"  base deexx disjoint ....... {deexx_targets_disjoint(ofsbeta, ityp, nh, nkb)}")
    rc = 0
    for offset in (True, False):
        tag = "offset ON " if offset else "offset OFF"
        ok_nl = nl_is_injective(replicate_nl(nl, nnr, args.rep, offset), args.rep * nnr)
        rep_ofs = replicate_ofsbeta(ofsbeta, nkb, args.rep, offset)
        ok_dx = deexx_targets_disjoint(rep_ofs, tile_1d(ityp, args.rep), nh, args.rep * nkb)
        print(f"  {tag}: addusxx nl injective={ok_nl}  newdxx deexx disjoint={ok_dx}")
        if ok_nl is not offset or ok_dx is not offset:
            rc = 1
    print("SELFCHECK:", "FAIL" if rc else "PASS (offset ON clean, offset OFF collides)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
