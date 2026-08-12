"""Specialise ``velocity_no_nproma.sdfgz`` into the 4 ``(lvn_only, istep)`` variants.

Runs on main DaCe. Uses ``propagate_if_cond`` to freeze known config symbols
and prune dead branches, then saves each variant as
``variants/velocity_no_nproma_if_prop_lvn_only_<L>_istep_<S>.sdfgz``.
"""

import argparse
import copy
import itertools
from pathlib import Path

import dace

from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from utils.propagate_if_cond import propagate_if_cond
from utils.unique_names import unique_names


_FIXED = {"lextra_diffu": "1", "ldeepatmo": "0"}


def specialize_vt(sdfg: dace.SDFG, config: dict) -> dace.SDFG:
    """Return a deep-copy of ``sdfg`` with ``config`` folded in, trivially
    true/false branches lifted, and dead branches pruned."""
    out = copy.deepcopy(sdfg)
    out.name = (
        sdfg.name + f"_if_prop_lvn_only_{config['lvn_only']}_istep_{config['istep']}"
    )
    propagate_if_cond(out, out, {**_FIXED, **config}, verbose=False)
    out.validate()
    if LiftTrivialIf is not None:
        LiftTrivialIf().apply_pass(out, {})
        out.validate()
    out.simplify(skip=["ScalarToSymbolPromotion"])
    out.validate()
    return out


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--input", default="velocity_no_nproma.sdfgz")
    argp.add_argument("--output-dir", default="variants")
    args = argp.parse_args()

    sdfg = dace.SDFG.from_file(args.input)
    sdfg.name = Path(args.input).stem
    sdfg.validate()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    variants = []
    for lvn_only, istep in itertools.product(("0", "1"), ("1", "2")):
        cfg = {"lvn_only": lvn_only, "istep": istep}
        print(f"Specialising for {cfg}")
        variants.append(specialize_vt(sdfg, cfg))

    unique_names(variants)
    for v in variants:
        path = outdir / f"{v.name}.sdfgz"
        v.save(path, compress=True)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
