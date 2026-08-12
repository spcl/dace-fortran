"""Generate the 4 baseline SDFG variants straight from the single-TU Fortran.

Replaces the old f2dace path (phase 0 ``tools/sdfg_from_velocity_f90.py`` plus steps
1-5 of this file) with one call into the dace-fortran HLFIR frontend on

    $REPO/tests/icon/atmosphere/velocity_advection_inlined_no_loop_exchange_single_tu.f90

The frontend already emits struct-free SoA, so ``StructToContainerGroups``,
``PrepareBaseline``, ``DealiasSymbols``, ``ResolveExtentOffsets`` and
``ResolveExtentSizes`` have nothing to match: there are no ``tmp_struct_symbol_*``,
``__f2dace_SA*``/``__f2dace_SOA*`` or ``__CG_*`` names, no ``Structure`` descriptors and
no global_code/init_code to parse. That also drops the ``data_r02b05`` / ``lbounds.csv``
dependency -- extents come out of the frontend as symbols, not as observed lbounds.

What survives, renumbered:

1. frontend: single TU -> SDFG (cached as ``<name>_frontend.sdfgz``)
2. save the pre-specialise baseline
3. ``specialize_vt`` x4 over ``{lvn_only, istep}`` (CONFIG PROP)
4. ``unify_variant_signatures`` + ``unique_names``, save each

Nothing here matches on a data name: the frontend's names differ from f2dace's
(``nproma`` not ``__CG_global_data__m_nproma``, ``p_diag_vt`` not ``__CG_p_diag__m_vt``,
``nlev`` a free symbol rather than the folded literal 91). The one name that is fixed is
the SDFG's own, because ``utils/stages/common.py`` derives the 4 variant filenames from
``velocity_no_nproma``; ``--name`` follows it if that template ever changes.
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

import dace
from dace import data
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion

# The two halves of this script need DIFFERENT dace checkouts, so nothing dace-version
# specific may be imported at module scope. Step 1 runs the dace-fortran frontend, which
# needs FaCe (``dace.sdfg.utils.specialize_symbols``); steps 2-4 run VTP's passes, which
# need yakup/dev (``dace.codegen.targets.unroller``, ``lift_trivial_if``). Neither branch
# has the other's module, so a single process cannot do both -- run ``--frontend-only``
# under FaCe first, then re-run with yakup/dev on PYTHONPATH to consume the cache.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "samples")]

DEFAULT_TU = REPO / "tests" / "icon" / "atmosphere" / "velocity_advection_inlined_no_loop_exchange_single_tu.f90"
DEFAULT_ENTRY = "mo_velocity_advection::velocity_tendencies"
CONFIG_KEYS = ("lvn_only", "istep", "lextra_diffu", "ldeepatmo")


def frontend_sdfg(tu: Path, entry: str, name: str, build_dir: Path) -> dace.SDFG:
    from _util import build_sdfg, have_flang
    if not have_flang():
        raise SystemExit("no LLVM flang on PATH -- source samples/env.sh first")
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(tu.read_text(), build_dir, name=name, entry=entry).build()


def normalize_config_reads(sdfg: dace.SDFG, keys=CONFIG_KEYS) -> int:
    """``propagate_if_cond`` substitutes config knobs as bare identifiers, so a length-1
    container read as ``x[0]`` would become ``1[0]``. Rewrite ``x[0]`` -> ``x`` first.

    A key only qualifies if it resolves to a length-1 descriptor on that SDFG, so this is
    a no-op for the ones the frontend hands over as symbols.
    """
    total = 0
    for nested in sdfg.all_sdfgs_recursive():
        live = [k for k in keys
                if isinstance(nested.arrays.get(k), data.Data) and nested.arrays[k].total_size == 1]
        if not live:
            continue
        pats = [(re.compile(r"(?<!\w)" + re.escape(k) + r"\s*\[\s*0\s*\]"), k) for k in live]

        def sub(text: str) -> str:
            nonlocal total
            for pat, k in pats:
                text, n = pat.subn(k, text)
                total += n
            return text

        for edge in nested.all_interstate_edges():
            if edge.data.assignments:
                edge.data.assignments = {k: sub(v) if isinstance(v, str) else v
                                         for k, v in edge.data.assignments.items()}
            if edge.data.condition is not None:
                edge.data.condition = CodeBlock(sub(edge.data.condition.as_string))
        for block in nested.all_control_flow_blocks():
            if isinstance(block, ConditionalBlock):
                block.branches[:] = [(CodeBlock(sub(c.as_string)) if c is not None else c, b)
                                     for c, b in block.branches]
        for region in nested.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                for attr in ("loop_condition", "init_statement", "update_statement"):
                    code = getattr(region, attr, None)
                    if code is not None:
                        setattr(region, attr, CodeBlock(sub(code.as_string)))
    return total


def specialize(sdfg: dace.SDFG, config: dict) -> dace.SDFG:
    """``utils.specialize_vt.specialize_vt`` plus a block-label de-dup between
    ``LiftTrivialIf`` and each validate."""
    import copy

    from dace.transformation.passes.lift_trivial_if import LiftTrivialIf

    from utils.propagate_if_cond import propagate_if_cond
    from utils.specialize_vt import _FIXED

    out = copy.deepcopy(sdfg)
    out.name = f"{sdfg.name}_if_prop_lvn_only_{config['lvn_only']}_istep_{config['istep']}"
    propagate_if_cond(out, out, {**_FIXED, **config}, verbose=False)
    out.validate()
    out.simplify()
    out.validate()
    return out


def assert_no_structs(sdfg: dace.SDFG):
    bad = [f"{n.name}.{k}" for n in sdfg.all_sdfgs_recursive() for k, d in n.arrays.items()
           if isinstance(d, data.Structure)]
    if bad:
        raise AssertionError(f"{sdfg.name!r} still carries struct descriptors: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tu", type=Path, default=DEFAULT_TU, help="single translation unit to compile")
    ap.add_argument("--entry", default=DEFAULT_ENTRY)
    ap.add_argument("--output-dir", type=Path, default=HERE / "baseline")
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="frontend scratch (default <output-dir>/frontend_build)")
    ap.add_argument("--name", default="velocity_no_nproma", help="must match utils/stages/common.py's template")
    ap.add_argument("--from-sdfg", type=Path, default=None, help="skip the frontend, load this .sdfgz instead")
    ap.add_argument("--force", action="store_true", help="rebuild the frontend SDFG even if cached")
    ap.add_argument("--frontend-only", action="store_true",
                    help="run step 1 and stop -- the half that needs the FaCe dace")
    ap.add_argument("--compile", action="store_true", help="compile each variant as a checkpoint")
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{args.name}_frontend.sdfgz"

    if args.from_sdfg is not None:
        print(f"step 1/4: loading {args.from_sdfg}", flush=True)
        sdfg = dace.SDFG.from_file(str(args.from_sdfg))
    elif cache.exists() and not args.force:
        print(f"step 1/4: reusing cached frontend SDFG {cache}", flush=True)
        sdfg = dace.SDFG.from_file(str(cache))
    else:
        if not args.tu.is_file():
            raise SystemExit(f"translation unit not found: {args.tu}")
        print(f"step 1/4: dace-fortran frontend on {args.tu.name} (entry {args.entry})", flush=True)
        build_dir = (args.build_dir or out / "frontend_build").resolve()
        sdfg = frontend_sdfg(args.tu, args.entry, args.name, build_dir)
        sdfg.save(str(cache), compress=True)
        print(f"  cached -> {cache}", flush=True)

    if args.frontend_only:
        return

    from dace.transformation.passes import RemoveUnusedSymbols

    from utils.passes.unify_variant_signatures import unify_variant_signatures
    from utils.unique_names import unique_names

    sdfg.name = args.name
    sdfg.validate()
    assert_no_structs(sdfg)
    print(f"  arrays={len(sdfg.arrays)} symbols={len(sdfg.symbols)} free={len(sdfg.free_symbols)}", flush=True)
    print(f"  normalized {normalize_config_reads(sdfg)} config-knob subscript read(s)", flush=True)
    sdfg.validate()

    pre = out / f"{args.name}_post_aos_soa.sdfgz"
    sdfg.save(str(pre), compress=True)
    print(f"step 2/4: saved pre-specialise baseline -> {pre}", flush=True)

    print("step 3/4: specialize_vt x4 (config prop)", flush=True)
    variants = []
    for lvn_only, istep in itertools.product(("0", "1"), ("1", "2")):
        cfg = {"lvn_only": lvn_only, "istep": istep}
        print(f"  {cfg}", flush=True)
        variants.append(specialize(sdfg, cfg))
    for v in variants:
        RemoveUnusedSymbols().apply_pass(v, {})
        v.validate()

    print("step 4/4: unify_variant_signatures + unique_names", flush=True)
    unify_variant_signatures(sdfg, variants)
    for v in variants:
        v.validate()
        assert_no_structs(v)
    unique_names(variants)
    for v in variants:
        path = out / f"{v.name}.sdfgz"
        v.save(str(path), compress=True)
        v.validate()
        if args.compile:
            v.compile()
        print(f"  -> {path}  arrays={len(v.arrays)}", flush=True)


if __name__ == "__main__":
    main()
