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

# Loop-exchange ON, matching the ICON integration lib (lowered with -D__LOOP_EXCHANGE) and the
# Fortran reference (VT_ACC_TU=loopexch).  The two TUs are not a loop swap: __LOOP_EXCHANGE also
# transposes the local work arrays -- zeta(nlev, nproma, nblks_v) here vs zeta(nproma, nlev,
# nblks_v) in the no-exchange TU -- so mixing them compares different memory layouts.
DEFAULT_TU = REPO / "tests" / "icon" / "atmosphere" / "velocity_advection_inlined_single_tu.f90"
DEFAULT_ENTRY = "mo_velocity_advection::velocity_tendencies"
CONFIG_KEYS = ("lvn_only", "istep", "lextra_diffu", "ldeepatmo", "lvert_nest")

# R02B05, single non-nested domain. None of these four is in the dumps (`p_patch.*.data`
# carries only nblks_c/nblks_e/nblks_v/cells/edges/verts), so the driver cannot read them
# back -- they are grid constants and have to be folded in here. f2dace did the same, which
# is why its generated code carried a literal 91 for nlevp1.
FIXED_SYMBOLS = {"p_patch_nlev": 90, "p_patch_nlevp1": 91, "p_patch_nshift": 0, "timers_level": 0}

# Length-1 arrays rather than symbols, so they fold through propagate_if_cond like the other
# knobs. lvert_nest gates `l_vert_nested`, whose only consumer guards the sole read of
# p_diag_vn_ie_ubc -- false here drops that whole upper-boundary path.
FIXED_KNOBS = {"lvert_nest": "0"}


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
    propagate_if_cond(out, out, {**_FIXED, **FIXED_KNOBS, **config}, verbose=False)
    out.validate()
    out.simplify()
    out.validate()
    return out


def specialize_knobs(sdfg: dace.SDFG, knobs: dict) -> tuple:
    """Fold each length-1 signature knob without moving the ABI.

    ``preserve_abi`` leaves the array on the signature -- the driver keeps passing
    ``lvert_nest`` -- and stages it into a fresh transient scalar carrying every body read.
    Only that staged copy is promoted to a symbol and specialised, so the branch folds while
    the argument stays. Must run BEFORE :func:`normalize_config_reads`: the pass matches the
    ``knob[0]`` subscript form, which normalisation would already have rewritten away.
    """
    from dace.transformation.passes import ConvertLengthOneArraysToScalars
    from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion

    seen = set(sdfg.arrays)
    ConvertLengthOneArraysToScalars(preserve_abi=True, filter=set(knobs)).apply_pass(sdfg, {})
    staged = {new: knobs[k] for new in set(sdfg.arrays) - seen for k in knobs if new.startswith(k)}
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    folded = {k: v for k, v in staged.items() if k in sdfg.symbols or k in sdfg.free_symbols}
    sdfg.specialize(folded)
    return staged, folded


def constant_lbounds(csv_path: Path) -> dict:
    """``(leaf, dim) -> value`` for every lbound the R02B05 dumps show as constant.

    Rows carry a semicolon-separated set of observed values; a single value means the
    extent never varies across the dumped timesteps and can be folded. 73 of the 77 lbound
    rows are constantly 1. Measured on R02B05, but the offsets are identical on the other
    grids (R02B06+), so folding them stays valid for the ICON sweep, not just this driver.
    """
    table = {}
    for line in csv_path.read_text().splitlines()[1:]:
        parts = line.split(",", 3)
        if len(parts) != 4:
            continue
        kind, field, dim, values = parts
        if kind != "lbound" or ";" in values:
            continue
        table[(field, int(dim))] = int(values)
    return table


def offset_constants(sdfg: dace.SDFG, table: dict) -> dict:
    """Match ``offset_<container>_<path>_<leaf>_d<N>`` against the leaf-keyed lbound table.

    The flat name's split point is unknown, so the leaf is recovered by longest matching
    suffix -- that keeps ``area_edge`` from being shadowed by ``area``.
    """
    live = {}
    for sym in set(map(str, sdfg.free_symbols)) | set(map(str, sdfg.symbols)):
        m = re.fullmatch(r"offset_(.+)_d(\d+)", sym)
        if m is None:
            continue
        base, dim = m.group(1), int(m.group(2))
        cands = [f for (f, d) in table if d == dim and (base == f or base.endswith("_" + f))]
        if cands:
            live[sym] = table[(max(cands, key=len), dim)]
    return live


def save_binding_metadata(sdfg: dace.SDFG, out: Path, name: str) -> Path | None:
    """Persist the builder's Fortran interface + flatten plan as JSON.

    Both are plain attributes the SDFGBuilder attaches (``builder/__init__.py:1002``) and
    neither survives a ``.sdfgz`` roundtrip, so a later stage that only has the saved SDFG
    cannot recover them. They are the authoritative struct-component -> flat-argument map:
    the emitted Fortran binding performs exactly the marshalling the C++ driver has to
    reproduce by hand, so the call site is derived from this rather than from splitting
    flat names on underscores.
    """
    import json

    iface = getattr(sdfg, "_fortran_interface_raw", None)
    plan = getattr(sdfg, "_flatten_plan_raw", None)
    if iface is None and plan is None:
        return None
    path = out / f"{name}_bindings.json"
    path.write_text(json.dumps({"fortran_interface": iface, "flatten_plan": plan}, indent=1, default=str))
    print(f"  binding metadata -> {path}", flush=True)
    return path


def emit_codegen(variant: dace.SDFG, codegen_root: Path) -> Path:
    """Write one variant's DaCe program folder under ``codegen/baseline/<variant>/``.

    The generated ``include/<variant>.h`` is the authoritative signature the C++ driver
    must call, so it is produced here rather than as a side effect of a later compile.
    """
    from dace.codegen import compiler

    folder = codegen_root / variant.name
    variant.build_folder = str(folder)
    compiler.generate_program_folder(variant, variant.generate_code(), str(folder))
    variant.save(str(folder / f"{variant.name}.sdfgz"), compress=True)
    return folder / "include" / f"{variant.name}.h"


def emit_fortran_binding(variant: dace.SDFG, codegen_root: Path, meta_path: Path, entry: str) -> Path | None:
    """Emit the Fortran binding module beside the generated header.

    The binding marshals the Fortran derived types into the flat argument list, so it is the
    readable form of the struct-component -> argument mapping and has to be regenerated
    whenever specialisation changes the signature. Arg order comes from ``arglist()``, which is
    what codegen emits, rather than from the frozen snapshot.
    """
    import json

    if not meta_path.is_file():
        return None
    from dace_fortran.bindings.emit_bindings import emit_bindings
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from dace_fortran.bindings.frozen_signature import refreeze

    meta = json.loads(meta_path.read_text())
    iface = build_auto_interface(meta["fortran_interface"], entry)
    plan = FlattenPlan.from_dict(meta["flatten_plan"] or {})
    out = codegen_root / variant.name / f"{variant.name}_bindings.f90"
    emit_bindings(refreeze(variant), iface, plan, str(out),
                  dace_arglist=tuple(variant.arglist().keys()))
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
    ap.add_argument("--codegen-dir", type=Path, default=HERE / "codegen" / "baseline",
                    help="where each variant's DaCe program folder (and its header) lands")
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="frontend scratch (default <output-dir>/frontend_build)")
    ap.add_argument("--name", default="velocity_no_nproma", help="must match utils/stages/common.py's template")
    ap.add_argument("--from-sdfg", type=Path, default=None, help="skip the frontend, load this .sdfgz instead")
    ap.add_argument("--force", action="store_true", help="rebuild the frontend SDFG even if cached")
    ap.add_argument("--frontend-only", action="store_true",
                    help="run step 1 and stop -- the half that needs the FaCe dace")
    ap.add_argument("--lbounds-csv", type=Path, default=HERE / "baseline" / "lbounds.csv",
                    help="observed extents from tools/analyze_lbounds; constant lbounds get folded")
    ap.add_argument("--no-specialize", action="store_true",
                    help="keep the R02B05 grid constants (FIXED_SYMBOLS/FIXED_KNOBS) as runtime arguments")
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
        save_binding_metadata(sdfg, out, args.name)
        print(f"  cached -> {cache}", flush=True)

    if args.frontend_only:
        return

    from dace.transformation.passes import RemoveUnusedSymbols

    from dace_fortran.bindings.frozen_signature import refreeze
    from utils.passes.unify_variant_signatures import unify_variant_signatures
    from utils.unique_names import unique_names

    sdfg.name = args.name
    sdfg.validate()
    assert_no_structs(sdfg)
    print(f"  arrays={len(sdfg.arrays)} symbols={len(sdfg.symbols)} free={len(sdfg.free_symbols)}", flush=True)
    if not args.no_specialize:
        before = len(sdfg.arglist())
        staged, folded = specialize_knobs(sdfg, FIXED_KNOBS)
        print(f"  staged {staged}, folded {folded}", flush=True)
    print(f"  normalized {normalize_config_reads(sdfg)} config-knob subscript read(s)", flush=True)
    if not args.no_specialize:
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf

        from utils.passes.specialize_constants import specialize_constants

        live = {k: v for k, v in FIXED_SYMBOLS.items() if k in sdfg.symbols or k in sdfg.free_symbols}
        sdfg.specialize(live)
        print(f"  specialized {len(live)} grid symbol(s): {live}", flush=True)

        if args.lbounds_csv.is_file():
            offs = offset_constants(sdfg, constant_lbounds(args.lbounds_csv))
            sdfg.specialize(offs)
            print(f"  specialized {len(offs)} constant offset(s) from {args.lbounds_csv.name}", flush=True)

        # Before the simplify below, not after: the specialize calls above only record constants.
        consts = specialize_constants(sdfg)
        print(f"  folded {len(consts)} constant(s) to literals", flush=True)

        sdfg.simplify()
        LiftTrivialIf().apply_pass(sdfg, {})
        sdfg.simplify()
        print(f"  arglist {before} -> {len(sdfg.arglist())}", flush=True)
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
        # Specialisation folded scalars and symbols away, so the snapshot the frontend
        # attached no longer matches. Re-snapshot BEFORE saving: the .sdfgz is what a
        # later stage reads the signature back out of.
        refreeze(v)
        v.save(str(out / f"{v.name}.sdfgz"), compress=True)
        v.validate()
        if args.compile:
            v.compile()
        print(f"  -> {out / f'{v.name}.sdfgz'}  arrays={len(v.arrays)}", flush=True)

    print(f"step 5/5: codegen -> {args.codegen_dir}", flush=True)
    sigs = {}
    meta_path = out / f"{args.name}_bindings.json"
    for v in variants:
        hdr = emit_codegen(v, args.codegen_dir.resolve())
        sigs[v.name] = hdr.read_text().replace(v.name, "VARIANT")
        print(f"  -> {hdr}  arglist={len(v.arglist())}", flush=True)
        try:
            f90 = emit_fortran_binding(v, args.codegen_dir.resolve(), meta_path, args.entry.split("::")[-1])
        except Exception as exc:
            print(f"     fortran binding FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"     -> {f90}" if f90 else f"     (no binding: {meta_path.name} missing, "
              f"rerun --frontend-only --force)", flush=True)
    # The 4 variants are dispatched behind one call site, so their headers must agree
    # once the per-variant state-struct name is normalised away.
    odd = [n for n, t in sigs.items() if t != next(iter(sigs.values()))]
    if odd:
        raise AssertionError(f"variant headers disagree beyond the state struct: {odd}")
    print("  all 4 headers identical modulo the state struct name", flush=True)


if __name__ == "__main__":
    main()
