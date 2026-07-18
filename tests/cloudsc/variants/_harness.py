"""Single-TU extraction + call scaffolding for the CLOUDSC-GPU variant tests: inlines
cloudsc_outer + dependency closure into one TU, builds the SDFG via the HLFIR bridge, and
runs the same TU through gfortran/f2py as the value baseline on identical seeded inputs.
"""

import re
from pathlib import Path

import numpy as np

from _util import build_sdfg
from cloudsc.full._harness import f2py_argnames, lower_keys, sdfg_call_args
from cloudsc.full._registries import get_inputs_physical, get_outputs
from dace_fortran import inline_to_single_tu

SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)
ENTRY = "cloudscouter"

HERE = Path(__file__).resolve().parent
DWARF_SRC = HERE / "dwarf-p-cloudsc" / "src"
GPU_SRC = DWARF_SRC / "cloudsc_gpu"
COMMON_MODULE = DWARF_SRC / "common" / "module"
COMMON_INCLUDE = DWARF_SRC / "common" / "include"

# Compute-path closure shared by both variants. FILE_IO_MOD intentionally excluded:
# tolerate_external_uses prunes the *_LOAD_PARAMETERS importers that would otherwise need it.
COMPUTE_CLOSURE = (
    GPU_SRC / "cloudsc_gpu_scc_k_caching_mod.F90",
    GPU_SRC / "cloudsc_driver_gpu_scc_k_caching_mod.F90",
    COMMON_MODULE / "parkind1.F90",
    COMMON_MODULE / "yomcst.F90",
    COMMON_MODULE / "yoethf.F90",
    COMMON_MODULE / "yoecldp.F90",
    COMMON_MODULE / "yomphyder.F90",
    COMMON_MODULE / "cloudsc_mpi_mod.F90",
    COMMON_MODULE / "timer_mod.F90",
)

# Timer stubs noop'd: f2py can't wrap a derived-type-dummy module proc (segfaults on import).
TIMER_NOOPS = tuple(("timer_mod", name) for name in (
    "performance_timer_start",
    "performance_timer_end",
    "performance_timer_thread_start",
    "performance_timer_thread_end",
    "performance_timer_thread_log",
    "performance_timer_print_performance",
))

# get_thread_num is a FUNCTION (result used, can't noop); no derived-type dummy, f2py-safe.
TIMER_STUBS = ("get_thread_num", )

# YOECLDP species-index PARAMETERs; must bake to compile-time literals, never runtime args.
BAKED_PARAMETERS = frozenset({"nclv", "ncldql", "ncldqi", "ncldqr", "ncldqs", "ncldqv"})


def extract_variant_tu(wrapper: Path, out_dir: Path, name: str, extra_sources: tuple = ()) -> str:
    """Inline one variant into a single self-contained TU; return its text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tu = inline_to_single_tu(
        [wrapper, *COMPUTE_CLOSURE, *extra_sources],
        entry=ENTRY,
        out_dir=out_dir,
        name=name,
        expand_cpp=True,
        # force_double_precision: SELECTED_REAL_KIND(...)->8 so JPRB=JPRL=JPRM=fp64. The
        # kernel's SC2026 JPRM(fp32) power/exp helpers become fp64, giving a bit-exact
        # SDFG-vs-gfortran comparison. Native mixed precision (JPRM=fp32) still matches to
        # ~1 fp32 ulp but not 1e-12; verify at full fp64.
        force_double_precision=True,
        include_dirs=[COMMON_INCLUDE],
        make_noop=list(TIMER_NOOPS),
        do_not_emit=TIMER_STUBS,
        tolerate_external_uses=True,
        # This TU is f2py-wrapped as the value baseline; an emptied stub type (PERFORMANCE_TIMER)
        # would make f2py emit a NULL module wrapper that segfaults on import.
        f2py_safe=True,
    )
    text = Path(tu).read_text()
    # YRECLDP allocatable->static patch: bridge can't flatten an allocatable derived-type
    # global; applied to both legs so the comparison contract is unaffected.
    text, nsub = re.subn(
        r"TYPE\s*\(\s*TECLDP\s*\)\s*,\s*ALLOCATABLE\s*::\s*YRECLDP",
        "TYPE(TECLDP) :: YRECLDP",
        text,
        flags=re.IGNORECASE,
    )
    assert nsub == 1, f"expected exactly one allocatable YRECLDP declaration in the TU, patched {nsub}"
    # Upstream scc_k_caching bug: ZPOW(ZTP1(JK),3) reads the 2-slot k-cache buffer at the full
    # level index JK (OOB for JK>2) where every sibling access uses JK_I. Undefined in both legs
    # (each reads different garbage past ZTP1(2)); patch to the intended JK_I so the comparison is
    # on defined behaviour. Applied to both legs identically.
    text, nztp = re.subn(r"zpow\(\s*ztp1\(\s*jk\s*\)", "zpow(ztp1(jk_i)", text, flags=re.IGNORECASE)
    assert nztp == 1, f"expected exactly one ZPOW(ZTP1(JK)) OOB site in the TU, patched {nztp}"
    return text


def assert_species_parameters_baked(sdfg):
    """YOECLDP species PARAMETERs must be compile-time literals: no free symbols, no arguments."""
    leaked = ({str(s).lower() for s in sdfg.free_symbols} | {k.lower() for k in sdfg.arglist()}) & BAKED_PARAMETERS
    assert not leaked, (f"YOECLDP PARAMETER constants leaked into the SDFG interface instead of "
                        f"config-propagating to literals: {sorted(leaked)}")


def run_cloudsc_gpu(tu_text: str,
                    name: str,
                    f2py_ref,
                    sdfg_dir: Path,
                    *,
                    seed: int = 42,
                    state_arrays: tuple = (),
                    simplify: bool = False):
    """Build the SDFG from the TU and run both legs on identical seeded physical inputs.

    state_arrays get per-leg copies (both start identical) and are returned alongside outputs.
    """
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(tu_text, sdfg_dir, name=name, entry=ENTRY).build()
    assert_species_parameters_baked(sdfg)
    if simplify:
        # "with simplify" leg: same comparison contract must hold after DaCe's simplify pipeline.
        sdfg.simplify()

    rng = np.random.default_rng(seed)
    inputs = lower_keys(get_inputs_physical(rng))
    outputs_ref = lower_keys(get_outputs(rng))
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}
    state_names = tuple(s.lower() for s in state_arrays)
    state_ref = {k: np.array(inputs[k], copy=True, order="F") for k in state_names}
    state_sdfg = {k: v.copy(order="F") for k, v in state_ref.items()}

    accepted = f2py_argnames(f2py_ref.cloudscouter)
    all_kw = {**inputs, **outputs_ref, **state_ref}
    f2py_ref.cloudscouter(**{k: v for k, v in all_kw.items() if k in accepted})

    scalars = {k: v for k, v in inputs.items() if isinstance(v, SCALAR_TYPES)}
    # cloudsc_mpi_mod rank/size survive as lifted scalar args; default to single-process values
    # matching the TU's own module initializers (f2py leg reads those globals directly).
    scalars.setdefault("irank", np.int32(0))
    scalars.setdefault("numproc", np.int32(1))
    kwargs = {k: v for k, v in inputs.items() if not isinstance(v, SCALAR_TYPES)}
    kwargs.update(outputs_sdfg)
    kwargs.update(state_sdfg)
    kwargs.update(sdfg_call_args(sdfg, scalars))
    arglist = sdfg.arglist()
    # Non-arglist registry entries are the baked YOECLDP PARAMETERs (asserted above);
    # anything else missing/unprovided is a bug.
    kwargs = {k: v for k, v in kwargs.items() if k in arglist}
    missing = sorted(set(arglist) - set(kwargs))
    assert not missing, f"SDFG arguments not covered by the registries: {missing}"
    sdfg(**kwargs)

    return {**outputs_sdfg, **state_sdfg}, {**outputs_ref, **state_ref}


def mismatch_report(outputs_sdfg: dict, outputs_ref: dict, names, *, rtol: float, atol: float) -> list:
    """Element-wise comparison over names; one report line per mismatch (empty == exact)."""
    report = []
    for name in names:
        a = np.asarray(outputs_sdfg[name.lower()])
        b = np.asarray(outputs_ref[name.lower()])
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        nbad = int(bad.sum())
        if nbad:
            report.append(f"{name}: {nbad} cell(s) exceed rtol={rtol} "
                          f"(max |delta|={np.abs(a - b)[bad].max():.3e})")
    return report
