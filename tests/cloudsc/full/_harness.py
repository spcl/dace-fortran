"""Shared CLOUDSC test scaffolding: build the SDFG, run f2py on identical seeded physical inputs, route scalars via the Scalar-vs-length-1-Array ABI convention (mismatch policy stays per-test)."""

import copy
import re
from pathlib import Path

import numpy as np
from dace import SDFG

from dace_fortran.pipelines import accepted_call_args, verify_numerics
from tests._util import build_sdfg, f2py_compile
from tests.cloudsc.full._registries import CLOUDSC_F90FLAGS, get_inputs_physical, get_outputs

_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)
_ENTRY = "cloudscouter"
_FULL_SOURCE = Path(__file__).resolve().parent / "cloudsc.F90"


def lower_keys(d: dict) -> dict:
    """Lowercase every key (flang HLFIR identifiers are case-sensitive; f2py wrappers expect lowercase kwargs)."""
    return {k.lower(): v for k, v in d.items()}


def f2py_argnames(fn) -> set:
    """Argument names f2py exposes, parsed from its generated docstring; bracketed entries (auto-derived shape symbols) are kept as accepted."""
    doc = fn.__doc__ or ""
    m = re.match(r"\s*\w+\((.*?)\)", doc, re.DOTALL)
    if not m:
        return set()
    al = m.group(1)
    opt = set()
    for mm in re.finditer(r"\[([^\]]+)\]", al):
        opt.update(s.strip() for s in mm.group(1).split(","))
    al = re.sub(r"\[[^\]]*\]", "", al)
    return {s.strip() for s in al.split(",") if s.strip()} | opt


def sdfg_call_args(sdfg, scalar_values: dict) -> dict:
    """Route each scalar to a plain Python scalar (SDFG Scalar/symbol) or a length-1 numpy array (intent out/inout) of
    the dtype the SDFG declares (a LOGICAL is its storage, e.g. ``uint32``)."""
    from dace.data import Scalar

    arglist = sdfg.arglist()
    out = {}
    for k, v in scalar_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=desc.dtype.as_numpy_dtype())
    return out


def f2py_reference(out_dir: Path):
    """The untouched full ``cloudsc.F90`` through gfortran/f2py at ``CLOUDSC_F90FLAGS``, exposing ``cloudscouter`` only.

    ``only``: the inner CLOUDSC's ``TYPE(TOMCST/...)`` dummies map to ``void`` and crash f2py's crackfortran.
    """
    return f2py_compile(
        _FULL_SOURCE.read_text(), out_dir, "cloudsc_ref", extra_f90flags=CLOUDSC_F90FLAGS, only=(_ENTRY,)
    )


def run_cloudsc(src: str, name: str, f2py_ref, sdfg_dir: Path, *, seed: int = 42):
    """Build the SDFG of ``src`` and :func:`run_against_reference` it as built."""
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return run_against_reference(build_sdfg(src, sdfg_dir, name=name, entry=_ENTRY).build(), f2py_ref, seed=seed)


def run_against_reference(sdfg: SDFG, f2py_ref, *, unoptimized: SDFG | None = None, seed: int = 42):
    """Run the f2py reference and ``sdfg`` on identical seeded physical inputs.

    ``unoptimized`` is ``sdfg`` as built, before an optimization pipeline ran on it. It is replayed
    on the same inputs and must agree with ``sdfg`` BIT-EXACTLY: a strictly different question from
    the returned reference comparison, isolating "did the pipeline change a value" from "does the
    frontend match gfortran" at a bar no reference tolerance can hold. Names it accepts that ``sdfg``
    does not (specialization bakes them out of the signature) are dropped from ``sdfg``'s call.

    Returns ``(outputs_sdfg, outputs_ref)`` -- lowercase-keyed dicts for the caller to compare under its own mismatch policy.
    """
    baked = accepted_call_args(unoptimized) - accepted_call_args(sdfg) if unoptimized is not None else set()

    rng = np.random.default_rng(seed)
    inputs = get_inputs_physical(rng)
    outputs_ref = lower_keys(get_outputs(rng))
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = f2py_argnames(f2py_ref.cloudscouter)
    all_kw = {**lower_keys(inputs), **lower_keys(outputs_ref)}
    f2py_ref.cloudscouter(**{k: v for k, v in all_kw.items() if k in accepted})

    scalars = {k.lower(): v for k, v in inputs.items() if isinstance(v, _SCALAR_TYPES)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _SCALAR_TYPES)}
    sdfg_kwargs.update(lower_keys(outputs_sdfg))
    sdfg_kwargs.update(sdfg_call_args(sdfg, scalars))
    # Snapshot BEFORE the run: the kernel writes through its arguments, so replaying from the
    # post-run dict would start both SDFGs from already-computed tendencies instead of the inputs.
    # Scalars are re-routed against the PRE-optimize descriptors -- specialization can change a
    # name's Scalar-vs-length-1-Array kind, and the snapshot must be called by its own convention.
    verify_kwargs = {}
    if unoptimized is not None:
        verify_kwargs = {**copy.deepcopy(sdfg_kwargs), **sdfg_call_args(unoptimized, scalars)}  # fresh buffers
    sdfg(**{k: v for k, v in sdfg_kwargs.items() if k not in baked})

    if unoptimized is not None:
        # Unfiltered: verify_numerics drops per-SDFG whatever each signature does not accept, so the
        # snapshot still receives the names specialization baked out of the optimized one.
        verify_numerics(unoptimized, sdfg, verify_kwargs)

    return outputs_sdfg, outputs_ref
