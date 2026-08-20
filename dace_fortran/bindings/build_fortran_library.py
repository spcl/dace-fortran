# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a Fortran-callable shared library from a built SDFG.

The dace-fortran-only entrypoint tying the binding contract together;
``dace.SDFG.compile`` stays vanilla DaCe.  Order: compile the SDFG,
verify it still matches the ``FrozenSignature`` snapshot (raises
:class:`SignatureDriftError` before any binding is emitted), emit the
``<entry>_bindings.f90`` wrapper, then gfortran-link it with the
kernel ``.so`` and any extra sources into one shared library.
"""
import ctypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from dace_fortran.bindings.bind_c_shim import emit_bind_c_shim
from dace_fortran.bindings.emit_bindings import emit_bindings
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface, build_auto_interface

#: Mandatory flags -- a shared, position-independent, long-line module.
_SHARED_FLAGS = ("-shared", "-fPIC", "-ffree-line-length-none")

#: True when a source line is a module-level variable declaration carrying
#: ALLOCATABLE (i.e. a deferred-shape array) without a TARGET attribute.
_TARGET_RE = re.compile(r"\bTARGET\b", re.IGNORECASE)
_DECL_TYPE_RE = re.compile(r"^\s*(?:REAL|INTEGER|LOGICAL|COMPLEX|DOUBLE\s+PRECISION|CHARACTER)\b", re.IGNORECASE)
_ALLOCATABLE_RE = re.compile(r"\bALLOCATABLE\b", re.IGNORECASE)
_POINTER_RE = re.compile(r"\bPOINTER\b", re.IGNORECASE)
_NON_VAR_RE = re.compile(
    r"^\s*(?:TYPE|INTERFACE|ABSTRACT\s+INTERFACE|PROCEDURE|MODULE\s+PROCEDURE|"
    r"SUBROUTINE|FUNCTION|END\b)", re.IGNORECASE)

_TYPE_DEF_START_RE = re.compile(r"^\s*TYPE\s+(?:::|,?\s*\w)", re.IGNORECASE)
_END_TYPE_RE = re.compile(r"^\s*END\s*TYPE\b", re.IGNORECASE)


def _ensure_target_on_module_deferred_arrays(text: str) -> str:
    """Add ``TARGET`` to module-level allocatable array declarations that lack it.

    The generated binding aliases contiguous module arrays with a wrapper-local
    ``pointer, contiguous`` and ``X => X__mod``; this requires the host variable
    to carry the ``TARGET`` attribute.  Inserting it in the prelude sources is
    safe: it only enables pointer association, it does not change the variable's
    type, layout, or lifetime.

    ``POINTER`` variables are left untouched: Fortran prohibits both ``POINTER``
    and ``TARGET`` on the same entity, and a pointer can already be the target of
    another pointer association.  Derived-type components are also untouched.
    """
    lines = text.splitlines(keepends=True)
    in_module = False
    in_spec = True
    in_type = False
    out: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if re.match(r"^\s*MODULE\s+\w+", raw, re.IGNORECASE):
            in_module = True
            in_spec = True
            in_type = False
        elif in_module and re.match(r"^\s*CONTAINS\b", raw, re.IGNORECASE):
            in_spec = False
            in_type = False
        elif in_module and re.match(r"^\s*END\s*MODULE\b", raw, re.IGNORECASE):
            in_module = False
            in_spec = True
            in_type = False
        elif in_spec and _TYPE_DEF_START_RE.match(raw):
            in_type = True
        elif in_type and _END_TYPE_RE.match(raw):
            in_type = False
        elif in_spec and not in_type and _DECL_TYPE_RE.match(raw) and _ALLOCATABLE_RE.search(raw) \
                and not _POINTER_RE.search(raw) and not _TARGET_RE.search(raw) and "::" in raw \
                and not _NON_VAR_RE.match(raw):
            # Insert TARGET right before the first :: on the line.
            raw = re.sub(r"(\S)(\s*)(::)", r"\1, target\2\3", raw, count=1)
        out.append(raw)
    return "".join(out)


#: Optimised + debug info + strict IEEE (no fast-math/fp-contract,
#: rounding-aware) so SDFG-vs-reference comparisons stay bit-reproducible.
_DEBUG_FLAGS = ("-O3", "-g", "-fno-fast-math", "-ffp-contract=off", "-frounding-math")

#: -O3 -ffast-math -- trades IEEE reproducibility for speed.
_RELEASE_FLAGS = ("-O3", "-ffast-math")

_MODE_FLAGS = {"debug": _DEBUG_FLAGS, "release": _RELEASE_FLAGS}


@dataclass
class FortranLibrary:
    """A built Fortran-callable shared library: linked ``.so``, its SDFG
    kernel ``.so``, the emitted bindings wrapper, and (if requested) the
    bind(c) shim source."""

    so_path: Path
    sdfg_so: Path
    bindings_f90: Path
    bind_c_shim_f90: Path = None

    def load(self) -> ctypes.CDLL:
        """Open the library with :class:`ctypes.CDLL`.

        If the kernel needs OpenMP, supply it via ``LD_PRELOAD`` --
        deliberately not hard-coded here so any runtime works."""
        return ctypes.CDLL(str(self.so_path))


def build_fortran_library(
        sdfg,
        iface: OriginalInterface = None,
        plan: FlattenPlan = None,
        out_dir: str = None,
        *,
        name: str = None,
        prelude_sources: Sequence = (),
        extra_sources: Sequence = (),
        mode: str = "debug",
        flags: Sequence = None,
        extra_flags: Sequence = (),
        verify: bool = True,
        bind_c_shim: bool = False,
        bind_c_shim_debug_prints: bool = False,
        bind_c_shim_module_symbol_forward=(),
) -> FortranLibrary:
    """Emit + verify + link a Fortran-callable library for ``sdfg``.

    ``iface``/``plan`` default to the SDFG's stamped snapshots when
    omitted.  ``prelude_sources`` compile BEFORE the binding (deps the
    binding ``use``s); ``extra_sources`` compile AFTER (callers/shims
    that ``use`` the binding) -- order matters for gfortran's
    left-to-right module resolution.  ``mode`` picks debug (bit-
    reproducible) or release flags unless ``flags`` overrides it
    entirely; ``extra_flags`` appends rather than replaces.
    ``bind_c_shim=True`` auto-generates and links a ``bind(c)`` C-ABI
    entry point (flat + inline-flat-struct dummies only; raises
    :class:`UnsupportedShimInterfaceError` otherwise).

    :raises SignatureDriftError: live SDFG drifted from the snapshot.
    """
    if out_dir is None:
        raise ValueError("build_fortran_library: out_dir is required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or sdfg.name

    if flags is not None:
        opt_flags = tuple(flags)
    elif mode in _MODE_FLAGS:
        opt_flags = _MODE_FLAGS[mode]
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'debug', "
                         f"'release', or an explicit flags= list")

    frozen = getattr(sdfg, "_frozen_signature", None)
    if frozen is None:
        raise ValueError("build_fortran_library requires an SDFG built by "
                         "SDFGBuilder.build() (no _frozen_signature attached). "
                         "Plain SDFGs are unaffected -- dace-core codegen is "
                         "vanilla; the drift contract is dace-fortran-only.")
    # Refuse to emit a wrapper that disagrees with the kernel -- checked
    # before compile so it never produces a stale binding.
    if verify:
        frozen.verify_against(sdfg)

    # Auto-derive when not given, after the drift gate so a drift error
    # surfaces first; iface is built against final ``name`` for symbol match.
    if plan is None:
        raw = getattr(sdfg, "_flatten_plan_raw", None)
        if raw is None:
            raise ValueError("build_fortran_library: no plan given and the SDFG "
                             "carries no _flatten_plan_raw (build via SDFGBuilder).")
        plan = FlattenPlan.from_dict(raw)
    if iface is None:
        raw = getattr(sdfg, "_fortran_interface_raw", None)
        if raw is None:
            raise ValueError("build_fortran_library: no iface given and the SDFG "
                             "carries no _fortran_interface_raw (build via SDFGBuilder).")
        iface = build_auto_interface(raw, name)

    compiled = sdfg.compile()
    sdfg_so = Path(compiled._lib._library_filename)

    # Authoritative __program_<entry> arg order comes live from
    # CompiledSDFG._sig (codegen output, transform-dependent -- NOT
    # snapshotted in FrozenSignature).  Empty -> falls back to frozen.args.
    dace_arglist = tuple(getattr(compiled, "_sig", None) or ())

    bindings_f90 = out_dir / f"{name}_bindings.f90"
    emit_bindings(frozen, iface, plan, str(bindings_f90), dace_arglist)

    # Threaded between the binding (which the shim USEs) and extra_sources
    # -- gfortran compiles strictly left-to-right by module dependency.
    shim_f90 = None
    if bind_c_shim:
        shim_f90 = out_dir / f"{iface.entry}_c.f90"
        emit_bind_c_shim(iface,
                         str(shim_f90),
                         debug_prints=bind_c_shim_debug_prints,
                         module_symbol_forward=bind_c_shim_module_symbol_forward,
                         plan=plan)

    so_path = out_dir / f"lib{name}.so"
    # Copy and patch prelude sources in the build dir so module arrays used by
    # the binding carry TARGET; the originals are left untouched.
    patched_preludes: List[Path] = []
    for src in prelude_sources:
        src_path = Path(src)
        patched = out_dir / f"{src_path.stem}_target{src_path.suffix}"
        patched.write_text(_ensure_target_on_module_deferred_arrays(src_path.read_text()))
        patched_preludes.append(patched)

    # gfortran compiles left-to-right, no reordering: deps before, users after.
    cmd = [
        "gfortran", *_SHARED_FLAGS, *opt_flags, *extra_flags, "-fopenmp", f"-J{out_dir}",
        *[str(s) for s in patched_preludes],
        str(bindings_f90), *([str(shim_f90)] if shim_f90 else []), *[str(s) for s in extra_sources], "-o",
        str(so_path), f"-L{sdfg_so.parent}", f"-Wl,-rpath,{sdfg_so.parent}", f"-l:{sdfg_so.name}"
    ]
    subprocess.check_call(cmd, cwd=out_dir)
    return FortranLibrary(so_path=so_path, sdfg_so=sdfg_so, bindings_f90=bindings_f90, bind_c_shim_f90=shim_f90)
