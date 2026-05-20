"""Public entry point for building a :class:`dace.SDFG` from Fortran.

This is the one documented surface for turning Fortran into an SDFG;
it composes the pieces (module merge / text rewrites -> ``flang-new``
HLFIR -> the MLIR/C++ bridge -> ``SDFGBuilder``) so callers do not
wire them by hand:

- :func:`build_sdfg` -- a single inline source string.
- :func:`build_sdfg_from_files` -- a multi-file project (a driver
  plus the modules it ``USE``s, in any order); the file defining
  ``entry`` is the root, the rest are merged in via
  ``merge_used_modules``.

Both return a built, validated :class:`dace.SDFG`.  ``entry`` is the
mangled Flang symbol of the target procedure (``_QPrun`` for a free
subroutine, ``_QMmodPbar`` for a module procedure); it selects which
procedure the SDFG represents (and, for the multi-file form, which
input file is the root).

External (separately compiled) ``bind(c)`` functions a kernel calls
are declared through :mod:`dace_fortran.external`
(``register_external``); they are re-exported here for convenience.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

from dace import SDFG

from dace_fortran.build_bridge import hb  # noqa: F401  -- ensures the bridge is built
from dace_fortran.external import Arg, ExternalSignature, keep_external, register_external  # noqa: F401
from dace_fortran.hlfir_to_sdfg import DEFAULT_PIPELINE, SDFGBuilder
from dace_fortran.preprocess import preprocess_fortran_source

__all__ = [
    "build_sdfg",
    "build_sdfg_from_files",
    "register_external",
    "keep_external",
    "ExternalSignature",
    "Arg",
]


def _find_flang() -> str:
    """Locate ``flang-new-21`` or raise a clear error."""
    bin_ = shutil.which("flang-new-21")
    if bin_ is None:
        raise RuntimeError("flang-new-21 not on PATH; install LLVM/Flang 21 to use "
                           "the HLFIR frontend.")
    return bin_


def _entry_proc_name(entry: Optional[str]) -> Optional[str]:
    """Demangle a Flang entry symbol to its Fortran procedure name.

    Flang mangles uppercase tag letters (``M`` module, ``F`` function,
    ``S`` submodule, ``P`` procedure) around lowercased identifiers,
    so the procedure name is the segment after the last ``P``
    (``_QPmain`` -> ``main``; ``_QMmymodPbar`` -> ``bar``).  ``None``
    passes through (no root selection needed).
    """
    if not entry:
        return None
    return entry.rsplit("P", 1)[-1] if "P" in entry else entry


_PROC_RE = re.compile(r"^\s*(?:(?:recursive|pure|impure|elemental|module)\s+)*"
                      r"(?:[\w*()]+\s+)*?(subroutine|function)\s+(\w+)",
                      re.IGNORECASE)
_MOD_RE = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*end\s*(module|interface|subroutine|function)?\b", re.IGNORECASE)
_IFACE_RE = re.compile(r"^\s*(?:abstract\s+)?interface\b", re.IGNORECASE)


def _resolve_entry(source: str, entry: Optional[str]) -> str:
    """Return the mangled Flang entry symbol.

    ``entry`` given -> used verbatim.  ``None`` -> auto-resolved by
    scanning ``source`` for procedure *definitions* (``interface``
    blocks and ``end`` lines are skipped): exactly one -> derive its
    mangled name (``_QP<name>`` for a free procedure, ``_QM<mod>P<name>``
    for a module procedure); none or more than one -> a clear error
    (an SDFG targets one specific procedure -- no "first of many"
    guessing).

    :raises ValueError: if no procedure is found, or more than one is
        found and no explicit ``entry`` was given.
    """
    if entry:
        return entry
    cur_mod: Optional[str] = None
    iface_depth = 0
    procs = []  # (module|None, name)
    for raw in source.splitlines():
        line = raw.strip()
        if _IFACE_RE.match(line):
            iface_depth += 1
            continue
        m_end = _END_RE.match(line)
        if m_end:
            kind = (m_end.group(1) or "").lower()
            if kind == "interface" and iface_depth:
                iface_depth -= 1
            elif kind == "module":
                cur_mod = None
            continue
        m_mod = _MOD_RE.match(line)
        if m_mod and m_mod.group(1).lower() != "procedure":
            cur_mod = m_mod.group(1)
            continue
        if iface_depth:
            continue
        m_p = _PROC_RE.match(line)
        if m_p:
            procs.append((cur_mod, m_p.group(2)))
    if not procs:
        raise ValueError("build: no SUBROUTINE/FUNCTION definition found to use as the "
                          "SDFG entry; pass entry= explicitly")
    if len(procs) > 1:
        shown = ", ".join(n for _, n in procs)
        raise ValueError(f"build: source defines multiple procedures ({shown}); pass "
                          f"entry= (the mangled Flang symbol of the target one)")
    mod, name = procs[0]
    return f"_QM{mod.lower()}P{name.lower()}" if mod else f"_QP{name.lower()}"


def _emit_hlfir(source: str, out_dir: Path, name: str, *, merge: bool, preprocess: bool) -> Path:
    """Write ``source`` to ``<out_dir>/<name>.f90``, preprocess
    (module-merge + opt-in rewrites), ``flang -emit-hlfir`` it, and
    return the ``.hlfir`` path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{name}.f90"
    src.write_text(preprocess_fortran_source(source, search_dirs=[out_dir],
                                             merge=merge, if_intvar=preprocess))
    hlfir = out_dir / f"{name}.hlfir"
    subprocess.check_call([_find_flang(), "-fc1", "-emit-hlfir", str(src), "-o", str(hlfir)])
    return hlfir


def make_builder(source: str,
                 *,
                 entry: Optional[str] = None,
                 name: str = "sdfg",
                 pipeline: Optional[str] = None,
                 out_dir: Optional[Union[str, Path]] = None,
                 preprocess: bool = False) -> SDFGBuilder:
    """Resolve the entry, lower ``source`` to HLFIR, and return a
    configured (not yet built) :class:`SDFGBuilder`.

    This is the shared core of the build entry points -- the test
    harness (``tests/_util.build_sdfg``) calls it too so every test
    goes through one real implementation (entry auto-resolution and
    all) while still wrapping the builder for per-test xdist naming.

    ``entry`` may be ``None`` -> :func:`_resolve_entry` derives it from
    the single procedure in ``source`` (error if none / ambiguous).
    The ``.hlfir`` is parsed into the bridge module at ``SDFGBuilder``
    construction, so a temporary scratch dir is fine.
    """
    # ``_resolve_entry`` *validates* the auto case: it raises when an
    # entry-less source has zero or >1 procedures (the contract: no
    # "first of many").  For a validated single-procedure source we
    # forward the caller's original ``entry`` (``None``) unchanged --
    # passing a synthesised symbol would call ``set_entry_symbol`` and
    # perturb the single-proc lowering for no reason (every existing
    # entry-less single-proc test must stay byte-identical).  An
    # explicit ``entry`` is always forwarded (multi-proc / multi-file
    # need it to privatise the non-entry procedures).
    _resolve_entry(source, entry)
    pipeline = pipeline or DEFAULT_PIPELINE
    if out_dir is not None:
        hlfir = _emit_hlfir(source, Path(out_dir), name, merge=True, preprocess=preprocess)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=entry)
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        hlfir = _emit_hlfir(source, Path(td), name, merge=True, preprocess=preprocess)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=entry)


def build_sdfg(source: str,
               *,
               entry: Optional[str] = None,
               name: str = "sdfg",
               pipeline: Optional[str] = None,
               out_dir: Optional[Union[str, Path]] = None,
               preprocess: bool = False) -> SDFG:
    """Build a :class:`dace.SDFG` from a single inline Fortran source.

    :param source: Fortran source as one string.
    :param entry: mangled Flang symbol of the target procedure
        (``_QPrun`` for a free subroutine, ``_QMmodPbar`` for a module
        procedure).  ``None`` (default) -> auto-resolved from the
        single procedure in ``source``; an error is raised if the
        source has no procedure or more than one (an SDFG targets one
        specific procedure -- no "first of many" guessing).
    :param name: base filename for the scratch ``.f90`` / ``.hlfir``.
    :param pipeline: MLIR pass pipeline; defaults to
        ``DEFAULT_PIPELINE``.
    :param out_dir: scratch directory; a temporary one is used and
        removed when omitted.
    :param preprocess: also run the opt-in ``IF (intvar)`` rewrite
        (off by default so clean source is untouched).
    :returns: a built, validated SDFG.
    :raises ValueError: if ``entry`` is ``None`` and the source has no
        procedure or is ambiguous (more than one).
    """
    return make_builder(source, entry=entry, name=name, pipeline=pipeline,
                        out_dir=out_dir, preprocess=preprocess).build()


def build_sdfg_from_files(files: Sequence[Union[str, Path]],
                          *,
                          entry: Optional[str] = None,
                          name: str = "sdfg",
                          pipeline: Optional[str] = None,
                          out_dir: Optional[Union[str, Path]] = None,
                          preprocess: bool = False) -> SDFG:
    """Build a :class:`dace.SDFG` from a multi-file Fortran project.

    The files (a driver/root plus the modules it ``USE``s, in any
    order) are staged into the scratch dir; ``merge_used_modules``
    inlines every ``USE``-d module into the root's translation unit
    so flang sees one self-contained TU.  The root is the file that
    defines ``entry`` 's procedure.

    :param files: ``.f90`` paths (one defines ``entry``; the rest are
        its ``USE``-d modules).
    :param entry: mangled Flang symbol of the target procedure --
        **required**; it selects the root file.
    :param name: base filename for the merged ``.f90`` / ``.hlfir``.
    :param pipeline: MLIR pass pipeline; defaults to
        ``DEFAULT_PIPELINE``.
    :param out_dir: scratch directory; a temporary one is used and
        removed when omitted.
    :param preprocess: also run the opt-in ``IF (intvar)`` rewrite.
    :returns: a built, validated SDFG.
    :raises ValueError: ``entry`` missing, or no file defines its
        procedure.
    """
    if not entry:
        raise ValueError("build_sdfg_from_files requires entry= (it selects the root file)")
    paths = [Path(f) for f in files]
    proc = _entry_proc_name(entry)
    _def = re.compile(rf"^\s*(?:[\w()*]+\s+)*?(?:subroutine|function)\s+{re.escape(proc)}\b",
                      re.IGNORECASE | re.MULTILINE)
    roots = [p for p in paths if _def.search(p.read_text())]
    if not roots:
        raise ValueError(f"no input file defines procedure {proc!r} (entry {entry!r}); "
                          f"given {[p.name for p in paths]}")

    def _do(d: Path) -> SDFG:
        d.mkdir(parents=True, exist_ok=True)
        for p in paths:
            (d / p.name).write_text(p.read_text())
        return build_sdfg(roots[0].read_text(), entry=entry, name=name,
                           pipeline=pipeline, out_dir=d, preprocess=preprocess)

    if out_dir is not None:
        return _do(Path(out_dir))
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        return _do(Path(td))
