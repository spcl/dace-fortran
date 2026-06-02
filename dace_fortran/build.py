"""Public entry points for building a :class:`dace.SDFG` from Fortran.

Three entry points layered by how much the bridge does on the
caller's behalf:

- :func:`build_sdfg` -- a single inline source string; the bridge
  composes module merge / text rewrites -> ``flang-new`` HLFIR ->
  the MLIR/C++ bridge -> ``SDFGBuilder``.
- :func:`build_sdfg_from_files` -- a multi-file project (a driver
  plus the modules it ``USE``s, in any order); the file defining
  ``entry`` is the root, the rest are merged in via
  ``merge_used_modules`` so flang sees one self-contained TU.
- :func:`build_sdfg_from_hlfir` (tier 3) -- the bridge does not
  drive flang at all; the user's own build system emits ``.hlfir``
  via :mod:`dace_fortran.emit_hlfir` and the bridge consumes it.
  This is the canonical path for codebases too large or dep-tangled
  for the bridge to compile alone (ICON-scale, real external
  libraries).
- :func:`build_sdfg_from_project` (tier 3) -- the one-call form of
  the above: hand it a built project's ``compile_commands.json``
  and it emits + lowers in a single step.

All three return a built, validated :class:`dace.SDFG`.  ``entry``
is the mangled Flang symbol of the target procedure (``_QPrun`` for
a free subroutine, ``_QMmodPbar`` for a module procedure); it
selects which procedure the SDFG represents (and, for the
multi-file / multi-``.hlfir`` forms, which input the bridge starts
from).

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
    "build_sdfg_from_hlfir",
    "build_sdfg_from_project",
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


def _flang_intrinsic_modules_path(flang_bin: str) -> Optional[Path]:
    """Locate the LLVM-flang intrinsic-modules directory shipped beside ``flang_bin``.

    Probed paths (first match wins):

    * ``<flang_install_root>/include/flang/iso_c_binding.mod``  --  the
      Ubuntu / Debian / Fedora layout (``flang-new-21`` -> ``/usr/lib/llvm-21/bin``;
      modules under ``/usr/lib/llvm-21/include/flang``).
    * ``<flang_install_root>/../share/flang/include`` -- some custom builds.

    Returns the **directory** that contains ``iso_c_binding.mod`` (and the
    matching ``ieee_*`` / ``omp_lib`` / ... modules), or ``None`` if no
    match is found.  When called with ``-fc1``, flang skips the driver-side
    auto-population of this path -- without an explicit
    ``-fintrinsic-modules-path`` every ``USE iso_c_binding`` fails at
    semantic analysis with a ``No explicit type declared for 'c_int'``
    error.

    The probe is best-effort: a missing match leaves the bridge to fall
    back on flang's compiled-in default, which usually fails for
    ``-fc1`` invocations but works for tests that don't ``USE`` any
    intrinsic module.
    """
    flang_real = Path(flang_bin).resolve()
    install_root = flang_real.parent.parent  # /usr/lib/llvm-21
    candidates = (
        install_root / "include" / "flang",
        install_root / "share" / "flang" / "include",
    )
    for cand in candidates:
        if (cand / "iso_c_binding.mod").exists():
            return cand
    return None


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
    """Return the mangled Flang entry symbol for ``entry``.

    Three input forms, all keyed off a scan of ``source`` for procedure
    *definitions* (``interface`` blocks and ``end`` lines are skipped):

    - a mangled symbol (``_Q...`` ) -> returned verbatim;
    - a plain Fortran name (``solve_nh`` or ``mod::proc`` ) -> resolved
      against the scan to its mangled form;
    - ``None`` -> auto-resolved, requiring exactly one definition (an SDFG
      targets one specific procedure -- no "first of many" guessing).

    Mangling is ``_QP<name>`` for a free procedure and ``_QM<mod>P<name>``
    for a module procedure.

    :raises ValueError: if the named procedure is not found or is ambiguous,
        or (``None`` case) the source has no or more than one procedure.
    """
    if entry and entry.startswith("_Q"):
        return entry  # already a mangled symbol
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

    def _mangle(mod, name):
        return f"_QM{mod.lower()}P{name.lower()}" if mod else f"_QP{name.lower()}"

    # Plain Fortran name given (``proc`` or ``mod::proc``): resolve against
    # the scanned definitions so callers need not hand-write the mangled symbol.
    if entry:
        want_mod, _, want_proc = entry.lower().rpartition("::")
        want_mod = want_mod or None
        matches = {(m, n) for (m, n) in procs
                   if n.lower() == want_proc
                   and (want_mod is None or (m or "").lower() == want_mod)}
        if not matches:
            raise ValueError(f"build: no procedure {entry!r} defined in the source")
        if len(matches) > 1:
            cands = ", ".join(f"{(m or '<free>')}::{n}" for m, n in sorted(matches))
            raise ValueError(f"build: entry {entry!r} is ambiguous ({cands}); "
                             f"qualify it as module::proc")
        return _mangle(*matches.pop())

    # entry=None: derive from the single procedure definition.
    if not procs:
        raise ValueError("build: no SUBROUTINE/FUNCTION definition found to use as the "
                          "SDFG entry; pass entry= explicitly")
    if len(procs) > 1:
        shown = ", ".join(n for _, n in procs)
        raise ValueError(f"build: source defines multiple procedures ({shown}); pass "
                          f"entry= (the Fortran name or mangled symbol of the target one)")
    return _mangle(*procs[0])


def _emit_hlfir(source: str, out_dir: Path, name: str, *, merge: bool,
                preprocess: bool, defines: Sequence[str] = ()) -> Path:
    """Write ``source`` to ``<out_dir>/<name>.F90``, preprocess
    (module-merge + opt-in rewrites), ``flang -fc1 -cpp -emit-hlfir``
    it, and return the ``.hlfir`` path.

    ``out_dir`` is the merge search path -- ``merge_used_modules``
    inlines any ``.f90`` staged there by ``build_sdfg_from_files`` so
    flang sees one self-contained TU.  For codebases too large /
    dep-tangled for the text-merge to handle cleanly, use
    :func:`build_sdfg_from_hlfir` instead and let the project's own
    build system emit the ``.hlfir``.

    flang runs with ``-cpp -U_OPENMP -U_OPENACC`` so kernels that ship
    ``#ifdef`` blocks are still consumable; the macros are forced
    undefined to match what :func:`strip_openmp_directives` already
    applies at the Fortran-text level.  ``defines`` are extra ``-D`` cpp
    macros (``NAME`` / ``NAME=val``) for sources whose ``#if`` branches
    select code by a build-time configuration -- the no-build-system
    equivalent of the ``-D`` flags tier 3 reads from
    ``compile_commands.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # ``.F90`` routes through flang's built-in preprocessor by extension
    # convention; combined with ``-cpp`` this stays consistent if a
    # future flang changes its default.
    src = out_dir / f"{name}.F90"
    src.write_text(preprocess_fortran_source(source, search_dirs=[out_dir],
                                             merge=merge, if_intvar=preprocess))
    hlfir = out_dir / f"{name}.hlfir"
    cmd = [_find_flang(), "-fc1", "-cpp", "-U_OPENMP", "-U_OPENACC",
           "-I", str(out_dir)]
    intrinsic_path = _flang_intrinsic_modules_path(cmd[0])
    if intrinsic_path is not None:
        cmd += ["-fintrinsic-modules-path", str(intrinsic_path)]
    for d in defines:
        cmd += ["-D", d]
    cmd += ["-emit-hlfir", str(src), "-o", str(hlfir)]
    # flang resolves ``USE iso_c_binding`` (and the other intrinsic modules)
    # by first checking cwd for a matching ``.mod`` file, then falling back
    # on the install's intrinsic-modules path.  An earlier failed build can
    # leave a stale stub ``iso_c_binding.mod`` (or similar) in cwd that
    # checksums different from the install's ``__fortran_builtins`` -- flang
    # then refuses to resolve ``c_int`` etc. with a misleading
    # ``No explicit type declared`` error.  Run flang in a clean scratch
    # cwd to avoid the cwd-precedence trap.
    subprocess.check_call(cmd, cwd=str(out_dir))
    return hlfir


def make_builder(source: str,
                 *,
                 entry: Optional[str] = None,
                 name: str = "sdfg",
                 pipeline: Optional[str] = None,
                 out_dir: Optional[Union[str, Path]] = None,
                 preprocess: bool = False,
                 defines: Sequence[str] = ()) -> SDFGBuilder:
    """Resolve the entry, lower ``source`` to HLFIR, and return a
    configured (not yet built) :class:`SDFGBuilder`.

    This is the shared core of the build entry points -- the test
    harness (``tests/_util.build_sdfg``) calls it too so every test
    goes through one real implementation (entry auto-resolution and
    all) while still wrapping the builder for per-test xdist naming.

    ``entry`` may be ``None`` (:func:`_resolve_entry` derives it from the
    single procedure in ``source``; error if none / ambiguous), a plain
    Fortran name (``proc`` or ``mod::proc`` ), or a mangled ``_Q...`` symbol.
    The ``.hlfir`` is parsed into the bridge module at ``SDFGBuilder``
    construction, so a temporary scratch dir is fine.
    """
    # ``_resolve_entry`` validates the auto case (it raises when an
    # entry-less source has zero or >1 procedures -- the contract is no
    # "first of many") and resolves a plain Fortran name to its mangled
    # symbol.  ``entry=None`` is forwarded unchanged: a validated
    # single-procedure source must not call ``set_entry_symbol`` (it would
    # perturb the single-proc lowering -- every entry-less test stays
    # byte-identical).  A given entry (mangled or plain name) is forwarded
    # as the resolved symbol so multi-proc / multi-file builds privatise
    # the non-entry procedures.
    resolved = _resolve_entry(source, entry)
    fwd = None if entry is None else resolved
    pipeline = pipeline or DEFAULT_PIPELINE
    if out_dir is not None:
        hlfir = _emit_hlfir(source, Path(out_dir), name, merge=True,
                            preprocess=preprocess, defines=defines)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=fwd)
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        hlfir = _emit_hlfir(source, Path(td), name, merge=True,
                            preprocess=preprocess, defines=defines)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=fwd)


def build_sdfg(source: str,
               *,
               entry: Optional[str] = None,
               name: str = "sdfg",
               pipeline: Optional[str] = None,
               out_dir: Optional[Union[str, Path]] = None,
               preprocess: bool = False,
               defines: Sequence[str] = ()) -> SDFG:
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
    :param defines: extra ``-D`` cpp macros (``NAME`` / ``NAME=val``)
        for flang's preprocessor -- set the build configuration when
        there is no build system to read it from (tier 3 reads the same
        flags from ``compile_commands.json`` automatically).
    :returns: a built, validated SDFG.
    :raises ValueError: if ``entry`` is ``None`` and the source has no
        procedure or is ambiguous (more than one).
    """
    return make_builder(source, entry=entry, name=name, pipeline=pipeline,
                        out_dir=out_dir, preprocess=preprocess,
                        defines=defines).build()


#: ``func.func @<symbol>(`` -- the MLIR opener for a procedure
#: **definition** inside a flang-emitted ``.hlfir``.  The negative
#: lookahead drops the ``func.func private @<sym>(...)`` form, which
#: is just a forward declaration for a cross-TU call site (flang
#: emits one of those in every ``.hlfir`` whose source calls the
#: procedure); only the file with the actual body should match.
_HLFIR_FUNC_RE = re.compile(r"func\.func\s+(?!private\b)@([A-Za-z_]\w*)\s*\(")


def _resolve_hlfir_for_entry(root: Path, entry: str) -> Path:
    """Walk ``root`` for ``.hlfir`` files and return the one whose
    MLIR text defines ``func.func @<entry>(...)``.

    Real build systems (cmake / make / fpm) emit one ``.hlfir`` per
    translation unit into a build / artefact tree -- the caller knows
    the entry symbol, not which TU it lives in.  This scan keeps the
    caller from having to track that mapping by hand.
    """
    matches = []
    for p in sorted(root.rglob("*.hlfir")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in _HLFIR_FUNC_RE.finditer(txt):
            if m.group(1) == entry:
                matches.append(p)
                break
    if not matches:
        raise FileNotFoundError(
            f"no .hlfir under {root} defines func.func @{entry}; "
            f"check that the build emitted HLFIR for the TU containing "
            f"the entry (and that the entry symbol is correctly mangled)")
    if len(matches) > 1:
        raise ValueError(
            f"multiple .hlfir under {root} define @{entry} -- pick one explicitly: "
            f"{[str(p) for p in matches]}")
    return matches[0]


def build_sdfg_from_hlfir(hlfir_path: Union[str, Path],
                          *,
                          entry: Optional[str] = None,
                          pipeline: Optional[str] = None) -> SDFG:
    """Build a :class:`dace.SDFG` from a pre-emitted ``.hlfir`` file
    produced by the project's own build system (the tier-3 path; see
    the module docstring for when to reach for it).

    ``hlfir_path`` may be:

    * a path to a specific ``.hlfir`` file -- consumed directly; or
    * a directory -- walked recursively for the one ``.hlfir`` whose
      MLIR contains ``func.func @<entry>(...)``.  ``entry`` is
      **required** in this form (it is the match key); this saves the
      caller from tracking which per-TU ``.hlfir`` holds the entry.

    Inlining is intra-TU only: flang emits one ``.hlfir`` per
    translation unit, so a procedure ``USE``-d from another TU stays
    an external symbol reference in the SDFG (the right contract for
    halo exchanges / I/O routines deliberately left external).

    :param hlfir_path: path to a ``.hlfir`` file, or a directory
        containing one.
    :param entry: mangled Flang symbol of the target procedure.
        Optional when ``hlfir_path`` is a single file with one
        procedure; **required** when ``hlfir_path`` is a directory
        (it selects which ``.hlfir`` to load).
    :param pipeline: MLIR pass pipeline; defaults to
        ``DEFAULT_PIPELINE``.
    :returns: a built, validated SDFG.
    :raises FileNotFoundError: directory has no ``.hlfir`` containing
        ``func.func @<entry>``.
    :raises ValueError: directory passed without ``entry``, or several
        ``.hlfir`` files define the same entry symbol.
    """
    pipeline = pipeline or DEFAULT_PIPELINE
    p = Path(hlfir_path)
    if p.is_dir():
        if not entry:
            raise ValueError("build_sdfg_from_hlfir requires entry= when given a "
                              "directory (it selects which .hlfir to load)")
        p = _resolve_hlfir_for_entry(p, entry)
    return SDFGBuilder(str(p), pipeline=pipeline, entry=entry).build()


def build_sdfg_from_project(compile_commands: Union[str, Path],
                            *,
                            entry: str,
                            stubs: Sequence[Union[str, Path]] = (),
                            out_dir: Optional[Union[str, Path]] = None,
                            pipeline: Optional[str] = None,
                            flang: str = "flang-new-21") -> SDFG:
    """Build a :class:`dace.SDFG` from a built project's
    ``compile_commands.json`` in one call -- tier 3.

    Collapses the two tier-3 steps (emit HLFIR for the project's TUs,
    then lower the entry) into one.  Equivalent to::

        from dace_fortran.emit_hlfir import emit
        emit(compile_commands=cc, stubs=stubs, out_dir=hlfir_dir, flang=flang)
        sdfg = build_sdfg_from_hlfir(hlfir_dir, entry=entry, pipeline=pipeline)

    Use the explicit two-step form when you want to emit once and
    lower several entries, or to inspect the intermediate ``.hlfir``
    files; this wrapper re-emits each call.

    The ``compile_commands.json`` artefact comes from the project's
    own build (``cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON``, or
    ``bear -- make`` for autotools / plain-make builds like ICON);
    see the README's *Building an SDFG from a real project* section.

    :param compile_commands: path to the build's ``compile_commands.json``.
    :param entry: the target procedure -- either the mangled Flang symbol
        (``_QMmo_solve_nonhydroPsolve_nh``) or the plain Fortran name
        (``solve_nh`` / ``mo_solve_nonhydro::solve_nh``), resolved against
        the project sources.  Selects which emitted ``.hlfir`` to lower and
        restricts emission to its USE-closure.
    :param stubs: flang-buildable stub sources for modules flang has
        no shipped ``.mod`` for (``mpi`` / ``netcdf`` / ``hdf5`` / ...),
        compiled before the project TUs.
    :param out_dir: directory for the emitted ``.hlfir`` / ``.mod``
        files; a temporary one is used and removed when omitted.
    :param pipeline: MLIR pass pipeline; defaults to ``DEFAULT_PIPELINE``.
    :param flang: flang binary to drive (default ``flang-new-21``).
    :returns: a built, validated SDFG.
    """
    from dace_fortran.emit_hlfir import emit, resolve_entry, _parse_compile_commands

    # Accept a plain Fortran name (``solve_nh`` / ``mod::proc``) and resolve
    # it to the mangled symbol against the project's own sources, so callers
    # need not hand-write ``_QMmo_solve_nonhydroPsolve_nh``.
    sources = [s for s, _, _ in _parse_compile_commands(Path(compile_commands))]
    entry_sym = resolve_entry(entry, sources)

    def _do(d: Path) -> SDFG:
        # Pass ``entry_sym`` so the emitter restricts a whole-project
        # compile_commands.json to the entry's USE-closure (a codebase
        # like ICON lists ~900 TUs; we only need the entry's plus what
        # it transitively USEs).
        emit(compile_commands=Path(compile_commands),
             stubs=[Path(s) for s in stubs], out_dir=d, entry=entry_sym, flang=flang)
        return build_sdfg_from_hlfir(d, entry=entry_sym, pipeline=pipeline)

    if out_dir is not None:
        return _do(Path(out_dir))
    with tempfile.TemporaryDirectory(prefix="hlfir_project_") as td:
        return _do(Path(td))


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
