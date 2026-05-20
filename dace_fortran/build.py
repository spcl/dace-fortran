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
from typing import List, Optional, Sequence, Union

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


def _collect_include_dirs(roots: Sequence[Path]) -> List[Path]:
    """Find every ``include`` directory under ``roots`` plus the roots
    themselves -- both forms appear in real codebases (ICON keeps its
    ``omp_definitions.inc`` / ``icon_definitions.inc`` under
    ``src/include/``; the carved-out kernels stage their ``*_t0.inc``
    next to the kernel ``.f90``).  Used to derive flang's ``-I`` flags
    without forcing callers to enumerate every subdirectory.
    """
    out: list = []
    seen: set = set()

    def _add(p: Path):
        p = p.resolve()
        if p not in seen and p.is_dir():
            seen.add(p)
            out.append(p)

    for r in roots:
        p = Path(r)
        _add(p)
        if p.is_dir():
            for sub in p.rglob("include"):
                _add(sub)
    return out


#: ``MODULE <name>`` opener -- one definition per file in the common
#: case (one-module-per-file).  Used to build the ``module -> path``
#: index for the ``.mod`` compile path.
_MODULE_DEF_RE = re.compile(r"^\s*MODULE\s+([A-Za-z_]\w*)\s*$",
                            re.IGNORECASE | re.MULTILINE)

#: ``USE <name>`` -- captured to walk the dependency graph from the
#: entry source through the search roots.  The ``intrinsic`` prefix
#: form is accepted; intrinsic / compiler-supplied modules
#: (``iso_c_binding``, ``mpi``, ...) are filtered out separately.
_USE_DEP_RE = re.compile(r"^\s*USE[\s,]*(?:INTRINSIC\s*::\s*)?\s*([A-Za-z_]\w*)",
                         re.IGNORECASE | re.MULTILINE)


def _index_module_files(roots: Sequence[Path]) -> dict:
    """Build a ``module-name -> file path`` map for every ``.f90`` /
    ``.F90`` source under ``roots`` (skipping ``build/`` /
    ``CMakeFiles/`` artifact directories that mirror the upstream
    sources but are stale at parse time).  First-seen wins, so caller
    can order ``roots`` to prefer one tree over another."""
    out: dict = {}
    for root in roots:
        root = Path(root)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(list(root.rglob("*.f90")) + list(root.rglob("*.F90")))
        else:
            continue
        for f in files:
            sp = str(f)
            if "/build/" in sp or "/CMakeFiles/" in sp:
                continue
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            for m in _MODULE_DEF_RE.finditer(txt):
                out.setdefault(m.group(1).lower(), f)
    return out


def _topo_dep_order(source: str, mod_index: dict) -> List[Path]:
    """Return the ``.mod`` compile order for ``source`` 's USE closure,
    deps-first (dependencies before their dependents), with file paths
    de-duplicated -- when one ``.f90`` file defines several modules
    (a common pattern in ICON externals: ``libmtime.f90`` defines
    ``mtime_utilities`` + ``mtime_eventgroups``), compiling that file
    once produces all the ``.mod`` files at once and a re-compile
    would clobber / fail on the already-emitted ones.

    DFS over the ``USE`` graph starting from ``source`` 's directly-used
    modules; nodes outside ``mod_index`` (intrinsics, externals we have
    no source for) are skipped silently -- flang will diagnose them
    when it can't resolve a USE in the entry compile.  A cycle guard
    marks a node "visiting" when entered and "done" after; revisits
    while "visiting" are dropped (a USE cycle is rare in well-formed
    Fortran and not worth a hard failure here).
    """
    order: list = []
    state: dict = {}  # name -> "visiting" / "done"
    seen_paths: set = set()  # resolved file paths already in ``order``

    def _visit(name: str):
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            return  # cycle -- skip
        path = mod_index.get(name)
        if path is None:
            return  # external / intrinsic
        state[name] = "visiting"
        try:
            txt = path.read_text(errors="ignore")
        except OSError:
            state[name] = "done"
            return
        for m in _USE_DEP_RE.finditer(txt):
            _visit(m.group(1).lower())
        state[name] = "done"
        rp = path.resolve()
        if rp not in seen_paths:
            seen_paths.add(rp)
            order.append(path)

    for m in _USE_DEP_RE.finditer(source):
        _visit(m.group(1).lower())
    return order


def _compile_dep_mods(source: str,
                      mod_dir: Path,
                      include_flags: List[str],
                      search_dirs: Sequence[Path]) -> int:
    """Compile ``source`` 's USE-closure to ``.mod`` files under
    ``mod_dir`` in dep order, one at a time -- the same model flang
    (and clang for C++) uses for normal multi-file builds.

    flang does not ship a multi-file driver or a dep-extractor like
    ``clang-scan-deps``; the standard practice in the Fortran world
    (``makedepf90`` / ``fortdepend`` / ``fpm``) is to regex-scan
    ``USE`` statements, topo-sort, and drive ``flang -fsyntax-only``
    one-file-at-a-time -- which is what this does.

    Each invocation is independent (``-fsyntax-only -J<mod_dir>``):
    flang reads the already-built ``.mod`` files for the module's own
    deps and writes a fresh ``.mod`` for it.  ``-fhermetic-module-files``
    asks flang to emit each ``.mod`` self-contained (the transitive
    USE info gets embedded), so a later consumer that misses a
    transitive ``.mod`` still resolves the symbols it needs --
    important on a real upstream tree where some leaf modules
    legitimately fail to build in our scope (missing externals like
    ``netcdf`` / ``yaxt`` / ``hdf5``).

    Modules whose source isn't in ``search_dirs`` (intrinsics,
    externals) are silently skipped; modules that fail to compile in
    isolation (unresolvable transitive USEs, missing platform
    includes) are skipped too so a single broken leaf doesn't tank
    the whole closure -- the entry compile will diagnose any
    genuinely-needed symbol that didn't resolve.

    :returns: count of modules successfully compiled.
    """
    mod_dir.mkdir(parents=True, exist_ok=True)
    mod_index = _index_module_files(search_dirs)
    paths = _topo_dep_order(source, mod_index)
    compiled = 0
    for path in paths:
        flags = ["-fc1", "-cpp", "-U_OPENMP", "-U_OPENACC", "-fsyntax-only",
                 "-fhermetic-module-files",
                 *include_flags, "-J", str(mod_dir), "-I", str(mod_dir),
                 str(path)]
        r = subprocess.run([_find_flang(), *flags], capture_output=True)
        if r.returncode == 0:
            compiled += 1
    return compiled


def _emit_hlfir(source: str,
                out_dir: Path,
                name: str,
                *,
                merge: bool,
                preprocess: bool,
                search_dirs: Sequence[Path] = ()) -> Path:
    """Write ``source`` to ``<out_dir>/<name>.F90``, lower it to HLFIR,
    and return the ``.hlfir`` path.

    Two compile models, selected by whether ``search_dirs`` is empty:

    * **empty** (the default, all existing inline / staged tests):
      run the text-level ``merge_used_modules`` pass so any ``.f90``
      files in ``out_dir`` are inlined into ``source`` as one TU,
      then flang emits HLFIR for the whole TU.  This is the legacy
      behaviour and preserves callee inlining the multi-file tests
      depend on.
    * **non-empty** (the ICON / upstream-tree path): follow flang's
      native model -- walk the entry source's USE closure, compile
      each dep to a ``.mod`` file in dep order
      (:func:`_compile_dep_mods`), then compile the entry alone
      against ``-I<mod_dir>``.  No textual merge, so the merge's
      preamble / between-module artefacts (which were tripping
      flang-21 on the 91 k-line ICON TU) are out of the picture --
      this is the model flang and clang/gcc normally use.  The
      emitted HLFIR is the entry's only; callees referenced through
      ``USE`` stay external symbols (handled downstream by
      ``emit_call`` / ``ExternalCall``).

    flang runs with ``-cpp -U_OPENMP -U_OPENACC`` so legacy codebases
    that ship ``#ifdef`` blocks or ``#include`` cpp directives (ICON,
    ECRAD, ...) are consumable; forcing the OMP/ACC macros undefined
    matches the convention :func:`strip_openmp_directives` already
    applies at the Fortran-text level so the two layers agree.  Every
    search root (plus any ``include`` directory under it) becomes a
    ``-I`` so cpp resolves real ``#include`` directives without callers
    enumerating subdirectories.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use ``.F90`` so flang's default file-by-extension detection
    # routes through its built-in preprocessor; combined with ``-cpp``
    # this stays consistent if a future flang changes its default.
    src = out_dir / f"{name}.F90"
    all_dirs = [out_dir, *[Path(d) for d in search_dirs]]
    if search_dirs:
        # Native model: write the entry verbatim, compile deps to .mod.
        # Drop OMP/ACC sentinels + the OMP cpp include from the entry
        # source so flang's cpp sees clean text (``strip_openmp_directives``
        # is the same pass the merge path runs as part of
        # ``preprocess_fortran_source``).
        from dace_fortran.preprocess import strip_openmp_directives
        src.write_text(strip_openmp_directives(source))
    else:
        src.write_text(preprocess_fortran_source(source, search_dirs=all_dirs,
                                                 merge=merge, if_intvar=preprocess))
    include_flags: list = []
    for d in _collect_include_dirs(all_dirs):
        include_flags += ["-I", str(d)]
    if search_dirs:
        mod_dir = out_dir / f"{name}_mods"
        _compile_dep_mods(src.read_text(), mod_dir, include_flags, list(search_dirs))
        include_flags += ["-I", str(mod_dir)]
    hlfir = out_dir / f"{name}.hlfir"
    subprocess.check_call([_find_flang(),
                           "-fc1", "-cpp", "-U_OPENMP", "-U_OPENACC",
                           *include_flags,
                           "-emit-hlfir", str(src), "-o", str(hlfir)])
    return hlfir


def make_builder(source: str,
                 *,
                 entry: Optional[str] = None,
                 name: str = "sdfg",
                 pipeline: Optional[str] = None,
                 out_dir: Optional[Union[str, Path]] = None,
                 preprocess: bool = False,
                 search_dirs: Sequence[Union[str, Path]] = ()) -> SDFGBuilder:
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

    ``search_dirs`` are extra roots scanned recursively for module
    definitions; the scratch / ``out_dir`` is always searched too.
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
    sdirs = [Path(d) for d in search_dirs]
    if out_dir is not None:
        hlfir = _emit_hlfir(source, Path(out_dir), name, merge=True,
                            preprocess=preprocess, search_dirs=sdirs)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=entry)
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        hlfir = _emit_hlfir(source, Path(td), name, merge=True,
                            preprocess=preprocess, search_dirs=sdirs)
        return SDFGBuilder(str(hlfir), pipeline=pipeline, entry=entry)


def build_sdfg(source: str,
               *,
               entry: Optional[str] = None,
               name: str = "sdfg",
               pipeline: Optional[str] = None,
               out_dir: Optional[Union[str, Path]] = None,
               preprocess: bool = False,
               search_dirs: Sequence[Union[str, Path]] = ()) -> SDFG:
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
    :param search_dirs: extra roots scanned recursively for module
        definitions (point at an unmodified upstream tree like an ICON
        checkout to resolve ``USE`` chains without staging copies).
        ``out_dir`` is always part of the search path; ``search_dirs``
        adds to it, it does not replace it.
    :returns: a built, validated SDFG.
    :raises ValueError: if ``entry`` is ``None`` and the source has no
        procedure or is ambiguous (more than one).
    """
    return make_builder(source, entry=entry, name=name, pipeline=pipeline,
                        out_dir=out_dir, preprocess=preprocess,
                        search_dirs=search_dirs).build()


def build_sdfg_from_files(files: Sequence[Union[str, Path]],
                          *,
                          entry: Optional[str] = None,
                          name: str = "sdfg",
                          pipeline: Optional[str] = None,
                          out_dir: Optional[Union[str, Path]] = None,
                          preprocess: bool = False,
                          search_dirs: Sequence[Union[str, Path]] = ()) -> SDFG:
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
    :param search_dirs: extra roots scanned recursively for module
        definitions, on top of the staged scratch dir (same semantics
        as :func:`build_sdfg` 's ``search_dirs``).
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
                           pipeline=pipeline, out_dir=d, preprocess=preprocess,
                           search_dirs=search_dirs)

    if out_dir is not None:
        return _do(Path(out_dir))
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        return _do(Path(td))
