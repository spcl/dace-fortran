"""The tier-3 HLFIR emitter: drive flang per translation unit from a
build's ``compile_commands.json`` (or an explicit ``--source`` list)
so the user need not wire flang into their build by hand.  CLI usage
and flags are in :func:`main`'s argparse help; the README's
*Building an SDFG from a real project* section is the walkthrough.

Two file-list sources, mutually exclusive:

* ``compile_commands`` -- the preferred path: build order and per-TU
  ``-I`` / ``-D`` flags come straight from the artefact, so the
  emitter never guesses.  Produced by ``cmake
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`` or, for autotools / plain make,
  ``bear -- make``.
* ``sources`` -- a fallback for builds with no such artefact; order is
  a ``USE``-graph topo-sort (regex scan, the shape ``makedepf90`` /
  ``fortdepend`` / ``fpm`` use).

``stubs`` are flang-buildable stand-ins for modules flang ships no
``.mod`` for (``mpi`` / ``netcdf`` / ``hdf5`` / ...); compiled first
so the project's ``USE`` lines resolve.  The emitted directory feeds
:func:`dace_fortran.build_sdfg_from_hlfir` /
:func:`dace_fortran.build_sdfg_from_project`.
"""
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

#: ``MODULE <name>`` opener at the top of a Fortran source.  Used for
#: the (fallback) ``--source`` topo-sort when no ``compile_commands.json``
#: artefact is supplied.
_MODULE_DEF_RE = re.compile(r"^\s*MODULE\s+([A-Za-z_]\w*)\s*$",
                            re.IGNORECASE | re.MULTILINE)
#: ``USE <name>`` -- captured for the same fallback topo-sort.
_USE_DEP_RE = re.compile(r"^\s*USE[\s,]*(?:INTRINSIC\s*::\s*)?\s*([A-Za-z_]\w*)",
                         re.IGNORECASE | re.MULTILINE)


def _topo_order(sources: Sequence[Path]) -> List[Path]:
    """USE-graph topo-sort over an explicit file list (fallback path
    for projects without ``compile_commands.json``).  Files defining
    no module sort last.  Multi-module files are emitted once.  A
    ``USE`` cycle (rare in well-formed Fortran) is broken silently --
    the ``"visiting"`` re-entry just returns, leaving the back-edge
    file wherever the recursion first reached it.
    """
    file_modules: dict = {}
    module_owner: dict = {}
    file_uses: dict = {}
    for src in sources:
        txt = src.read_text(errors="ignore")
        file_modules[src] = [m.group(1).lower() for m in _MODULE_DEF_RE.finditer(txt)]
        file_uses[src] = [m.group(1).lower() for m in _USE_DEP_RE.finditer(txt)]
        for nm in file_modules[src]:
            module_owner[nm] = src

    order: list = []
    state: dict = {}

    def _visit(src: Path):
        if state.get(src) == "done":
            return
        if state.get(src) == "visiting":
            return
        state[src] = "visiting"
        for nm in file_uses[src]:
            owner = module_owner.get(nm)
            if owner is not None and owner is not src:
                _visit(owner)
        state[src] = "done"
        order.append(src)

    for src in sources:
        _visit(src)
    return order


def _parse_compile_commands(cc_path: Path):
    """Return ``[(source_path, include_dirs, cpp_defines), ...]``
    in the order cmake / ninja recorded -- they topo-sort Fortran
    via the same scanner the regular build uses, so reusing that
    order is robust and matches the user's expectations exactly.

    Only Fortran entries (``.f90`` / ``.F90``) are kept; C / C++
    entries that share the same ``compile_commands.json`` (mixed
    project) are filtered.  ``-I`` / ``-D`` flags are extracted from
    the recorded command so the flang invocation sees the same cpp
    surface gfortran did.
    """
    with open(cc_path) as f:
        entries = json.load(f)
    out: list = []
    for e in entries:
        src = Path(e["file"])
        # Fortran TUs only -- a mixed project's C/C++ entries (yaxt,
        # cdi, ...) share the same compile_commands.json and must be
        # skipped.  Suffix already lower-cased, so ``.F90`` is covered.
        if src.suffix.lower() not in (".f90", ".f", ".for"):
            continue
        # Recorded command may be a string ("cc -I/x foo.c") or a list.
        cmd = e["command"] if "command" in e else " ".join(e.get("arguments", []))
        tokens = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        includes: list = []
        defines: list = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "-I":
                includes.append(tokens[i + 1]); i += 2; continue
            if t.startswith("-I"):
                includes.append(t[2:]); i += 1; continue
            if t == "-D":
                defines.append(tokens[i + 1]); i += 2; continue
            if t.startswith("-D"):
                defines.append(t[2:]); i += 1; continue
            i += 1
        out.append((src, includes, defines))
    return out


def _entry_module(entry: str) -> Optional[str]:
    """Module name from a mangled entry symbol, or ``None`` for a free
    subroutine.  ``_QM<module>P<proc>`` (module subroutine) / ``...F<proc>``
    (function) -> ``<module>``; ``_QP<proc>`` has no module."""
    m = re.match(r"_QM([a-z0-9_]+?)[PF][a-z0-9_]+$", entry, re.IGNORECASE)
    return m.group(1).lower() if m else None


def demangle_entry(sym: str) -> str:
    """Flang mangled symbol -> the Fortran name.  ``_QM<mod>P<proc>`` /
    ``_QM<mod>F<proc>`` -> ``<mod>::<proc>``; ``_QP<proc>`` -> ``<proc>``."""
    m = re.match(r"_QM([a-z0-9_]+?)[PF]([a-z0-9_]+)$", sym, re.IGNORECASE)
    if m:
        return f"{m.group(1)}::{m.group(2)}"
    m = re.match(r"_Q[PF]([a-z0-9_]+)$", sym, re.IGNORECASE)
    return m.group(1) if m else sym


_MOD_OPEN_RE = re.compile(r"^\s*module\s+([a-z_]\w*)\s*$", re.IGNORECASE)
_MOD_END_RE = re.compile(r"^\s*end\s+module\b", re.IGNORECASE)
_INTERFACE_RE = re.compile(r"^\s*(abstract\s+)?interface\b", re.IGNORECASE)
_END_INTERFACE_RE = re.compile(r"^\s*end\s+interface\b", re.IGNORECASE)
_SUBR_DEF_RE = re.compile(
    r"^\s*(?:(?:recursive|pure|impure|elemental|module)\s+)*subroutine\s+([a-z_]\w*)",
    re.IGNORECASE)


def _scan_subroutine_defs(text: str):
    """``[(subroutine_lower, enclosing_module_lower_or_None), ...]`` for the
    subroutine *definitions* in ``text``.  Tracks ``MODULE`` / ``END MODULE``
    for the qualifier and skips ``INTERFACE`` blocks (those are declarations,
    not definitions)."""
    defs = []
    mod = None
    in_iface = False
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        if _INTERFACE_RE.match(line):
            in_iface = True
            continue
        if _END_INTERFACE_RE.match(line):
            in_iface = False
            continue
        if in_iface:
            continue
        mo = _MOD_OPEN_RE.match(line)
        if mo and not re.match(r"^\s*module\s+procedure\b", line, re.IGNORECASE):
            mod = mo.group(1).lower()
            continue
        if _MOD_END_RE.match(line):
            mod = None
            continue
        sm = _SUBR_DEF_RE.match(line)
        if sm:
            defs.append((sm.group(1).lower(), mod))
    return defs


def resolve_entry(name: str, sources) -> str:
    """Resolve a Fortran procedure name to its mangled flang symbol by
    scanning ``sources`` for the subroutine definition.  Pass-through if
    ``name`` is already mangled (``_Q...``).  Accepts ``module::proc`` to
    disambiguate; a bare ``proc`` resolves uniquely or raises.

    Currently resolves SUBROUTINES (the usual SDFG entry, incl. the ICON
    dycore ``solve_nh``); pass a function's mangled symbol directly.

    :raises ValueError: ``name`` is not found, or is ambiguous across
        modules (the message lists the candidates so the caller can
        qualify it ``module::proc``).
    """
    if name.startswith("_Q"):
        return name
    want_mod, _, want_proc = name.lower().rpartition("::")
    want_mod = want_mod or None

    matches = set()
    for src in sources:
        # Each source is a file path (read it) or already-inline source text
        # (tier-1 passes the kernel string straight through).
        try:
            text = Path(src).read_text(errors="ignore") if Path(src).is_file() else str(src)
        except (OSError, ValueError):
            text = str(src)
        for proc, mod in _scan_subroutine_defs(text):
            if proc == want_proc and (want_mod is None or mod == want_mod):
                matches.add((proc, mod))
    if not matches:
        raise ValueError(f"resolve_entry: no subroutine {name!r} found in the sources")
    if len(matches) > 1:
        cands = ", ".join(f"{m or '<free>'}::{p}" for p, m in sorted(matches))
        raise ValueError(f"resolve_entry: {name!r} is ambiguous ({cands}); "
                         f"qualify it as module::proc or pass the mangled symbol")
    proc, mod = matches.pop()
    return f"_QM{mod}P{proc}" if mod else f"_QP{proc}"


def _select_use_closure(parsed, root_module: str):
    """Filter parsed ``(src, includes, defines)`` entries down to the TU
    that defines ``root_module`` plus the transitive ``USE``-closure it
    needs, preserving the original (build) order.

    A whole project's ``compile_commands.json`` lists every TU (ICON's is
    ~900); emitting all of them to lower one entry is wasteful and drags
    in modules flang need never see.  The bridge inlines only the entry's
    call tree, so it only needs the entry's TU + the modules it ``USE``s
    (transitively) for the ``.mod`` chain.  Falls back to the full list
    if the root module's defining TU isn't found (safe over-emit)."""
    module_owner: dict = {}
    file_uses: dict = {}
    for src, _, _ in parsed:
        txt = src.read_text(errors="ignore")
        for m in _MODULE_DEF_RE.finditer(txt):
            module_owner[m.group(1).lower()] = src
        file_uses[src] = [u.group(1).lower() for u in _USE_DEP_RE.finditer(txt)]

    root = module_owner.get(root_module)
    if root is None:
        return parsed

    needed: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        if s in needed:
            continue
        needed.add(s)
        for use in file_uses.get(s, []):
            owner = module_owner.get(use)
            if owner is not None:
                stack.append(owner)
    return [t for t in parsed if t[0] in needed]


def _flang_emit(flang: str,
                src: Path,
                out_dir: Path,
                includes: Sequence[str],
                defines: Sequence[str]):
    """Run one ``flang -fc1 -emit-hlfir`` invocation.  ``cwd`` and
    ``-J/-I`` all point at ``out_dir`` so flang only ever sees the
    ``.mod`` files it wrote itself -- the gfortran-format binary
    ``.mod`` files the user's regular build emits next door cannot
    collide via flang's implicit ``.`` lookup."""
    cmd = [flang, "-fc1", "-cpp", "-U_OPENMP", "-U_OPENACC",
           "-fhermetic-module-files",
           "-J", str(out_dir), "-I", str(out_dir)]
    for d in includes:
        cmd += ["-I", str(d)]
    for d in defines:
        cmd += ["-D", d]
    cmd += ["-emit-hlfir", str(src), "-o", str(out_dir / f"{src.stem}.hlfir")]
    subprocess.check_call(cmd, cwd=str(out_dir))


def emit(*,
         compile_commands: Optional[Path] = None,
         sources: Sequence[Path] = (),
         stubs: Sequence[Path] = (),
         out_dir: Path,
         extra_includes: Sequence[Path] = (),
         extra_defines: Sequence[str] = (),
         entry: Optional[str] = None,
         flang: str = "flang-new-21") -> List[Path]:
    """Emit ``.hlfir`` files under ``out_dir``.  Exactly one of
    ``compile_commands`` or ``sources`` must drive the file list:

    * ``compile_commands`` -- a path to a cmake/ninja- or
      ``bear``/autotools-emitted ``compile_commands.json``; build order
      + per-TU ``-I`` / ``-D`` flags come from there (recommended).
    * ``sources`` -- explicit ``.f90`` list, topo-sorted by ``USE``;
      ``extra_includes`` becomes ``-I`` for every invocation.

    ``extra_defines`` are extra ``-D`` cpp macros (``NAME`` or
    ``NAME=val``) passed to flang on top of whatever the build flags
    already carry -- the only way to set the cpp configuration when
    there is no build system to read it from (the ``sources`` path), and
    an override/augment for the ``compile_commands`` path.

    ``stubs`` is a list of flang-buildable stub sources for external
    modules flang has no shipped ``.mod`` for (``mpi`` / ``netcdf``
    / ``hdf5`` / ...); they are emitted first (in the order given)
    so the project sources' ``USE`` lines resolve.

    ``entry`` (a mangled symbol like ``_QMmodPproc``) restricts the
    ``compile_commands`` run to the entry's TU + its transitive
    ``USE``-closure -- the only TUs the bridge needs to lower that entry.
    Without it every TU in the database is emitted (fine for small
    projects, wasteful for a whole-codebase ``compile_commands.json``).

    :returns: the emitted ``.hlfir`` paths in build order.
    """
    # XOR: exactly one of the two file-list sources must drive the run.
    if (compile_commands is None) == (not sources):
        raise ValueError("emit() takes exactly one of compile_commands= or sources=")
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_defs = list(extra_defines)
    emitted: list = []
    # 1. stubs first (USE-order across stubs; usually a flat list).
    for src in _topo_order([Path(s) for s in stubs]):
        _flang_emit(flang, src, out_dir, (), extra_defs)
        emitted.append(out_dir / f"{src.stem}.hlfir")
    # 2. project sources.
    if compile_commands is not None:
        parsed = _parse_compile_commands(Path(compile_commands))
        if entry is not None:
            # Accept a plain Fortran name (resolved against the sources) or
            # a mangled symbol (passthrough), then keep only its USE-closure.
            entry = resolve_entry(entry, [t[0] for t in parsed])
            mod = _entry_module(entry)
            if mod is not None:
                parsed = _select_use_closure(parsed, mod)
        for src, incs, defs in parsed:
            _flang_emit(flang, src, out_dir, incs, list(defs) + extra_defs)
            emitted.append(out_dir / f"{src.stem}.hlfir")
    else:
        incs = [str(d) for d in extra_includes]
        for src in _topo_order([Path(s) for s in sources]):
            _flang_emit(flang, src, out_dir, incs, extra_defs)
            emitted.append(out_dir / f"{src.stem}.hlfir")
    return emitted


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m dace_fortran.emit_hlfir",
        description="Emit HLFIR for a Fortran project so "
                    "dace_fortran.build_sdfg_from_hlfir can consume it.")
    p.add_argument("compile_commands", nargs="?", type=Path,
                   help="path to a cmake / ninja compile_commands.json "
                        "(use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON when "
                        "configuring) -- preferred path; build order + "
                        "-I/-D flags inferred from this artefact.")
    p.add_argument("--source", action="append", default=[], type=Path,
                   dest="sources",
                   help="fallback: explicit .f90 file (repeat) when no "
                        "compile_commands.json is available; ordering "
                        "derived by USE-graph topo-sort.")
    p.add_argument("--stub", action="append", default=[], type=Path,
                   dest="stubs",
                   help="flang-buildable stub source for an external "
                        "module flang has no shipped .mod for "
                        "(mpi / netcdf / hdf5 / ...); compiled first.")
    p.add_argument("--include", action="append", default=[], type=Path,
                   dest="extra_includes",
                   help="extra -I path (--source mode only; "
                        "compile_commands inherits its own -I list).")
    p.add_argument("--define", "-D", action="append", default=[],
                   dest="extra_defines", metavar="NAME[=val]",
                   help="extra -D cpp macro for flang's preprocessor "
                        "(repeat); the only way to set cpp config in "
                        "--source mode, and an augment in compile_commands "
                        "mode.")
    p.add_argument("--entry", default=None,
                   help="mangled entry symbol (_QMmodPproc); restricts a "
                        "compile_commands run to that entry's USE-closure "
                        "instead of emitting every TU.")
    p.add_argument("--out", required=True, type=Path,
                   help="output directory; .hlfir + .mod files land here.")
    p.add_argument("--flang", default="flang-new-21",
                   help="flang binary to drive (default: flang-new-21).")
    args = p.parse_args(argv)
    if shutil.which(args.flang) is None:
        p.error(f"flang binary {args.flang!r} not on PATH")
    if (args.compile_commands is None) == (not args.sources):
        p.error("pass either compile_commands.json (positional) or one or "
                "more --source paths, not both / neither")
    out = emit(compile_commands=args.compile_commands,
               sources=args.sources,
               stubs=args.stubs,
               out_dir=args.out,
               extra_includes=args.extra_includes,
               extra_defines=args.extra_defines,
               entry=args.entry,
               flang=args.flang)
    print(f"emitted {len(out)} .hlfir under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
