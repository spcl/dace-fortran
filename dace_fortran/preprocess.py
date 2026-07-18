"""Fortran-source-level pre-processor.

Text-only rewrites that must run before ``flang-new -fc1 -emit-hlfir`` sees
the source, since they change what flang accepts or which arithmetic a
backend picks:

* ``rewrite_integer_powers`` -- ``x**2.0`` -> ``(x*x)``.  Unconditional in
  ``compile_to_hlfir``: algebraically exact, avoids a backend-dependent
  ``pow()`` vs ``x*x`` rounding diff against the gfortran reference.

* ``promote_real_literals_to_double`` -- ``2.0`` -> ``2.0D0``.  Standalone
  utility, not wired into the build path.

* ``strip_openmp_directives`` -- drops OpenMP/OpenACC sentinels and the ICON
  ``omp_definitions.inc`` include.  Unconditional in
  ``preprocess_fortran_source``: semantic no-op without ``-fopenmp``, but the
  cpp ``#include`` would otherwise crash flang (bridge doesn't run cpp).

* ``preprocess_fortran`` -- ``IF (intvar)`` -> ``IF (intvar /= 0)``.
  flang-new-21 rejects bare INTEGER as an IF condition; legacy ECRAD /
  CloudSC / ICON code ships this shape.  Opt-in
  (``compile_to_hlfir(..., preprocess=True)``).

SED-style text rewrites, not a Fortran parser -- deliberately narrow
(single-identifier IF guards; powers with a primary base only) and brittle
by construction.  Comment/string safety shared via ``_scan_line``.
"""

import re
from pathlib import Path
from typing import Iterable, Optional

# Whole-number REAL exponent only (``**2.0``, ``**2.0_JPRB``) -- backends are
# free to round ``pow(x, 2.0)`` differently from gfortran's ``x*x``.
# Bare-integer ``x**2`` (flang lowers it bit-identically to gfortran already)
# and fractional powers (``**0.5``) are left alone.  Requires a digit before
# the dot so a bare integer never matches.
_REAL_EXP = r"\d+\.\d*(?:[eEdD][+-]?\d+)?(?:_[A-Za-z]\w*|_\d+)?"
_INT_POW_RE = re.compile(r"\*\*\s*(?:\(\s*(" + _REAL_EXP + r")\s*\)|(" + _REAL_EXP + r")(?![\w.]))")

# Identifier + ``(`` -- a call/array ref; a power base containing one must
# not be duplicated.
_CALL_IN_BASE = re.compile(r"[A-Za-z_]\w*\s*\(")

# REAL literal: needs a frac point or exponent so a bare integer never
# matches; captures are mantissa/exponent/kind.  Lookbehind/ahead keep off
# identifiers (``R2ES``) and kind selectors.
_REAL_LIT_RE = re.compile(r"(?<![\w.])"
                          r"(\d+\.\d*|\.\d+|\d+)"  # mantissa
                          r"([eEdD][+-]?\d+)?"  # optional exponent
                          r"(_[A-Za-z]\w*|_\d+)?"  # optional kind suffix
                          r"(?![\w.])")
# Kind suffixes already double precision -- left alone.
_DOUBLE_KINDS = {"jprb", "jprd", "dp", "8", "16", "r8", "qp"}

_INTEGER_DECL_RE = re.compile(
    r"\bINTEGER\b(?:\s*\([^)]*\))?(?:\s*,\s*[A-Z_]+(?:\s*\([^)]*\))?)*\s*::\s*([^\n!]+)",
    re.IGNORECASE,
)
_BARE_IF_RE = re.compile(r"\b(IF\s*\(\s*)([A-Za-z_]\w*)(\s*\))", re.IGNORECASE)


def _scan_line(body: str):
    """Comment start + character-string spans of one line; shared so a ``!``
    or ``**`` inside a literal is never treated as code.  Returns
    ``(comment_index, [(start, end), ...])``; doubled quotes (``''``) stay
    inside one span.
    """
    spans, i, n = [], 0, len(body)
    while i < n:
        c = body[i]
        if c in "'\"":
            j = i + 1
            while j < n:
                if body[j] == c:
                    if j + 1 < n and body[j + 1] == c:
                        j += 2  # doubled quote -> escaped, stay in string
                        continue
                    break
                j += 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
        elif c == "!":
            return i, spans
        else:
            i += 1
    return n, spans


def _collect_integer_scalar_names(source: str) -> set[str]:
    """INTEGER scalar identifiers declared in ``source``, lowercased.  Array
    declarations are skipped -- can't be the bare operand of an ``IF``.
    """
    names: set[str] = set()
    for m in _INTEGER_DECL_RE.finditer(source):
        decl = m.group(1).split('!', 1)[0]
        for tok in decl.split(','):
            head = tok.strip().split('=', 1)[0].strip()
            # Skip array forms (``name(...)``) -- can't be the bare argument of IF.
            if '(' in head:
                continue
            name = head.split()[0] if head else ''
            if name and name.replace('_', '').isalnum() and not name[0].isdigit():
                names.add(name.lower())
    return names


def _extract_power_base(code: str, star: int):
    """Base (left primary) of a ``**`` operator: scans leftward over a
    parenthesised group, identifier, array/function reference
    (``name(...)``), or ``%`` component chain (``a%b(i)%c``).  Returns
    ``(begin, end)`` slice, or ``None`` when no base is found or parens are
    unbalanced.
    """
    i = star
    while i > 0 and code[i - 1] in " \t":
        i -= 1
    end = i
    while True:
        if i > 0 and code[i - 1] == ")":
            depth, k = 0, i
            while k > 0:
                k -= 1
                if code[k] == ")":
                    depth += 1
                elif code[k] == "(":
                    depth -= 1
                    if depth == 0:
                        break
            if depth != 0:
                return None  # unbalanced -- refuse to rewrite
            i = k  # now at the matching '('
            while i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
                i -= 1  # consume the array/function name, if any
        elif i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
            while i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
                i -= 1
        else:
            break
        if i > 0 and code[i - 1] == "%":
            i -= 1  # designator chain -- keep walking the components
            continue
        break
    return None if i == end else (i, end)


def _real_exp_int_value(tok: str):
    """Integer value of a REAL-literal exponent token (``2.0`` -> 2), or
    ``None`` when it isn't a whole number >= 1 (``2.5``, ``0.0``).
    """
    mant = re.sub(r"(_[A-Za-z]\w*|_\d+)$", "", tok)
    try:
        val = float(mant.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None
    return int(val) if val >= 1 and val == int(val) else None


def rewrite_integer_powers(source: str) -> str:
    """Expand integer-valued REAL-literal powers to repeated multiply:
    ``base**2.0`` -> ``(base*base)``.

    Adds a single outer paren pair -- ``_extract_power_base`` always returns
    a Fortran primary, so this preserves precedence in any context
    (``2.0*x**2.0`` -> ``2.0*(x*x)``, ``-x**2.0`` -> ``-(x*x)``).
    Bare-integer ``x**2`` (flang already lowers it bit-identically to
    gfortran) and fractional powers (``**0.5``) are never touched, nor is a
    base containing a function/array ref (duplicating it could call twice).
    Overlapping (stacked ``a**2.0**2.0``) matches are skipped.  Idempotent.
    """
    out = []
    for line in source.splitlines(keepends=True):
        nl = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(nl)]
        cut, strings = _scan_line(body)  # string-aware, shared
        code, tail = body[:cut], body[cut:]
        edits = []
        for m in _INT_POW_RE.finditer(code):
            if any(s <= m.start() < e for s, e in strings):
                continue  # ``**`` inside a character literal
            n = _real_exp_int_value(m.group(1) or m.group(2))
            if n is None:
                continue
            span = _extract_power_base(code, m.start())
            if span is None:
                continue
            begin, base_end = span
            if edits and begin < edits[-1][1]:
                continue  # overlaps a stacked power -- leave both
            base = code[begin:base_end]
            if _CALL_IN_BASE.search(base):
                # Base has a call/array ref (``f(x)``) -- duplicating could call
                # twice; the inliner also shares the callee's accumulator across
                # copies (observed: ``custom_sum(d)**2.0`` -> 2500 not 625).
                continue
            repl = "(" + "*".join(base for _ in range(n)) + ")"
            edits.append((begin, m.end(), repl))
        for begin, fin, repl in reversed(edits):  # right-to-left: stable idx
            code = code[:begin] + repl + code[fin:]
        out.append(code + tail + nl)
    return "".join(out)


def _promote_one(m: re.Match):
    """Rewrite one ``_REAL_LIT_RE`` match to double precision; returns the
    match unchanged when it's an integer or already double.
    """
    mant, expo, kind = m.group(1), m.group(2) or "", m.group(3) or ""
    if "." not in mant and not expo:
        return m.group(0)  # bare integer -- not a real literal
    if expo[:1] in ("d", "D"):
        return m.group(0)  # already double via D-exponent
    if kind and kind[1:].lower() in _DOUBLE_KINDS:
        return m.group(0)  # already double via kind suffix
    if expo:
        return f"{mant}D{expo[1:]}"  # E-exponent -> D-exponent
    return f"{mant}D0"  # bare/single -> append D0


def promote_real_literals_to_double(source: str) -> str:
    """Rewrite single/default REAL literals to double precision (``2.0`` ->
    ``2.0D0``, ``1.0_JPRM`` -> ``1.0D0``).  Already-double literals (``D``
    exponent / double kind suffix) and integers are left untouched.
    Comments and strings are never modified.  Idempotent.
    """
    out = []
    for line in source.splitlines(keepends=True):
        nl = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(nl)]
        cut, strings = _scan_line(body)
        code, tail = body[:cut], body[cut:]

        def _repl(m: re.Match) -> str:
            if any(s <= m.start() < e for s, e in strings):
                return m.group(0)  # inside a character string
            return _promote_one(m)

        out.append(_REAL_LIT_RE.sub(_repl, code) + tail + nl)
    return "".join(out)


#: OpenMP/OpenACC/CUDA-Fortran sentinel (``!$omp``/``!$acc``/``!$cuf``) at
#: line start; also matches ``!$OMP&`` continuations (same prefix).
_OMP_SENTINEL_RE = re.compile(r"^\s*!\s*\$\s*(?:omp|acc|cuf)\b", re.IGNORECASE)

#: ``!$ <stmt>`` OpenMP-conditional line -- no-op without ``-fopenmp``.
#: Requires a space after ``!$`` so ``!$OMP``/``!$ACC`` fall through to the
#: sentinel rule above.
_OMP_COND_LINE_RE = re.compile(r"^\s*!\s*\$[ \t]+\S")

#: Vendor directives (``!DIR$ IVDEP`` etc.) -- flang-new warns on every one
#: (ICON's dycore: hundreds of ``[-Wignored-directive]``); never honoured, so
#: dropped like the accelerator sentinels above.
_VENDOR_DIRECTIVE_RE = re.compile(r"^\s*!\s*DIR\$", re.IGNORECASE)

#: ICON's ``omp_definitions.inc`` / ``hamocc_omp_definitions.inc`` cpp
#: include -- bridge doesn't run cpp so the raw ``#include`` would crash
#: flang; safe to drop since its macros only expand to OpenMP sentinels
#: (stripped above too).
_OMP_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"][^>"]*omp_definitions[^>"]*\.inc[>"]\s*$', re.IGNORECASE)

#: Macros ``strip_openmp_directives`` treats as undefined (``#ifdef`` arm
#: dropped, ``#ifndef`` passes).  Other ``#ifdef`` macros (``__SWAPDIM``,
#: ``_CRAYFTN``) flow through untouched -- a separate preprocessing step.
_OMP_ACC_MACROS = frozenset({"_OPENMP", "_OPENACC"})

#: ``#if[n]def MACRO`` and the ``#if [!]defined(MACRO)`` aliases -- the
#: directive openers we elide when ``MACRO`` is in :data:`_OMP_ACC_MACROS`.
_CPP_IFDEF_RE = re.compile(r'^\s*#\s*(ifn?def)\s+(\w+)', re.IGNORECASE)
_CPP_IFDEFINED_RE = re.compile(r'^\s*#\s*if\s+(!)?\s*defined\s*\(\s*(\w+)\s*\)\s*$', re.IGNORECASE)
_CPP_IF_RE = re.compile(r'^\s*#\s*if\b', re.IGNORECASE)
_CPP_ELSE_RE = re.compile(r'^\s*#\s*else\b', re.IGNORECASE)
_CPP_ELIF_RE = re.compile(r'^\s*#\s*elif\b', re.IGNORECASE)
_CPP_ENDIF_RE = re.compile(r'^\s*#\s*endif\b', re.IGNORECASE)


def strip_openmp_directives(source: str) -> str:
    """Drop OpenMP/OpenACC/CUDA-Fortran sentinels, the ICON
    ``omp_definitions.inc`` include, and ``#ifdef _OPENMP``/``#ifdef
    _OPENACC`` blocks (keeping the ``#else`` arm when present) -- all
    already inert without ``-fopenmp``, but the raw cpp forms would crash
    flang outright.  Elision is scoped to ``_OPENMP``/``_OPENACC``; other
    ``#ifdef`` macros pass through untouched.  ``#if [!]defined(...)`` is
    recognised as a ``#ifdef``/``#ifndef`` alias.  Idempotent.
    """
    out = []
    # Stack of (is_omp_acc_block, dropping_now) per open cpp conditional --
    # non-OMP/ACC ``#if`` pushes (False, False) so unrelated blocks pass through.
    stack: list = []
    for line in source.splitlines(keepends=True):
        # (False, ...) top-of-stack: the branches below do nothing, so the
        # line just falls through to the emit at the end.
        m_ifdef = _CPP_IFDEF_RE.match(line)
        m_ifdef_paren = _CPP_IFDEFINED_RE.match(line) if not m_ifdef else None
        if m_ifdef:
            kind, macro = m_ifdef.group(1).lower(), m_ifdef.group(2)
            if macro in _OMP_ACC_MACROS:
                # undefined macro: ``#ifdef`` arm drops, ``#ifndef`` arm keeps.
                stack.append((True, kind == "ifdef"))
                continue
            stack.append((False, False))
        elif m_ifdef_paren:
            negate, macro = bool(m_ifdef_paren.group(1)), m_ifdef_paren.group(2)
            if macro in _OMP_ACC_MACROS:
                stack.append((True, not negate))  # defined()->drop, !defined()->keep
                continue
            stack.append((False, False))
        elif _CPP_IF_RE.match(line):
            stack.append((False, False))
        elif _CPP_ELSE_RE.match(line):
            if stack and stack[-1][0]:
                stack[-1] = (True, not stack[-1][1])  # flip to the other arm
                continue
        elif _CPP_ELIF_RE.match(line):
            if stack and stack[-1][0]:
                # KNOWN LIMITATION: ``#elif FOO`` after ``#ifdef _OPENMP`` is
                # dropped too (macro treated as purely undefined); rare in
                # practice since OMP/ACC blocks seldom carry a meaningful ``#elif``.
                stack[-1] = (True, True)
                continue
        elif _CPP_ENDIF_RE.match(line):
            # Unbalanced ``#endif`` (its ``#if`` predates ``source``) tolerated as a no-op.
            if stack and stack[-1][0]:
                stack.pop()
                continue
            if stack:
                stack.pop()
        if any(is_omp and drop for is_omp, drop in stack):
            continue
        if _OMP_SENTINEL_RE.match(line):
            continue
        if _OMP_COND_LINE_RE.match(line):
            continue
        if _OMP_INCLUDE_RE.match(line):
            continue
        if _VENDOR_DIRECTIVE_RE.match(line):
            continue
        out.append(line)
    return "".join(out)


# Default precision-kind aliases -- unresolved kind symbol -> IEEE byte
# width.  ``wp``/``rp``/``sp``/``dp``/``qp`` cover conventions seen in
# climate/NWP code (ECMWF/ICON/CLOUDSC).  ``kind_map=`` to
# ``normalize_kind_parameters`` extends/overrides this table; ``None`` for
# an alias leaves it untouched.
_DEFAULT_KIND_ALIASES = {
    "wp": 8,
    "rp": 8,
    "sp": 4,
    "dp": 8,
    "qp": 16,
}

# ``KIND = sym`` -- group 1 = prefix (kept verbatim), group 2 = symbol.
_KIND_EQ_RE = re.compile(r"\b(KIND\s*=\s*)([A-Za-z_]\w*)\b", re.IGNORECASE)

# ``REAL(sym)``/``INTEGER(sym)``/``COMPLEX(sym)``/``LOGICAL(sym)`` sole-arg
# type-spec form.  Groups: 1=keyword, 2=``(``, 3=symbol, 4=``)``.
_TYPE_PAREN_RE = re.compile(
    r"\b(REAL|INTEGER|COMPLEX|LOGICAL)(\s*\(\s*)([A-Za-z_]\w*)(\s*\))",
    re.IGNORECASE,
)

# Literal-kind suffix (``1.0_wp``): group 1 = numeric, group 2 = kind symbol.
# Leading word boundary keeps the match off the middle of identifiers.
_LITERAL_KIND_RE = re.compile(r"\b(\d+(?:\.\d*)?(?:[eEdD][+-]?\d+)?)_([A-Za-z_]\w*)\b")

# ``INTEGER, PARAMETER :: sym = <int>`` locally-bound form -- caught aliases
# are dropped from substitution (local binding wins).  Non-integer RHS
# (``SELECTED_REAL_KIND(...)``) doesn't match, so it falls through to the
# rewrite -- the intended case.
_PARAM_BIND_RE = re.compile(r"\bINTEGER\b[^:\n]*::\s*([A-Za-z_]\w*)\s*=\s*(\d+)\b", re.IGNORECASE)


def _local_kind_bindings(source: str) -> dict:
    """Locally-defined ``INTEGER, PARAMETER :: sym = <int>`` bindings, as
    ``{lowercase-symbol: int}``.  Only literal-integer RHS is captured;
    other forms (intrinsics) fall through to the alias-substitution path.
    """
    out: dict = {}
    for raw in source.splitlines():
        m = _PARAM_BIND_RE.search(_code_of(raw))
        if m:
            out.setdefault(m.group(1).lower(), int(m.group(2)))
    return out


def normalize_kind_parameters(source: str, *, kind_map: dict = None, passthrough: bool = False) -> str:
    """Substitute symbolic precision kind aliases (``wp``, ``JPRB``, ...) with
    literal kind ints (default fp64 ``8``) at every use site
    (``REAL(KIND=wp)``, ``1.0_wp``).

    flang only resolves a kind alias when its defining constants module is in
    the TU; a single-file slice (probe / extracted kernel) lacks it and flang
    errors out.  Locally-bound ``INTEGER, PARAMETER`` aliases are skipped
    (flang already handles those).  Idempotent and no-op-safe -- safe to
    leave default-on even when an upstream pipeline resolves kinds itself.

    ``kind_map`` overrides/extends the default alias table (per-alias
    ``None`` leaves it untouched); ``passthrough=True`` disables the rewrite
    entirely.
    """
    if passthrough:
        return source

    aliases = dict(_DEFAULT_KIND_ALIASES)
    if kind_map:
        for k, v in kind_map.items():
            aliases[k.lower()] = v
    # Drop aliases the caller opted out of and any locally-bound ones.
    aliases = {k: v for k, v in aliases.items() if v is not None}
    for nm in _local_kind_bindings(source):
        aliases.pop(nm, None)
    if not aliases:
        return source

    def _resolve_kind_eq(m):
        sym = m.group(2).lower()
        return f"{m.group(1)}{aliases[sym]}" if sym in aliases else None

    def _resolve_type_paren(m):
        sym = m.group(3).lower()
        if sym not in aliases:
            return None
        return f"{m.group(1)}{m.group(2)}{aliases[sym]}{m.group(4)}"

    def _resolve_literal(m):
        sym = m.group(2).lower()
        return f"{m.group(1)}_{aliases[sym]}" if sym in aliases else None

    rules = (
        (_KIND_EQ_RE, _resolve_kind_eq),
        (_TYPE_PAREN_RE, _resolve_type_paren),
        (_LITERAL_KIND_RE, _resolve_literal),
    )

    out = []
    for line in source.splitlines(keepends=True):
        nl = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(nl)]
        cut, strings = _scan_line(body)
        code, tail = body[:cut], body[cut:]
        edits = []
        for pat, fn in rules:
            for m in pat.finditer(code):
                if any(s <= m.start() < e for s, e in strings):
                    continue  # inside a character literal
                repl = fn(m)
                if repl is None:
                    continue
                # Skip if this span overlaps an edit already queued.
                if any(not (m.end() <= eb or m.start() >= ee) for eb, ee, _ in edits):
                    continue
                edits.append((m.start(), m.end(), repl))
        for begin, fin, repl in sorted(edits, key=lambda e: e[0], reverse=True):
            code = code[:begin] + repl + code[fin:]
        out.append(code + tail + nl)
    return "".join(out)


def preprocess_fortran(source: str) -> str:
    """Rewrite ``IF (intvar)`` to ``IF (intvar /= 0)`` for any INTEGER scalar
    declared in ``source``.  Idempotent.
    """
    int_names = _collect_integer_scalar_names(source)
    if not int_names:
        return source

    def _rewrite(m: re.Match) -> str:
        ident = m.group(2)
        if ident.lower() in int_names:
            return f"{m.group(1)}{ident} /= 0{m.group(3)}"
        return m.group(0)

    return _BARE_IF_RE.sub(_rewrite, source)


# Intrinsic/compiler-provided modules -- flang supplies them, so a ``USE``
# of one is left untouched.
_INTRINSIC_MODULES = frozenset({
    "iso_c_binding",
    "iso_fortran_env",
    "ieee_arithmetic",
    "ieee_exceptions",
    "ieee_features",
    "omp_lib",
    "omp_lib_kinds",
    "openacc",
    "mpi",
    "mpi_f08",
})

# ``use [, intrinsic] [::] <name>`` -- code part only (comments/strings
# stripped by ``_scan_line``).
_USE_RE = re.compile(r"^\s*use\b\s*(?:,\s*intrinsic\s*)?(?:::)?\s*([A-Za-z]\w*)", re.IGNORECASE)
# ``module <name>`` opener -- excludes ``module procedure``/``subroutine``/
# ``function`` and ``submodule (...)``.
_MODULE_OPEN_RE = re.compile(r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)([A-Za-z]\w*)\s*$", re.IGNORECASE)
_MODULE_END_RE = re.compile(r"^\s*end\s*module\b", re.IGNORECASE)


def _code_of(line: str) -> str:
    """Code portion of one line with character literals blanked (spaces), so
    keyword scans never trip on a ``!``/module name inside a string.
    """
    cut, strings = _scan_line(line)
    code = list(line[:cut])
    for s, e in strings:
        for i in range(s, min(e, cut)):
            code[i] = " "
    return "".join(code)


#: Non-Fortran preamble line carried with the next ``MODULE`` opener (cpp
#: directive / ``!`` comment / blank) -- e.g. ICON's ``#include
#: "icon_definitions.inc"`` must survive extraction or its macros
#: (``start_sync_timer``, ...) stay unexpanded and flang errors on the bare
#: invocation.  Used by :func:`_module_blocks`.
_PREAMBLE_LINE_RE = re.compile(r"^\s*(?:#|!|$)")

#: cpp conditional directives, classified for balancing an extracted block.
_CPP_OPEN_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b", re.IGNORECASE)
_CPP_CLOSE_RE = re.compile(r"^\s*#\s*endif\b", re.IGNORECASE)
_CPP_MID_RE = re.compile(r"^\s*#\s*(?:else|elif)\b", re.IGNORECASE)


def _balance_cpp(block: str) -> str:
    """Drop cpp conditionals left unbalanced by module-block extraction -- a
    whole-module ``#ifdef...#endif`` wrapper splits across the block
    boundary, orphaning one side.  Unmatched directives are removed but their
    guarded content is kept unconditionally (the module was already
    build-selected).  Conditionals fully contained in a block stay untouched.
    """
    lines = block.splitlines(keepends=True)
    open_idx: list = []
    drop: set = set()
    for i, ln in enumerate(lines):
        if _CPP_OPEN_RE.match(ln):
            open_idx.append(i)
        elif _CPP_CLOSE_RE.match(ln):
            if open_idx:
                open_idx.pop()
            else:
                drop.add(i)  # orphan #endif
        elif _CPP_MID_RE.match(ln):
            if not open_idx:
                drop.add(i)  # orphan #else / #elif
    drop.update(open_idx)  # unmatched #if openers
    if not drop:
        return block
    return "".join(ln for i, ln in enumerate(lines) if i not in drop)


def _module_blocks(text: str):
    """Yield ``(name_lower, block_text)`` per top-level ``module`` in
    ``text`` (``submodule``/``module procedure`` excluded; modules don't
    nest).

    Each block also captures its contiguous preamble (cpp/comment/blank
    lines, see :data:`_PREAMBLE_LINE_RE`) so a leading ``#include`` survives
    extraction; the walk-back stops at the prior module's ``END MODULE`` or
    any real statement, so bodies never bleed together.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    last_end = 0  # next-after the previous module's END MODULE (or 0)
    while i < n:
        m = _MODULE_OPEN_RE.match(_code_of(lines[i].rstrip("\r\n")))
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        # Walk back over the contiguous preamble so a leading ``#include`` is carried.
        start = i
        while start > last_end and _PREAMBLE_LINE_RE.match(lines[start - 1]):
            start -= 1
        i += 1
        while i < n and not _MODULE_END_RE.match(_code_of(lines[i].rstrip("\r\n"))):
            i += 1
        end = min(i, n - 1)
        # Balance cpp conditionals split across the block boundary (see _balance_cpp).
        yield name, _balance_cpp("".join(lines[start:end + 1]))
        last_end = end + 1
        i = end + 1


def _used_modules(text: str) -> list:
    """Ordered, de-duplicated lowercase names of modules ``USE``-d in
    ``text`` (intrinsic modules excluded).
    """
    seen, out = set(), []
    for raw in text.splitlines():
        m = _USE_RE.match(_code_of(raw))
        if not m:
            continue
        nm = m.group(1).lower()
        if nm in _INTRINSIC_MODULES or nm in seen:
            continue
        seen.add(nm)
        out.append(nm)
    return out


#: ``SUBROUTINE``/``FUNCTION`` opener (prefix keywords + typed-function
#: forms), capturing the name.  Shared by the procedure indexer and the body stubber.
_PROC_OPEN_RE = re.compile(
    r"^\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+|IMPURE\s+)*"
    r"(?:SUBROUTINE|FUNCTION|REAL\s*FUNCTION|"
    r"INTEGER\s*FUNCTION|LOGICAL\s*FUNCTION|"
    r"DOUBLE\s+PRECISION\s+FUNCTION|"
    r"(?:REAL|INTEGER|LOGICAL|COMPLEX|CHARACTER)\s*\(\s*[^)]*\)\s+FUNCTION)\s+"
    r"([A-Za-z]\w*)",
    re.IGNORECASE,
)

#: ``END SUBROUTINE`` / ``END FUNCTION`` (optionally naming the procedure).
_PROC_END_RE = re.compile(r"^\s*END\s*(?:SUBROUTINE|FUNCTION)\b", re.IGNORECASE)

#: A ``CONTAINS`` line opening a scope's internal-subprogram part.
_CONTAINS_RE = re.compile(r"^\s*CONTAINS\b", re.IGNORECASE)

#: ``INTERFACE`` block opener (named/operator/``ABSTRACT``).  Its nested
#: ``SUBROUTINE``/``FUNCTION`` decls are spec, not body -- the stubber
#: consumes the whole block as a unit so they aren't mistaken for body start
#: (ICON's ``bind(c)`` halo wrappers carry this shape).
_INTERFACE_OPEN_RE = re.compile(r"^\s*(?:ABSTRACT\s+)?INTERFACE\b", re.IGNORECASE)
_INTERFACE_END_RE = re.compile(r"^\s*END\s*INTERFACE\b", re.IGNORECASE)

#: Specification-part statement (decl/attribute/interface) -- legal before a
#: procedure's first executable statement.  The stubber keeps these (dummy
#: args stay declared) and drops everything after the first non-matching
#: line.  Leading ``&`` keeps continuation lines.
_SPEC_LINE_RE = re.compile(
    r"^\s*(?:&|USE\b|IMPLICIT\b|INTEGER\b|REAL\b|DOUBLE\s+PRECISION\b|COMPLEX\b|LOGICAL\b|"
    r"CHARACTER\b|TYPE\b|CLASS\b|PROCEDURE\b|INTERFACE\b|END\s+INTERFACE\b|END\s+TYPE\b|"
    r"MODULE\s+PROCEDURE\b|PARAMETER\b|DIMENSION\b|ALLOCATABLE\b|POINTER\b|TARGET\b|"
    r"INTENT\b|OPTIONAL\b|SAVE\b|EXTERNAL\b|INTRINSIC\b|COMMON\b|DATA\b|NAMELIST\b|"
    r"IMPORT\b|VALUE\b|VOLATILE\b|ASYNCHRONOUS\b|SEQUENCE\b|GENERIC\b|ENUM\b|ENUMERATOR\b|"
    r"EQUIVALENCE\b|BIND\b|CONTIGUOUS\b)",
    re.IGNORECASE,
)


def _stub_procedure_bodies(text: str, names) -> str:
    """Empty the executable body of every procedure matching ``names`` --
    exactly, or as the generic an ICON interface dispatches over
    (``sync_patch_array`` -> ``sync_patch_array_3d_dp``) -- keeping its
    opener, spec part, and matching ``END``.

    Regex-merge analogue of the fparser inliner's ``make_noop``
    (:func:`dace_fortran.fparser_inliner._keep_external_noop_specs`): dummy
    args stay declared (in-TU call site stays legal) but internals
    (halo/MPI/I/O) never enter the TU.  Nesting-aware ``END`` scan handles
    internal ``CONTAINS`` subprograms.  Case-insensitive; names come from the
    caller's policy.
    """
    targets = {n.lower() for n in names}
    if not targets:
        return text

    def _is_target(nm: str) -> bool:
        nm = nm.lower()
        return nm in targets or any(nm.startswith(t + "_") for t in targets)

    lines = text.splitlines(keepends=True)
    out: list = []
    i, n = 0, len(lines)
    while i < n:
        code = _code_of(lines[i].rstrip("\r\n"))
        m = _PROC_OPEN_RE.match(code)
        if not m or not _is_target(m.group(1)):
            out.append(lines[i])
            i += 1
            continue
        # Keep the opener + contiguous spec part (dummy args stay declared);
        # stop at the first executable stmt, nested subprogram, CONTAINS, or matching END.
        out.append(lines[i])
        i += 1
        while i < n:
            c = _code_of(lines[i].rstrip("\r\n"))
            if _INTERFACE_OPEN_RE.match(c):
                # Keep the INTERFACE block verbatim (nesting-aware) -- its nested
                # proc decls are spec, not body, so they must not end spec collection.
                depth_if = 0
                while i < n:
                    ci = _code_of(lines[i].rstrip("\r\n"))
                    out.append(lines[i])
                    i += 1
                    if _INTERFACE_OPEN_RE.match(ci):
                        depth_if += 1
                    elif _INTERFACE_END_RE.match(ci):
                        depth_if -= 1
                        if depth_if == 0:
                            break
                continue
            if _PROC_END_RE.match(c) or _CONTAINS_RE.match(c) or _PROC_OPEN_RE.match(c):
                break
            if c.strip() and not _SPEC_LINE_RE.match(c):
                break  # first executable statement -- body ends here
            out.append(lines[i])
            i += 1
        # Drop up to (and keep) the matching END, tracking nesting so an
        # internal subprogram's own END isn't mistaken for it.
        depth = 1
        while i < n:
            c = _code_of(lines[i].rstrip("\r\n"))
            if _PROC_OPEN_RE.match(c):
                depth += 1
            elif _PROC_END_RE.match(c):
                depth -= 1
                if depth == 0:
                    out.append(lines[i])
                    i += 1
                    break
            i += 1
    return "".join(out)


def merge_used_modules(source: str, *, search_dirs=(), external_functions=(), do_not_emit=()) -> str:
    """Inline every ``USE``-d module's real source into ``source`` -- one
    self-contained TU, fparser-free (transitive ``USE``-graph resolve +
    dependency-ordered splice, de-duplicated).

    Pass-through when nothing external is resolvable (self-contained input);
    idempotent -- re-running adds nothing.

    ``external_functions``/``do_not_emit`` names are never inlined: their
    bodies are stubbed empty (:func:`_stub_procedure_bodies`, the regex
    analogue of the fparser inliner's ``make_noop``) so halo/MPI/I/O
    internals never enter the TU.  ``external_functions`` gets an EMITted
    external call, ``do_not_emit`` gets the call DROPped.
    """
    from pathlib import Path

    from dace_fortran.external_functions import dont_inline_names, validate
    validate(external_functions, do_not_emit)
    dont_inline = dont_inline_names(external_functions, do_not_emit)

    in_source = {nm for nm, _ in _module_blocks(source)}
    index: dict = {}
    for d in search_dirs:
        d = Path(d)
        files = [
            d
        ] if d.is_file() else sorted(list(d.rglob("*.f90")) + list(d.rglob("*.F90")) + list(d.rglob("*.incf")))
        for f in files:
            try:
                txt = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for nm, blk in _module_blocks(txt):
                if nm not in in_source:
                    index.setdefault(nm, blk)

    # Post-order DFS toposort (deps before dependents): each name is pushed
    # unexpanded, then re-pushed ``expanded`` so it's appended only after its
    # deps are visited.  A ``USE`` cycle's back-edge is silently dropped.
    order: list = []
    placed = set(in_source)
    stack = [(nm, False) for nm in reversed(_used_modules(source))]
    while stack:
        nm, expanded = stack.pop()
        if nm in placed or nm not in index:
            continue
        if expanded:
            placed.add(nm)
            order.append(index[nm])
            continue
        stack.append((nm, True))
        for dep in reversed(_used_modules(index[nm])):
            if dep not in placed and dep in index:
                stack.append((dep, False))

    if not order:
        return _stub_procedure_bodies(source, dont_inline) if dont_inline else source
    # Guard against a module's final line lacking ``\n`` (common in
    # hand-edited files) gluing into the next block's ``MODULE`` opener.
    parts = []
    for blk in order:
        parts.append(blk)
        if not blk.endswith("\n"):
            parts.append("\n")
    parts.append("\n")
    parts.append(source)
    merged = "".join(parts)
    return _stub_procedure_bodies(merged, dont_inline) if dont_inline else merged


# ``EXTERNAL`` declaration, anchored at line start so it never matches
# mid-statement.  Group ``names`` = the comma-separated name list (raw).
_EXTERNAL_DECL_RE = re.compile(
    r"^(?P<indent>\s*)EXTERNAL\s*(?:::\s*)?(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)

# ``SUBROUTINE``/``FUNCTION`` scope opener -- synthesised ``USE`` lines are
# inserted right after it, stacking with any imports already present.
_SCOPE_OPEN_RE = re.compile(
    r"^\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+|IMPURE\s+)*"
    r"(?:SUBROUTINE|FUNCTION|REAL\s*FUNCTION|"
    r"INTEGER\s*FUNCTION|LOGICAL\s*FUNCTION|"
    r"DOUBLE\s+PRECISION\s+FUNCTION|"
    r"(?:REAL|INTEGER|LOGICAL|COMPLEX|CHARACTER)\s*\(\s*[^)]*\)\s+FUNCTION)\s+"
    r"([A-Za-z]\w*)",
    re.IGNORECASE,
)


def _index_procedures_in_modules(search_dirs) -> dict:
    """Index every ``SUBROUTINE``/``FUNCTION`` declared inside a ``MODULE``
    block across ``search_dirs``, as ``{procedure_name_lower:
    module_name_lower}``.  Sibling of :func:`merge_used_modules`'s scan;
    feeds :func:`replace_external_with_modules`.
    """
    from pathlib import Path
    index: dict = {}
    proc_open = _PROC_OPEN_RE
    for d in search_dirs:
        d = Path(d)
        if d.is_file():
            files = [d]
        else:
            files = sorted(list(d.rglob("*.f90")) + list(d.rglob("*.F90")) + list(d.rglob("*.incf")))
        for f in files:
            try:
                txt = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for mod_name, blk in _module_blocks(txt):
                for raw in blk.splitlines():
                    m = proc_open.match(_code_of(raw))
                    if m:
                        index.setdefault(m.group(1).lower(), mod_name)
    return index


# Type decl naming one function result with no variable-defining attributes --
# detects the ``REAL(8) :: dscale`` companion of ``EXTERNAL :: dscale``, which
# becomes a "use-associated, cannot be re-declared" flang error once
# ``dscale`` is imported via ``USE``.
_FUNC_RESULT_TYPE_DECL_RE = re.compile(
    r"^\s*(?P<type>(?:REAL|INTEGER|LOGICAL|COMPLEX|DOUBLE\s+PRECISION)"
    r"(?:\s*\(\s*[^)]*\s*\))?)\s*"
    r"(?:::\s*)?"
    r"(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)


def _scope_already_uses(scope_lines, mod_name: str) -> bool:
    """``True`` when ``scope_lines`` already has a ``USE <mod_name>`` (any
    casing) -- avoids a redundant synthesised ``USE ..., ONLY:`` import.
    """
    target = mod_name.lower()
    for raw in scope_lines:
        m = _USE_RE.match(_code_of(raw))
        if m and m.group(1).lower() == target:
            return True
    return False


def replace_external_with_modules(source: str, *, search_dirs=()) -> str:
    """Replace ``EXTERNAL <name>, ...`` declarations with the equivalent
    ``USE <module>, ONLY: <name>, ...`` when the procedure is defined in a
    module visible via ``search_dirs``.

    ``EXTERNAL`` only promises a linker symbol; flang then routes the call
    through its implicit-interface path, which the bridge lowers less
    faithfully than an explicit module import (type/shape info lost).
    Resolution is per-scope, grouped by defining module, and inserted right
    after the scope opener (collapsed with any ``USE`` already present).

    Conservative: an ``EXTERNAL`` line with any unresolved name is left in
    place verbatim (even its resolved names) -- an unresolved name means the
    build is missing a source file.  Pass-through when ``search_dirs`` is
    empty or nothing resolves.  Idempotent.
    """
    if not search_dirs:
        return source
    index = _index_procedures_in_modules(search_dirs)
    if not index:
        return source

    lines = source.splitlines(keepends=True)

    # Pass 1: find scope-opener indices + each scope's ``EXTERNAL`` lines. A
    # scope opens at SUBROUTINE/FUNCTION, closes at the next one or its END.
    scopes = []  # list of (opener_idx, end_idx_exclusive)
    cur_open = None
    for i, raw in enumerate(lines):
        code = _code_of(raw)
        if _SCOPE_OPEN_RE.match(code):
            if cur_open is not None:
                scopes.append((cur_open, i))
            cur_open = i
        elif re.match(r"^\s*END\s+(SUBROUTINE|FUNCTION)\b", code, re.IGNORECASE):
            if cur_open is not None:
                scopes.append((cur_open, i + 1))
                cur_open = None
    if cur_open is not None:
        scopes.append((cur_open, len(lines)))

    # Pass 2: resolve EXTERNAL names per scope; build USE-insertions + EXTERNAL-deletions.
    delete_idx = set()
    insert_at_idx: dict = {}  # opener_idx -> list of synthesized USE lines
    for opener_idx, end_idx in scopes:
        scope_lines = lines[opener_idx:end_idx]
        # Collect EXTERNAL declarations + resolve, grouped by defining module.
        per_mod_names: dict = {}
        ext_line_idxs: list = []  # absolute indices to delete
        all_resolved_names: set = set()
        for li, raw in enumerate(scope_lines, start=opener_idx):
            code = _code_of(raw)
            m = _EXTERNAL_DECL_RE.match(code)
            if not m:
                continue
            names = [n.strip() for n in m.group("names").split(",")]
            unresolved = [n for n in names if n.lower() not in index]
            if unresolved:
                # Leave the EXTERNAL line alone -- conservative.
                continue
            for n in names:
                mod = index[n.lower()]
                per_mod_names.setdefault(mod, []).append(n)
                all_resolved_names.add(n.lower())
            ext_line_idxs.append(li)
        if not per_mod_names:
            continue
        # Function-result decls now use-associated must go too (flang rejects
        # ``REAL(8) :: dscale`` once ``dscale`` is USE-imported). Only a
        # single-name decl is dropped whole; a mixed line is left alone.
        for li, raw in enumerate(scope_lines, start=opener_idx):
            code = _code_of(raw)
            # Skip EXTERNAL / attribute-bearing decls -- the function-result
            # regex only matches the bare ``<type> :: <name>`` shape.
            if _EXTERNAL_DECL_RE.match(code):
                continue
            if re.search(
                    r"(?i)\b(intent|parameter|dimension|allocatable|pointer|target|save|optional|public|private|value)\b",
                    code):
                continue
            m = _FUNC_RESULT_TYPE_DECL_RE.match(code)
            if not m:
                continue
            decl_names = [n.strip().lower() for n in m.group("names").split(",")]
            # Drop only if EVERY name on the line is a resolved external.
            if all(n in all_resolved_names for n in decl_names):
                delete_idx.add(li)
        # Suppress imports for modules the scope already USEs.
        synth_use_lines = []
        for mod, names in per_mod_names.items():
            if _scope_already_uses(scope_lines, mod):
                continue  # EXTERNAL becomes a no-op delete
            # Indent = opener's leading whitespace + 2 spaces (conventional continuation).
            opener_indent = re.match(r"^(\s*)", lines[opener_idx]).group(1)
            body_indent = opener_indent + "  "
            synth_use_lines.append(f"{body_indent}USE {mod}, ONLY: {', '.join(names)}\n")
        if synth_use_lines:
            insert_at_idx[opener_idx + 1] = synth_use_lines
        for idx in ext_line_idxs:
            delete_idx.add(idx)

    if not delete_idx and not insert_at_idx:
        return source

    # Pass 3: walk every line, emitting inserts before it, dropping deletes.
    out = []
    for i, raw in enumerate(lines):
        if i in insert_at_idx:
            out.extend(insert_at_idx[i])
        if i in delete_idx:
            continue
        out.append(raw)
    return "".join(out)


# CHARACTER dummy decl (string used as an enum).  Recognises
# ``CHARACTER(LEN=*/LEN=N/*/N)`` and bare ``CHARACTER`` -- all valid Fortran
# string-arg shapes.
_CHARACTER_INTENT_IN_RE = re.compile(
    r"^(?P<lead>\s*)CHARACTER(?:\s*\(\s*[^)]+\s*\))?\s*"
    r",\s*INTENT\s*\(\s*IN\s*\)\s*"
    r"::\s*(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)


def _line_clip_comment(raw: str) -> str:
    """``raw`` truncated at the first ``!`` outside a string literal.
    Unlike :func:`_code_of`, string contents are preserved (needed by the
    enum detector).
    """
    cut, _ = _scan_line(raw)
    return raw[:cut]


def _scan_string_enum_uses(scope_body: str, var: str) -> dict:
    """Enum mapping ``{lowercase_literal: int}`` for every distinct literal
    ``var`` is compared against in ``scope_body``: ``var == 'lit'``,
    ``'lit' == var``, or ``CASE ('lit')`` inside ``SELECT CASE (var)``.
    Case-insensitive grouping (``'c'``/``'C'`` collapse to one entry);
    literals are enumerated in first-appearance order for a deterministic map.
    """
    seen_order: list = []
    mapping: dict = {}

    # ``<var> == 'lit'`` / ``<var> .EQ. 'lit'`` -- two operand orders.
    var_eq_re = re.compile(
        rf"\b{re.escape(var)}\s*(?:==|\.EQ\.)\s*['\"]([^'\"]*)['\"]",
        re.IGNORECASE,
    )
    eq_var_re = re.compile(
        rf"['\"]([^'\"]*)['\"]\s*(?:==|\.EQ\.)\s*\b{re.escape(var)}\b",
        re.IGNORECASE,
    )

    def _record(lit: str):
        key = lit.lower()
        if key in mapping:
            return
        mapping[key] = len(seen_order)
        seen_order.append(key)

    for line in scope_body.splitlines():
        code = _line_clip_comment(line)
        for m in var_eq_re.finditer(code):
            _record(m.group(1))
        for m in eq_var_re.finditer(code):
            _record(m.group(1))

    # SELECT CASE (var) block scan: collect CASE literals within the block;
    # single-pass, re-anchored per opener so nested blocks aren't conflated.
    in_block = False
    block_re = re.compile(rf"^\s*SELECT\s+CASE\s*\(\s*{re.escape(var)}\s*\)", re.IGNORECASE)
    end_block_re = re.compile(r"^\s*END\s+SELECT\b", re.IGNORECASE)
    case_lit_re = re.compile(r"^\s*CASE\s*\(\s*['\"]([^'\"]*)['\"]\s*\)", re.IGNORECASE)
    for line in scope_body.splitlines():
        code = _line_clip_comment(line)
        if not in_block:
            if block_re.match(code):
                in_block = True
            continue
        if end_block_re.match(code):
            in_block = False
            continue
        cm = case_lit_re.match(code)
        if cm:
            _record(cm.group(1))

    return mapping


def rewrite_string_enum_to_integer(source: str) -> tuple:
    """Convert ``CHARACTER(LEN=...), INTENT(IN)`` dummy args used as enum
    switches to ``INTEGER, INTENT(IN)`` plus integer-valued comparisons (QE's
    ``addusxx_g`` ``flag`` shape).  Returns ``(rewritten_source, enum_maps)``
    where ``enum_maps`` is ``{proc: {arg: {literal_lower: int}}}`` --
    consumed by the bindings layer to expose a string-typed wrapper and
    normalise to the integer value at the SDFG boundary.

    Scope-narrow by design: only single-name CHARACTER decls, only
    INTENT(IN), only bare ``SELECT CASE (var)`` (not ``TRIM(var)``).  Call
    sites are NOT rewritten -- the bindings layer converts at the SDFG
    boundary, so a caller invoking the Fortran directly must pass an integer.

    Pass-through (``source``, ``{}``) when no enum-switch dummy is found.
    Idempotent.
    """
    lines = source.splitlines(keepends=True)

    # Reuse the EXTERNAL pass's scope extraction.
    scopes: list = []
    cur_open = None
    cur_name = None
    for i, raw in enumerate(lines):
        code = _code_of(raw)
        m = _SCOPE_OPEN_RE.match(code)
        if m:
            if cur_open is not None:
                scopes.append((cur_open, i, cur_name))
            cur_open = i
            cur_name = m.group(1).lower()
        elif re.match(r"^\s*END\s+(SUBROUTINE|FUNCTION)\b", code, re.IGNORECASE):
            if cur_open is not None:
                scopes.append((cur_open, i + 1, cur_name))
                cur_open = None
                cur_name = None
    if cur_open is not None:
        scopes.append((cur_open, len(lines), cur_name))

    enum_maps: dict = {}
    delete_idx: dict = {}  # line idx -> replacement string (or None for delete)

    for opener_idx, end_idx, proc_name in scopes:
        scope_lines = lines[opener_idx:end_idx]
        scope_body = "".join(scope_lines)

        # Find every CHARACTER(...) INTENT(IN) declaration in this scope.
        decl_targets: list = []  # (line_idx, var_name, lead_indent)
        for li, raw in enumerate(scope_lines, start=opener_idx):
            code = _code_of(raw)
            m = _CHARACTER_INTENT_IN_RE.match(code)
            if not m:
                continue
            names = [n.strip() for n in m.group("names").split(",")]
            if len(names) != 1:
                continue  # mixed-decl line -- skip
            decl_targets.append((li, names[0], m.group("lead")))

        if not decl_targets:
            continue

        # Skip a dummy the scope never compares against a string literal.
        for li, var, lead in decl_targets:
            mapping = _scan_string_enum_uses(scope_body, var)
            if not mapping:
                continue

            enum_maps.setdefault(proc_name, {})[var] = dict(mapping)

            delete_idx[li] = f"{lead}INTEGER, INTENT(IN) :: {var}\n"

            # Rewrite every comparison + CASE literal in scope.
            var_eq_re = re.compile(
                rf"(\b{re.escape(var)}\b\s*(?:==|\.EQ\.)\s*)['\"]([^'\"]*)['\"]",
                re.IGNORECASE,
            )
            eq_var_re = re.compile(
                rf"['\"]([^'\"]*)['\"](\s*(?:==|\.EQ\.)\s*\b{re.escape(var)}\b)",
                re.IGNORECASE,
            )
            case_re = re.compile(r"^(\s*CASE\s*\(\s*)['\"]([^'\"]*)['\"](\s*\))", re.IGNORECASE)

            in_block = False
            block_re = re.compile(rf"^\s*SELECT\s+CASE\s*\(\s*{re.escape(var)}\s*\)", re.IGNORECASE)
            end_block_re = re.compile(r"^\s*END\s+SELECT\b", re.IGNORECASE)

            for lj, raw in enumerate(scope_lines, start=opener_idx):
                if lj == li:
                    continue  # already rewritten above
                clipped = _line_clip_comment(raw)
                if not clipped.strip():
                    continue

                # Track SELECT CASE block scope for case-literal rewrite.
                if not in_block and block_re.match(clipped):
                    in_block = True
                elif in_block and end_block_re.match(clipped):
                    in_block = False

                new_line = raw
                # CASE ('lit') inside SELECT CASE (var)
                if in_block:
                    cm = case_re.match(clipped)
                    if cm:
                        lit_key = cm.group(2).lower()
                        if lit_key in mapping:
                            # Rebuild, preserving any trailing comment past the match end.
                            new_line = (cm.group(1) + str(mapping[lit_key]) + cm.group(3) + raw[len(cm.group(0)):])

                # ``<var> == 'lit'`` / ``<var> .EQ. 'lit'``
                def _replace_var_eq(m):
                    lit_key = m.group(2).lower()
                    if lit_key not in mapping:
                        return m.group(0)
                    return f"{m.group(1)}{mapping[lit_key]}"

                def _replace_eq_var(m):
                    lit_key = m.group(1).lower()
                    if lit_key not in mapping:
                        return m.group(0)
                    return f"{mapping[lit_key]}{m.group(2)}"

                new_line = var_eq_re.sub(_replace_var_eq, new_line)
                new_line = eq_var_re.sub(_replace_eq_var, new_line)

                if new_line != raw:
                    delete_idx[lj] = new_line

    if not enum_maps:
        return source, {}

    out = []
    for i, raw in enumerate(lines):
        out.append(delete_idx[i] if i in delete_idx else raw)
    return "".join(out), enum_maps


def _fparser_merge(source: str,
                   *,
                   search_dirs=(),
                   entry: Optional[str] = None,
                   external_names: Iterable[str] = ()) -> str:
    """Single-TU merge via the fparser inliner engine (opt-in via
    ``merge_engine="fparser"``; the regex splicer stays default).

    Sibling of :func:`merge_used_modules`: parses ``source`` + every file
    under ``search_dirs`` into one fparser AST, resolves ``USE``, inlines,
    desugars, and serialises back to ``.f90`` text.  ``entry`` restricts
    pruning to its USE-closure (``None`` keeps every top-level subprogram).
    ``external_names`` are stubbed empty (inliner's ``make_noop``) so their
    internals never enter the TU.
    """
    from dace_fortran.fparser_inliner import inline_to_ast, strip_builtin_stub_modules

    src_map = {}
    for d in search_dirs:
        d = Path(d)
        files = ([d]
                 if d.is_file() else sorted(list(d.rglob("*.f90")) + list(d.rglob("*.F90")) + list(d.rglob("*.incf"))))
        for f in files:
            try:
                src_map.setdefault(str(f), f.read_text())
            except (OSError, UnicodeDecodeError):
                continue
    # Inject ``source`` under a synthetic key only if not already staged (a
    # multi-file build already stages the root file; injecting unconditionally
    # would duplicate every procedure in it).
    if not any(txt == source for txt in src_map.values()):
        src_map["__entry__.f90"] = source
    # Inject intrinsic-module stubs so fparser resolves ``USE iso_c_binding``
    # etc. (it hard-requires every USE-d module).  They don't leak into flang's
    # output: the inliner prunes unreachable stubs and resolves intrinsic
    # kinds inline (``REAL(c_double)`` -> ``REAL(KIND=8)``).
    # ``optimize=False``: skip the inliner's const-propagation -- flang/the
    # bridge already do constant-folding, and this avoids optimizer
    # fragilities on inlined-call patterns.
    ast = inline_to_ast(src_map, entry, include_builtins=True, optimize=False, do_not_emit=external_names)
    # Strip injected intrinsic-module stubs (no-op if ``entry`` already pruned
    # them) so flang's own ``iso_c_binding``/``iso_fortran_env`` are used.
    ast = strip_builtin_stub_modules(ast)
    return ast.tofortran()


def preprocess_fortran_source(source: str,
                              *,
                              search_dirs=(),
                              merge: bool = True,
                              merge_engine: str = "regex",
                              merge_entry: Optional[str] = None,
                              external_names: Iterable[str] = (),
                              if_intvar: bool = False,
                              kind_map: dict = None,
                              kind_passthrough: bool = False) -> str:
    """Single entrypoint for all Fortran-source preprocessing before flang.

    Order matters -- composes:

    1. ``merge_used_modules`` (if ``merge``) -- inline ``USE``-d modules into
       one TU.  ``merge_engine="fparser"`` routes through
       :func:`_fparser_merge` instead (also desugars/prunes; ``merge_entry``
       scopes its pruning, ignored by the regex engine).
    2. ``strip_openmp_directives`` -- drop OpenMP/OpenACC sentinels + the
       ICON include (bridge runs no cpp, no ``-fopenmp``).
    3. ``normalize_kind_parameters`` -- unresolved kind aliases -> literal
       ints (``kind_map`` overrides; ``kind_passthrough=True`` skips it).
    4. ``rewrite_integer_powers`` -- ``x**2.0`` -> ``x*x``.
    5. ``preprocess_fortran`` (if ``if_intvar``) -- opt-in ``IF (intvar)``
       rewrite.

    ``external_names``: kept external (not inlined) at the merge stage in
    BOTH engines -- stubbed empty so halo/MPI/I/O internals never enter the TU.
    """
    if merge:
        if merge_engine == "fparser":
            source = _fparser_merge(source, search_dirs=search_dirs, entry=merge_entry, external_names=external_names)
        elif merge_engine == "regex":
            source = merge_used_modules(source, search_dirs=search_dirs, do_not_emit=external_names)
        else:
            raise ValueError(f"merge_engine must be 'regex' or 'fparser', got {merge_engine!r}")
    source = strip_openmp_directives(source)
    source = normalize_kind_parameters(source, kind_map=kind_map, passthrough=kind_passthrough)
    source = rewrite_integer_powers(source)
    if if_intvar:
        source = preprocess_fortran(source)
    return source
