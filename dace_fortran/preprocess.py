"""Fortran-source-level pre-processor.

Some transforms have to happen on the Fortran *text*, before
``flang-new -fc1 -emit-hlfir`` runs, because they either change what
flang accepts or what arithmetic each backend is free to pick.  This
module holds the independent text rewrites:

* ``rewrite_integer_powers`` -- expands an integer-valued REAL-literal
  power (``x**2.0`` -> ``(x*x)``).  Runs **unconditionally** in
  ``compile_to_hlfir``: the rewrite is algebraically exact and removes
  a backend-dependent ``pow(x, 2.0)`` vs ``x*x`` rounding difference
  against the gfortran reference.

* ``promote_real_literals_to_double`` -- rewrites single/default REAL
  literals to an explicit double form (``2.0`` -> ``2.0D0``).  A
  standalone utility, applied directly to kernel source on disk when a
  codebase must be globally double; **not** wired into the build path.

* ``strip_openmp_directives`` -- drops ``!$OMP`` / ``!$ACC`` / ``!$``
  sentinel lines and the ICON ``#include "*omp_definitions*.inc"``
  bring-ins of OpenMP macros.  Runs **unconditionally** in
  ``preprocess_fortran_source``: without ``-fopenmp`` flang already
  treats sentinels as comments, so the rewrite is a semantic no-op,
  but it keeps the merged source free of accelerator noise (and
  removes the cpp ``#include`` that flang would otherwise refuse
  because the bridge does not run cpp).

* ``preprocess_fortran`` -- rewrites ``IF (intvar)`` to
  ``IF (intvar /= 0)`` for INTEGER scalars.  flang-new-21 rejects bare
  INTEGER as an IF condition (only LOGICAL is legal); legacy ECRAD /
  CloudSC / ICON code ships this shape.  **Opt-in** per call site
  (``compile_to_hlfir(..., preprocess=True)``) -- off by default so we
  don't paper over real issues in clean source.

These are pragmatic SED-style transforms, NOT a Fortran parser; they
are deliberately narrow (single-identifier IF guards only; powers with
a primary base only) and brittle by construction.  Comment- and
string-awareness is shared via ``_scan_line`` so a ``!`` or ``**``
inside a character literal is never touched.
"""

import re

# A REAL-literal exponent whose value is a whole number (``**2.0``,
# ``**3.0``, ``**2.0_JPRB``, ``**2.0D0``, ``**2.``).  Only this form is
# expanded to repeated multiplication: it is the case where each backend
# is free to pick ``pow(x, 2.0)`` and round differently from gfortran's
# ``x*x``.  A bare-integer exponent (``x**2``) is deliberately left
# alone -- flang already lowers it to the integer-power (multiply) path
# bit-identically to gfortran -- and genuine fractional powers
# (``**0.5``, ``**0.333``) must stay as ``pow()``.  ``_REAL_EXP``
# requires a digit before the dot so a bare integer never matches.
_REAL_EXP = r"\d+\.\d*(?:[eEdD][+-]?\d+)?(?:_[A-Za-z]\w*|_\d+)?"
_INT_POW_RE = re.compile(r"\*\*\s*(?:\(\s*(" + _REAL_EXP + r")\s*\)|(" + _REAL_EXP + r")(?![\w.]))")

# An identifier immediately followed by ``(`` -- a function call or an
# array reference.  A power base containing one must not be duplicated.
_CALL_IN_BASE = re.compile(r"[A-Za-z_]\w*\s*\(")

# A Fortran REAL literal: needs a fractional point or an exponent (so a
# bare integer never matches).  ``mantissa`` is groups 1-3, ``kind`` the
# optional ``_KIND`` / ``_8`` suffix.  Lookbehind/ahead keep us off
# identifiers (``R2ES``) and kind selectors.
_REAL_LIT_RE = re.compile(r"(?<![\w.])"
                          r"(\d+\.\d*|\.\d+|\d+)"  # mantissa
                          r"([eEdD][+-]?\d+)?"  # optional exponent
                          r"(_[A-Za-z]\w*|_\d+)?"  # optional kind suffix
                          r"(?![\w.])")
# Kind suffixes that are already double precision -- leave those alone.
_DOUBLE_KINDS = {"jprb", "jprd", "dp", "8", "16", "r8", "qp"}

_INTEGER_DECL_RE = re.compile(
    r"\bINTEGER\b(?:\s*\([^)]*\))?(?:\s*,\s*[A-Z_]+(?:\s*\([^)]*\))?)*\s*::\s*([^\n!]+)",
    re.IGNORECASE,
)
_BARE_IF_RE = re.compile(r"\b(IF\s*\(\s*)([A-Za-z_]\w*)(\s*\))", re.IGNORECASE)


def _scan_line(body: str):
    """Locate the comment start and the character-string spans of one
    physical Fortran line.  Shared by every text rewrite so a ``!`` or
    ``**`` inside a character literal is never treated as code.

    :param body: the line without its newline.
    :returns: ``(comment_index, [(start, end), ...])`` -- ``comment_index``
        is ``len(body)`` when the line has no comment; the span list
        covers ``'...'`` / ``"..."`` literals (Fortran ``''`` / ``""``
        doubling stays inside one span).
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
    """Return the set of INTEGER scalar identifiers declared in
    ``source``.  Skip array declarations -- those can't be the bare
    operand of an ``IF`` anyway.  All names are lowercased for
    case-insensitive matching.

    :param source: full Fortran source text.
    :returns: lowercased INTEGER scalar names.
    """
    names: set[str] = set()
    for m in _INTEGER_DECL_RE.finditer(source):
        decl = m.group(1).split('!', 1)[0]
        for tok in decl.split(','):
            head = tok.strip().split('=', 1)[0].strip()
            # Skip array forms ("name(...)") and assumed-shape (":") --
            # an array can't be the bare argument of IF.
            if '(' in head:
                continue
            name = head.split()[0] if head else ''
            if name and name.replace('_', '').isalnum() and not name[0].isdigit():
                names.add(name.lower())
    return names


def _extract_power_base(code: str, star: int):
    """Find the base (left primary) of a ``**`` operator.

    Scans leftward from the ``**`` over a Fortran *primary*: a
    parenthesised group, an identifier, an array/function reference
    (``name(...)``) and ``%`` component chains (``a%b(i)%c``).

    :param code: the comment-stripped source line.
    :param star: index of the first ``*`` of the ``**`` token.
    :returns: ``(begin, end)`` slice of the base in ``code``, or
        ``None`` when no base is found or parens are unbalanced.
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
    """Integer value of a REAL-literal exponent token, or ``None``.

    :param tok: e.g. ``2.0``, ``3.0_JPRB``, ``2.0D0``, ``2.5``.
    :returns: the int ``n`` when ``tok`` is a whole number >= 1
        (``2.0`` -> 2), else ``None`` (``2.5`` / ``0.0``).
    """
    mant = re.sub(r"(_[A-Za-z]\w*|_\d+)$", "", tok)
    try:
        val = float(mant.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None
    return int(val) if val >= 1 and val == int(val) else None


def rewrite_integer_powers(source: str) -> str:
    """Expand integer-valued REAL-literal powers to repeated multiply:
    ``base**2.0`` -> ``(base*base)``, ``base**3.0_JPRB`` ->
    ``(base*base*base)``.

    Only one outer pair of parentheses is added -- the minimal change
    that keeps the diff close to the source.  ``_extract_power_base``
    always returns a Fortran *primary* (identifier, ``a%b(i)`` chain,
    array/function reference, or an already-parenthesised group), so
    each copied factor is safe to juxtapose with ``*`` without its own
    wrapping.  The single outer pair preserves precedence in every
    surrounding context (``2.0*x**2.0`` -> ``2.0*(x*x)``, ``a/b**2.0``
    -> ``a/(b*b)``, ``-x**2.0`` -> ``-(x*x)``, ``(p-q)**3.0`` ->
    ``((p-q)*(p-q)*(p-q))``).  Only a whole-number REAL exponent is
    matched: bare-integer ``x**2`` is left for flang's (correct)
    integer-power lowering, and genuine fractional powers (``**0.5``,
    ``**0.333``) are never altered.  A base containing a function /
    array reference (``f(x)``, ``arr(i,j)``, ``a%b(i)%c``) is also
    left alone -- duplicating it would call twice (impure functions /
    shared inlined accumulators).  Comments and overlapping (stacked
    ``a**2.0**2.0``) matches are skipped.

    Idempotent: the output contains no ``**<real>`` left to match, so
    a second pass returns its input unchanged.

    :param source: full Fortran source text.
    :returns: source with integer-valued REAL powers expanded.
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
                # Base contains a function / array reference
                # (``f(x)``, ``arr(i,j)``, ``a%b(i)%c``).  Duplicating
                # it would invoke the call twice -- unsafe for impure
                # functions, and the bridge's call-inlining shares the
                # callee's accumulator across the copies (observed:
                # ``custom_sum(d)**2.0`` -> 2500 instead of 625).  Leave
                # such powers for flang's own lowering.
                continue
            repl = "(" + "*".join(base for _ in range(n)) + ")"
            edits.append((begin, m.end(), repl))
        for begin, fin, repl in reversed(edits):  # right-to-left: stable idx
            code = code[:begin] + repl + code[fin:]
        out.append(code + tail + nl)
    return "".join(out)


def _promote_one(m: re.Match):
    """Rewrite a single real-literal match to a double-precision form.

    :param m: a ``_REAL_LIT_RE`` match.
    :returns: the double literal text, or the original match when it is
        an integer or already double precision.
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
    """Rewrite every single-precision / default REAL literal to an
    explicit double-precision form (``2.0`` -> ``2.0D0``, ``0.85E5`` ->
    ``0.85D5``, ``1.0_JPRM`` -> ``1.0D0``).

    Literals already double -- a ``D`` exponent or a double kind suffix
    (``_JPRB``, ``_8``, ...) -- and integer literals are left untouched.
    Comments and character strings are never modified.

    Idempotent: a promoted literal carries a ``D`` exponent, which the
    classifier treats as already-double on a second pass.

    :param source: full Fortran source text.
    :returns: source with single/default REAL literals doubled.
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


#: Free-form OpenMP / OpenACC / CUDA-Fortran sentinel: ``!$<tag>`` at the
#: start of a (possibly-indented) line, where ``<tag>`` is ``omp`` /
#: ``acc`` / ``cuf``.  Matches both the directive opener (``!$OMP DO``)
#: and a continuation (``!$OMP&``) because both share the prefix.
_OMP_SENTINEL_RE = re.compile(r"^\s*!\s*\$\s*(?:omp|acc|cuf)\b", re.IGNORECASE)

#: Fortran-OpenMP conditional-compilation line: ``!$ <stmt>`` -- compiled
#: only when ``_OPENMP`` is defined.  Without ``-fopenmp`` flang treats
#: it as a comment, so dropping the line is a semantic no-op.  The
#: pattern requires a space (or tab) after ``!$`` so a directive (``!$OMP``,
#: ``!$ACC``) is left for the sentinel rule to handle.
_OMP_COND_LINE_RE = re.compile(r"^\s*!\s*\$[ \t]+\S")

#: ICON cpp include that pulls in the ``ICON_OMP_*`` / ``ICON_HAMOCC_OMP_*``
#: macros (``#include "omp_definitions.inc"``, ``"hamocc_omp_definitions.inc"``).
#: With the bridge not running cpp, the include line itself would crash
#: flang -- dropping it is the safe choice because every macro it
#: defines expands to an OpenMP sentinel comment (and those are stripped
#: above too).
_OMP_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"][^>"]*omp_definitions[^>"]*\.inc[>"]\s*$',
                             re.IGNORECASE)

#: Macros treated as undefined by ``strip_openmp_directives`` -- ``#ifdef``
#: blocks gated on these are dropped, ``#ifndef`` blocks pass through.
#: Limiting elision to this set keeps the pass narrow: unrelated cpp
#: conditionals (``#ifdef __SWAPDIM``, ``#ifdef _CRAYFTN``) flow through
#: untouched (and will surface as flang errors if no real cpp runs --
#: that is the next preprocessing step to address, not this one's job).
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
    """Drop OpenMP / OpenACC / CUDA-Fortran sentinel lines, the ICON
    ``omp_definitions.inc`` cpp include, and ``#ifdef _OPENMP`` /
    ``#ifdef _OPENACC`` conditional blocks (taking their ``#else`` body
    when present).

    The bridge does not run cpp and does not pass ``-fopenmp`` to flang,
    so accelerator sentinels (``!$OMP``, ``!$ACC``, ``!$CUF``, ``!$ ...``)
    are already inert comments, while the cpp ``#include`` and
    ``#ifdef _OPENMP`` lines themselves crash flang outright.  This pass
    removes all of them so the merged source the bridge writes to disk
    is free of accelerator noise and free of OpenMP / OpenACC cpp
    constructs that flang cannot consume.

    Block elision is scoped to ``_OPENMP`` / ``_OPENACC`` (see
    :data:`_OMP_ACC_MACROS`).  Other ``#ifdef`` macros (``__SWAPDIM``,
    ``_CRAYFTN``, ...) pass through unchanged -- evaluating those is a
    separate preprocessing step.  ``#if defined(_OPENMP)`` and
    ``#if !defined(_OPENMP)`` are recognised as aliases of
    ``#ifdef`` / ``#ifndef``; any other ``#if`` form passes through.

    Idempotent: a second invocation finds no sentinel / include / OMP
    conditional lines left to drop and returns the input unchanged.

    :param source: full Fortran source text.
    :returns: source with OpenMP / OpenACC sentinels, includes, and
        ``#ifdef _OPENMP`` / ``#ifdef _OPENACC`` blocks removed.
    """
    out = []
    # Stack of (is_omp_acc_block, dropping_now).  A non-OMP/ACC ``#if``
    # pushes ``(False, False)`` so we keep nesting straight and don't
    # touch unrelated cpp blocks.
    stack: list = []
    for line in source.splitlines(keepends=True):
        m_ifdef = _CPP_IFDEF_RE.match(line)
        m_ifdef_paren = _CPP_IFDEFINED_RE.match(line) if not m_ifdef else None
        if m_ifdef:
            kind, macro = m_ifdef.group(1).lower(), m_ifdef.group(2)
            if macro in _OMP_ACC_MACROS:
                stack.append((True, kind == "ifdef"))  # drop the matching body
                continue
            stack.append((False, False))
        elif m_ifdef_paren:
            negate, macro = bool(m_ifdef_paren.group(1)), m_ifdef_paren.group(2)
            if macro in _OMP_ACC_MACROS:
                stack.append((True, not negate))
                continue
            stack.append((False, False))
        elif _CPP_IF_RE.match(line):
            stack.append((False, False))
        elif _CPP_ELSE_RE.match(line):
            if stack and stack[-1][0]:
                stack[-1] = (True, not stack[-1][1])
                continue
        elif _CPP_ELIF_RE.match(line):
            if stack and stack[-1][0]:
                # An OMP/ACC ``#elif`` always drops (the macro stays
                # undefined whichever branch we are on).
                stack[-1] = (True, True)
                continue
        elif _CPP_ENDIF_RE.match(line):
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
        out.append(line)
    return "".join(out)


def preprocess_fortran(source: str) -> str:
    """Rewrite ``IF (intvar)`` to ``IF (intvar /= 0)`` for any INTEGER
    scalar declared in ``source``.

    Idempotent: a second invocation finds no bare-identifier IF guards
    left to rewrite and returns the input unchanged.

    :param source: full Fortran source text.
    :returns: source with bare-INTEGER IF guards made LOGICAL.
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


# Intrinsic / compiler-provided modules: never resolved or merged --
# flang supplies them itself, so a ``USE`` of one is left untouched.
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

# ``use [, intrinsic] [::] <name>`` -- captured from the code part of a
# line only (``_scan_line`` strips comments / character literals first).
_USE_RE = re.compile(r"^\s*use\b\s*(?:,\s*intrinsic\s*)?(?:::)?\s*([A-Za-z]\w*)", re.IGNORECASE)
# ``module <name>`` opening a module definition -- excludes
# ``module procedure`` / ``module subroutine`` / ``module function``
# and the ``submodule (...)`` form.
_MODULE_OPEN_RE = re.compile(r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)([A-Za-z]\w*)\s*$", re.IGNORECASE)
_MODULE_END_RE = re.compile(r"^\s*end\s*module\b", re.IGNORECASE)


def _code_of(line: str) -> str:
    """Return the code portion of one physical line with character
    literals blanked, so keyword scans never trip on a ``!`` / module
    name inside a string or comment.

    :param line: one physical Fortran line (no newline).
    :returns: the pre-comment text with ``'...'`` / ``"..."`` spans
        replaced by spaces.
    """
    cut, strings = _scan_line(line)
    code = list(line[:cut])
    for s, e in strings:
        for i in range(s, min(e, cut)):
            code[i] = " "
    return "".join(code)


#: A non-Fortran-statement line that should be carried with the next
#: ``MODULE`` opener -- a cpp directive (``#include`` /
#: ``#define`` / ``#ifdef`` / ...), a Fortran ``!`` comment, or a
#: blank line.  Used by :func:`_module_blocks` so that leading cpp
#: includes (ICON: ``#include "icon_definitions.inc"`` above the
#: ``MODULE mo_sync`` opener) survive module extraction; without
#: that, the macros those headers define (``start_sync_timer``,
#: ``HANDLE_MPI_ERROR``, ...) stay unexpanded in the merged source
#: and flang errors on the bare macro invocations.
_PREAMBLE_LINE_RE = re.compile(r"^\s*(?:#|!|$)")


def _module_blocks(text: str):
    """Yield ``(name_lower, block_text)`` for every top-level ``module``
    definition in ``text`` (modules do not nest; ``submodule`` and
    ``module procedure`` are not matched).

    The yielded block also captures any contiguous cpp / comment /
    blank lines immediately preceding the ``MODULE`` opener, so a
    top-of-file ``#include "<defs>.inc"`` (and the macro definitions
    behind it) is preserved when the bridge inlines the module into a
    merged translation unit.  The capture walks back only over the
    preamble shape (lines matching :data:`_PREAMBLE_LINE_RE`); it
    stops at the previous module's ``END MODULE`` or any real
    Fortran statement, so it never bleeds an earlier module's body
    into the next one.

    :param text: Fortran source.
    :returns: generator of ``(lowercase module name, verbatim block)``.
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
        # Walk back over the contiguous preamble (cpp / comment / blank
        # lines) so a leading ``#include`` is carried with the module.
        start = i
        while start > last_end and _PREAMBLE_LINE_RE.match(lines[start - 1]):
            start -= 1
        i += 1
        while i < n and not _MODULE_END_RE.match(_code_of(lines[i].rstrip("\r\n"))):
            i += 1
        end = min(i, n - 1)
        yield name, "".join(lines[start:end + 1])
        last_end = end + 1
        i = end + 1


def _used_modules(text: str) -> list:
    """Ordered, de-duplicated lowercase names of modules ``USE``-d in
    ``text`` (intrinsic modules excluded).

    :param text: Fortran source.
    :returns: list of module names in first-appearance order.
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


def merge_used_modules(source: str, *, search_dirs=()) -> str:
    """Inline every ``USE``-d module's real source into ``source``,
    producing one self-contained translation unit.

    A minimal, fparser-free port of the f2dace single-TU concept: scan
    ``search_dirs`` for module definitions, resolve the ``USE`` graph
    transitively from ``source``, and prepend each needed module's
    verbatim block in dependency order (deps first), de-duplicated.

    Pass-through (returns ``source`` unchanged) when nothing external is
    resolvable -- a self-contained single-file input, every ``USE``
    being intrinsic or defined in ``source`` itself.  This makes the
    pass safe to run by default: only genuine multi-file projects are
    transformed.  Idempotent: re-running finds the modules already
    inlined and adds nothing.

    :param source: the entry Fortran source text.
    :param search_dirs: directories scanned (recursively, ``*.f90`` /
        ``*.F90`` / ``*.incf``) for module definitions.
    :returns: a single-TU source, or ``source`` unchanged.
    """
    from pathlib import Path

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
        return source
    # Ensure a newline between every block so a module whose final
    # ``END MODULE`` line lacks a trailing ``\n`` (a common shape for
    # human-edited files) does not glue into the next block's
    # ``MODULE <next>`` opener.
    parts = []
    for blk in order:
        parts.append(blk)
        if not blk.endswith("\n"):
            parts.append("\n")
    parts.append("\n")
    parts.append(source)
    return "".join(parts)


def preprocess_fortran_source(source: str, *, search_dirs=(), merge: bool = True, if_intvar: bool = False) -> str:
    """Single entrypoint for all Fortran-source preprocessing the HLFIR
    frontend applies before handing source to flang.

    Composes, in order:

    1. ``merge_used_modules`` (when ``merge``) -- inline every
       externally-``USE``-d module into one translation unit.
    2. ``strip_openmp_directives`` -- drop OpenMP / OpenACC sentinels
       and the ICON ``omp_definitions.inc`` cpp include (the bridge
       does not run cpp and does not pass ``-fopenmp``).
    3. ``rewrite_integer_powers`` -- expand integer-valued real powers
       (``x**2.0`` -> ``x*x``) for byte-identical arithmetic.
    4. ``preprocess_fortran`` (when ``if_intvar``) -- the opt-in
       ``IF (intvar)`` -> ``IF (intvar /= 0)`` rewrite.

    The individual stages remain importable for their unit tests; this
    is the production composition.

    :param source: Fortran source text.
    :param search_dirs: directories searched by ``merge_used_modules``.
    :param merge: run the ``USE``-merge stage (default on).
    :param if_intvar: also run the opt-in integer-IF rewrite.
    :returns: the fully preprocessed Fortran source.
    """
    if merge:
        source = merge_used_modules(source, search_dirs=search_dirs)
    source = strip_openmp_directives(source)
    source = rewrite_integer_powers(source)
    if if_intvar:
        source = preprocess_fortran(source)
    return source
