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
from pathlib import Path
from typing import Iterable, Optional

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
_OMP_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"][^>"]*omp_definitions[^>"]*\.inc[>"]\s*$', re.IGNORECASE)

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
        # ``stack`` holds one ``(is_omp_acc, dropping)`` per open cpp
        # conditional.  ``is_omp_acc`` marks blocks gated on a macro in
        # ``_OMP_ACC_MACROS``; ``dropping`` is whether the *current* arm
        # is the one to elide.  A non-OMP ``#if`` pushes ``(False, False)``
        # so its lines and its ``#else`` / ``#elif`` pass through verbatim
        # -- those branches below intentionally do nothing for a
        # ``(False, ...)`` top-of-stack (the only ``else`` is "emit the
        # line").
        m_ifdef = _CPP_IFDEF_RE.match(line)
        m_ifdef_paren = _CPP_IFDEFINED_RE.match(line) if not m_ifdef else None
        if m_ifdef:
            kind, macro = m_ifdef.group(1).lower(), m_ifdef.group(2)
            if macro in _OMP_ACC_MACROS:
                # _OPENMP/_OPENACC are undefined: ``#ifdef`` arm drops,
                # ``#ifndef`` arm keeps.
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
                # KNOWN LIMITATION: in ``#ifdef _OPENMP / #elif FOO``, the
                # ``#elif FOO`` arm is dropped too -- the macro is treated
                # purely as undefined, so a sibling arm gated on an
                # unrelated ``FOO`` is discarded.  Acceptable because
                # OMP/ACC blocks rarely carry a meaningful ``#elif``.
                stack[-1] = (True, True)
                continue
        elif _CPP_ENDIF_RE.match(line):
            # Pop on close.  An unbalanced ``#endif`` (empty stack -- e.g.
            # its ``#if`` predates ``source``) is tolerated as a no-op.
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


# Default precision-kind aliases.  Each maps an unresolved kind symbol
# (one not locally bound to a literal integer) to the IEEE byte width
# the bridge should substitute.  Covers the four conventions seen in
# climate / NWP code: ``wp`` (working precision, ECMWF / ICON / CLOUDSC),
# ``sp``/``dp`` (single / double, generic), ``qp`` (quad), ``rp`` (real
# precision, less common).  ``kind_map=`` to ``normalize_kind_parameters``
# both extends and overrides this table; pass ``None`` for an alias to
# leave it untouched.
_DEFAULT_KIND_ALIASES = {
    "wp": 8,
    "rp": 8,
    "sp": 4,
    "dp": 8,
    "qp": 16,
}

# ``KIND = sym`` -- with arbitrary spacing.  Group 1 is the
# ``KIND<sp>=<sp>`` prefix (preserved verbatim), group 2 the symbol.
_KIND_EQ_RE = re.compile(r"\b(KIND\s*=\s*)([A-Za-z_]\w*)\b", re.IGNORECASE)

# ``REAL(sym)`` / ``INTEGER(sym)`` / ``COMPLEX(sym)`` / ``LOGICAL(sym)``
# -- the sole-argument numeric-type-spec form.  Group 1 = type keyword,
# group 2 = ``(<sp>``, group 3 = symbol, group 4 = ``<sp>)``.
_TYPE_PAREN_RE = re.compile(
    r"\b(REAL|INTEGER|COMPLEX|LOGICAL)(\s*\(\s*)([A-Za-z_]\w*)(\s*\))",
    re.IGNORECASE,
)

# Literal-kind suffix: ``1.0_wp``, ``1.0E0_wp``, ``1_wp``.  Group 1 is
# the numeric portion, group 2 the kind symbol.  A leading word
# boundary keeps the match off the middle of identifiers.
_LITERAL_KIND_RE = re.compile(r"\b(\d+(?:\.\d*)?(?:[eEdD][+-]?\d+)?)_([A-Za-z_]\w*)\b")

# ``INTEGER, PARAMETER ... :: sym = <int_literal>`` -- the locally-bound
# form the bridge can already evaluate.  Aliases caught here are dropped
# from the substitution set (the local binding wins).  Any non-integer
# RHS (``SELECTED_REAL_KIND(...)``, ``KIND(0.0D0)``) doesn't match and
# falls through to the rewrite, which is exactly the target case.
_PARAM_BIND_RE = re.compile(r"\bINTEGER\b[^:\n]*::\s*([A-Za-z_]\w*)\s*=\s*(\d+)\b", re.IGNORECASE)


def _local_kind_bindings(source: str) -> dict:
    """Collect locally-defined ``INTEGER, PARAMETER :: sym = <int>``
    bindings.

    Only the literal-integer form is captured -- those are the bindings
    flang already lowers without help.  Any other RHS (intrinsics,
    arithmetic) is opaque to this scan and falls through to the
    alias-substitution path.

    :param source: full Fortran source text.
    :returns: ``{lowercase-symbol: literal int}``.
    """
    out: dict = {}
    for raw in source.splitlines():
        m = _PARAM_BIND_RE.search(_code_of(raw))
        if m:
            out.setdefault(m.group(1).lower(), int(m.group(2)))
    return out


def normalize_kind_parameters(source: str, *, kind_map: dict = None, passthrough: bool = False) -> str:
    """Substitute symbolic precision kind aliases with literal kind ints.

    Climate / NWP Fortran tends to thread one symbolic kind alias
    (``wp`` in CLOUDSC / ICON, ``JPRB`` via ECMWF's ``PARKIND1``,
    ``REAL_KIND`` in legacy ECRAD) through every type spec
    (``REAL(KIND=wp)``, ``REAL(wp)``) and every numeric literal
    (``1.0_wp``).  The kind alias is itself defined in a tiny
    constants module via ``SELECTED_REAL_KIND`` / ``KIND(0.0D0)`` and
    pulled in via ``USE``.  flang resolves these aliases at parse time
    *only if* the defining module is in the translation unit -- when
    the bridge runs against a single-file slice (a probe, a kernel
    extracted for a microbenchmark) the constants module is absent and
    flang errors out.

    This rewrite makes single-file slices self-contained: every
    unresolved kind alias is replaced by the literal IEEE byte width
    (default fp64 ``8``) at every use site.  Locally-bound integer
    parameter aliases (``INTEGER, PARAMETER :: wp = 8``) are skipped --
    flang already handles those.

    Integrating into an existing build pipeline that already supplies
    the constants module (and so resolves ``wp`` itself):

    1. ``normalize_kind_parameters`` is **idempotent and no-op-safe**.
       A second pass over already-substituted source finds no
       symbolic aliases left and returns the input unchanged; running
       it on source where ``wp`` is locally bound to an integer
       literal also returns the input unchanged.  It is therefore
       safe to leave default-on in the bridge even when the upstream
       pipeline resolves kinds.
    2. To bind a single alias to a non-default precision (e.g. an
       fp32 build that defines ``wp = 4``), pass
       ``kind_map={"wp": 4}``.
    3. To leave one specific alias alone (the upstream pipeline
       resolves it correctly and the default would be wrong), pass
       ``kind_map={"wp": None}``.
    4. To disable the rewrite entirely (upstream guarantees every
       kind is already a literal), pass ``passthrough=True``.

    The pass is comment- and string-aware via ``_scan_line``: a kind
    alias that appears inside a character literal or a ``!`` comment
    is never touched.

    :param source: Fortran source text.
    :param kind_map: optional override / extension of
        ``_DEFAULT_KIND_ALIASES``; per-alias ``None`` disables it.
    :param passthrough: ``True`` returns ``source`` unchanged.
    :returns: source with kind aliases replaced by literal integers.
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

#: cpp conditional directives, classified for balancing an extracted block.
_CPP_OPEN_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b", re.IGNORECASE)
_CPP_CLOSE_RE = re.compile(r"^\s*#\s*endif\b", re.IGNORECASE)
_CPP_MID_RE = re.compile(r"^\s*#\s*(?:else|elif)\b", re.IGNORECASE)


def _balance_cpp(block: str) -> str:
    """Drop cpp conditional directives left unbalanced by module-block
    extraction.  A whole-module ``#ifdef GUARD ... MODULE ... END MODULE ...
    #endif`` wrapper splits across the block boundary -- the opener lands in one
    block's preamble and the ``#endif`` in the next block's -- leaving each
    block with an orphan ``#if`` or ``#endif`` that breaks cpp once the blocks
    are concatenated into one TU.  Remove the unmatched directives (and orphan
    ``#else`` / ``#elif``), keeping their guarded content: every module pulled
    into a merged USE-closure was already selected by the real build, so it is
    wanted unconditionally.  Conditionals fully contained in the block (a normal
    in-body ``#if/#else/#endif``) stay balanced and untouched."""
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
        # Balance cpp conditionals: a whole-module ``#ifdef..#endif`` wrapper
        # splits across the block boundary (opener in this block's preamble,
        # ``#endif`` swept into the next block's), so drop the orphan side.
        yield name, _balance_cpp("".join(lines[start:end + 1]))
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


#: A ``SUBROUTINE`` / ``FUNCTION`` opener (any leading prefix keywords +
#: typed-function forms), capturing the procedure name.  Shared by the
#: procedure indexer and the external-procedure body stubber.
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

#: An ``INTERFACE`` block opener (named / operator / ``ABSTRACT`` forms).  Its
#: nested ``SUBROUTINE`` / ``FUNCTION`` declarations are specification part, not
#: the enclosing procedure's executable body -- the stubber consumes the whole
#: ``INTERFACE`` ... ``END INTERFACE`` block as a unit so those nested openers
#: are not mistaken for the start of the body (the ICON ``bind(c)`` halo
#: wrappers that forward to a C++ impl carry exactly this shape).
_INTERFACE_OPEN_RE = re.compile(r"^\s*(?:ABSTRACT\s+)?INTERFACE\b", re.IGNORECASE)
_INTERFACE_END_RE = re.compile(r"^\s*END\s*INTERFACE\b", re.IGNORECASE)

#: A specification-part statement (declarations / attributes / interfaces) --
#: everything legal *before* the first executable statement of a procedure.
#: The external-body stubber keeps these (so the empty stub still declares its
#: dummy arguments) and drops everything after the first non-matching line.  A
#: leading ``&`` keeps continuation lines of a multi-line declaration / opener.
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
    """Empty the executable body of every procedure whose name matches
    ``names`` -- exactly, or as the generic an ICON interface dispatches over
    (``sync_patch_array`` -> ``sync_patch_array_3d_dp``) -- keeping its opener,
    specification part, and matching ``END``.

    The regex-merge analogue of the fparser inliner's ``make_noop``
    (:func:`dace_fortran.fparser_inliner._keep_external_noop_specs`): a kept-
    external procedure stays *declared* (its dummy arguments are still typed,
    so the in-TU call site is legal Fortran), but its internals -- halo
    exchange, MPI, I/O -- never enter the translation unit.  The bridge then
    lowers the call through its external registry.  A stubbed procedure's whole
    body is dropped -- including any internal ``CONTAINS`` subprograms (the
    nesting-aware ``END`` scan keeps only the procedure's own closing ``END``);
    external halo/sync procedures are leaves, so this rarely matters.  Matching
    is case-insensitive and nothing ICON-specific is hardcoded -- the names
    come from the caller's policy."""
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
        # Keep the opener, then the contiguous specification part (so the dummy
        # arguments stay declared); stop at the first executable statement, a
        # nested subprogram, ``CONTAINS``, or the matching ``END``.
        out.append(lines[i])
        i += 1
        while i < n:
            c = _code_of(lines[i].rstrip("\r\n"))
            if _INTERFACE_OPEN_RE.match(c):
                # Keep the whole INTERFACE block verbatim (nesting-aware): its
                # nested procedure declarations are spec, not the body, so the
                # ``_PROC_OPEN_RE`` lines inside must NOT end spec collection.
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
        # Drop everything up to (and keep) the matching ``END`` of this
        # procedure, tracking nesting so an internal subprogram's own ``END``
        # is not mistaken for it.
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

    ``external_functions`` / ``do_not_emit`` are the external-function policy
    (see :mod:`dace_fortran.external_functions`): names of procedures that must
    NOT be inlined.  When their defining module is spliced in, their bodies are
    stubbed to empty (:func:`_stub_procedure_bodies`) so the halo-exchange /
    MPI / I/O internals never enter the TU -- the regex-merge parallel of the
    fparser inliner's ``make_noop``.  The bridge lowers the surviving call
    through its external registry.

    :param source: the entry Fortran source text.
    :param search_dirs: directories scanned (recursively, ``*.f90`` /
        ``*.F90`` / ``*.incf``) for module definitions.
    :param external_functions: :class:`ExternalFunction` specs (don't-inline +
        the bridge EMITs an external call); only their ``name`` is read here.
    :param do_not_emit: plain names (don't-inline + the bridge DROPs the call).
    :returns: a single-TU source, or ``source`` unchanged.
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

    # Post-order DFS toposort (deps emitted before their dependents):
    # each name is pushed once unexpanded, then re-pushed ``expanded``
    # so it is appended to ``order`` only after its deps were visited
    # (the classic gray/black marking).  A ``USE`` cycle drops its
    # back-edge silently -- the already-``placed`` node is skipped --
    # which is fine for well-formed Fortran (acyclic module graph).
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
    merged = "".join(parts)
    return _stub_procedure_bodies(merged, dont_inline) if dont_inline else merged


# ``EXTERNAL`` declaration: ``EXTERNAL`` keyword followed by zero or
# more attribute commas, optional ``::``, then a comma-separated name
# list.  Anchored at the leading-whitespace start of a line so it
# never matches mid-statement.  Captures: group(1) = the name list as
# a raw string (commas + names + optional whitespace).
_EXTERNAL_DECL_RE = re.compile(
    r"^(?P<indent>\s*)EXTERNAL\s*(?:::\s*)?(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)

# A ``SUBROUTINE`` / ``FUNCTION`` opener at the start of a scope.  The
# pass inserts the synthesised ``USE`` lines right after this opener
# (and after any ``USE`` lines already present, so the insertion
# stacks with the existing imports rather than displacing them).
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
    """Scan ``search_dirs`` recursively for ``.f90`` / ``.F90`` /
    ``.incf`` files; index every ``SUBROUTINE`` / ``FUNCTION`` declared
    inside a ``MODULE`` block by lowercase procedure name.

    Sibling of :func:`merge_used_modules`'s module-block scan: walks
    the same files but indexes the procedures within each module
    instead of inlining the module bodies.  Used by
    :func:`replace_external_with_modules` to resolve every ``EXTERNAL
    <name>`` declaration to its defining module.

    :param search_dirs: iterable of directories (each scanned
        recursively) and / or individual file paths.
    :returns: dict ``{procedure_name_lower: module_name_lower}``.
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


# A type declaration that names ONE function result without any
# variable-defining attributes (no ``INTENT``, no ``::`` followed by
# more than the function name, no array spec).  Used to detect the
# ``REAL(8) :: dscale`` companion that often sits next to ``EXTERNAL
# :: dscale``; once ``dscale`` is imported via ``USE``, this line
# becomes a "use-associated, cannot be re-declared" error in flang.
_FUNC_RESULT_TYPE_DECL_RE = re.compile(
    r"^\s*(?P<type>(?:REAL|INTEGER|LOGICAL|COMPLEX|DOUBLE\s+PRECISION)"
    r"(?:\s*\(\s*[^)]*\s*\))?)\s*"
    r"(?:::\s*)?"
    r"(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)


def _scope_already_uses(scope_lines, mod_name: str) -> bool:
    """``True`` when one of ``scope_lines`` is ``USE <mod_name>`` (any
    casing, with or without ``ONLY``).  Used to suppress duplicate
    imports: if the kernel already imports the module under a plain
    ``USE``, the synthesised ``USE module, ONLY: name`` would be
    redundant (and some compilers warn on it)."""
    target = mod_name.lower()
    for raw in scope_lines:
        m = _USE_RE.match(_code_of(raw))
        if m and m.group(1).lower() == target:
            return True
    return False


def replace_external_with_modules(source: str, *, search_dirs=()) -> str:
    """Replace ``EXTERNAL <name1>, <name2>`` declarations with the
    equivalent ``USE <module>, ONLY: <name1>, <name2>`` imports
    whenever the referenced procedures are defined in modules visible
    via ``search_dirs``.

    Background:
        Legacy Fortran code -- and machine-translated code like
        QE's ``f2dace-qe-source`` pruned single-TU sources --
        often declares one procedure ``EXTERNAL`` and another via
        ``USE``, even when both procedures live in the same
        module that's reachable from the build's source tree.
        ``EXTERNAL`` carries only the linker-symbol promise;
        flang then routes the call through its implicit-interface
        path, which the bridge can't lower as faithfully as a
        proper module import (type / shape promises lost,
        polymorphism inferred per call site, etc.).  Converting
        every resolvable ``EXTERNAL`` to a ``USE`` restores the
        explicit interface and unblocks downstream lowering.

    Algorithm:
        1. Index every procedure declared inside any ``MODULE``
           block in ``search_dirs`` (recursive).
        2. Find each ``EXTERNAL`` line in ``source``; resolve each
           name through the index to a defining module.
        3. Within the enclosing scope (the ``SUBROUTINE`` /
           ``FUNCTION`` containing the ``EXTERNAL`` line, or the
           top-level module-procedure scope), group resolved
           names by their defining module and synthesise one
           ``USE <mod>, ONLY: <names...>`` line per module.
        4. Insert the synthesised ``USE`` lines right after the
           scope opener (collapsing with any existing ``USE``
           imports of the same module so the resulting source
           has no duplicate imports).
        5. Delete every ``EXTERNAL`` declaration whose names
           were fully resolved.  An ``EXTERNAL`` line carrying
           any unresolved name is left in place verbatim (with
           the resolved names also left in the declaration) --
           the pass is conservative: an unresolved ``EXTERNAL``
           means the build is missing a source file the user
           knows about.

    Pass-through:
        Returns ``source`` unchanged when ``search_dirs`` is empty
        or no ``EXTERNAL`` declaration in ``source`` resolves to a
        module in the index.  Idempotent: a second invocation has
        nothing left to rewrite.  Comment- and string-aware via
        ``_scan_line``: an ``EXTERNAL`` token inside a character
        literal or after ``!`` is never touched.

    :param source: Fortran source text.
    :param search_dirs: directories (recursive) and / or file paths
        scanned for procedure definitions, same convention as
        ``merge_used_modules``'s ``search_dirs``.
    :returns: rewritten source with every resolvable ``EXTERNAL``
        replaced by the equivalent module import.
    """
    if not search_dirs:
        return source
    index = _index_procedures_in_modules(search_dirs)
    if not index:
        return source

    lines = source.splitlines(keepends=True)

    # Pass 1: identify scope-opener line indices and the ``EXTERNAL``
    # lines that belong to each scope.  A scope opens at a
    # ``SUBROUTINE`` / ``FUNCTION`` line and closes at the next one;
    # ``END SUBROUTINE`` / ``END FUNCTION`` closes the current scope
    # without opening a new one.
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

    # Pass 2: per scope, resolve the EXTERNAL names + build the
    # USE-line additions and EXTERNAL-line deletions.
    delete_idx = set()
    insert_at_idx: dict = {}  # opener_idx -> list of synthesized USE lines
    for opener_idx, end_idx in scopes:
        scope_lines = lines[opener_idx:end_idx]
        # First pass: collect EXTERNAL declarations + their resolution.
        # Group resolved names by defining module.
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
        # Function-result type declarations whose name is now
        # use-associated must also be deleted: flang rejects
        # ``REAL(8) :: dscale`` once ``dscale`` is imported via
        # ``USE``.  Only a single-name declaration is safe to drop
        # whole; a mixed ``REAL(8) :: dscale, scratch_local`` is
        # left alone (would need a more surgical rewrite to split
        # the line, which is rare enough to defer).
        for li, raw in enumerate(scope_lines, start=opener_idx):
            code = _code_of(raw)
            # Skip lines that are EXTERNAL / INTENT-bearing decls /
            # any ``::``-attribute-bearing declarations -- the
            # function-result-decl regex matches only the bare
            # ``<type> :: <name>`` shape with no attributes.
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
            # Only drop when EVERY name on this line is a resolved
            # external -- a mixed line keeps the bridge safe.
            if all(n in all_resolved_names for n in decl_names):
                delete_idx.add(li)
        # Suppress imports for modules the scope already USEs.
        synth_use_lines = []
        for mod, names in per_mod_names.items():
            if _scope_already_uses(scope_lines, mod):
                continue  # EXTERNAL becomes a no-op delete
            # Pick the indent matching the opener's leading whitespace
            # plus two spaces -- the conventional Fortran continuation.
            opener_indent = re.match(r"^(\s*)", lines[opener_idx]).group(1)
            body_indent = opener_indent + "  "
            synth_use_lines.append(f"{body_indent}USE {mod}, ONLY: {', '.join(names)}\n")
        if synth_use_lines:
            insert_at_idx[opener_idx + 1] = synth_use_lines
        for idx in ext_line_idxs:
            delete_idx.add(idx)

    if not delete_idx and not insert_at_idx:
        return source

    # Pass 3: produce the rewritten source.  Walk every line; when an
    # index has an insertion, emit the inserts BEFORE the line; when
    # an index is in delete_idx, drop it.
    out = []
    for i, raw in enumerate(lines):
        if i in insert_at_idx:
            out.extend(insert_at_idx[i])
        if i in delete_idx:
            continue
        out.append(raw)
    return "".join(out)


# A CHARACTER dummy declaration -- the LHS shape of a procedure
# parameter that's a string used as an enum.  Captures the dummy
# names (group ``names``) and the full type spec (group ``type``).
# Recognises ``CHARACTER(LEN=*)``, ``CHARACTER(LEN=N)``,
# ``CHARACTER(*)``, ``CHARACTER(N)``, and the bare ``CHARACTER``
# form -- all of which are valid Fortran string-arg shapes.
_CHARACTER_INTENT_IN_RE = re.compile(
    r"^(?P<lead>\s*)CHARACTER(?:\s*\(\s*[^)]+\s*\))?\s*"
    r",\s*INTENT\s*\(\s*IN\s*\)\s*"
    r"::\s*(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*$",
    re.IGNORECASE,
)


def _line_clip_comment(raw: str) -> str:
    """Return ``raw`` truncated at the first ``!`` that is outside a
    character literal.  Unlike :func:`_code_of` this preserves the
    contents of character literals (which the string-enum detector
    NEEDS to capture)."""
    cut, _ = _scan_line(raw)
    return raw[:cut]


def _scan_string_enum_uses(scope_body: str, var: str) -> dict:
    """Return the enum mapping ``{lowercase_literal: int}`` for every
    distinct literal ``var`` is compared against.

    Recognises three shapes within ``scope_body``:
      * ``<var> == '<literal>'``  /  ``<var> .EQ. '<literal>'``
      * ``'<literal>' == <var>``  /  ``'<literal>' .EQ. <var>``
      * ``CASE ('<literal>')`` inside a ``SELECT CASE (<var>)`` block
        (the block scope is tracked by ``_select_case_blocks``).

    Case-insensitive grouping is applied so a Fortran source that
    accepts both ``'c'`` and ``'C'`` collapses to one integer entry
    (the same downstream branch fires for either).  Literals are
    enumerated in first-appearance order so the resulting map is
    deterministic between runs.
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

    # ``SELECT CASE (<var>)`` block scan: collect CASE-branch literals
    # within the block.  Single-pass; nested SELECT CASE on different
    # variables don't get conflated because we re-anchor on each
    # block opener.
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
    """Convert ``CHARACTER(LEN=...), INTENT(IN) :: <var>`` dummy args
    that act as enum-style switches to ``INTEGER, INTENT(IN)`` plus
    integer-valued comparisons.

    Pattern (QE's ``addusxx_g`` ``flag`` shape):

      SUBROUTINE k(out, flag)
        CHARACTER(LEN=1), INTENT(IN) :: flag
        IF (flag == 'c' .OR. flag == 'C') THEN ...
        IF (flag == 'r' .OR. flag == 'R') THEN ...
      END SUBROUTINE

    becomes

      SUBROUTINE k(out, flag)
        INTEGER, INTENT(IN) :: flag
        IF (flag == 0 .OR. flag == 0) THEN ...
        IF (flag == 1 .OR. flag == 1) THEN ...
      END SUBROUTINE

    where ``0 -> {'c', 'C'}``, ``1 -> {'r', 'R'}``, etc. (the case-
    insensitive grouping collapses upper/lower variants to the
    same integer).

    Sidecar mapping:
        Returns ``(rewritten_source, enum_maps)`` where ``enum_maps``
        is ``{procedure_name: {arg_name: {literal_lower: int}}}``.
        The bindings layer consumes ``enum_maps`` to expose a string-
        typed wrapper (``run(flag='c', ...)``) at the Python boundary
        and normalises the string to the integer value before calling
        the SDFG.

    Limitations (deliberately scope-narrow for the first cut):
      * Only single-name CHARACTER declarations  --  a mixed
        ``CHARACTER :: a, b`` is left alone (would need a per-name
        split).
      * Only INTENT(IN) dummies  --  locals, INTENT(OUT) / (INOUT)
        and module variables are skipped (could be enum-promoted
        too but the call-site rewrite story is harder).
      * Call sites are NOT rewritten  --  the bindings layer
        converts the string argument to the integer at the SDFG
        boundary.  A caller invoking the rewritten Fortran
        directly would have to pass an integer literal.
      * SELECT CASE on TRIM(<var>) etc. is not detected  --  only
        the bare ``SELECT CASE (<var>)`` shape.

    Pass-through:
        Returns ``(source, {})`` when no procedure has an INTENT(IN)
        CHARACTER dummy that's used purely as an enum switch.
        Idempotent: a second invocation finds no CHARACTER-INTENT(IN)
        dummies left to rewrite and returns its input + ``{}``.
        Comment- and string-aware via ``_scan_line``: an ``==``
        comparison inside a string literal or after ``!`` is never
        treated as code.

    :param source: Fortran source text.
    :returns: ``(rewritten_source, enum_maps)``.
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

        # For each candidate dummy, derive its enum mapping; skip if
        # the scope never compares it against any string literal.
        for li, var, lead in decl_targets:
            mapping = _scan_string_enum_uses(scope_body, var)
            if not mapping:
                continue

            # Record the mapping for the bindings layer.
            enum_maps.setdefault(proc_name, {})[var] = dict(mapping)

            # Rewrite the declaration line: CHARACTER... -> INTEGER.
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
                            # Rebuild line preserving any trailing
                            # comment past the match end.
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


def _fparser_merge(source: str, *, search_dirs=(), entry: Optional[str] = None,
                   external_names: Iterable[str] = ()) -> str:
    """Single-TU merge via the fparser inliner engine (opt-in).

    Sibling of :func:`merge_used_modules`: instead of the regex
    text-splicer, this parses ``source`` plus every Fortran file under
    ``search_dirs`` into one fparser AST, resolves the ``USE`` graph,
    inlines the needed modules, runs the desugaring pipeline, and
    serialises back to a single ``.f90`` text.  Selected by
    ``merge_engine="fparser"``; the regex engine stays the default.

    ``entry`` (when given) restricts pruning to the entry's USE-closure;
    ``None`` keeps every top-level subprogram (a faithful whole-project
    single-TU merge).

    ``external_names`` are procedures to keep external (NOT inlined): each is
    stubbed to an empty body so its internals never enter the TU (the inliner's
    ``make_noop`` path).  The build path sources these from the bridge's
    external registry, so a ``keep_external`` / ``apply_external_functions``
    declaration drives both the merge and the SDFG-construction step.
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
    # The multi-file build stages the root file (the one defining the entry)
    # into ``search_dirs`` as well, so it is already in ``src_map``; only
    # inject ``source`` under a synthetic key when it is NOT already present
    # (the single-source build passes no staged files).  Injecting it
    # unconditionally would duplicate every procedure in the root file.
    if not any(txt == source for txt in src_map.values()):
        src_map["__entry__.f90"] = source
    # Inject the intrinsic-module stubs so the inliner's fparser parse
    # resolves ``USE iso_c_binding`` / ``iso_fortran_env`` (it hard-requires
    # every ``USE``-d module in the closure).  The stubs do NOT leak into the
    # output fed to flang: the inliner prunes the unreachable stub modules and
    # resolves intrinsic kinds inline (``REAL(c_double)`` -> ``REAL(KIND=8)``),
    # so flang still supplies its own intrinsic modules without a collision.
    # ``optimize=False``: the merge only needs a valid inlined single TU --
    # flang and the bridge do their own constant-folding / dead-branch
    # elimination, so skip the inliner's const-propagation optimizers (which
    # also matches the legacy regex merge's "splice and let flang inline"
    # semantics and avoids optimizer fragilities on inlined-call patterns).
    ast = inline_to_ast(src_map, entry, include_builtins=True, optimize=False,
                        do_not_emit=external_names)
    # A whole-project merge (``entry is None``) keeps every top-level unit,
    # including the injected intrinsic-module stubs; strip them so flang's own
    # ``iso_c_binding`` / ``iso_fortran_env`` are used without a collision.
    # (No-op when an entry point already pruned them.)
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
    """Single entrypoint for all Fortran-source preprocessing the HLFIR
    frontend applies before handing source to flang.

    Composes, in order:

    1. ``merge_used_modules`` (when ``merge``) -- inline every
       externally-``USE``-d module into one translation unit.
    2. ``strip_openmp_directives`` -- drop OpenMP / OpenACC sentinels
       and the ICON ``omp_definitions.inc`` cpp include (the bridge
       does not run cpp and does not pass ``-fopenmp``).
    3. ``normalize_kind_parameters`` -- substitute unresolved precision
       aliases (``wp``, ``sp``, ``dp``, ``qp``) with literal kind ints
       (default fp64).  No-op when every alias is locally bound or the
       caller passes ``kind_passthrough=True``.
    4. ``rewrite_integer_powers`` -- expand integer-valued real powers
       (``x**2.0`` -> ``x*x``) for byte-identical arithmetic.
    5. ``preprocess_fortran`` (when ``if_intvar``) -- the opt-in
       ``IF (intvar)`` -> ``IF (intvar /= 0)`` rewrite.

    The individual stages remain importable for their unit tests; this
    is the production composition.

    :param source: Fortran source text.
    :param search_dirs: directories searched by ``merge_used_modules``.
    :param merge: run the ``USE``-merge stage (default on).
    :param merge_engine: which merge implementation to use --
        ``"regex"`` (default) is the fparser-free
        :func:`merge_used_modules` text-splicer; ``"fparser"`` routes
        through :func:`_fparser_merge`, the fparser AST inliner
        (:mod:`dace_fortran.fparser_inliner`), which additionally desugars
        and prunes.
    :param merge_entry: entry procedure for the ``"fparser"`` engine's
        pruning (plain name / ``module::proc`` / mangled symbol); ``None``
        keeps every top-level subprogram.  Ignored by the regex engine.
    :param external_names: procedures to keep external (NOT inlined) at the
        merge stage -- their bodies are stubbed to empty in BOTH engines (the
        regex :func:`_stub_procedure_bodies` / the fparser ``make_noop``), so
        halo-exchange / MPI / I/O internals never enter the TU.  The build path
        sources these from the bridge's external registry so a single
        ``keep_external`` / ``apply_external_functions`` declaration governs
        both the merge and the SDFG-construction step (no hand-syncing).
    :param if_intvar: also run the opt-in integer-IF rewrite.
    :param kind_map: per-alias override forwarded to
        ``normalize_kind_parameters`` -- ``{"wp": 4}`` for an fp32
        build, ``{"wp": None}`` to leave one alias alone, etc.
    :param kind_passthrough: ``True`` skips ``normalize_kind_parameters``
        entirely -- for build pipelines that already resolve every kind
        alias upstream and want the bridge to assume nothing.
    :returns: the fully preprocessed Fortran source.
    """
    if merge:
        if merge_engine == "fparser":
            source = _fparser_merge(source, search_dirs=search_dirs, entry=merge_entry,
                                    external_names=external_names)
        elif merge_engine == "regex":
            source = merge_used_modules(source, search_dirs=search_dirs,
                                        do_not_emit=external_names)
        else:
            raise ValueError(f"merge_engine must be 'regex' or 'fparser', got {merge_engine!r}")
    source = strip_openmp_directives(source)
    source = normalize_kind_parameters(source, kind_map=kind_map, passthrough=kind_passthrough)
    source = rewrite_integer_powers(source)
    if if_intvar:
        source = preprocess_fortran(source)
    return source
