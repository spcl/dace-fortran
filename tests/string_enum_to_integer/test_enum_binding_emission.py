"""Coverage for enum-aware binding emission -- closes the Pattern 2 loop at the binding
layer (the binding accepts ``flag='c'`` but calls the SDFG with ``flag=0``; the SDFG
itself takes only integer args).

Feeds ``emit_bindings`` an ``enum_maps`` table and asserts the emitted
``<entry>_bindings.f90`` has: a ``CHARACTER(LEN=N)`` outer dummy sized to the longest
literal (not the SDFG's INTEGER type); a local ``INTEGER(c_int) :: dace_enum_<arg>``
scratch; a ``SELECT CASE`` block converting string to int; and an SDFG call using
``dace_enum_<arg>``, not the outer CHARACTER dummy.

Pure emitter test -- no SDFG build / no f2py run.
"""
from pathlib import Path

from dace_fortran.bindings import (
    FlattenPlan,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)

# ---------------------------------------------------------------------------
# Fixtures: a kernel ``run(out_val, flag)`` where ``flag`` is the enum dummy
# ---------------------------------------------------------------------------


def _enum_kernel_signature(tmp_path: Path) -> tuple:
    """Frozen + Iface + Plan for ``run(out_val, flag)`` where ``flag`` is CHARACTER on
    the caller side, INTEGER on the SDFG side (already rewritten by
    :func:`rewrite_string_enum_to_integer`); the binding layer overrides the outer decl
    back to CHARACTER using ``enum_maps``."""
    frozen = FrozenSignature(
        entry="run",
        mangled="_QPrun",
        args=(
            FrozenArg(fortran_name="out_val",
                      sdfg_name="out_val",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("1", ),
                      intent="out"),
            FrozenArg(fortran_name="flag", sdfg_name="flag", kind="scalar", dtype="int32", rank=0, intent="in"),
        ),
        free_symbols=(),
    )
    iface = OriginalInterface(
        entry="run",
        args=(
            # post-preprocess flang sees flag as INTEGER; the emitter restores the
            # CHARACTER outer surface via enum_maps.
            OriginalArg(name="out_val", fortran_type="real(c_double)", rank=1, shape=("1", ), intent="out"),
            OriginalArg(name="flag", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
        struct_types={},
    )
    plan = FlattenPlan(entries=())
    return frozen, iface, plan


def _emit(tmp_path: Path, enum_maps: dict = None) -> str:
    """Run ``emit_bindings`` and return the rendered source text."""
    frozen, iface, plan = _enum_kernel_signature(tmp_path)
    out = tmp_path / "run_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out), dace_arglist=("out_val", "flag"), enum_maps=enum_maps)
    return out.read_text()


# ---------------------------------------------------------------------------
# Outer signature: CHARACTER replaces INTEGER for the enum arg
# ---------------------------------------------------------------------------


def test_enum_arg_outer_dummy_becomes_character(tmp_path):
    """The binding's outer ``flag`` dummy is CHARACTER, not INTEGER --
    the caller passes a string."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0, "r": 1, "i": 2}})
    # The outer dummy decl spans ``character(len=N), intent(in) :: flag``.
    import re
    assert re.search(r"character\s*\(\s*len\s*=\s*1\s*\)\s*,\s*intent\s*\(\s*in\s*\)\s*::\s*flag",
                     src, re.IGNORECASE), \
        f"expected ``character(len=1), intent(in) :: flag``.  got:\n{src}"
    # And the original ``integer(c_int) ... :: flag`` outer dummy is GONE.
    assert not re.search(
        r"integer\s*\([^)]*\)\s*,\s*intent\s*\(\s*in\s*\)\s*,\s*target\s*::\s*flag\b",
        src, re.IGNORECASE), \
        "outer dummy should be CHARACTER, not INTEGER+target"


def test_enum_arg_character_length_matches_longest_literal(tmp_path):
    """``CHARACTER(LEN=N)`` where ``N`` is the longest enum literal."""
    import re
    # Three literals of length 7, 8, 4: max length 8.
    src = _emit(tmp_path, enum_maps={"flag": {"forward": 0, "backward": 1, "zero": 2}})
    assert re.search(r"character\s*\(\s*len\s*=\s*8\s*\)", src, re.IGNORECASE), \
        f"expected ``character(len=8)`` for longest-literal sizing.  got:\n{src}"


# ---------------------------------------------------------------------------
# Internal scratch: integer(c_int) :: <arg>__enum
# ---------------------------------------------------------------------------


def test_enum_arg_internal_integer_scratch_is_declared(tmp_path):
    """A local ``integer(c_int) :: dace_enum_flag`` holds the converted
    integer that reaches the SDFG."""
    import re
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0}})
    assert re.search(r"integer\s*\(\s*c_int\s*\)\s*::\s*dace_enum_flag",
                     src, re.IGNORECASE), \
        f"expected ``integer(c_int) :: dace_enum_flag`` scratch decl.  got:\n{src}"


# ---------------------------------------------------------------------------
# Body: SELECT CASE conversion is emitted
# ---------------------------------------------------------------------------


def test_body_contains_select_case_with_both_casings(tmp_path):
    """SELECT CASE matches both lowercase and uppercase of each literal -- mirrors the
    QE ``flag == 'c' .OR. flag == 'C'`` shape the preprocess pass collapses to one entry."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0, "r": 1, "i": 2}})
    src_lower = src.lower()
    # SELECT CASE on the outer string dummy.
    assert "select case (flag)" in src_lower, \
        f"expected ``select case (flag)``.  got:\n{src}"
    # CASE ('c', 'C') / ('r', 'R') / ('i', 'I').
    assert "case ('c', 'c')" in src_lower or "case ('c','c')" in src_lower or \
           "case ('c'" in src_lower and "'C'" in src, \
        "expected case-insensitive CASE clause for 'c'"
    # The converted assignment ``dace_enum_flag = N`` follows.
    assert "dace_enum_flag = 0" in src_lower
    assert "dace_enum_flag = 1" in src_lower
    assert "dace_enum_flag = 2" in src_lower


def test_body_select_case_has_default_fallback(tmp_path):
    """Unknown strings hit ``CASE DEFAULT`` and assign a sentinel (-1) -- permissive
    default-fallthrough rather than synthesising an abort."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0}})
    src_lower = src.lower()
    assert "case default" in src_lower
    assert "dace_enum_flag = -1" in src_lower


def test_body_case_clauses_ordered_by_integer_value(tmp_path):
    """Stable, integer-ordered CASE branches help diff-readability
    and match the order ``rewrite_string_enum_to_integer`` assigned."""
    src = _emit(tmp_path, enum_maps={"flag": {"r": 1, "c": 0, "i": 2}})
    src_lower = src.lower()
    pos_c = src_lower.find("dace_enum_flag = 0")
    pos_r = src_lower.find("dace_enum_flag = 1")
    pos_i = src_lower.find("dace_enum_flag = 2")
    assert 0 < pos_c < pos_r < pos_i, \
        f"expected integer-ordered case branches; got positions c={pos_c} r={pos_r} i={pos_i}"


# ---------------------------------------------------------------------------
# SDFG call: integer scratch replaces the outer CHARACTER dummy
# ---------------------------------------------------------------------------


def test_sdfg_call_passes_integer_scratch_not_outer_character(tmp_path):
    """The SDFG call's actuals reference ``dace_enum_flag`` (the converted
    integer scratch), not ``flag`` (the outer CHARACTER dummy)."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0}})
    import re
    # nested parens in c_loc(out_val) make a single .+? regex stop too early -- slice
    # from the call line to the next `end subroutine` instead.
    call_start = re.search(r"call\s+dace_program_run\s*\(", src, re.IGNORECASE)
    assert call_start, f"expected ``call dace_program_run(...)`` in:\n{src}"
    tail_start = call_start.end()
    end_match = re.search(r"end\s+subroutine", src[tail_start:], re.IGNORECASE)
    assert end_match, "no end subroutine after the call -- truncated source?"
    call_block_text = src[tail_start:tail_start + end_match.start()]
    assert "dace_enum_flag" in call_block_text, \
        f"expected dace_enum_flag in call args.  got:\n{call_block_text}"
    # outer CHARACTER flag must not be passed (incompatible type); allow the substring
    # only as part of dace_enum_flag.
    leftover = call_block_text.replace("dace_enum_flag", "")
    assert not re.search(r"\bflag\b", leftover), \
        f"outer CHARACTER 'flag' must not appear in the SDFG call args (only dace_enum_flag).  call args:\n{call_block_text}"


# ---------------------------------------------------------------------------
# Pass-through: no enum_maps -> emitter behaviour unchanged
# ---------------------------------------------------------------------------


def test_empty_enum_maps_emits_integer_outer_dummy_as_before(tmp_path):
    """Without ``enum_maps`` the emitter behaves identically to the
    pre-feature path: the outer dummy stays INTEGER."""
    src = _emit(tmp_path, enum_maps=None)
    import re
    assert re.search(r"integer\s*\([^)]*\)\s*,\s*intent\s*\(\s*in\s*\)\s*,\s*target\s*::\s*flag\b",
                     src, re.IGNORECASE), \
        f"without enum_maps the outer dummy should stay INTEGER.  got:\n{src}"
    assert not re.search(r"character\s*\([^)]*\)\s*,\s*intent\s*\(\s*in\s*\)\s*::\s*flag\b",
                         src, re.IGNORECASE), \
        "without enum_maps no CHARACTER decl for flag"
    assert "select case (flag)" not in src.lower()


def test_enum_maps_missing_iface_arg_is_silent(tmp_path):
    """An ``enum_maps`` key absent from ``iface.args`` (e.g. an arg the flatten pass
    renamed/removed) is silently ignored -- no spurious decl, SELECT CASE, or call-site
    substitution."""
    src = _emit(tmp_path, enum_maps={"not_an_arg": {"x": 0}})
    assert "not_an_arg" not in src
    assert "dace_enum_" not in src


# ---------------------------------------------------------------------------
# Synthetic-name collision-resistance
# ---------------------------------------------------------------------------


def test_synthesised_local_uses_dace_prefix_namespace(tmp_path):
    """Synthesised INTEGER scratch lives in the ``dace_`` namespace the binding layer
    reserves for its own emitted names -- avoids false collision with a user-plausible
    ``<arg>__enum`` shape (``flag__enum`` is Fortran-legal).  A user variable named
    exactly ``dace_enum_<arg>`` would still collide, but that's outside user namespace
    and fails loudly rather than silently shadowing."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0}})
    assert "dace_enum_flag" in src
    # The old ``__enum`` shape is gone.
    assert "flag__enum" not in src
