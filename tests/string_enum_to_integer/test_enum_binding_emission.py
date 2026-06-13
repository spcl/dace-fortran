"""Coverage for the enum-aware binding emission.

Closes the Pattern 2 loop at the binding layer (per user's
instruction: "the binding should call flag=0 but accept only
flag='c'").  The SDFG itself takes only integer args; the binding
is the ONLY place where the string is accepted.

These tests feed ``emit_bindings`` an ``enum_maps`` table and
assert the emitted ``<entry>_bindings.f90`` has:

  * A ``CHARACTER(LEN=N)`` outer dummy declaration sized to the
    longest enum literal (NOT the SDFG's ``INTEGER`` type).
  * A local ``INTEGER(c_int) :: <arg>__enum`` scratch.
  * A ``SELECT CASE (<arg>) ... CASE ('lit', 'LIT') ...
    <arg>__enum = N ... END SELECT`` block in the wrapper body.
  * The SDFG call uses ``<arg>__enum`` (the converted integer),
    NOT the outer CHARACTER dummy.

No SDFG build / no f2py run -- this is a pure emitter test.  The
end-to-end run-through-Fortran path is covered downstream in the
existing binding integration suite once an enum-bearing kernel
lands there.
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
    """Frozen + Iface + Plan for ``run(out_val, flag)`` where
    ``flag`` is the enum-mapped CHARACTER on the caller side
    and ``INTEGER`` on the SDFG side.  The bridge already
    rewrote the kernel source via
    :func:`rewrite_string_enum_to_integer` so by the time we
    snapshot the iface, ``flag`` reads as INTEGER -- the
    binding layer overrides the outer decl back to CHARACTER
    using ``enum_maps``."""
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
            # Post-preprocess: the iface flang sees treats ``flag`` as
            # INTEGER.  The binding emitter restores the CHARACTER
            # outer surface via ``enum_maps`` -- without the override
            # the caller would have to pass an integer too.
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
    """The SELECT CASE matches both lowercase and uppercase variants of
    each literal -- matches the QE ``flag == 'c' .OR. flag == 'C'``
    shape the preprocess pass collapses to one integer entry."""
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
    """Unknown strings hit the ``CASE DEFAULT`` arm and assign a
    sentinel (-1).  The bridge sticks with the source's permissive
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
    # Find the call to the C-bound SDFG program.  Nested parens in
    # ``c_loc(out_val)`` make a single regex with ``.+?`` stop too
    # early, so we slice the source from the call line up to the
    # next ``end subroutine`` marker and check what's in that range.
    call_start = re.search(r"call\s+dace_program_run\s*\(", src, re.IGNORECASE)
    assert call_start, f"expected ``call dace_program_run(...)`` in:\n{src}"
    tail_start = call_start.end()
    end_match = re.search(r"end\s+subroutine", src[tail_start:], re.IGNORECASE)
    assert end_match, "no end subroutine after the call -- truncated source?"
    call_block_text = src[tail_start:tail_start + end_match.start()]
    assert "dace_enum_flag" in call_block_text, \
        f"expected dace_enum_flag in call args.  got:\n{call_block_text}"
    # The outer CHARACTER ``flag`` dummy must NOT be passed -- it has
    # an incompatible type for the integer C-bound parameter.  Allow
    # the substring ``flag`` only as part of ``dace_enum_flag``.
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
    # Plain INTEGER outer dummy survives.
    assert re.search(r"integer\s*\([^)]*\)\s*,\s*intent\s*\(\s*in\s*\)\s*,\s*target\s*::\s*flag\b",
                     src, re.IGNORECASE), \
        f"without enum_maps the outer dummy should stay INTEGER.  got:\n{src}"
    # No CHARACTER for flag.
    assert not re.search(r"character\s*\([^)]*\)\s*,\s*intent\s*\(\s*in\s*\)\s*::\s*flag\b",
                         src, re.IGNORECASE), \
        "without enum_maps no CHARACTER decl for flag"
    # No SELECT CASE.
    assert "select case (flag)" not in src.lower()


def test_enum_maps_missing_iface_arg_is_silent(tmp_path):
    """An ``enum_maps`` key that doesn't appear in ``iface.args``
    (e.g. an arg the flatten pass renamed or removed) is silently
    ignored -- no spurious decl, no SELECT CASE, no call-site
    substitution."""
    src = _emit(tmp_path, enum_maps={"not_an_arg": {"x": 0}})
    assert "not_an_arg" not in src
    assert "dace_enum_" not in src


# ---------------------------------------------------------------------------
# Synthetic-name collision-resistance
# ---------------------------------------------------------------------------


def test_synthesised_local_uses_dace_prefix_namespace(tmp_path):
    """The synthesised INTEGER scratch lives in the ``dace_`` namespace
    that the rest of the binding layer reserves for its own emitted
    names (``dace_handle``, ``dace_program_<entry>``, ...).  This
    avoids the false-collision risk of a ``<arg>__enum`` shape that a
    user kernel could plausibly use itself (``flag__enum``,
    ``column__enum``  --  both Fortran-legal identifiers).  A user
    variable explicitly named ``dace_enum_<arg>`` would still collide,
    but the convention puts that outside the user namespace and the
    resulting Fortran would fail to compile loudly rather than
    silently shadow."""
    src = _emit(tmp_path, enum_maps={"flag": {"c": 0}})
    assert "dace_enum_flag" in src
    # The old ``__enum`` shape is gone.
    assert "flag__enum" not in src
