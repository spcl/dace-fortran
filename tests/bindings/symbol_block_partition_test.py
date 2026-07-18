"""Regression: the wrapper's early/late symbol-population split must move each
``if (guard) then / sym=<shape> / else / sym=0 / end if`` guard block as a UNIT.

A presence-guarded, buffer-derived-extent symbol (ICON's deferred POINTER struct
members, e.g. ocean_state%p_prog(nold)%h) emits such a block whose guard AND true
branch both read the wrapper-allocated flat companion. Splitting per LINE sends
those two lines late and the bare ``else``/``sym=0``/``end if`` early, orphaning
the ELSE -- gfortran rejects it ("Unexpected ELSE statement"), which
non-deterministically broke the solve_free_sfc bindings compile.
"""
from dace_fortran.bindings.block_builders import partition_symbol_blocks


def _balanced(lines):
    depth = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("if (") and s.endswith("then"):
            depth += 1
        elif s == "end if":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _guard_block(guard, sym, shape):
    return [
        f"    if ({guard}) then", f"      {sym} = int({shape}, c_int)", "    else", f"      {sym} = 0", "    end if"
    ]


def test_buffer_guarded_block_moves_late_intact():
    # Guard AND true-branch both read the flat buffer 'hbuf' -> the whole block is late.
    sym_lines = ["    a = int(othermod, c_int)"] \
        + _guard_block("associated(hbuf)", "h_d0", "size(hbuf, dim=1)") \
        + ["    c = int(modval, c_int)"]
    early, late = partition_symbol_blocks(sym_lines, {"hbuf"})
    assert _balanced(early) and _balanced(late), (early, late)
    # Whole guard block lands in late (no partial block in either half).
    assert _guard_block("associated(hbuf)", "h_d0", "size(hbuf, dim=1)") == [l for l in late]
    assert early == ["    a = int(othermod, c_int)", "    c = int(modval, c_int)"]


def test_non_buffer_guarded_block_stays_early_intact():
    # Guard + shape read a module global, not a wrapper buffer -> whole block early.
    sym_lines = _guard_block("present(opt)", "k", "mod_extent")
    early, late = partition_symbol_blocks(sym_lines, {"hbuf"})
    assert _balanced(early) and _balanced(late)
    assert late == []
    assert early == sym_lines


def test_multiple_interleaved_buffer_guard_blocks():
    sym_lines = _guard_block("associated(hbuf)", "h_d0", "size(hbuf, dim=1)") \
        + ["    n = int(nproma, c_int)"] \
        + _guard_block("associated(vnbuf)", "vn_d0", "size(vnbuf, dim=1)")
    early, late = partition_symbol_blocks(sym_lines, {"hbuf", "vnbuf"})
    assert _balanced(early) and _balanced(late)
    assert early == ["    n = int(nproma, c_int)"]
    # Both blocks intact and late: two if/then, two end if, no orphan else.
    assert sum(1 for l in late if l.strip().endswith("then")) == 2
    assert sum(1 for l in late if l.strip() == "end if") == 2
    assert "    else" not in early


def test_old_per_line_split_would_shear_the_block():
    # Pin the failure mode the fix prevents: a naive per-line split orphans the ELSE.
    import re
    sym_lines = _guard_block("associated(hbuf)", "h_d0", "size(hbuf, dim=1)")
    buf = {"hbuf"}
    bd = lambda s: any(re.search(r'\b' + re.escape(b) + r'\b', s) for b in buf)
    bad_early = [l for l in sym_lines if not bd(l)]  # else / h_d0=0 / end if -> orphan
    assert not _balanced(bad_early), "per-line split must orphan the ELSE (else the test proves nothing)"
    # The block-aware partition does NOT.
    early, late = partition_symbol_blocks(sym_lines, buf)
    assert _balanced(early) and _balanced(late)
