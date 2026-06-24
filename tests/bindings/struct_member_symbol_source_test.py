"""A struct dummy's member used ONLY symbolically -- a loop bound
(``do ig = 1, dfftt%ngm``) or an array extent
(``g(3, dfftt%ngm)`` / ``size(dfftt%nl_d)``) -- is lifted by
``hlfir-flatten-structs`` to a free SDFG symbol with NO
``FlattenEntry``.  Only members read as *values* get a data entry
(cf. ``struct_of_scalars`` ``cst%rg``), so the plan-driven
``scalar_member`` / ``flat_shapes`` sourcing in ``_build_symbol_assigns``
never sees these.

Before the fix the symbol-population block emitted
``! TODO: no plan entry gives size for free symbol 'dfftt_ngm'`` and the
symbol defaulted to 0 at runtime -> the SDFG indexed a zero-extent array
-> OOB segfault (the QE ``vexx`` real-exchange path).  The fix rebuilds
the bridge's member-symbol names from the static
``OriginalInterface.struct_types`` layout and reads each member straight
from the caller's struct.
"""
from dace_fortran.bindings import (
    DerivedType,
    FlattenPlan,
    FrozenSignature,
    Member,
    OriginalArg,
    OriginalInterface,
)
from dace_fortran.bindings.block_builders import (
    _build_symbol_assigns,
    _struct_member_symbol_sources,
)


def _fft_iface() -> OriginalInterface:
    """A QE-shaped FFT descriptor dummy: two scalar members used as loop
    bounds (``ngm`` / ``nnr``) and one dynamic-shape array member used
    only for its extent (``nl_d``)."""
    return OriginalInterface(
        entry="vexx",
        args=(OriginalArg(name="dfftt",
                          fortran_type="type(fft_type_descriptor)",
                          rank=0,
                          intent="in",
                          struct_type="fft_type_descriptor"), ),
        struct_types={
            "fft_type_descriptor":
            DerivedType(
                name="fft_type_descriptor",
                module="fft_types",
                members=(
                    Member(name="ngm", fortran_type="integer(c_int)", rank=0),
                    Member(name="nnr", fortran_type="integer(c_int)", rank=0),
                    Member(name="nl_d", fortran_type="integer(c_int)", rank=1, shape=("?", )),
                ),
            )
        },
    )


def test_struct_member_symbol_sources_maps_scalar_and_extent():
    """``_struct_member_symbol_sources`` regenerates the bridge's
    single-underscore member-symbol names and pairs each with its ``%``
    read: scalar member -> the member itself, array member -> a per-dim
    ``size`` (extent) and ``lbound`` (offset)."""
    src = _struct_member_symbol_sources(_fft_iface())
    assert src["dfftt_ngm"] == "dfftt%ngm"
    assert src["dfftt_nnr"] == "dfftt%nnr"
    assert src["dfftt_nl_d_d0"] == "size(dfftt%nl_d, dim=1)"
    assert src["offset_dfftt_nl_d_d0"] == "lbound(dfftt%nl_d, dim=1)"


def test_struct_member_symbol_sources_recurses_nested():
    """A nested derived-type member contributes the joined path
    (``outer%inner%cnt`` -> ``outer_inner_cnt``)."""
    iface = OriginalInterface(
        entry="k",
        args=(OriginalArg(name="outer", fortran_type="type(t_outer)", rank=0, intent="in", struct_type="t_outer"), ),
        struct_types={
            "t_outer":
            DerivedType(name="t_outer",
                        module="m",
                        members=(Member(name="inner", fortran_type="type(t_inner)", rank=0, struct_name="t_inner"), )),
            "t_inner":
            DerivedType(name="t_inner",
                        module="m",
                        members=(Member(name="cnt", fortran_type="integer(c_int)", rank=0), )),
        },
    )
    src = _struct_member_symbol_sources(iface)
    assert src["outer_inner_cnt"] == "outer%inner%cnt"


def test_symbol_population_sources_struct_member_loop_bounds():
    """End of the ``_build_symbol_assigns`` ladder: a struct-member free
    symbol with NO plan entry is sourced from the static layout, NOT left
    as an unresolved-symbol TODO."""
    iface = _fft_iface()
    frozen = FrozenSignature(
        entry="vexx",
        mangled="_QPvexx",
        # The struct's data members are irrelevant to symbol sourcing;
        # what matters is the free-symbol list the SDFG folded the
        # member loop-bounds / extents into.
        args=(),
        free_symbols=("dfftt_ngm", "dfftt_nnr", "dfftt_nl_d_d0"),
    )
    lines = _build_symbol_assigns(frozen, FlattenPlan(entries=()), {"dfftt"}, iface)
    text = "\n".join(lines)
    assert "dfftt_ngm = int(dfftt%ngm, c_int)" in text
    assert "dfftt_nnr = int(dfftt%nnr, c_int)" in text
    assert "dfftt_nl_d_d0 = int(size(dfftt%nl_d, dim=1), c_int)" in text
    assert "TODO: no plan entry gives size" not in text


def test_plan_entry_still_wins_over_struct_layout_fallback():
    """A member that DOES carry a plan entry (read as a value) keeps the
    precise plan-driven sourcing; the struct-layout fallback only fires
    for the plan-less symbolic-use members."""
    iface = _fft_iface()
    frozen = FrozenSignature(
        entry="vexx",
        mangled="_QPvexx",
        args=(),
        # ``dfftt_ngm`` has a plan entry (rank-0 value read); ``dfftt_nnr``
        # is symbolic-only.
        free_symbols=("dfftt_ngm", "dfftt_nnr"),
    )
    from dace_fortran.bindings import FlattenEntry, FlattenRecipe
    plan = FlattenPlan(
        entries=(FlattenEntry(outer_expr="dfftt%ngm",
                              outer_type="integer(c_int)",
                              writeback_intent="in",
                              recipe=FlattenRecipe(flat_names=("dfftt_ngm", ), read_exprs=("dfftt%ngm", ), rank=0)), ))
    text = "\n".join(_build_symbol_assigns(frozen, plan, {"dfftt"}, iface))
    # Plan-driven scalar_member path (read_exprs[0], no $i strip needed).
    assert "dfftt_ngm = int(dfftt%ngm, c_int)" in text
    # Struct-layout fallback for the plan-less one.
    assert "dfftt_nnr = int(dfftt%nnr, c_int)" in text
    assert "TODO: no plan entry gives size" not in text
