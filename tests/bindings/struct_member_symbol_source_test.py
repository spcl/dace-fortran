"""A struct member used ONLY symbolically (loop bound / array extent) is lifted by
``hlfir-flatten-structs`` to a free SDFG symbol with NO ``FlattenEntry`` -- the
plan-driven sourcing in ``_build_symbol_assigns`` never sees these.

Before the fix the symbol defaulted to 0 at runtime -> OOB segfault (QE ``vexx``
real-exchange path).  Fix rebuilds member-symbol names from the static
``OriginalInterface.struct_types`` layout and reads each member from the caller's
struct directly.
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
    """QE-shaped FFT descriptor: two scalar loop-bound members (``ngm``/``nnr``) and one
    dynamic-shape array member used only for its extent (``nl_d``)."""
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
    """Regenerates member-symbol names paired with their ``%`` read: scalar member -> the
    member itself, array member -> per-dim ``size``/``lbound``."""
    src, paths = _struct_member_symbol_sources(_fft_iface())
    assert src["dfftt_ngm"] == "dfftt%ngm"
    assert src["dfftt_nnr"] == "dfftt%nnr"
    assert src["dfftt_nl_d_d0"] == "size(dfftt%nl_d, dim=1)"
    assert src["offset_dfftt_nl_d_d0"] == "lbound(dfftt%nl_d, dim=1)"
    # member_paths carries every leaf's bare %-path so a nested pointer/allocatable
    # member's ASSOCIATED/ALLOCATED tracker can spell the caller-side access.
    assert paths["dfftt_ngm"] == "dfftt%ngm"
    assert paths["dfftt_nl_d"] == "dfftt%nl_d"


def test_struct_member_symbol_sources_recurses_nested():
    """A nested derived-type member contributes the joined path (``outer%inner%cnt`` ->
    ``outer_inner_cnt``)."""
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
    src, paths = _struct_member_symbol_sources(iface)
    assert src["outer_inner_cnt"] == "outer%inner%cnt"
    assert paths["outer_inner_cnt"] == "outer%inner%cnt"


def test_symbol_population_sources_struct_member_loop_bounds():
    """A struct-member free symbol with NO plan entry is sourced from the static layout,
    not left as an unresolved-symbol TODO."""
    iface = _fft_iface()
    frozen = FrozenSignature(
        entry="vexx",
        mangled="_QPvexx",
        # Only the free-symbol list (member loop-bounds/extents) matters here.
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
    """A member with a plan entry (read as a value) keeps precise plan-driven sourcing;
    the struct-layout fallback only fires for plan-less symbolic-use members."""
    iface = _fft_iface()
    frozen = FrozenSignature(
        entry="vexx",
        mangled="_QPvexx",
        args=(),
        # dfftt_ngm has a plan entry (rank-0 value read); dfftt_nnr is symbolic-only.
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
