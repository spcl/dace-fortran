"""Coverage for ``install_enum_arg_translation``  --  the binding-
layer follow-up that lets ``sdfg(flag='c', ...)`` work even after
``rewrite_string_enum_to_integer`` changed the kernel's dummy from
``CHARACTER`` to ``INTEGER``.

Test layout:

  * Unit tests on the wrapper mechanism (no SDFG build needed)
    use a minimal stub class to verify ``__call__`` substitutes
    correctly, falls through for unknown values, is case-
    insensitive, and composes with another class-assignment
    wrapper.

  * One end-to-end test runs the full pipeline:
    Pattern 2 rewrite -> build SDFG -> install translation ->
    call with string arg -> compare against a direct int-arg
    call.  This is the contract the bindings layer relies on.
"""
import sys
from pathlib import Path

import pytest

import dace

from dace_fortran.builder.enum_arg_symbols import (_EnumArgMixin, install_enum_arg_translation)
from dace_fortran.preprocess import rewrite_string_enum_to_integer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import build_sdfg, have_flang  # noqa: E402

# ---------------------------------------------------------------------------
# Mechanism: a stub SDFG-like object exercises the mixin directly
# ---------------------------------------------------------------------------


class _RecorderSDFG(dace.SDFG):
    """Bottom-of-MRO stand-in for ``dace.SDFG.__call__``: records the
    kwargs it was called with so the test can compare against the
    expected post-translation values.

    Inherits from ``dace.SDFG`` so the layout matches for
    ``__class__`` reassignment in :func:`_wrap_with_enum`."""

    def __call__(self, *args, **kwargs):
        self.last_kwargs = dict(kwargs)
        return "ok"


def _make_recorder() -> _RecorderSDFG:
    sdfg = dace.SDFG("recorder")
    sdfg.__class__ = _RecorderSDFG
    sdfg.last_kwargs = None
    return sdfg


def _wrap_with_enum(stub, lit_map: dict):
    """Class-assign the mixin onto a stub instance with the given
    per-arg enum table, mirroring ``install_enum_arg_translation``'s
    runtime class-rewrite."""
    cls = type(f"_Wrapped_{type(stub).__name__}", (_EnumArgMixin, type(stub)), {})
    stub.__class__ = cls
    stub._enum_arg_maps = {arg: {k.lower(): int(v) for k, v in m.items()} for arg, m in lit_map.items()}
    return stub


def test_string_kwarg_gets_translated_to_int():
    """A string ``'c'`` becomes integer ``0`` when ``c -> 0`` is in
    the table."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0, "r": 1}})
    stub(flag="c", other=42)
    assert stub.last_kwargs == {"flag": 0, "other": 42}


def test_case_insensitive_lookup():
    """``'C'`` matches the ``'c'`` table entry  --  the lookup is
    case-folded at runtime, matching the lowercase grouping the
    preprocess pass applies."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0}})
    stub(flag="C")
    assert stub.last_kwargs == {"flag": 0}


def test_int_kwarg_passes_through_unchanged():
    """When the caller already passes an integer the translation
    is a no-op  --  the type guard skips non-string values."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0}})
    stub(flag=42)
    assert stub.last_kwargs == {"flag": 42}


def test_unknown_string_value_flows_through_unchanged():
    """A string that isn't in the enum table is left in ``kwargs``
    verbatim.  Downstream the SDFG will probably reject it with a
    type error, surfacing a clean ``unsupported value`` instead of
    silently substituting ``0``."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0}})
    stub(flag="zzz")
    assert stub.last_kwargs == {"flag": "zzz"}


def test_kwarg_not_in_enum_table_is_untouched():
    """``other`` isn't an enum-mapped argument  --  the wrapper
    leaves it strictly alone, even when it's a string."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0}})
    stub(flag="c", other="just a string")
    assert stub.last_kwargs == {"flag": 0, "other": "just a string"}


def test_bytes_value_is_decoded_and_translated():
    """A ``bytes`` arg (e.g. ``b'c'``) is decoded as ASCII before
    lookup; bytes shows up when a caller round-trips a string
    through a C interop boundary."""
    stub = _wrap_with_enum(_make_recorder(), {"flag": {"c": 0}})
    stub(flag=b"c")
    assert stub.last_kwargs == {"flag": 0}


def test_multiple_enum_kwargs_all_translated():
    """The mixin walks every kwarg in the table  --  not just one."""
    stub = _wrap_with_enum(_make_recorder(), {
        "flag": {
            "c": 0,
            "r": 1
        },
        "mode": {
            "forward": 0,
            "backward": 1
        },
    })
    stub(flag="r", mode="backward")
    assert stub.last_kwargs == {"flag": 1, "mode": 1}


def test_install_with_empty_enum_maps_is_noop():
    """``install_enum_arg_translation(sdfg, {})`` returns the SDFG
    unmodified  --  no class rewrite, no instance state change."""
    sdfg = dace.SDFG("noop")
    before_cls = type(sdfg)
    out = install_enum_arg_translation(sdfg, {})
    assert out is sdfg
    assert type(sdfg) is before_cls
    assert not hasattr(sdfg, "_enum_arg_maps")


def test_install_flattens_per_procedure_to_per_arg():
    """``rewrite_string_enum_to_integer`` returns ``{proc: {arg:
    table}}``; the installer flattens this to ``{arg: table}`` on
    the SDFG since the resulting SDFG is one entry."""
    sdfg = dace.SDFG("flat")
    install_enum_arg_translation(sdfg, {"run": {"flag": {"c": 0, "r": 1}}})
    assert sdfg._enum_arg_maps == {"flag": {"c": 0, "r": 1}}


# ---------------------------------------------------------------------------
# Composition: stacks correctly with another class-assignment wrapper
# ---------------------------------------------------------------------------


def test_mixin_composes_with_existing_class_assignment():
    """``install_enum_arg_translation`` is called after a previous
    class-assignment wrapper (e.g. ``install_auto_dim_symbols``).
    The mixin must insert itself ABOVE the existing wrapper in the
    MRO so its ``super().__call__`` chains through the existing
    wrapper before reaching the underlying class."""
    chain_log = []

    class _PriorWrapper(dace.SDFG):

        def __call__(self, *args, **kwargs):
            chain_log.append("prior")
            return super().__call__(*args, **kwargs)

    class _RecorderBase(dace.SDFG):

        def __call__(self, *args, **kwargs):
            chain_log.append("base")
            self.last_kwargs = dict(kwargs)
            return "ok"

    # Stack the prior wrapper first (as install_auto_dim_symbols does).
    stub = dace.SDFG("compose")
    prior_class = type("_PriorOver", (_PriorWrapper, _RecorderBase), {})
    stub.__class__ = prior_class
    stub.last_kwargs = None
    # Now install enum on top.
    _wrap_with_enum(stub, {"flag": {"c": 0}})
    stub(flag="c")
    # Both wrappers' ``__call__`` fired, in order; the final kwargs
    # at the base layer had the int substitution applied.
    assert chain_log == ["prior", "base"]
    assert stub.last_kwargs == {"flag": 0}


# ---------------------------------------------------------------------------
# End-to-end: real SDFG built from the rewritten kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_string_arg_end_to_end_matches_int_arg(tmp_path):
    """The contract the binding layer commits to: after running
    Pattern 2's rewrite + the bridge build + this installer, a
    string-typed call returns identical output to the equivalent
    integer call.  The string ``'c'`` and the integer ``0`` must
    drive the SAME branch of the kernel's enum switch."""
    import numpy as np

    src = (_HERE / "string_enum_basic_example.f90").read_text()
    rewritten, enum_maps = rewrite_string_enum_to_integer(src)
    # Sanity: the preprocess pass actually produced a map.
    assert enum_maps and "run" in enum_maps

    sdfg = build_sdfg(rewritten, tmp_path / "sdfg", name="enum_run", entry="_QPrun").build()
    sdfg = install_enum_arg_translation(sdfg, enum_maps)

    # ``run(out_val, action)``  --  action=0 ('c') -> out_val=1.0,
    # action=1 ('r') -> 2.0, action=2 ('i') -> 3.0, anything else
    # -> 0.0.  Run with the string form first, then with the int.
    for literal, expected in (
        ("c", 1.0),
        ("C", 1.0),  # case-insensitive
        ("r", 2.0),
        ("i", 3.0),
    ):
        out_str = np.zeros(1, dtype=np.float64)
        sdfg(out_val=out_str, action=literal)
        out_int = np.zeros(1, dtype=np.float64)
        sdfg(out_val=out_int, action=enum_maps["run"]["action"][literal.lower()])
        assert out_str[0] == out_int[0] == expected, \
            f"literal={literal!r}: str-call={out_str[0]} int-call={out_int[0]} expected={expected}"
