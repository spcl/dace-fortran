# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the external-function policy data model
(:mod:`dace_fortran.external_functions`) -- the pure-stdlib declaration the
inliner and the bridge share.  No dace / fparser dependency, so these run
anywhere."""
import pytest

from dace_fortran.external_functions import ExternalFunction, dont_inline_names, validate


def test_defaults_symbol_is_name():
    c = ExternalFunction("sync_patch_array")
    assert c.name == "sync_patch_array"
    assert c.c_function is None
    assert c.library is None
    assert c.symbol == "sync_patch_array"  # symbol defaults to name


def test_explicit_binding():
    c = ExternalFunction("foo", c_function="my_c_abi_fn", library="/path/libfoo.so")
    assert c.symbol == "my_c_abi_fn"
    assert c.library == "/path/libfoo.so"


def test_frozen():
    c = ExternalFunction("foo")
    with pytest.raises(Exception):
        c.name = "bar"


def test_rejects_empty_name():
    with pytest.raises(ValueError):
        ExternalFunction("")
    with pytest.raises(ValueError):
        ExternalFunction("   ")


def test_dont_inline_names_is_lowercased_union():
    fns = [ExternalFunction("Sync_Patch_Array"), ExternalFunction("exchange_data")]
    ignore = ["Finish", "timer_start"]
    assert dont_inline_names(fns, ignore) == {"sync_patch_array", "exchange_data", "finish", "timer_start"}


def test_dont_inline_names_empty():
    assert dont_inline_names() == set()
    assert dont_inline_names([ExternalFunction("a")]) == {"a"}
    assert dont_inline_names(do_not_emit=["b"]) == {"b"}


def test_validate_accepts_clean_policy():
    validate([ExternalFunction("a"), ExternalFunction("b")], ["c", "d"])  # must not raise


def test_validate_rejects_duplicate_emit_name():
    with pytest.raises(ValueError, match="duplicate"):
        validate([ExternalFunction("a"), ExternalFunction("A")], [])  # case-insensitive dupe


def test_validate_rejects_name_in_both_lists():
    with pytest.raises(ValueError, match="both"):
        validate([ExternalFunction("sync_patch_array")], ["Sync_Patch_Array"])
