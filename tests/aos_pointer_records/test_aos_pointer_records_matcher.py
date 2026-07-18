"""Recognition tests for ``hlfir-lift-aos-pointer-records``: pins that the
AoS-of-pointer-records pattern emits ``hlfir.aos_ptr_records.<aos_decl>``, ahead of materialisation.
"""
from pathlib import Path
import sys

import pytest

import dace_fortran
from dace_fortran.build import make_builder

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _post_pass_module(probe_name: str, entry: str, tmp_path):
    """Return the post-pass HLFIR module text for assertion."""
    src = (_HERE / probe_name).read_text()
    builder = make_builder(src, entry=entry, name=entry, out_dir=str(tmp_path))
    return builder.module.dump()


def _has_attr(module_str: str, attr_substring: str) -> bool:
    """The matcher emits ``hlfir.aos_ptr_records.<uniq_name> = [...]``."""
    return any(attr_substring in line for line in module_str.splitlines())


def test_single_pointer_member_recognised(tmp_path):
    mod = _post_pass_module("aos_single_pointer_member_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records."), \
        "matcher did not emit the recognition attribute for single-member shape"


def test_two_pointer_members_recognised(tmp_path):
    """Two pointer members of different rank both surface as rebind entries."""
    mod = _post_pass_module("aos_two_pointer_members_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records."), \
        "matcher did not emit the recognition attribute for two-member shape"
    # both members should appear in the rebind table
    assert 'member = "p"' in mod and 'member = "x"' in mod, \
        "two-member rebind table missing one of the member entries"


def test_runtime_index_recognised(tmp_path):
    """Runtime-index reads don't change matching: the matcher is rebind-driven."""
    mod = _post_pass_module("aos_runtime_index_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records.")


def test_write_through_pointer_recognised(tmp_path):
    """Writes through the alias don't affect matching either."""
    mod = _post_pass_module("aos_write_through_pointer_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records.")


def _materialisation_landed(module_str: str) -> bool:
    """The materialisation step injects a concat declare named ``_QXaos_lift_*``."""
    return any('_QXaos_lift_' in line for line in module_str.splitlines())


def test_materialisation_emits_concat_declare(tmp_path):
    """The pass allocates a per-member concat transient when materialisation fires.

    The recognised candidate's pointer member gets a top-level
    ``hlfir.declare`` whose uniq_name carries the ``_QXaos_lift_<aos>_<member>_<n>``
    convention.  This verifies the pass progresses past the recognition stage
    into the actual IR rewrite.
    """
    mod = _post_pass_module("aos_single_pointer_member_probe.f90", "m::run", tmp_path)
    assert _materialisation_landed(mod), ("expected an `_QXaos_lift_` concat declare in the post-pass module; "
                                          "the materialisation step did not emit one")


def test_materialisation_handles_two_members(tmp_path):
    """Both pointer members get their own concat transient."""
    mod = _post_pass_module("aos_two_pointer_members_probe.f90", "m::run", tmp_path)
    concat_lines = [l for l in mod.splitlines() if '_QXaos_lift_' in l]
    # 2 concat declares (one per member)
    assert sum(1 for l in concat_lines if 'hlfir.declare' in l) >= 2, \
        f"expected 2 concat declares; got: {concat_lines!r}"


def test_allocatable_member_aos_not_matched(tmp_path):
    """An AoS whose member is ALLOCATABLE (``fir.heap``), not POINTER
    (``fir.ptr``), must NOT be matched by ``hlfir-lift-aos-pointer-records``.

    The pass lifts the ``q(c)%m => target`` pointer-rebind pattern; an
    allocatable member is never rebound with ``=>`` -- it is allocated with
    ``allocate(a(i)%w(...))`` and belongs to flatten-structs' Phase 5c-A
    padded-companion path.  Matching it here fires the ``rebinds.empty()``
    "dead exchange scaffolding" branch, which mints a bogus size-1 placeholder
    companion ``_QXaos_lift_..._w_0`` (shape ``(N, 1)``) and rewrites the live
    ``a(i)%w(j)`` reads onto it -- producing an out-of-bounds
    ``a_w_0[1, 1]`` memlet (and, at an SDFG boundary, an unbound
    ``a_w_0_d0`` free symbol).  Regression guard for that double-companion
    bug: assert neither the recognition attribute nor an ``_QXaos_lift_``
    companion is emitted for the allocatable-member shape.
    """
    mod = _post_pass_module("aos_allocatable_member_probe.f90", "m::run", tmp_path)
    assert not _has_attr(mod, "hlfir.aos_ptr_records."), \
        "matcher wrongly recognised an allocatable-member AoS as pointer-records"
    assert not _materialisation_landed(mod), \
        "matcher wrongly minted an _QXaos_lift_ companion for an allocatable-member AoS"
