"""Frontend-recognition tests for the AoS-of-pointer-records pass scaffold.

``hlfir-lift-aos-pointer-records`` detects the
``TYPE(...) :: q(N)`` + ``REAL, POINTER :: m(:[, :])`` pattern that
Graupel's ``t_qx_ptr%x`` uses and records its findings as a
``hlfir.aos_ptr_records.<aos_decl>`` attribute on the enclosing
function.  These tests drive a handful of isolated shapes through
the bridge and verify the attribute lands -- the materialisation
step (copy-in / copy-out + access rewrite) is a follow-up commit;
this matcher test family pins the recognition so the materialisation
PR can iterate against real probes without ambiguity about which
sites should be transformed.
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
