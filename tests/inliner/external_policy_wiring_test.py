# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The external-function policy (:mod:`dace_fortran.external_functions`) wired into both
inliner engines: the fparser pipeline (:func:`inline_to_ast`) and the regex text-splicer
(:func:`merge_used_modules`).

``external_functions``/``do_not_emit`` name procedures that stay declared but get their
executable body emptied (halo/MPI/I/O internals never enter the TU). The deprecated
``keep_external=`` (fparser only) is a thin shim -- these tests assert byte-identical output
plus a warning.
"""
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dace_fortran.external_functions import ExternalFunction
from dace_fortran.fparser_inliner import inline_to_ast
from dace_fortran.preprocess import merge_used_modules, preprocess_fortran_source


def _have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


def _gfortran_compiles(src_text: str) -> bool:
    with TemporaryDirectory() as td:
        f = Path(td) / "tu.f90"
        f.write_text(src_text)
        r = subprocess.run(["gfortran", "-fsyntax-only", "-ffree-line-length-none",
                            str(f)],
                           cwd=td,
                           capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode())
        return r.returncode == 0


# Kernel's enclosing module also defines "halo_exchange"; keeping it external must empty its
# body (+1.0 update) while leaving the kernel's own body (*2.0) and the call site intact.
_HALO_MODULE = """
module mo_halo
  implicit none
contains
  subroutine do_step(a, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      a(i) = a(i) * 2.0
    end do
    call halo_exchange(a)
  end subroutine do_step

  subroutine halo_exchange(a)
    real, intent(inout) :: a(:)
    integer :: i
    do i = 1, size(a)
      a(i) = a(i) + 1.0
    end do
  end subroutine halo_exchange
end module mo_halo
"""

# ---------------------------------------------------------------------------
# fparser engine -- external_functions / do_not_emit / keep_external parity
# ---------------------------------------------------------------------------


def _fparser_out(**kw) -> str:
    return inline_to_ast({"mo_halo.f90": _HALO_MODULE}, entry="mo_halo::do_step", **kw).tofortran().lower()


def test_fparser_baseline_keeps_external_body():
    """Without a policy, the called procedure survives WITH its body -- the contrast the policy removes."""
    base = _fparser_out().replace(" ", "")
    assert "callhalo_exchange" in base
    assert "a(i)=a(i)+1.0" in base, "halo_exchange body present without a policy"


def test_fparser_external_functions_stub_body():
    """``external_functions`` empties the procedure body but keeps the call."""
    out = _fparser_out(external_functions=[ExternalFunction("halo_exchange")]).replace(" ", "")
    assert "callhalo_exchange" in out, "the external call must survive"
    assert "a(i)=a(i)+1.0" not in out, "the external body must be emptied"
    assert "a(i)=a(i)*2.0" in out, "the kernel's own body must remain"


def test_fparser_external_functions_do_not_emit_keep_external_are_identical():
    """All three spellings drive the same ``make_noop`` -- byte-identical TUs (the inliner only needs the name; emit-vs-drop is the bridge's concern)."""
    via_ext = _fparser_out(external_functions=[ExternalFunction("halo_exchange")])
    via_dne = _fparser_out(do_not_emit=["halo_exchange"])
    with pytest.warns(DeprecationWarning):
        via_legacy = _fparser_out(keep_external=["halo_exchange"])
    assert via_ext == via_dne == via_legacy


def test_fparser_keep_external_warns_but_works():
    """The deprecated ``keep_external=`` still stubs the body, with a warning."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        out = _fparser_out(keep_external=["halo_exchange"]).replace(" ", "")
    assert "a(i)=a(i)+1.0" not in out


def test_fparser_validate_rejects_name_in_both():
    """A name in both emitted and ``do_not_emit`` is an inconsistent policy -- rejected before any parsing."""
    with pytest.raises(ValueError, match="both"):
        _fparser_out(external_functions=[ExternalFunction("halo_exchange")], do_not_emit=["halo_exchange"])


# ---------------------------------------------------------------------------
# regex engine -- merge_used_modules body-stubbing
# ---------------------------------------------------------------------------

_CALLER = """
module mo_user
  use mo_halo, only: do_step
  implicit none
contains
  subroutine run(a, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: n
    call do_step(a, n)
  end subroutine run
end module mo_user
"""


def _write_halo(d: Path) -> Path:
    (d / "mo_halo.f90").write_text(_HALO_MODULE)
    return d


def test_regex_merge_stubs_external_body(tmp_path):
    """``merge_used_modules`` splices ``mo_halo`` in; with the policy, spliced ``halo_exchange``
    is emptied (opener+spec+END kept, ``+1.0`` body gone) while the call site survives."""
    _write_halo(tmp_path)
    merged = merge_used_modules(_CALLER, search_dirs=[tmp_path], external_functions=[ExternalFunction("halo_exchange")])
    flat = merged.replace(" ", "").lower()
    assert "subroutinehalo_exchange" in flat, "the procedure stays declared"
    assert "callhalo_exchange" in flat, "the call site survives"
    assert "a(i)=a(i)+1.0" not in flat, "the external body is emptied"
    assert "a(i)=a(i)*2.0" in flat, "do_step's body is untouched"


def test_regex_merge_do_not_emit_same_as_external_functions(tmp_path):
    """``do_not_emit`` stubs identically to ``external_functions`` (names only)."""
    _write_halo(tmp_path)
    a = merge_used_modules(_CALLER, search_dirs=[tmp_path], external_functions=[ExternalFunction("halo_exchange")])
    b = merge_used_modules(_CALLER, search_dirs=[tmp_path], do_not_emit=["halo_exchange"])
    assert a == b


def test_regex_merge_no_policy_keeps_body(tmp_path):
    """Without a policy the spliced body is kept verbatim (the default)."""
    _write_halo(tmp_path)
    merged = merge_used_modules(_CALLER, search_dirs=[tmp_path]).replace(" ", "").lower()
    assert "a(i)=a(i)+1.0" in merged


def test_regex_merge_generic_prefix_match(tmp_path):
    """A generic policy name (``sync_patch_array``) stubs every concrete specific (``sync_patch_array_3d_dp``) -- the ICON interface pattern."""
    mod = """
module mo_sync
  implicit none
contains
  subroutine sync_patch_array_3d_dp(a)
    real, intent(inout) :: a(:)
    a = a + 1.0
  end subroutine sync_patch_array_3d_dp
end module mo_sync
"""
    (tmp_path / "mo_sync.f90").write_text(mod)
    caller = ("module mo_c\n  use mo_sync, only: sync_patch_array_3d_dp\n"
              "contains\n  subroutine r(a)\n    real, intent(inout) :: a(:)\n"
              "    call sync_patch_array_3d_dp(a)\n  end subroutine r\nend module mo_c\n")
    merged = merge_used_modules(caller,
                                search_dirs=[tmp_path],
                                external_functions=[ExternalFunction("sync_patch_array")])
    flat = merged.replace(" ", "").lower()
    assert "subroutinesync_patch_array_3d_dp" in flat
    assert "a=a+1.0" not in flat, "the generic specific's body must be stubbed"


def test_regex_merge_stubs_body_with_interface_in_spec(tmp_path):
    """A stubbed procedure whose spec contains an ``INTERFACE`` block (ICON's ``bind(c)`` halo
    wrapper forwarding to C++) keeps the whole block and drops only the executable ``call``.
    Regression: the nested ``subroutine`` inside must not be mistaken for the body's start
    (would orphan the ``interface`` opener -> uncompilable).
    """
    mod = """
module mo_wrap
  use iso_c_binding
  implicit none
contains
  subroutine halo_via_c(tag, d0, field_p) bind(c, name='halo_via_c')
    integer(c_int), value :: tag, d0
    type(c_ptr), value :: field_p
    interface
      subroutine halo_impl(tag, d0, field_p) bind(c, name='halo_cpp')
        use iso_c_binding
        integer(c_int), value :: tag, d0
        type(c_ptr), value :: field_p
      end subroutine
    end interface
    call halo_impl(tag, d0, field_p)
  end subroutine halo_via_c
end module mo_wrap
"""
    (tmp_path / "mo_wrap.f90").write_text(mod)
    caller = ("module mo_c\n  use mo_wrap, only: halo_via_c\n  use iso_c_binding\n"
              "contains\n  subroutine r(tag, d0, p)\n    integer(c_int), value :: tag, d0\n"
              "    type(c_ptr), value :: p\n    call halo_via_c(tag, d0, p)\n"
              "  end subroutine r\nend module mo_c\n")
    merged = merge_used_modules(caller, search_dirs=[tmp_path], external_functions=[ExternalFunction("halo_via_c")])
    flat = merged.replace(" ", "").lower()
    assert "subroutinehalo_via_c" in flat, "the procedure stays declared"
    assert "interface" in flat and "endinterface" in flat, "the interface block survives whole"
    assert "callhalo_impl" not in flat, "the executable body is emptied"
    if _have_gfortran():
        assert _gfortran_compiles(merged), "stubbed TU with an interface-in-spec must compile"


@pytest.mark.skipif(not _have_gfortran(), reason="gfortran not on PATH")
def test_regex_merge_stubbed_tu_compiles(tmp_path):
    """The stubbed single-TU is still valid Fortran: an empty-bodied ``halo_exchange`` with its dummy argument declared compiles standalone."""
    _write_halo(tmp_path)
    merged = merge_used_modules(_CALLER, search_dirs=[tmp_path], external_functions=[ExternalFunction("halo_exchange")])
    assert _gfortran_compiles(merged), "stubbed merged TU must compile"


# ---------------------------------------------------------------------------
# build-path threading -- preprocess_fortran_source forwards external_names to both merge
# engines; the build path sources them from the bridge registry
# ---------------------------------------------------------------------------


def test_preprocess_source_threads_external_names_regex(tmp_path):
    """``preprocess_fortran_source`` forwards ``external_names`` to the regex merge -- the spliced external body is stubbed."""
    _write_halo(tmp_path)
    out = preprocess_fortran_source(_CALLER,
                                    search_dirs=[tmp_path],
                                    merge_engine="regex",
                                    external_names=["halo_exchange"]).replace(" ", "").lower()
    assert "subroutinehalo_exchange" in out
    assert "a(i)=a(i)+1.0" not in out, "regex merge must stub the external body"


def test_preprocess_source_threads_external_names_fparser(tmp_path):
    """Same through the fparser engine (``make_noop`` path)."""
    _write_halo(tmp_path)
    out = preprocess_fortran_source(_CALLER,
                                    search_dirs=[tmp_path],
                                    merge_engine="fparser",
                                    merge_entry="mo_user::run",
                                    external_names=["halo_exchange"]).replace(" ", "").lower()
    assert "callhalo_exchange" in out, "the external call must survive"
    assert "a(i)=a(i)+1.0" not in out, "fparser merge must stub the external body"


def test_preprocess_source_no_external_names_keeps_body(tmp_path):
    """Default (no policy) keeps the spliced body -- the threading is opt-in."""
    _write_halo(tmp_path)
    out = preprocess_fortran_source(_CALLER, search_dirs=[tmp_path], merge_engine="regex").replace(" ", "").lower()
    assert "a(i)=a(i)+1.0" in out


def test_build_path_sources_external_names_from_registry():
    """The build path's merge sources keep-external names from the bridge's external registry:
    a ``keep_external`` registration is in the merge set automatically, unioned (de-duplicated)
    with any explicit names."""
    from dace_fortran.build import _merge_external_names
    from dace_fortran.external import clear_external_registry, keep_external

    clear_external_registry()
    try:
        keep_external("sync_patch_array", c_name="sync_patch_array_c")
        names = _merge_external_names()
        assert "sync_patch_array" in names, "registered external must drive the merge"
        # explicit names union with the registry, de-duplicated + order-stable.
        names2 = _merge_external_names(["extra_fn", "sync_patch_array"])
        assert names2[:2] == ["extra_fn", "sync_patch_array"]
        assert names2.count("sync_patch_array") == 1
    finally:
        clear_external_registry()
