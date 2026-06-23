"""Regression coverage for the QE ``deexx`` indirect-index family.

These pin the ``ijkb0 = ofsbeta(na)`` indexing shape from QE's exchange
kernels (``paw_newdxx`` / ``add_nlxx_pot`` / ``addusxx_g`` in
``vexx_bp_k_gpu``): a beta-offset table is read indirectly, a per-atom
base ``ijkb0`` is derived from it, and the loop body gathers / scatters
through ``ikb = ijkb0 + ih``.

History -- this was once hypothesised as bug **M3** ("``buildIndexExpr``
re-materialise pre-pass for an indirect index operand"), but a prior
bring-up session established that hypothesis was WRONG: the real QE
``?``-leak cascade had three unrelated root causes (local-allocatable
``box_dims`` section bounds, ``PRESENT()`` on an inlined optional, and a
flattened struct scalar member used as an array size), plus the gate-4a
pointer-member indexed read -- all since fixed.  The indirect-index shape
itself compiles AND is numerically correct.  These tests lock that in
end-to-end against an f2py reference so the landed fixes can't silently
regress to the "drops a dim, masked" failure mode that family once had.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _module_wrap(src: str, free_sub_decl: str, mod_name: str) -> str:
    """Wrap a free subroutine in a module so the SDFG can build the
    ``<mod>::<sub>`` entry, while the free form stays the f2py reference."""
    # count=1: ``free_sub_decl`` ("subroutine foo") is a substring of the
    # matching ``end subroutine foo`` too -- wrap only the opening line.
    return src.replace(free_sub_decl, f"module {mod_name}\ncontains\n{free_sub_decl}", 1).rstrip() \
        + f"\nend module {mod_name}\n"


# ---------------------------------------------------------------------------
# 1. Indirect scatter-accumulate -- the paw_newdxx body shape.
#    ijkb0 = ofsbeta(na) ; ikb = ijkb0 + ih ; deexx(ikb) += w*becphi(jkb)*becphi(ikb)
# ---------------------------------------------------------------------------
_SCATTER = """
subroutine paw_accum(weight, becphi, deexx, ofsbeta, nat, nh, nkb)
  integer, intent(in) :: nat, nh, nkb
  real(8), intent(in) :: weight
  integer, intent(in) :: ofsbeta(nat)
  real(8), intent(in) :: becphi(nkb)
  real(8), intent(inout) :: deexx(nkb)
  integer :: ijkb0, ih, jh, na, ikb, jkb
  do na = 1, nat
    ijkb0 = ofsbeta(na)
    do jh = 1, nh
      jkb = ijkb0 + jh
      do ih = 1, nh
        ikb = ijkb0 + ih
        deexx(ikb) = deexx(ikb) + weight * becphi(jkb) * becphi(ikb)
      end do
    end do
  end do
end subroutine paw_accum
"""


def test_indirect_scatter_accumulate_matches_reference(tmp_path: Path):
    """``deexx(ikb) += ...`` scatter-accumulate through ``ikb = ofsbeta(na)+ih``
    must match the gfortran/f2py reference -- a dropped record / wrong
    indirect base would diverge, not round off."""
    sdfg = build_sdfg(_module_wrap(_SCATTER, "subroutine paw_accum", "paw_mod"),
                      tmp_path / "sdfg",
                      name="paw_accum",
                      entry="paw_mod::paw_accum").build()
    sdfg.validate()
    ref = f2py_compile(_SCATTER, tmp_path / "ref", "paw_ref", only=("paw_accum", ))

    nat, nh, nkb = 3, 4, 40
    rng = np.random.default_rng(7)
    ofsbeta = np.asfortranarray(np.array([0, 10, 20], dtype=np.int32))
    becphi = np.asfortranarray(rng.standard_normal(nkb))
    deexx0 = np.asfortranarray(rng.standard_normal(nkb))
    weight = 0.5

    d_ref = deexx0.copy(order="F")
    ref.paw_accum(weight, becphi, d_ref, ofsbeta, nh)  # nat/nkb derived by f2py

    d_sdfg = deexx0.copy(order="F")
    sdfg(weight=np.float64(weight),
         becphi=becphi,
         deexx=d_sdfg,
         ofsbeta=ofsbeta,
         nat=np.int32(nat),
         nh=np.int32(nh),
         nkb=np.int32(nkb))

    np.testing.assert_allclose(d_sdfg, d_ref, rtol=1e-12, atol=1e-12)
    # The accumulation actually happened (guard against a no-op pass).
    assert not np.allclose(d_sdfg, deexx0)


# ---------------------------------------------------------------------------
# 2. Inline indirect index -- the offset read sits directly in the subscript
#    (no named ijkb0 scalar to re-materialise from).
# ---------------------------------------------------------------------------
_INLINE = """
subroutine inline_scatter(src, out, ofsbeta, nat, nh, nkb)
  integer, intent(in) :: nat, nh, nkb
  integer, intent(in) :: ofsbeta(nat)
  real(8), intent(in) :: src(nkb)
  real(8), intent(out) :: out(nkb)
  integer :: ih, na
  do na = 1, nkb
    out(na) = 0.0d0
  end do
  do na = 1, nat
    do ih = 1, nh
      out(ofsbeta(na) + ih) = src(ofsbeta(na) + ih) * 3.0d0
    end do
  end do
end subroutine inline_scatter
"""


def test_inline_indirect_index_matches_reference(tmp_path: Path):
    """``out(ofsbeta(na)+ih) = src(ofsbeta(na)+ih)*3`` -- the indirect read
    inlined into the subscript arithmetic (no intermediate scalar)."""
    sdfg = build_sdfg(_module_wrap(_INLINE, "subroutine inline_scatter", "inl_mod"),
                      tmp_path / "sdfg",
                      name="inline_scatter",
                      entry="inl_mod::inline_scatter").build()
    sdfg.validate()
    ref = f2py_compile(_INLINE, tmp_path / "ref", "inl_ref", only=("inline_scatter", ))

    nat, nh, nkb = 3, 4, 40
    rng = np.random.default_rng(11)
    ofsbeta = np.asfortranarray(np.array([0, 10, 20], dtype=np.int32))
    src = np.asfortranarray(rng.standard_normal(nkb))

    o_ref = ref.inline_scatter(src, ofsbeta, nh)  # out intent(out) -> returned

    o_sdfg = np.zeros(nkb, dtype=np.float64, order="F")
    sdfg(src=src, out=o_sdfg, ofsbeta=ofsbeta, nat=np.int32(nat), nh=np.int32(nh), nkb=np.int32(nkb))

    np.testing.assert_allclose(o_sdfg, o_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. 2-D indirect read -- the add_nlxx_pot ``vkbp(ig, ikb)`` shape, where the
#    second subscript is the indirect-derived ``ikb`` and the first is a loop.
# ---------------------------------------------------------------------------
_READ2D = """
subroutine nlxx_apply(vkbp, deexx, ofsbeta, hpsi, nat, nh, nkb, npw)
  integer, intent(in) :: nat, nh, nkb, npw
  integer, intent(in) :: ofsbeta(nat)
  real(8), intent(in) :: vkbp(npw, nkb)
  real(8), intent(in) :: deexx(nkb)
  real(8), intent(inout) :: hpsi(npw)
  integer :: ih, na, ikb, ig
  do na = 1, nat
    do ih = 1, nh
      ikb = ofsbeta(na) + ih
      do ig = 1, npw
        hpsi(ig) = hpsi(ig) - deexx(ikb) * vkbp(ig, ikb)
      end do
    end do
  end do
end subroutine nlxx_apply
"""


def test_2d_indirect_read_matches_reference(tmp_path: Path):
    """``hpsi(ig) -= deexx(ikb)*vkbp(ig, ikb)`` with ``ikb = ofsbeta(na)+ih`` --
    a 2-D read whose column index is indirect-derived (add_nlxx_pot)."""
    sdfg = build_sdfg(_module_wrap(_READ2D, "subroutine nlxx_apply", "nlxx_mod"),
                      tmp_path / "sdfg",
                      name="nlxx_apply",
                      entry="nlxx_mod::nlxx_apply").build()
    sdfg.validate()
    ref = f2py_compile(_READ2D, tmp_path / "ref", "nlxx_ref", only=("nlxx_apply", ))

    nat, nh, nkb, npw = 3, 4, 40, 5
    rng = np.random.default_rng(13)
    ofsbeta = np.asfortranarray(np.array([0, 10, 20], dtype=np.int32))
    vkbp = np.asfortranarray(rng.standard_normal((npw, nkb)))
    deexx = np.asfortranarray(rng.standard_normal(nkb))
    hpsi0 = np.asfortranarray(rng.standard_normal(npw))

    h_ref = hpsi0.copy(order="F")
    ref.nlxx_apply(vkbp, deexx, ofsbeta, h_ref, nh)  # nat/nkb/npw derived

    h_sdfg = hpsi0.copy(order="F")
    sdfg(vkbp=vkbp,
         deexx=deexx,
         ofsbeta=ofsbeta,
         hpsi=h_sdfg,
         nat=np.int32(nat),
         nh=np.int32(nh),
         nkb=np.int32(nkb),
         npw=np.int32(npw))

    np.testing.assert_allclose(h_sdfg, h_ref, rtol=1e-12, atol=1e-12)
    assert not np.allclose(h_sdfg, hpsi0)


# ---------------------------------------------------------------------------
# 4. Module-global ``ofsbeta`` -- the actual QE shape (``USE uspp, ONLY:
#    ofsbeta``).  The offset table is an allocatable module global, so it
#    reaches the SDFG as a host-sourced kwarg rather than a dummy.
# ---------------------------------------------------------------------------
_MODGLOBAL = """
module uspp_dx
  implicit none
  integer, allocatable :: ofsbeta(:)
contains
  subroutine paw_accum_g(becphi, deexx, nat, nh, nkb)
    integer, intent(in) :: nat, nh, nkb
    real(8), intent(in) :: becphi(nkb)
    real(8), intent(inout) :: deexx(nkb)
    integer :: ijkb0, ih, na, ikb
    do na = 1, nat
      ijkb0 = ofsbeta(na)
      do ih = 1, nh
        ikb = ijkb0 + ih
        deexx(ikb) = deexx(ikb) + becphi(ikb) * 2.0d0
      end do
    end do
  end subroutine paw_accum_g
end module uspp_dx
"""


def test_module_global_ofsbeta_indirect_matches_reference(tmp_path: Path):
    """QE-faithful: ``ofsbeta`` is an allocatable MODULE global (``USE uspp``).
    It surfaces as a host-sourced kwarg on the SDFG; the f2py reference sets
    the module data directly.  Both must compute the same accumulation."""
    sdfg = build_sdfg(_MODGLOBAL, tmp_path / "sdfg", name="paw_accum_g", entry="uspp_dx::paw_accum_g").build()
    sdfg.validate()
    assert "ofsbeta" in sdfg.arglist(), "module-global ofsbeta must surface as a kwarg"

    ref = f2py_compile(_MODGLOBAL, tmp_path / "ref", "modg_ref", only=("paw_accum_g", ))

    nat, nh, nkb = 3, 4, 40
    rng = np.random.default_rng(17)
    ofsbeta = np.asfortranarray(np.array([0, 10, 20], dtype=np.int32))
    becphi = np.asfortranarray(rng.standard_normal(nkb))
    deexx0 = np.asfortranarray(rng.standard_normal(nkb))

    ref.uspp_dx.ofsbeta = ofsbeta  # f2py allocates + fills the module global
    d_ref = deexx0.copy(order="F")
    ref.uspp_dx.paw_accum_g(becphi, d_ref, nat, nh, nkb)

    d_sdfg = deexx0.copy(order="F")
    sdfg(becphi=becphi, deexx=d_sdfg, ofsbeta=ofsbeta, nat=np.int32(nat), nh=np.int32(nh), nkb=np.int32(nkb))

    np.testing.assert_allclose(d_sdfg, d_ref, rtol=1e-12, atol=1e-12)
    assert not np.allclose(d_sdfg, deexx0)
