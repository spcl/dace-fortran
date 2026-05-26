"""Quantum-ESPRESSO ``addusxx_g`` augmentation kernel, layout matrix.

This is the US/PAW exchange augmentation hot loop from QE's
``addusxx_g`` (SC26 ``Experiments/E5_USXX/usxx_kernels.cu``), reduced
to small symbolic sizes but kept structurally faithful: for each atom
of the active type, a doubly-nested projector contraction builds a
per-G accumulator, then a structure factor scales it and the result
is written into the charge grid.

    aux1(g)   = sum_jh  qgm(g, ijtoh(ih,jh)) * becpsi(ijkb0+jh)
    aux2(g)   = sum_ih  aux1(g) * conjg(becphi(ijkb0+ih))
    sf(g)     = eigqts(na) * eigts1(m1) * eigts2(m2) * eigts3(m3)
    rhoc(.)  += aux2(g) * sf(g)

It exercises, in one kernel, the pieces a layout transform has to get
right: complex(8) multiply-accumulate, ``conjg``, a column gather on
``qgm`` (via ``ijtoh``), a read gather on the ``eigts*`` phase tables
(via the Miller indices ``mill*``), and an accumulating write to
``rhoc``.

Layouts:

* **AoS** -- ``complex(8)`` arrays (re / im interleaved, the natural
  Fortran complex layout).
* **SoA** -- every complex array split into paired ``real(8)``
  ``*_re`` / ``*_im`` arrays with the complex arithmetic expanded by
  hand; the layout-transformed form the SC26 paper sweeps.

Indirection on the final write:

* **single** -- ``rhoc(g)``        (read gathers only; dense write).
* **double** -- ``rhoc(nlmap(g))`` (read gathers + scatter write).

The reference is a complex-space numpy reimplementation, NOT an
f2py build of each layout: it is the layout-*independent* ground
truth both AoS and SoA must reproduce, so it catches an SDFG bug and
a hand-expansion bug in the SoA arithmetic alike (an f2py reference
built from each layout's own source could not -- it would share the
bug).  Inputs come from the same xorshift64 stream the SC26 artifacts
use; see :mod:`_prng`.

The real kernel selects atoms by a string species flag; per the QE
porting convention that string flag is rewritten as an integer enum
(``ityp == nt``) at port time, not lowered as a string.
"""
import numpy as np
import pytest

from _prng import complex_stream
from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


# Small symbolic problem sizes -- a correctness test, not a benchmark.
# ``ncol = nh*nh`` qgm columns, ``nbeta = nat*nh`` projector entries.
_NGMS, _NH, _NAT, _NT, _NM, _NRHO = 5, 3, 3, 2, 7, 9


def _aos_src(write_idx: str) -> str:
    """AoS ``complex(8)`` augmentation kernel; ``write_idx`` is the
    ``rhoc`` subscript (``g`` dense / ``nlmap(g)`` scatter)."""
    return f"""
subroutine addusxx_aos(ngms, nh, nat, nt, nm, nrho, ofsbeta, ityp, ijtoh, &
                       becphi, becpsi, qgm, eigqts, eigts1, eigts2, eigts3, &
                       mill1, mill2, mill3, nlmap, rhoc)
  implicit none
  integer, intent(in) :: ngms, nh, nat, nt, nm, nrho
  integer, intent(in) :: ofsbeta(nat), ityp(nat), ijtoh(nh, nh)
  integer, intent(in) :: mill1(ngms), mill2(ngms), mill3(ngms), nlmap(ngms)
  complex(8), intent(in) :: becphi(nat*nh), becpsi(nat*nh)
  complex(8), intent(in) :: qgm(ngms, nh*nh), eigqts(nat)
  complex(8), intent(in) :: eigts1(nm, nat), eigts2(nm, nat), eigts3(nm, nat)
  complex(8), intent(inout) :: rhoc(nrho)
  complex(8) :: aux1(ngms), aux2(ngms), cbphi, sf
  integer :: na, ih, jh, g, ijkb0, col
  do na = 1, nat
    if (ityp(na) == nt) then
      ijkb0 = ofsbeta(na)
      do g = 1, ngms
        aux2(g) = (0.0d0, 0.0d0)
      end do
      do ih = 1, nh
        do g = 1, ngms
          aux1(g) = (0.0d0, 0.0d0)
        end do
        do jh = 1, nh
          col = ijtoh(ih, jh)
          do g = 1, ngms
            aux1(g) = aux1(g) + qgm(g, col) * becpsi(ijkb0 + jh)
          end do
        end do
        cbphi = conjg(becphi(ijkb0 + ih))
        do g = 1, ngms
          aux2(g) = aux2(g) + aux1(g) * cbphi
        end do
      end do
      do g = 1, ngms
        sf = eigqts(na) * eigts1(mill1(g), na) &
                        * eigts2(mill2(g), na) * eigts3(mill3(g), na)
        rhoc({write_idx}) = rhoc({write_idx}) + aux2(g) * sf
      end do
    end if
  end do
end subroutine addusxx_aos
"""


def _soa_src(write_idx: str) -> str:
    """SoA paired-real augmentation kernel: the same computation as
    :func:`_aos_src` with every complex op expanded into real / imag
    parts (multiply ``(ac-bd, ad+bc)``; ``conjg`` negates imag)."""
    return f"""
subroutine addusxx_soa(ngms, nh, nat, nt, nm, nrho, ofsbeta, ityp, ijtoh, &
                       becphi_re, becphi_im, becpsi_re, becpsi_im, &
                       qgm_re, qgm_im, eigqts_re, eigqts_im, &
                       eigts1_re, eigts1_im, eigts2_re, eigts2_im, &
                       eigts3_re, eigts3_im, mill1, mill2, mill3, nlmap, &
                       rhoc_re, rhoc_im)
  implicit none
  integer, intent(in) :: ngms, nh, nat, nt, nm, nrho
  integer, intent(in) :: ofsbeta(nat), ityp(nat), ijtoh(nh, nh)
  integer, intent(in) :: mill1(ngms), mill2(ngms), mill3(ngms), nlmap(ngms)
  real(8), intent(in) :: becphi_re(nat*nh), becphi_im(nat*nh)
  real(8), intent(in) :: becpsi_re(nat*nh), becpsi_im(nat*nh)
  real(8), intent(in) :: qgm_re(ngms, nh*nh), qgm_im(ngms, nh*nh)
  real(8), intent(in) :: eigqts_re(nat), eigqts_im(nat)
  real(8), intent(in) :: eigts1_re(nm, nat), eigts1_im(nm, nat)
  real(8), intent(in) :: eigts2_re(nm, nat), eigts2_im(nm, nat)
  real(8), intent(in) :: eigts3_re(nm, nat), eigts3_im(nm, nat)
  real(8), intent(inout) :: rhoc_re(nrho), rhoc_im(nrho)
  real(8) :: aux1_re(ngms), aux1_im(ngms), aux2_re(ngms), aux2_im(ngms)
  real(8) :: cbphi_re, cbphi_im, br, bi
  real(8) :: t1_re, t1_im, t2_re, t2_im, sf_re, sf_im
  integer :: na, ih, jh, g, ijkb0, col
  do na = 1, nat
    if (ityp(na) == nt) then
      ijkb0 = ofsbeta(na)
      do g = 1, ngms
        aux2_re(g) = 0.0d0
        aux2_im(g) = 0.0d0
      end do
      do ih = 1, nh
        do g = 1, ngms
          aux1_re(g) = 0.0d0
          aux1_im(g) = 0.0d0
        end do
        do jh = 1, nh
          col = ijtoh(ih, jh)
          br = becpsi_re(ijkb0 + jh)
          bi = becpsi_im(ijkb0 + jh)
          do g = 1, ngms
            aux1_re(g) = aux1_re(g) + qgm_re(g, col) * br - qgm_im(g, col) * bi
            aux1_im(g) = aux1_im(g) + qgm_re(g, col) * bi + qgm_im(g, col) * br
          end do
        end do
        cbphi_re = becphi_re(ijkb0 + ih)
        cbphi_im = -becphi_im(ijkb0 + ih)
        do g = 1, ngms
          aux2_re(g) = aux2_re(g) + aux1_re(g) * cbphi_re - aux1_im(g) * cbphi_im
          aux2_im(g) = aux2_im(g) + aux1_re(g) * cbphi_im + aux1_im(g) * cbphi_re
        end do
      end do
      do g = 1, ngms
        t1_re = eigqts_re(na) * eigts1_re(mill1(g), na) - eigqts_im(na) * eigts1_im(mill1(g), na)
        t1_im = eigqts_re(na) * eigts1_im(mill1(g), na) + eigqts_im(na) * eigts1_re(mill1(g), na)
        t2_re = t1_re * eigts2_re(mill2(g), na) - t1_im * eigts2_im(mill2(g), na)
        t2_im = t1_re * eigts2_im(mill2(g), na) + t1_im * eigts2_re(mill2(g), na)
        sf_re = t2_re * eigts3_re(mill3(g), na) - t2_im * eigts3_im(mill3(g), na)
        sf_im = t2_re * eigts3_im(mill3(g), na) + t2_im * eigts3_re(mill3(g), na)
        rhoc_re({write_idx}) = rhoc_re({write_idx}) + aux2_re(g) * sf_re - aux2_im(g) * sf_im
        rhoc_im({write_idx}) = rhoc_im({write_idx}) + aux2_re(g) * sf_im + aux2_im(g) * sf_re
      end do
    end if
  end do
end subroutine addusxx_soa
"""


def _inputs(seed_base: int = 0):
    """Generate one consistent input set shared by AoS and SoA.

    Integer index tables are 1-based (Fortran convention).  ``ityp`` is
    the integer-enum species flag; atoms with ``ityp == _NT`` are
    active.  ``ofsbeta(na) = (na-1)*nh`` slots each atom's ``nh``
    projector entries into ``becphi`` / ``becpsi``.  ``mill*`` gather
    the phase tables; ``nlmap`` is a distinct-target permutation sample
    so the scatter write has no intra-atom collisions and the
    comparison stays exact.
    """
    rng = np.random.default_rng(seed_base)
    ncol, nbeta = _NH * _NH, _NAT * _NH

    ofsbeta = (np.arange(_NAT) * _NH).astype(np.int32)
    ityp = np.array([1, _NT, _NT], dtype=np.int32)
    ijtoh = np.asfortranarray(
        (np.arange(_NH)[:, None] * _NH + np.arange(_NH) + 1).astype(np.int32))
    mill1 = rng.integers(1, _NM + 1, _NGMS, dtype=np.int32)
    mill2 = rng.integers(1, _NM + 1, _NGMS, dtype=np.int32)
    mill3 = rng.integers(1, _NM + 1, _NGMS, dtype=np.int32)
    nlmap = (rng.permutation(_NRHO)[:_NGMS] + 1).astype(np.int32)

    becphi = complex_stream(nbeta, seed=1)
    becpsi = complex_stream(nbeta, seed=2)
    qgm = np.asfortranarray(complex_stream(_NGMS * ncol, seed=3).reshape(_NGMS, ncol))
    eigqts = complex_stream(_NAT, seed=4)
    eigts1 = np.asfortranarray(complex_stream(_NM * _NAT, seed=5).reshape(_NM, _NAT))
    eigts2 = np.asfortranarray(complex_stream(_NM * _NAT, seed=6).reshape(_NM, _NAT))
    eigts3 = np.asfortranarray(complex_stream(_NM * _NAT, seed=7).reshape(_NM, _NAT))
    rhoc = complex_stream(_NRHO, seed=8)
    return dict(ofsbeta=ofsbeta, ityp=ityp, ijtoh=ijtoh, mill1=mill1, mill2=mill2,
                mill3=mill3, nlmap=nlmap, becphi=becphi, becpsi=becpsi, qgm=qgm,
                eigqts=eigqts, eigts1=eigts1, eigts2=eigts2, eigts3=eigts3, rhoc=rhoc)


def _ref(d: dict, scatter: bool) -> np.ndarray:
    """Complex-space ground truth -- the layout-independent reference."""
    out = d["rhoc"].copy()
    for na in range(_NAT):
        if d["ityp"][na] != _NT:
            continue
        ijkb0 = int(d["ofsbeta"][na])
        aux2 = np.zeros(_NGMS, dtype=np.complex128)
        for ih in range(_NH):
            aux1 = np.zeros(_NGMS, dtype=np.complex128)
            for jh in range(_NH):
                col = int(d["ijtoh"][ih, jh]) - 1
                aux1 += d["qgm"][:, col] * d["becpsi"][ijkb0 + jh]
            aux2 += aux1 * np.conj(d["becphi"][ijkb0 + ih])
        for g in range(_NGMS):
            sf = (d["eigqts"][na] * d["eigts1"][d["mill1"][g] - 1, na]
                  * d["eigts2"][d["mill2"][g] - 1, na] * d["eigts3"][d["mill3"][g] - 1, na])
            tgt = (int(d["nlmap"][g]) - 1) if scatter else g
            out[tgt] += aux2[g] * sf
    return out


def _dims() -> dict:
    """The SDFG's shape symbols, passed alongside the arrays."""
    return dict(ngms=np.int32(_NGMS), nh=np.int32(_NH), nat=np.int32(_NAT),
                nt=np.int32(_NT), nm=np.int32(_NM), nrho=np.int32(_NRHO))


@pytest.mark.parametrize("indir", ["single", "double"])
def test_addusxx_aos(tmp_path, indir):
    """AoS complex(8) augmentation -- read gathers, dense or scatter write."""
    d = _inputs()
    ref = _ref(d, scatter=(indir == "double"))
    rhoc = np.asfortranarray(d["rhoc"].copy())

    sdfg = build_sdfg(_aos_src("nlmap(g)" if indir == "double" else "g"),
                      tmp_path, name="addusxx_aos", entry="addusxx_aos").build()
    sdfg(**_dims(), ofsbeta=d["ofsbeta"], ityp=d["ityp"], ijtoh=d["ijtoh"],
         becphi=np.asfortranarray(d["becphi"]), becpsi=np.asfortranarray(d["becpsi"]),
         qgm=d["qgm"], eigqts=np.asfortranarray(d["eigqts"]),
         eigts1=d["eigts1"], eigts2=d["eigts2"], eigts3=d["eigts3"],
         mill1=d["mill1"], mill2=d["mill2"], mill3=d["mill3"], nlmap=d["nlmap"],
         rhoc=rhoc)
    np.testing.assert_allclose(rhoc, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("indir", ["single", "double"])
def test_addusxx_soa(tmp_path, indir):
    """SoA paired real(8) augmentation -- same matrix, layout-transformed."""
    d = _inputs()
    ref = _ref(d, scatter=(indir == "double"))
    rhoc_re = np.asfortranarray(d["rhoc"].real.copy())
    rhoc_im = np.asfortranarray(d["rhoc"].imag.copy())

    def split(name):
        return (np.asfortranarray(d[name].real.copy()),
                np.asfortranarray(d[name].imag.copy()))

    becphi_re, becphi_im = split("becphi")
    becpsi_re, becpsi_im = split("becpsi")
    qgm_re, qgm_im = split("qgm")
    eigqts_re, eigqts_im = split("eigqts")
    eigts1_re, eigts1_im = split("eigts1")
    eigts2_re, eigts2_im = split("eigts2")
    eigts3_re, eigts3_im = split("eigts3")

    sdfg = build_sdfg(_soa_src("nlmap(g)" if indir == "double" else "g"),
                      tmp_path, name="addusxx_soa", entry="addusxx_soa").build()
    sdfg(**_dims(), ofsbeta=d["ofsbeta"], ityp=d["ityp"], ijtoh=d["ijtoh"],
         becphi_re=becphi_re, becphi_im=becphi_im,
         becpsi_re=becpsi_re, becpsi_im=becpsi_im,
         qgm_re=qgm_re, qgm_im=qgm_im, eigqts_re=eigqts_re, eigqts_im=eigqts_im,
         eigts1_re=eigts1_re, eigts1_im=eigts1_im,
         eigts2_re=eigts2_re, eigts2_im=eigts2_im,
         eigts3_re=eigts3_re, eigts3_im=eigts3_im,
         mill1=d["mill1"], mill2=d["mill2"], mill3=d["mill3"], nlmap=d["nlmap"],
         rhoc_re=rhoc_re, rhoc_im=rhoc_im)
    np.testing.assert_allclose(rhoc_re, ref.real, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(rhoc_im, ref.imag, rtol=1e-11, atol=1e-12)
