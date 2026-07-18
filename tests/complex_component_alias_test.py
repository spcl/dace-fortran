"""Complex-as-2-reals sequence association (``REAL(2,N)`` dummy aliasing a
``COMPLEX`` array element) -- the QE ``qvan2`` ``qg(2,ngy) <- qgm(1,ijh)`` pattern.

DaCe can't View a ``complex128`` array as ``float64`` (no dtype reinterpret), so
each ``qg(c,i)`` access is rewritten to a component access of ``z[i-1]``:
``qg(1,i)`` -> ``re(z[i-1])``, ``qg(2,i)`` -> ``im(z[i-1])``, emitted via
``dace::math::re``/``im``; a component write is a read-modify-write reconstructing
the complex via ``re/im + 1j``.

Tier 1: DaCe-level tests pin the re/im codegen mechanism.  Tier 2: Fortran-level
tests drive the bridge end-to-end on the seq-assoc pattern.
"""
import numpy as np
import pytest

import dace


# ---------------------------------------------------------------------------
# Tier 1 -- the dace::math::re / im codegen mechanism.
# ---------------------------------------------------------------------------
def test_dace_re_im_read_components(tmp_path):
    """``re(_in)``/``im(_in)`` in a tasklet read a complex connector's real/imaginary parts."""
    sdfg = dace.SDFG('cc_read')
    sdfg.add_array('z', (4, ), dace.complex128)
    sdfg.add_array('outr', (4, ), dace.float64)
    sdfg.add_array('outi', (4, ), dace.float64)
    st = sdfg.add_state('s', is_start_block=True)
    rz = st.add_read('z')
    wr = st.add_write('outr')
    wi = st.add_write('outi')
    me, mx = st.add_map('m', dict(i='0:4'))
    t = st.add_tasklet('t', {'zin'}, {'orr', 'oii'}, 'orr = re(zin)\noii = im(zin)')
    st.add_memlet_path(rz, me, t, dst_conn='zin', memlet=dace.Memlet('z[i]'))
    st.add_memlet_path(t, mx, wr, src_conn='orr', memlet=dace.Memlet('outr[i]'))
    st.add_memlet_path(t, mx, wi, src_conn='oii', memlet=dace.Memlet('outi[i]'))
    sdfg.validate()
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex128)
    orr = np.zeros(4)
    oii = np.zeros(4)
    sdfg(z=z, outr=orr, outi=oii)
    assert orr.tolist() == [1, 3, 5, 7]
    assert oii.tolist() == [2, 4, 6, 8]


@pytest.mark.parametrize("ind,expect", [(1, 'real'), (2, 'imag')])
def test_dace_component_rmw_runtime_ind(ind, expect):
    """Component RMW with a RUNTIME index ``ind`` in {1,2}: adds 1 to re (ind==1) or im
    (ind==2) of ``z[i]``, reconstructed via ``component + 1j*other``.  QE ``qvan2`` shape."""
    sdfg = dace.SDFG('cc_rmw')
    sdfg.add_array('z', (4, ), dace.complex128)
    sdfg.add_symbol('ind', dace.int64)
    st = sdfg.add_state('s', is_start_block=True)
    rz = st.add_read('z')
    wz = st.add_write('z')
    me, mx = st.add_map('m', dict(i='0:4'))
    code = ("_cur = (re(zin) if (ind == 1) else im(zin))\n"
            "_new = _cur + 1.0\n"
            "zout = (_new + 1j*im(zin)) if (ind == 1) else (re(zin) + 1j*_new)")
    t = st.add_tasklet('rmw', {'zin'}, {'zout'}, code)
    st.add_memlet_path(rz, me, t, dst_conn='zin', memlet=dace.Memlet('z[i]'))
    st.add_memlet_path(t, mx, wz, src_conn='zout', memlet=dace.Memlet('z[i]'))
    sdfg.validate()
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex128)
    sdfg(z=z, ind=ind)
    if expect == 'real':  # re += 1
        assert [zz.real for zz in z] == [2, 4, 6, 8]
        assert [zz.imag for zz in z] == [2, 4, 6, 8]
    else:  # im += 1
        assert [zz.real for zz in z] == [1, 3, 5, 7]
        assert [zz.imag for zz in z] == [3, 5, 7, 9]


# ---------------------------------------------------------------------------
# Tier 2 -- the Fortran seq-assoc pattern end-to-end through the bridge.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import build_sdfg, have_flang  # noqa: E402

_needs_flang = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run_seq_assoc(tmp_path, ind, src_decls, call):
    """Build + run a ``REAL(2,N)`` dummy bound to a COMPLEX element kernel; ``ind``
    selects the component written (1=real, 2=imag)."""
    # fill stays external (implicit interface) for COMPLEX-element -> REAL(2,n) seq
    # assoc; only the entry driver is module-wrapped.
    src = f"""
module driver_mod
  implicit none
contains
  subroutine driver(z, n)
    implicit none
    integer, intent(in) :: n
    {src_decls}
    {call}
  end subroutine driver
end module driver_mod

subroutine fill(ngy, qg)
  implicit none
  integer, intent(in) :: ngy
  real(8), intent(inout) :: qg(2, ngy)
  integer :: ig, ind
  ind = {ind}
  do ig = 1, ngy
    qg(ind, ig) = qg(ind, ig) + 1.0d0
  end do
end subroutine fill
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    n = 4
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex128)
    sdfg(z=z, n=np.int32(n))
    return z


@_needs_flang
def test_seq_assoc_imag_component(tmp_path):
    """``ind=2`` -> imaginary part of each ``z`` element gets +1, real unchanged."""
    z = _run_seq_assoc(tmp_path, 2, "complex(8), intent(inout) :: z(n)", "call fill(n, z(1))")
    assert [zz.real for zz in z] == [1, 3, 5, 7]
    assert [zz.imag for zz in z] == [3, 5, 7, 9]


@_needs_flang
def test_seq_assoc_real_component(tmp_path):
    """``ind=1`` -> real part of each ``z`` element gets +1, imag unchanged."""
    z = _run_seq_assoc(tmp_path, 1, "complex(8), intent(inout) :: z(n)", "call fill(n, z(1))")
    assert [zz.real for zz in z] == [2, 4, 6, 8]
    assert [zz.imag for zz in z] == [2, 4, 6, 8]


def _build_run_fill(tmp_path, fill_body, z0):
    """Build + run a driver whose inner ``fill(ngy, qg)`` has the given body (``qg`` is
    the ``REAL(2,ngy)`` complex-component alias); returns the array seeded from ``z0``."""
    # fill stays external (implicit interface) for COMPLEX-element -> REAL(2,n) seq
    # assoc; only the entry is module-wrapped.
    src = f"""
module driver_mod
  implicit none
contains
  subroutine driver(z, n)
    implicit none
    integer, intent(in) :: n
    complex(8), intent(inout) :: z(n)
    call fill(n, z(1))
  end subroutine driver
end module driver_mod

subroutine fill(ngy, qg)
  implicit none
  integer, intent(in) :: ngy
  real(8), intent(inout) :: qg(2, ngy)
  integer :: ig
{fill_body}
end subroutine fill
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    z = np.array(z0, dtype=np.complex128)
    sdfg(z=z, n=np.int32(len(z0)))
    return z


@_needs_flang
def test_seq_assoc_whole_zero(tmp_path):
    """``qg = 0`` zeros BOTH components of every element; lowers as a memset of the
    COMPLEX view, written back to the aliased source slab."""
    z = _build_run_fill(tmp_path, "  qg = 0.0d0", [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
    assert [zz.real for zz in z] == [0, 0, 0, 0]
    assert [zz.imag for zz in z] == [0, 0, 0, 0]


@_needs_flang
def test_seq_assoc_rhs_reads_other_array(tmp_path):
    """Component RMW rhs references an ORDINARY array + literal alongside the ``qg``
    self-read: exercises per-occurrence connector wiring for non-``qg`` reads inside the
    complex-component tasklet (QE ``sig*ylmk0(ig,lp)*work`` shape)."""
    # fill stays external (implicit interface) for COMPLEX-element -> REAL(2,n) seq
    # assoc; only the entry is module-wrapped.
    src = """
module driver_mod
  implicit none
contains
  subroutine driver(z, w, n)
    implicit none
    integer, intent(in) :: n
    complex(8), intent(inout) :: z(n)
    real(8), intent(in) :: w(n)
    call fill(n, z(1), w)
  end subroutine driver
end module driver_mod

subroutine fill(ngy, qg, w)
  implicit none
  integer, intent(in) :: ngy
  real(8), intent(inout) :: qg(2, ngy)
  real(8), intent(in) :: w(ngy)
  integer :: ig, ind
  ind = 1
  do ig = 1, ngy
    qg(ind, ig) = qg(ind, ig) + w(ig) * 2.0d0
  end do
end subroutine fill
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex128)
    w = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    sdfg(z=z, w=w, n=np.int32(4))
    # re += w*2 ; imag unchanged
    assert [zz.real for zz in z] == [1 + 20, 3 + 40, 5 + 60, 7 + 80]
    assert [zz.imag for zz in z] == [2, 4, 6, 8]


@_needs_flang
def test_seq_assoc_zero_then_set_components(tmp_path):
    """``qg = 0`` then per-element ``qg(1,ig)=3``/``qg(2,ig)=4``: the memset and the
    component RMW writes compose -- every element ends at ``3 + 4j``."""
    body = ("  qg = 0.0d0\n"
            "  do ig = 1, ngy\n"
            "    qg(1, ig) = 3.0d0\n"
            "    qg(2, ig) = 4.0d0\n"
            "  end do")
    z = _build_run_fill(tmp_path, body, [1 + 2j, 9 - 1j, 5 + 6j, 0 + 0j])
    assert [zz.real for zz in z] == [3, 3, 3, 3]
    assert [zz.imag for zz in z] == [4, 4, 4, 4]
