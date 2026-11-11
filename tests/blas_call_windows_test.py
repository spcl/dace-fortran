# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
"""A BLAS call means its sizes, increments and flags: nothing beyond them, and nothing silently."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg


def lib_nodes(sdfg):
    return [node for node, parent in sdfg.all_nodes_recursive() if type(node).__module__.startswith("dace.libraries")]


def test_dscal_touches_only_its_window(tmp_path: Path):
    src = """
subroutine probe_scal(x)
  double precision, intent(inout) :: x(10)
  call dscal(5, 2d0, x, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_scal").build()
    x = np.ones(10)
    sdfg(x=x)
    assert x.tolist() == [2.0] * 5 + [1.0] * 5


def test_daxpy_length_is_its_argument(tmp_path: Path):
    src = """
subroutine probe_axpy(m, x, y)
  integer, intent(in) :: m
  double precision, intent(in) :: x(8)
  double precision, intent(inout) :: y(8)
  call daxpy(m, 2d0, x, 1, y, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_axpy").build()
    x, y = np.ones(8), np.zeros(8)
    sdfg(m=np.int32(3), x=x, y=y)
    assert y.tolist() == [2.0] * 3 + [0.0] * 5


def test_dgemv_reads_its_window(tmp_path: Path):
    src = """
subroutine probe_gemv(a, x, y)
  double precision, intent(in) :: a(4, 4), x(4)
  double precision, intent(inout) :: y(4)
  call dgemv('N', 2, 2, 1d0, a, 4, x, 1, 0d0, y, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_gemv").build()
    a = np.arange(1.0, 17.0).reshape(4, 4, order="F").copy(order="F")
    x, y = np.ones(4), np.full(4, -1.0)
    sdfg(a=a, x=x, y=y)
    assert y.tolist() == [6.0, 8.0, -1.0, -1.0]


COMPLEX_OPS = {"N": lambda m: m, "T": lambda m: m.T, "C": lambda m: m.conj().T}


@pytest.mark.parametrize("flag", COMPLEX_OPS)
def test_zgemm_transposes_and_conjugates_its_window(tmp_path: Path, flag):
    """QE's ``calbec`` shape: ``op(A)`` over the first ``n`` rows of a taller ``A``, with literal complex scalars."""
    rows, cols = ("n", "k") if flag == "N" else ("k", "n")
    src = f"""
subroutine probe_zgemm(a, b, c, n, k)
  integer, intent(in) :: n, k
  complex(8), intent(in) :: a(5, 5), b(5, 5)
  complex(8), intent(inout) :: c(5, 5)
  call zgemm('{flag}', 'N', n, n, k, (2.0d0, 0.0d0), a, 5, b, 5, (0.0d0, 0.0d0), c, 5)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_zgemm").build()
    rng = np.random.default_rng(0)
    a, b = (np.asfortranarray(rng.random((5, 5)) + 1j * rng.random((5, 5))) for _ in range(2))
    c = np.full((5, 5), -1.0 + 0j, order="F")
    n, k = 2, 3
    sdfg(a=a, b=b, c=c, n=np.int32(n), k=np.int32(k))
    extent = {"n": n, "k": k}
    expected = np.full((5, 5), -1.0 + 0j)
    expected[:n, :n] = 2.0 * COMPLEX_OPS[flag](a[: extent[rows], : extent[cols]]) @ b[:k, :n]
    np.testing.assert_allclose(c, expected)


@pytest.mark.parametrize("flag", COMPLEX_OPS)
def test_zgemv_transposes_and_conjugates(tmp_path: Path, flag):
    src = f"""
subroutine probe_zgemv(a, x, y)
  complex(8), intent(in) :: a(3, 3), x(3)
  complex(8), intent(inout) :: y(3)
  call zgemv('{flag}', 3, 3, (1.0d0, 0.0d0), a, 3, x, 1, (0.0d0, 0.0d0), y, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_zgemv").build()
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.random((3, 3)) + 1j * rng.random((3, 3)))
    x, y = rng.random(3) + 1j * rng.random(3), np.zeros(3, complex)
    sdfg(a=a, x=x, y=y)
    np.testing.assert_allclose(y, COMPLEX_OPS[flag](a) @ x)


@pytest.mark.parametrize("flag", ["N", "C"])
def test_zgemv_scales_by_a_complex_literal(tmp_path: Path, flag):
    """A literal ``alpha`` with an imaginary part: Gemv's symbolic ``alpha`` property cannot hold it."""
    src = f"""
subroutine probe_literal(a, x, y)
  complex(8), intent(in) :: a(3, 3), x(3)
  complex(8), intent(inout) :: y(3)
  call zgemv('{flag}', 3, 3, (0.5d0, 2.0d0), a, 3, x, 1, (1.0d0, 0.0d0), y, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_literal").build()
    rng = np.random.default_rng(4)
    a = np.asfortranarray(rng.random((3, 3)) + 1j * rng.random((3, 3)))
    x, y = rng.random(3) + 1j * rng.random(3), rng.random(3) + 1j * rng.random(3)
    expected = (0.5 + 2.0j) * COMPLEX_OPS[flag](a) @ x + y
    sdfg(a=a, x=x, y=y)
    np.testing.assert_allclose(y, expected)


@pytest.mark.parametrize("alpha", ["z", "conjg(z)", "-z"])
def test_zgemm_scales_by_a_runtime_alpha(tmp_path: Path, alpha):
    """h_psi's ``zgemm('C', 'N', ..., CONJG(phase), ...)``: ``alpha`` is a COMPLEX value known only at run time."""
    src = f"""
subroutine probe_alpha(z, a, b, c)
  complex(8), intent(in) :: z, a(3, 3), b(3, 3)
  complex(8), intent(inout) :: c(3, 3)
  call zgemm('C', 'N', 3, 3, 3, {alpha}, a, 3, b, 3, (1.0d0, 0.0d0), c, 3)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_alpha").build()
    rng = np.random.default_rng(2)
    a, b, c = (np.asfortranarray(rng.random((3, 3)) + 1j * rng.random((3, 3))) for _ in range(3))
    z = 0.5 - 2.0j
    expected = {"z": z, "conjg(z)": np.conj(z), "-z": -z}[alpha] * a.conj().T @ b + c
    sdfg(z=np.array([z]), a=a, b=b, c=c)
    np.testing.assert_allclose(c, expected)


@pytest.mark.parametrize("alpha", ["w(i, 2)", "-w(i, 2)"])
def test_zgemm_scales_by_an_array_element_alpha(tmp_path: Path, alpha):
    """h_psi's ``zgemm(..., fact_hepsi(ik, pdir), ...)``: ``alpha`` is one element of a COMPLEX array, at a run-time
    index."""
    src = f"""
subroutine probe_element(i, w, a, b, c)
  integer, intent(in) :: i
  complex(8), intent(in) :: w(3, 2), a(3, 3), b(3, 3)
  complex(8), intent(inout) :: c(3, 3)
  call zgemm('N', 'N', 3, 3, 3, {alpha}, a, 3, b, 3, (1.0d0, 0.0d0), c, 3)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_element").build()
    rng = np.random.default_rng(3)
    w, a, b, c = (np.asfortranarray(rng.random(s) + 1j * rng.random(s)) for s in ((3, 2), (3, 3), (3, 3), (3, 3)))
    sign = -1.0 if alpha.startswith("-") else 1.0
    expected = sign * w[1, 1] * a @ b + c
    sdfg(i=np.int32(2), w=w, a=a, b=b, c=c)
    np.testing.assert_allclose(c, expected)


def test_dgemv_sizes_and_alpha_are_array_elements(tmp_path: Path):
    """h_psi's ``dgemm('N', 'N', nh(nt), m, nh(nt), ...)``: a size (and here ``alpha``) is one element of an array at a
    run-time index, not its first element."""
    src = """
subroutine probe_elements(i, m, w, a, x, y)
  integer, intent(in) :: i, m(3)
  double precision, intent(in) :: w(3), a(4, 4), x(4)
  double precision, intent(inout) :: y(4)
  call dgemv('N', m(i), m(i), w(i), a, 4, x, 1, 0d0, y, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_elements").build()
    a = np.arange(1.0, 17.0).reshape(4, 4, order="F").copy(order="F")
    x, y = np.ones(4), np.full(4, -1.0)
    sdfg(i=np.int32(2), m=np.array([4, 2, 3], np.int32), w=np.array([5.0, 3.0, 7.0]), a=a, x=x, y=y)
    assert y.tolist() == [18.0, 24.0, -1.0, -1.0]


@pytest.mark.parametrize(
    "call, node_type, expected",
    [
        ("dsyrk('U', 'T', 4, 4, 1d0, a, 4, 0d0, c, 4)", "Syrk", {"uplo": True, "transA": True}),
        ("dsymm('R', 'U', 4, 4, 1d0, a, 4, b, 4, 0d0, c, 4)", "Symm", {"side": True, "uplo": True}),
    ],
)
def test_literal_flags_reach_the_node(tmp_path: Path, call, node_type, expected):
    src = f"""
subroutine probe_flags(a, b, c)
  double precision, intent(in) :: a(4, 4), b(4, 4)
  double precision, intent(inout) :: c(4, 4)
  call {call}
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe_flags").build()
    (node,) = [n for n in lib_nodes(sdfg) if type(n).__name__ == node_type]
    assert {key: getattr(node, key) for key in expected} == expected


def test_runtime_flag_is_refused(tmp_path: Path):
    src = """
subroutine probe_runtime(trans, a, x, y)
  character, intent(in) :: trans
  double precision, intent(in) :: a(4, 4), x(4)
  double precision, intent(inout) :: y(4)
  call dgemv(trans, 4, 4, 1d0, a, 4, x, 1, 0d0, y, 1)
end subroutine
"""
    with pytest.raises(NotImplementedError, match="run time"):
        build_sdfg(src, tmp_path, name="probe_runtime").build()


def test_negative_increment_is_refused(tmp_path: Path):
    src = """
subroutine probe_negative(a, x, y)
  double precision, intent(in) :: a(2, 2), x(2)
  double precision, intent(inout) :: y(2)
  call dgemv('N', 2, 2, 1d0, a, 2, x, -1, 0d0, y, 1)
end subroutine
"""
    with pytest.raises(NotImplementedError, match="negative increment"):
        build_sdfg(src, tmp_path, name="probe_negative").build()


def test_strided_library_section_is_refused(tmp_path: Path):
    src = """
subroutine probe_stride(x, y, r)
  double precision, intent(in) :: x(4), y(4)
  double precision, intent(out) :: r
  r = dot_product(x(4:1:-1), y(4:1:-1))
end subroutine
"""
    with pytest.raises(Exception, match="non-unit stride"):
        build_sdfg(src, tmp_path, name="probe_stride").build()


ZETA = 0.5 + 2.0j
UNMAPPED_COMPLEX = {
    "zaxpy": ("call zaxpy(3, (0.5d0, 2.0d0), x, 1, y, 1)", (3,), (3,), lambda x, y: ZETA * x + y),
    "zscal": ("call zscal(3, (0.5d0, 2.0d0), y, 1)", (3,), (3,), lambda x, y: ZETA * y),
    "zgeru": (
        "call zgeru(3, 3, (0.5d0, 2.0d0), x, 1, x, 1, y, 3)",
        (3,),
        (3, 3),
        lambda x, y: y + ZETA * np.outer(x, x),
    ),
    "zsymv": (
        "call zsymv('U', 3, (0.5d0, 2.0d0), a, 3, x, 1, (1.0d0, 0.0d0), y, 1)",
        (3,),
        (3,),
        lambda x, y: ZETA * SYM @ x + y,
    ),
    "zsymm": (
        "call zsymm('L', 'U', 3, 3, (0.5d0, 2.0d0), a, 3, x, 3, (1.0d0, 0.0d0), y, 3)",
        (3, 3),
        (3, 3),
        lambda x, y: ZETA * SYM @ x + y,
    ),
    "zsyrk": (
        "call zsyrk('U', 'N', 3, 3, (0.5d0, 2.0d0), x, 3, (1.0d0, 0.0d0), y, 3)",
        (3, 3),
        (3, 3),
        lambda x, y: np.triu(ZETA * x @ x.T + y) + np.tril(y, -1),
    ),
}
SYM = np.asfortranarray(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]) * (1 + 1j))


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="complex BLAS other than gemv/gemm is not mapped")
@pytest.mark.parametrize("routine", list(UNMAPPED_COMPLEX))
def test_unmapped_complex_blas_is_refused_loudly(tmp_path: Path, routine):
    """Each call refuses with ``NotImplementedError``; a mapping that gets the numbers right XPASSes and must drop the mark."""
    call, x_shape, y_shape, reference = UNMAPPED_COMPLEX[routine]
    src = f"""
subroutine probe_{routine}(a, x, y)
  complex(8), intent(in) :: a(3, 3), x({", ".join(map(str, x_shape))})
  complex(8), intent(inout) :: y({", ".join(map(str, y_shape))})
  {call}
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name=f"probe_{routine}").build()
    rng = np.random.default_rng(5)
    x = np.asfortranarray(rng.random(x_shape) + 1j * rng.random(x_shape))
    y = np.asfortranarray(rng.random(y_shape) + 1j * rng.random(y_shape))
    expected = reference(x, y)
    sdfg(a=SYM.copy(order="F"), x=x, y=y)
    np.testing.assert_allclose(y, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
