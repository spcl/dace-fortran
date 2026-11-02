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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
