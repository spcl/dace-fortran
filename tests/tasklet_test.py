"""Verbatim port of f2dace/dev:tests/fortran/tasklet_test.py.

The PROGRAM wrapper is stripped  --  FaCe runs on the SUBROUTINE directly.
"""

import numpy as np

from tests._util import build_sdfg


def test_fortran_frontend_tasklet(tmp_path):
    src = """
SUBROUTINE tasklet_test_function(d,res)
real, dimension(2) :: d
real, dimension(2) :: res
real :: temp


integer :: i
i=1
temp = 88
d(1)=d(1)*2
temp = MIN(d(i), temp)
res(1) = temp + 10

END SUBROUTINE tasklet_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name="tasklet_test_function").build()

    inp = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=inp, res=res, i=0)
    assert np.allclose(res, [94, 42])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
