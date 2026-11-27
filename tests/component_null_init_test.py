# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Derived-type locals whose type has allocatable / pointer components and no component default.

LLVM 23 nullifies each such component with ``fir.coordinate_of`` + ``fir.store`` (21/22: ``fir.copy`` from the
``DerivedInit`` global), which kept FlattenStructs from flattening the struct (``KeyError: 'fortran_s'``) until the
``erase-component-null-init`` pass erased those stores.
"""

import numpy as np

SRC = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
    real, pointer :: p(:)
    real, allocatable :: v(:)
  end type t
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res
  type(t) :: s
  integer :: i
  allocate(s%w(n), s%v(n))
  do i = 1, n
    s%w(i) = real(i)
    s%v(i) = 2.0 * real(i)
  end do
  res = sum(s%w) + sum(s%v)
  deallocate(s%w, s%v)
end subroutine main
"""


def test_allocatable_and_pointer_components_flatten(tmp_path):
    from tests._util import build_sdfg

    sdfg = build_sdfg(SRC, tmp_path, name="main", entry="main").build()
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=np.int32(4), res=res)
    assert res[0] == 30.0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_allocatable_and_pointer_components_flatten(Path(d))
