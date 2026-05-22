"""End-to-end bridge tests for the Fortran I/O patterns in FV3 / ICON.

Each test writes a *known* input (a data file or a namelist), runs the kernel
through the HLFIR bridge, and asserts the VALUES that were read match what was
written -- a build that drops the statement or mis-reads it must fail here, not
pass.  Filenames are hardcoded relative paths and the test ``chdir``s into its
``tmp_path``, so no character argument has to cross the SDFG ABI.

These pin the acceptance criteria for the ``_FortranAio*`` recognizer (which
maps Fortran I/O to ``dace.libraries.fortran_io`` nodes).  Until it lands the
bridge either crashes (string constants leak into array-name extraction) or
silently drops the statement, so every case is marked ``xfail``; they flip to
passing one I/O family at a time as the recognizer is built.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_NOT_YET = "bridge _FortranAio* recognizer not yet implemented"


def test_list_directed_read(tmp_path, monkeypatch):
    """``read(u,*) y`` of a whole array must yield the file's values."""
    src = """
module m
  implicit none
contains
  subroutine read_vals(y)
    real(8), intent(out) :: y(3)
    integer :: u
    open (newunit=u, file='data.txt', status='old')
    read (u, *) y
    close (u)
  end subroutine read_vals
end module m
"""
    (tmp_path / "data.txt").write_text("1.5 2.5 3.5\n")
    monkeypatch.chdir(tmp_path)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="read_vals", entry="_QMmPread_vals").build()
    y = np.zeros(3, dtype=np.float64)
    sdfg(y=y)
    np.testing.assert_allclose(y, [1.5, 2.5, 3.5])


@pytest.mark.xfail(reason=_NOT_YET, strict=False)
def test_namelist_read_external(tmp_path, monkeypatch):
    """``open`` + ``read(u, nml=cfg)`` + ``close`` (the FV3 init pattern, with a
    local namelist group) must read each member's value from the file."""
    src = """
module m
  implicit none
contains
  subroutine nml_read(y)
    real(8), intent(out) :: y(2)
    real(8) :: alpha, beta
    integer :: u
    namelist /cfg/ alpha, beta
    alpha = 0.0d0
    beta = 0.0d0
    open (newunit=u, file='cfg.nml', status='old')
    read (u, nml=cfg)
    close (u)
    y(1) = alpha
    y(2) = beta
  end subroutine nml_read
end module m
"""
    (tmp_path / "cfg.nml").write_text("&cfg\n  alpha = 1.5\n  beta = 2.5\n/\n")
    monkeypatch.chdir(tmp_path)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="nml_read", entry="_QMmPnml_read").build()
    y = np.zeros(2, dtype=np.float64)
    sdfg(y=y)
    np.testing.assert_allclose(y, [1.5, 2.5])


def test_list_directed_write(tmp_path, monkeypatch):
    """``write(u,*) x`` must actually emit the values (today it is dropped):
    read the written file back and check it round-trips."""
    src = """
module m
  implicit none
contains
  subroutine write_vals(x)
    real(8), intent(in) :: x(3)
    integer :: u
    open (newunit=u, file='out.txt', status='replace')
    write (u, *) x
    close (u)
  end subroutine write_vals
end module m
"""
    monkeypatch.chdir(tmp_path)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="write_vals", entry="_QMmPwrite_vals").build()
    x = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    sdfg(x=x)
    written = [float(tok) for tok in (tmp_path / "out.txt").read_text().split()]
    np.testing.assert_allclose(written, [4.0, 5.0, 6.0])
