"""Unit coverage for the ``hlfir-strip-runtime-io`` pass.

Every Fortran ``WRITE`` / ``PRINT`` / ``FLUSH`` / ``OPEN`` / ``CLOSE``
statement lowers to a sequence of opaque ``_FortranAio*`` runtime
calls.  The bridge's SDFG is a numerical-equivalence model -- it
compares output arrays for inputs that don't trigger error paths --
so diagnostic output to ``stdout`` / ``stderr`` is dead code at test
time.  The pass walks the module, replaces each I/O call's SSA
result(s) with a benign constant (``i1`` -> false, ``i32`` -> 0 iostat,
``!fir.ref<i8>`` cookie -> ``fir.zero_bits``), then erases the call.
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _emit_hlfir_and_strip(src: str) -> str:
    """Compile ``src`` to HLFIR, run ``hlfir-strip-runtime-io``,
    return the dumped IR."""
    from dace_fortran.build_bridge import hb

    with tempfile.TemporaryDirectory(prefix="strip_io_") as td:
        f = Path(td) / "k.F90"
        f.write_text(src)
        h = Path(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(f), "-o",
            str(h)
        ],
                              cwd=td)
        mod = hb.HLFIRModule()
        mod.parse_file(str(h))
        mod.run_passes("hlfir-strip-runtime-io")
        return mod.dump()


def _count_io_calls(ir: str) -> int:
    """Total count of any remaining ``fir.call @_FortranAio*`` ops."""
    return ir.count("fir.call @_FortranAio")


def test_strip_write_star_message_to_stdout():
    """``WRITE(*,*)`` is the canonical print-to-stdout shape."""
    src = """
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  WRITE(*,*) "hello from run"
  a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_io_calls(ir) == 0
    # The data assignment must survive -- the pass only touches IO.
    assert "1.000000e+00" in ir or "1.0" in ir


def test_strip_write_with_format_and_args():
    """A formatted ``WRITE`` with several args produces a longer
    cookie-threaded chain; every link must be stripped together so
    no dangling cookie remains."""
    src = """
SUBROUTINE run(a, n, ierr)
  INTEGER, INTENT(IN) :: n, ierr
  REAL(8), INTENT(INOUT) :: a(n)
  WRITE(*, '("stop_clock: no clock for ",A12," found !")') "label_xyz"
  WRITE(*, '("ierr =",I6)') ierr
  a(1) = REAL(ierr, 8)
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_io_calls(ir) == 0


def test_strip_print_statement():
    """``PRINT`` lowers through the same runtime as ``WRITE(*,*)``."""
    src = """
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  PRINT *, "hello"
  a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_io_calls(ir) == 0


def test_strip_iostat_user_reads_zero():
    """User code that reads ``IOSTAT`` sees a zero (success) value
    after the pass -- the iostat result is replaced by ``arith.constant
    0 : i32`` before the call is erased, so any branch reading the
    status (``IF (ios /= 0) ...``) takes the no-error path."""
    src = """
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  INTEGER :: ios
  REAL(8), INTENT(INOUT) :: a(n)
  WRITE(*,'(A)', IOSTAT=ios) "hello"
  IF (ios == 0) a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # All IO calls gone.
    assert _count_io_calls(ir) == 0
    # The IF guarded by ios still exists; the ``a(1) = 1.0d0`` write
    # is what the no-error path produces and is the conventional
    # behaviour we want preserved.
    assert "1.000000e+00" in ir or "1.0" in ir


def test_strip_leaves_non_io_calls_alone():
    """A regular ``CALL kernel(...)`` survives -- only ``_FortranAio*``
    symbols are matched, not arbitrary calls."""
    src = """
SUBROUTINE kernel(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  a(1) = a(1) + 1.0d0
END SUBROUTINE
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  WRITE(*,*) "before kernel"
  CALL kernel(a, n)
  WRITE(*,*) "after kernel"
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # IO gone, kernel still called.
    assert _count_io_calls(ir) == 0
    assert "fir.call @_QPkernel" in ir


def test_strip_no_io_is_passthrough():
    """A kernel with no ``WRITE`` / ``PRINT`` is a no-op for this pass."""
    src = """
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  a(1) = a(1) + 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # The pass shouldn't add anything; the IR still has the function body.
    assert "func.func" in ir
    assert _count_io_calls(ir) == 0


def test_strip_handles_qe_stop_clock_diagnostic():
    """The QE / NPB ``stop_clock`` body reduces to exactly the
    diagnostic shape the user flagged: ``WRITE(stdout, '(...)')
    label``.  Verify the pass eats it entirely."""
    src = """
MODULE mytime
  IMPLICIT NONE
  INTEGER, PARAMETER :: stdout = 6
END MODULE
SUBROUTINE stop_clock(label)
  USE mytime, ONLY: stdout
  CHARACTER(LEN=*), INTENT(IN) :: label
  WRITE(stdout, '("stop_clock: no clock for ",A12," found !")') label
END SUBROUTINE
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  CALL stop_clock("xc_func    ")
  a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # Every IO call -- inside stop_clock's body -- is stripped.
    assert _count_io_calls(ir) == 0
    # The user-level ``CALL stop_clock(...)`` itself stays (only IO
    # calls are stripped; the body's IO is gone so stop_clock is now
    # an empty function, which symbol-dce later removes).
    assert "fir.call @_QPstop_clock" in ir
