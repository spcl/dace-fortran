"""Unit coverage for the ``hlfir-strip-error-helpers`` pass.

The pass deletes every ``CALL`` to a known abort-style error helper
(``errore``, ``finish``, ``abor1``, ``upf_error``, ``radiation_abort``,
``dwarning``).  Those helpers all share the same shape:
``IF (ierr <= 0) RETURN`` + diagnostic ``WRITE`` + ``STOP 1``, which
``lift-cf-to-scf`` cannot structurize because ``STOP`` is a noreturn
terminator the ``scf`` dialect doesn't model.  Inlining the resulting
multi-block callee into a structured ``scf`` region crashes flang's
``mlir::inlineCall``.  Deleting the call sites before the inliner
sidesteps the crash and matches the bridge's numerical-equivalence
contract: the SDFG models the no-error path, the error path is dead
code at test time.

These tests exercise the pass in isolation against tiny single-file
sources.  The QE end-to-end anchor lives at
``tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py`` and confirms the pass
unblocks the full SDFG-build pipeline through ``hlfir-inline-all``.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _emit_hlfir_and_strip(src: str, *, env_extra: dict = None) -> str:
    """Compile ``src`` to HLFIR with flang, parse it into the bridge,
    and run ``hlfir-strip-error-helpers``.  Returns the dumped IR.

    Splits the work so the test can inspect IR before/after rather
    than going through the full ``build_sdfg`` path -- this pass is
    pre-pipeline and the rest of the pipeline isn't what we're
    testing here.
    """
    from dace_fortran.build_bridge import hb

    with tempfile.TemporaryDirectory(prefix="strip_err_") as td:
        f = Path(td) / "k.F90"
        f.write_text(src)
        h = Path(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(f), "-o",
            str(h)
        ],
                              cwd=td)
        if env_extra:
            for k, v in env_extra.items():
                os.environ[k] = v
        try:
            mod = hb.HLFIRModule()
            mod.parse_file(str(h))
            mod.run_passes("hlfir-strip-error-helpers")
            return mod.dump()
        finally:
            if env_extra:
                for k in env_extra:
                    os.environ.pop(k, None)


def _count_calls(ir: str, callee: str) -> int:
    """Count ``fir.call @<callee>`` occurrences in dumped MLIR."""
    return ir.count(f"fir.call @{callee}")


def test_strip_errore_call_site_is_removed():
    """The canonical QE error helper -- the call vanishes, the kernel
    body otherwise stays intact."""
    src = """
SUBROUTINE errore(routine, message, ierr)
  CHARACTER(*), INTENT(IN) :: routine, message
  INTEGER, INTENT(IN) :: ierr
  IF (ierr <= 0) RETURN
  WRITE(*,*) routine, message
  STOP 1
END SUBROUTINE
SUBROUTINE run(a, n, ierr)
  INTEGER, INTENT(IN) :: n, ierr
  REAL(8), INTENT(INOUT) :: a(n)
  INTEGER :: i
  CALL errore("run", "negative size", ierr)
  DO i = 1, n
    a(i) = a(i) + 1.0d0
  END DO
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # The kernel-side call to errore must be gone.
    assert "fir.call @_QPerrore" not in ir
    # The DO-loop body must still be present (we only deleted the call).
    assert "fir.do_loop" in ir or "scf.for" in ir or "+ %" in ir or "1.0" in ir


def test_strip_handles_multiple_call_sites_in_one_function():
    """All ``CALL errore(...)`` instances are removed, not just the first."""
    src = """
SUBROUTINE errore(routine, message, ierr)
  CHARACTER(*), INTENT(IN) :: routine, message
  INTEGER, INTENT(IN) :: ierr
  IF (ierr <= 0) RETURN
  STOP 1
END SUBROUTINE
SUBROUTINE run(a, n, e1, e2, e3)
  INTEGER, INTENT(IN) :: n, e1, e2, e3
  REAL(8), INTENT(INOUT) :: a(n)
  CALL errore("r", "first",  e1)
  CALL errore("r", "second", e2)
  CALL errore("r", "third",  e3)
  a(1) = 0.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_calls(ir, "_QPerrore") == 0


def test_strip_recognises_module_procedure_form():
    """``_QMmodPname`` mangling -- module procedure -- also matches.
    The pass demangles the trailing ``P``-segment before comparing."""
    src = """
MODULE m
CONTAINS
  SUBROUTINE errore(routine, message, ierr)
    CHARACTER(*), INTENT(IN) :: routine, message
    INTEGER, INTENT(IN) :: ierr
    IF (ierr <= 0) RETURN
    STOP 1
  END SUBROUTINE
  SUBROUTINE run(a, n, ierr)
    INTEGER, INTENT(IN) :: n, ierr
    REAL(8), INTENT(INOUT) :: a(n)
    CALL errore("m::run", "negative", ierr)
    a(1) = 1.0d0
  END SUBROUTINE
END MODULE
"""
    ir = _emit_hlfir_and_strip(src)
    # Module form: _QMmPerrore
    assert _count_calls(ir, "_QMmPerrore") == 0


def test_strip_leaves_non_error_calls_alone():
    """A non-matching call -- ``CALL compute(...)`` -- survives the pass."""
    src = """
SUBROUTINE compute(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  a(1) = a(1) + 1.0d0
END SUBROUTINE
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  CALL compute(a, n)
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_calls(ir, "_QPcompute") == 1


def test_strip_default_list_covers_icon_finish():
    """ICON's ``finish`` helper is in the default match list."""
    src = """
SUBROUTINE finish(routine, message)
  CHARACTER(*), INTENT(IN) :: routine, message
  WRITE(*,*) routine, message
  STOP 1
END SUBROUTINE
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  CALL finish("run", "bad input")
  a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_calls(ir, "_QPfinish") == 0


def test_strip_default_list_covers_qe_upf_error():
    """QE's ``upf_error`` -- separate from ``errore``, also default-on."""
    src = """
SUBROUTINE upf_error(routine, message, ierr)
  CHARACTER(*), INTENT(IN) :: routine, message
  INTEGER, INTENT(IN) :: ierr
  IF (ierr <= 0) RETURN
  STOP 1
END SUBROUTINE
SUBROUTINE run(a, n, ierr)
  INTEGER, INTENT(IN) :: n, ierr
  REAL(8), INTENT(INOUT) :: a(n)
  CALL upf_error("run", "bad", ierr)
  a(1) = 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    assert _count_calls(ir, "_QPupf_error") == 0


def test_strip_env_extra_helpers_appended_to_default_list():
    """``HLFIR_ERROR_HELPERS`` extends the default list at runtime."""
    src = """
SUBROUTINE my_panic(routine, message)
  CHARACTER(*), INTENT(IN) :: routine, message
  STOP 1
END SUBROUTINE
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  CALL my_panic("run", "bad")
  a(1) = 1.0d0
END SUBROUTINE
"""
    # Without env var: ``my_panic`` is not in the default list -> survives.
    ir_default = _emit_hlfir_and_strip(src)
    assert _count_calls(ir_default, "_QPmy_panic") == 1
    # With env var: ``my_panic`` is matched and stripped.
    ir_extended = _emit_hlfir_and_strip(src, env_extra={"HLFIR_ERROR_HELPERS": "my_panic"})
    assert _count_calls(ir_extended, "_QPmy_panic") == 0


def test_strip_refuses_function_style_helper():
    """A helper that returns a result is left alone -- the call has an
    SSA result that downstream code depends on.  The default list would
    flag ``error`` if used as a CALL, but a function-style use needs an
    explicit rewrite."""
    src = """
MODULE m
CONTAINS
  INTEGER FUNCTION error(routine, message, ierr) RESULT(rc)
    CHARACTER(*), INTENT(IN) :: routine, message
    INTEGER, INTENT(IN) :: ierr
    rc = ierr
  END FUNCTION
  SUBROUTINE run(a, n, ierr)
    INTEGER, INTENT(IN) :: n, ierr
    REAL(8), INTENT(INOUT) :: a(n)
    INTEGER :: rc
    rc = error("run", "msg", ierr)
    IF (rc > 0) a(1) = -1.0d0
  END SUBROUTINE
END MODULE
"""
    ir = _emit_hlfir_and_strip(src)
    # Function-style call has a result -> pass refuses to strip.
    assert _count_calls(ir, "_QMmPerror") == 1


def test_strip_no_match_is_passthrough():
    """When no candidate exists, the pass is a no-op."""
    src = """
SUBROUTINE run(a, n)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  a(1) = a(1) + 1.0d0
END SUBROUTINE
"""
    ir = _emit_hlfir_and_strip(src)
    # No call site to compare; the body must still be there.
    assert "func.func" in ir
