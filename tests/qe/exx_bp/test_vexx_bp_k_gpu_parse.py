"""Parse-stress anchor for QE's ``exx_bp::vexx_bp_k_gpu`` GPU kernel.

The fixture :file:`ast_v1_vexx_bp_k_gpu.f90` is the pre-processed
flat-Fortran checkpoint emitted by ``f2dace-qe-source``'s pruning
pipeline for the ``vexx_bp_k_gpu`` entry point (single TU, all
USE-closure modules inlined into one file, ~2k lines).  It is the
bridge-facing analogue of the cloudsc / ICON full-source tests,
scoped down to a single QE microkernel.

Two issues sat between the fixture and a clean flang parse:

* The pruning pipeline emits empty ``INTERFACE invfft / fwfft`` blocks
  inside ``MODULE fft_interfaces``, leaving the eight call sites
  (lines 1454 / 1455 / 1460 / 1529 / 1553 / 1598 / 1599 / 1605) with
  no specific subroutine to resolve against.  ``flang-new-21`` then
  reports ``No specific subroutine of generic 'invfft' matches the
  actual arguments``.
* ``MODULE fft_interfaces`` is also emitted BEFORE ``MODULE fft_types``
  and ``MODULE fft_param``, so inlining the upstream specifics in
  place fails to resolve their ``USE fft_types`` / ``USE fft_param``
  forwards.

This test loader restores the upstream ``fwfft_y`` / ``invfft_y``
specifics by deleting the empty ``fft_interfaces`` block and
re-emitting it AFTER ``END MODULE fft_types`` so the ``USE``
statements forward-resolve cleanly.  The fixture file itself stays
untouched -- the rewrite lives in this test only, behind the
``_restore_fft_interfaces`` helper.

The ``DOUBLE PRECISION :: max`` shadow at line 1404 of the fixture
is a warning under flang-21, not an error, so no rename is required
for parse.

When the f2dace pruning pipeline starts emitting the specifics
inline, ``_restore_fft_interfaces`` folds to a no-op and this test
continues to pass.  See ``ast_v1_vexx_bp_k_gpu.f90`` for the
checkpoint provenance.
"""
import re
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ast_v1_vexx_bp_k_gpu.f90"
_ENTRY = "exx_bp::vexx_bp_k_gpu"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


_FFT_INTERFACES_EMPTY_RE = re.compile(
    r"MODULE fft_interfaces\s*\n"
    r"  IMPLICIT NONE\s*\n"
    r"  INTERFACE invfft\s*\n  END INTERFACE\s*\n"
    r"  INTERFACE fwfft\s*\n  END INTERFACE\s*\n"
    r"END MODULE fft_interfaces\s*\n")

_FFT_INTERFACES_FULL = """MODULE fft_interfaces
  USE fft_types, ONLY: fft_type_descriptor
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  INTERFACE invfft
     SUBROUTINE invfft_y(fft_kind, f, dfft, howmany)
       USE fft_types, ONLY: fft_type_descriptor
       USE fft_param, ONLY: DP
       IMPLICIT NONE
       CHARACTER(LEN=*), INTENT(IN) :: fft_kind
       TYPE(fft_type_descriptor), INTENT(IN) :: dfft
       INTEGER, OPTIONAL, INTENT(IN) :: howmany
       COMPLEX(DP) :: f(:)
     END SUBROUTINE invfft_y
  END INTERFACE
  INTERFACE fwfft
     SUBROUTINE fwfft_y(fft_kind, f, dfft, howmany)
       USE fft_types, ONLY: fft_type_descriptor
       USE fft_param, ONLY: DP
       IMPLICIT NONE
       CHARACTER(LEN=*), INTENT(IN) :: fft_kind
       TYPE(fft_type_descriptor), INTENT(IN) :: dfft
       INTEGER, OPTIONAL, INTENT(IN) :: howmany
       COMPLEX(DP) :: f(:)
     END SUBROUTINE fwfft_y
  END INTERFACE
END MODULE fft_interfaces
"""


def _restore_fft_interfaces(source: str) -> str:
    """Re-emit ``fft_interfaces`` specifics after ``fft_types`` is in scope.

    The pruner emits the empty ``INTERFACE invfft / fwfft`` blocks
    BEFORE ``MODULE fft_types`` / ``MODULE fft_param``, so the
    upstream specifics' ``USE fft_types`` forward references can't
    resolve.  Solution: delete the empty block, then re-insert the
    upstream module body immediately after ``END MODULE fft_types``.

    :returns: rewritten source, or the original verbatim when the
        empty-interface pattern is already gone (future upstream
        pruner fix).
    """
    stripped, n1 = _FFT_INTERFACES_EMPTY_RE.subn("", source, count=1)
    if n1 == 0:
        return source
    out, n2 = re.subn(r"(END MODULE fft_types\s*\n)",
                      r"\1" + _FFT_INTERFACES_FULL, stripped, count=1)
    if n2 == 0:
        raise RuntimeError(
            "_restore_fft_interfaces: ``END MODULE fft_types`` anchor not "
            "found; the QE checkpoint's module order may have changed.  "
            "Inspect ast_v1_vexx_bp_k_gpu.f90 and update the anchor.")
    return out


def test_restore_fft_interfaces_unblocks_flang_parse(tmp_path):
    """``_restore_fft_interfaces`` rewrites the source to a flang-parseable form.

    The fixture without the rewrite triggers ``No specific subroutine
    of generic 'invfft' matches the actual arguments`` at every
    ``invfft`` / ``fwfft`` call site.  The rewrite restores the
    upstream specifics so flang processes the file with only the
    documented warnings (``DOUBLE PRECISION :: max`` intrinsic shadow
    and the ``qvan2`` implicit-interface argument-kind mismatch),
    neither of which is a parse error.  This test pins the
    preprocessing's correctness independently of the bridge: when the
    full SDFG path also lands, ``test_vexx_bp_k_gpu_parses`` flips.
    """
    import subprocess
    src = _restore_fft_interfaces(_SRC.read_text())
    rewritten = tmp_path / "vexx_bp_k_gpu_rewritten.F90"
    rewritten.write_text(src)
    out = tmp_path / "qe.hlfir"
    result = subprocess.run(
        ["/usr/bin/flang-new-21", "-fc1",
         "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang",
         "-emit-hlfir", str(rewritten), "-o", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, \
        f"flang rejected the rewritten source:\n{result.stderr[:2000]}"
    assert out.exists() and out.stat().st_size > 0, \
        "flang did not produce a HLFIR output"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Source-side restore lets flang parse the QE checkpoint cleanly "
        "(verified in test_restore_fft_interfaces_unblocks_flang_parse).  "
        "Four earlier pipeline-internal gates are now fixed: "
        "(1) the ``hlfir-inline-all`` SIGSEGV (multi-block error helpers "
        "like ``errore`` / ``upf_error`` get stripped / refused); "
        "(2) the ``hlfir-rewrite-pointer-assigns`` rejection of QE's "
        "``ptr(<lo>:..) => src(..)`` bounds-remap (marked as a View at "
        "extract_vars time); "
        "(3) the ``hlfir-lift-reduction-operands`` verifier crash on "
        "dimensional ``SUM(arr, DIM=k)`` reductions producing "
        "``!hlfir.expr<Nxf64>`` (lift now skips non-scalar reduction "
        "results); "
        "(4) the ``fir.do_loop`` non-constant step refusal (hoist + "
        "symbolic-step support landed).  AST-extraction / SDFG "
        "construction hits the next downstream gap: tasklet "
        "expression rendering with a ``?`` placeholder for an "
        "unresolved operand.  Anchored as a follow-up."))
def test_vexx_bp_k_gpu_parses(tmp_path):
    """End-to-end SDFG build for ``vexx_bp_k_gpu`` -- currently xfails on a
    downstream bridge crash, gated separately from the parse fix above."""
    src = _restore_fft_interfaces(_SRC.read_text())
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"),
                                   entry=_ENTRY, name="vexx_bp_k_gpu")
    sdfg.validate()
    assert sdfg is not None
    assert any('vexx_bp_k_gpu' in name for name in sdfg.arrays) or \
        'vexx_bp_k_gpu' in str(sdfg.label)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Gates on ``test_vexx_bp_k_gpu_parses`` flipping first: until the "
        "full SDFG-build path lands, there's no SDFG to run.  This test "
        "is scaffolded so it auto-flips once the build gate closes -- "
        "the gfortran reference + SDFG run + element-wise comparison is "
        "wired through; the xfail just guards the missing build step."))
def test_vexx_bp_k_gpu_numerical_correctness(tmp_path):
    """End-to-end numerical correctness against a gfortran reference.

    Pipeline:
      1. Apply ``_restore_fft_interfaces`` to the QE checkpoint.
      2. Build the SDFG via the bridge.
      3. Compile the SAME source with gfortran into a Fortran-callable
         ``libvexx_ref.so`` via the standard bridge test pattern
         (mirrors the harness used by ``tests/icon/full/test_velocity_full.py``).
      4. Seed every INTENT(IN[OUT]) array with a deterministic RNG
         (``np.random.default_rng(0)``).
      5. Call both with identical inputs, assert element-wise
         agreement on the OUTPUT arrays.

    Currently xfail-strict-false: until the SDFG-build gate closes,
    step 2 throws and the whole test short-circuits.  When the build
    starts succeeding the comparison fires automatically.  The
    fixture's INTEGER/COMPLEX dummy shapes (``lda, n, m, psi(lda*npol,
    max_ibands), hpsi(lda*npol, max_ibands), becpsi``) drive the input
    sizing -- we pick small values (``lda=4``, ``n=4``, ``m=1``) so the
    test stays fast.
    """
    import ctypes
    import shutil
    import subprocess
    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required for the reference build")

    # Step 1: parse-restored source.
    src = _restore_fft_interfaces(_SRC.read_text())

    # Step 2: SDFG build (the current xfail gate).
    sdfg = dace_fortran.build_sdfg(
        src, out_dir=str(tmp_path / "sdfg"),
        entry=_ENTRY, name="vexx_bp_k_gpu")

    # Step 3: gfortran reference via ctypes.  The QE checkpoint is a
    # flat single-TU file containing every USE-closure module inlined,
    # so it compiles standalone; the entry is in MODULE exx_bp so the
    # mangled symbol is ``__exx_bp_MOD_vexx_bp_k_gpu`` under gfortran.
    src_path = tmp_path / "qe_ref.f90"
    src_path.write_text(src)
    libpath = tmp_path / "libvexx_ref.so"
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0",
        "-fno-fast-math", "-ffp-contract=off",
        "-ffree-line-length-none",
        str(src_path), "-o", str(libpath)],
        cwd=str(tmp_path))
    lib = ctypes.CDLL(str(libpath))

    # Step 4: deterministic random inputs at a small problem size.
    # ``max_ibands``, ``npol``, ``nkb`` etc. are module-globals on the
    # QE side -- the SDFG side binds them via the auto-dim symbol
    # mechanism (caller passes them as integer kwargs).
    rng = np.random.default_rng(0)
    lda, n, m = 4, 4, 1
    npol, max_ibands = 1, 1
    shape = (lda * npol, max_ibands)
    psi_re = rng.standard_normal(shape)
    psi_im = rng.standard_normal(shape)
    psi_ref = np.asfortranarray(psi_re + 1j * psi_im, dtype=np.complex128)
    psi_sdfg = psi_ref.copy(order="F")
    hpsi_ref = np.zeros(shape, dtype=np.complex128, order="F")
    hpsi_sdfg = np.zeros(shape, dtype=np.complex128, order="F")

    # Step 5: run reference + SDFG.  gfortran's by-reference ABI takes
    # all-pointers; the bind(c)-free entry name is mangled.
    fn = lib.__exx_bp_MOD_vexx_bp_k_gpu
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # lda
        ctypes.POINTER(ctypes.c_int),    # n
        ctypes.POINTER(ctypes.c_int),    # m
        ctypes.c_void_p,                  # psi
        ctypes.c_void_p,                  # hpsi
        ctypes.c_void_p,                  # becpsi (OPTIONAL, pass NULL)
    ]
    fn.restype = None
    fn(ctypes.byref(ctypes.c_int(lda)),
       ctypes.byref(ctypes.c_int(n)),
       ctypes.byref(ctypes.c_int(m)),
       psi_ref.ctypes.data, hpsi_ref.ctypes.data, None)

    # SDFG side -- the auto-dim symbol mechanism fills the module-
    # globals from caller-supplied kwargs.
    sdfg(lda=np.int32(lda), n=np.int32(n), m=np.int32(m),
         psi=psi_sdfg, hpsi=hpsi_sdfg,
         max_ibands=np.int32(max_ibands), npol=np.int32(npol))

    # Element-wise comparison on the writeable output.  Tight tolerance
    # since both sides use the SAME floating-point sequence on identical
    # inputs (no fast-math, no -ffp-contract).
    np.testing.assert_allclose(hpsi_sdfg, hpsi_ref, rtol=1e-12, atol=1e-12)
