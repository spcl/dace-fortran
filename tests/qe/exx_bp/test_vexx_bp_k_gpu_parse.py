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

_FFT_INTERFACES_EMPTY_RE = re.compile(r"MODULE fft_interfaces\s*\n"
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
    out, n2 = re.subn(r"(END MODULE fft_types\s*\n)", r"\1" + _FFT_INTERFACES_FULL, stripped, count=1)
    if n2 == 0:
        raise RuntimeError("_restore_fft_interfaces: ``END MODULE fft_types`` anchor not "
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
    result = subprocess.run([
        "/usr/bin/flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
        str(rewritten), "-o",
        str(out)
    ],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, \
        f"flang rejected the rewritten source:\n{result.stderr[:2000]}"
    assert out.exists() and out.stat().st_size > 0, \
        "flang did not produce a HLFIR output"


@pytest.mark.xfail(strict=False,
                   reason=("Source-side restore lets flang parse the QE checkpoint cleanly "
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
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=_ENTRY, name="vexx_bp_k_gpu")
    sdfg.validate()
    assert sdfg is not None
    assert any('vexx_bp_k_gpu' in name for name in sdfg.arrays) or \
        'vexx_bp_k_gpu' in str(sdfg.label)


_CALLER = _HERE / "vexx_bp_k_gpu_caller.f90"


def _compile_reference(tmp_path):
    """Compile QE source + caller wrapper into a ctypes-loadable .so.

    Returns ``(ctypes.CDLL, init, run)`` where ``init`` and ``run`` are
    ready-to-call function objects.  The caller wrapper provides:

    * ``init_vexx_bp_k_gpu_state_c(lda, n, m, npol, max_ibands)`` --
      one-shot module-state init for the no-op path (noncolin=.false.,
      okvan=.false., okpaw=.false., negrp=1, nqs=0, nibands=[0]).
    * ``run_vexx_bp_k_gpu_c(lda, n, m, psi*, hpsi*)`` -- forwards to
      ``exx_bp::vexx_bp_k_gpu`` with the OPTIONAL ``becpsi`` omitted.
    * stubs for ``fwfft_y`` / ``invfft_y`` / ``f_tcpu`` / ``f_wall``
      (linker-only -- never entered on the no-op path).

    ``-fallow-argument-mismatch`` downgrades the ``qvan2`` argument-kind
    mismatch (COMPLEX(8) passed to REAL(8)) from a hard error to a
    warning, matching flang's permissive handling.  The mismatched
    call site sits behind the ``IF (okvan)`` guard and never executes.
    """
    import ctypes
    import shutil
    import subprocess

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required for the reference build")

    src = _restore_fft_interfaces(_SRC.read_text())
    src_path = tmp_path / "qe_ref.f90"
    src_path.write_text(src)
    libpath = tmp_path / "libvexx_ref.so"
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        "-fallow-argument-mismatch",
        str(src_path),
        str(_CALLER), "-o",
        str(libpath)
    ],
                          cwd=str(tmp_path))
    lib = ctypes.CDLL(str(libpath))

    init = lib.init_vexx_bp_k_gpu_state_c
    init.argtypes = [ctypes.c_int] * 5
    init.restype = None

    run = lib.run_vexx_bp_k_gpu_c
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = None
    return lib, init, run


def _make_random_inputs(lda, npol, max_ibands, *, seed=0):
    """Deterministic complex(:,:) psi / hpsi pair for the wrapper signature.

    Returns ``(psi, hpsi_initial)`` Fortran-ordered ``complex128`` arrays
    of shape ``(lda*npol, max_ibands)`` seeded by ``np.random.default_rng(seed)``.
    Both buffers are populated; the caller can ``.copy(order='F')`` to
    keep a pre-call snapshot for after-vs-before comparisons.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    shape = (lda * npol, max_ibands)
    psi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    hpsi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    return psi, hpsi


def test_vexx_bp_k_gpu_reference_runs(tmp_path):
    """gfortran reference reaches the kernel and returns the identity hpsi.

    With ``noncolin=.false.``, ``okvan=.false.``, ``okpaw=.false.``,
    ``negrp=1``, ``nqs=0``, ``nibands=[0]``, the kernel trace is:
    setup loop skips -> ``vexxmain`` skips -> ``result_sum`` no-ops
    (``negrp==1``) -> final accumulation skips (``iexx_istart(1)==0``)
    -> ``hpsi = hpsi_d`` (identity copy).  The expected post-call state
    is therefore ``hpsi_out == hpsi_in`` exactly (no floating-point
    arithmetic touched it).

    This pins the caller wrapper / state-init / linker-stub harness
    independently of the SDFG build: if it ever stops passing, the QE
    fixture or wrapper changed shape, not the bridge.
    """
    import numpy as np
    _, init, run = _compile_reference(tmp_path)
    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1
    init(lda, n, m, npol, max_ibands)
    psi, hpsi_in = _make_random_inputs(lda, npol, max_ibands)
    hpsi_out = hpsi_in.copy(order="F")
    run(lda, n, m, psi.ctypes.data, hpsi_out.ctypes.data)
    np.testing.assert_array_equal(hpsi_out, hpsi_in)


@pytest.mark.xfail(strict=False,
                   reason=("Gates on ``test_vexx_bp_k_gpu_parses`` flipping first: until "
                           "the SDFG build closes (downstream tasklet ``?`` placeholder), "
                           "there is no SDFG side to compare against.  Reference half of "
                           "the harness is pinned by ``test_vexx_bp_k_gpu_reference_runs``; "
                           "once ``build_sdfg`` returns, this test auto-flips."))
def test_vexx_bp_k_gpu_numerical_correctness(tmp_path):
    """End-to-end gfortran-reference vs SDFG element-wise comparison.

    Both sides see byte-identical deterministic complex inputs from a
    seeded ``numpy.random.default_rng``.  No-op path init (see
    ``test_vexx_bp_k_gpu_reference_runs``) makes the expected output
    the identity ``hpsi_out == hpsi_in``; the comparison is therefore
    a "did the SDFG also produce identity?" check, which is exactly
    what we want at this gate (a non-no-op path would force opinion on
    QE module-state semantics).

    SDFG module-level state is initialised by the bridge's auto-dim
    symbol / bindings layer from the integer kwargs ``max_ibands``,
    ``npol`` passed alongside ``lda`` / ``n`` / ``m``.  When the
    pruner starts inlining the ``fwfft_y`` / ``invfft_y`` specifics
    behind the empty ``fft_interfaces`` block, the linker-stub seam
    in the caller wrapper folds to a no-op too.
    """
    import numpy as np
    _, init, run = _compile_reference(tmp_path)

    # SDFG build (the current xfail gate).
    src = _restore_fft_interfaces(_SRC.read_text())
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=_ENTRY, name="vexx_bp_k_gpu")

    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1
    init(lda, n, m, npol, max_ibands)
    psi_ref, hpsi_ref = _make_random_inputs(lda, npol, max_ibands)
    psi_sdfg = psi_ref.copy(order="F")
    hpsi_sdfg = hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)
    sdfg(lda=np.int32(lda),
         n=np.int32(n),
         m=np.int32(m),
         psi=psi_sdfg,
         hpsi=hpsi_sdfg,
         max_ibands=np.int32(max_ibands),
         npol=np.int32(npol))
    np.testing.assert_allclose(hpsi_sdfg, hpsi_ref, rtol=1e-12, atol=1e-12)
