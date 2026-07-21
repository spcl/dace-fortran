"""Parse-stress anchor for QE's ``h_psi_module::h_psi`` Hamiltonian kernel.

The fixture :file:`ast_v1_h_psi.f90` is the pre-processed flat-Fortran
checkpoint emitted by ``f2dace-qe-source``'s pruning pipeline for the
``h_psi`` entry point (single TU, all USE-closure modules inlined into
one file, ~9.6k lines).  It is the bridge-facing analogue of the
cloudsc / ICON full-source tests, scoped down to the QE
apply-Hamiltonian microkernel, and the sibling of the
``exx_bp::vexx_bp_k_gpu`` checkpoint under ``../exx_bp``.

Two issues sit between the fixture and a clean flang parse:

* The pruning pipeline emits empty ``INTERFACE invfft / fwfft`` blocks
  inside ``MODULE fft_interfaces``, leaving the ``invfft`` / ``fwfft``
  call sites (reached on the local-potential path through
  ``vloc_psi_k_acc`` -> ``wave_g2r`` / ``wave_r2g``) with no specific
  subroutine to resolve against.  ``flang-new-21`` then reports ``No
  specific subroutine of generic 'invfft' matches the actual
  arguments``.  ``MODULE fft_interfaces`` is also emitted BEFORE
  ``MODULE fft_types`` / ``MODULE fft_param``, so inlining the upstream
  specifics in place fails to resolve their ``USE fft_types`` /
  ``USE fft_param`` forwards.  ``_restore_fft_interfaces`` (shared
  verbatim with the ``vexx_bp_k_gpu`` test) deletes the empty block and
  re-emits the specifics AFTER ``END MODULE fft_types``.

* ``change_data_structure`` (line 4479, behind the ``IF (is_exx)`` EXX
  data-layout switch -- never taken on the no-op path) passes the
  keyword actual ``nyfft = nyfft`` where the local ``nyfft`` lost its
  declaration in the prune; the adjacent ``"wave"`` call passes
  ``nyfft = ntask_groups``.  Both flang and gfortran reject the
  undeclared symbol (``No explicit type declared for 'nyfft'``), so
  ``_fix_change_data_structure_nyfft`` restores ``nyfft = ntask_groups``
  to match.  This is a pruner artifact, not a real-QE bug.

The ``DOUBLE PRECISION :: max`` shadow / ``qvan2`` / BLAS
(``dgemm`` / ``dger`` / ``dgemv``) implicit-interface argument-kind
mismatches are warnings under flang-21 and gfortran (the latter with
``-fallow-argument-mismatch``), not parse errors, so no rewrite is
required for them; every mismatched call sits behind a gate
(``IF (okvan)``, ``lda_plus_u``, ...) never taken on the no-op path.

When the f2dace pruning pipeline starts emitting the ``fft_interfaces``
specifics inline / restoring the ``nyfft`` declaration, the two
rewrites fold to no-ops and these tests continue to pass.  See
``ast_v1_h_psi.f90`` for the checkpoint provenance and
``h_psi_caller.f90`` for the C-callable driver harness.
"""
import re
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ast_v1_h_psi.f90"
_ENTRY = "h_psi_module::h_psi"

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

_NYFFT_BAD = "nyfft = nyfft, nmany = nmany_"
_NYFFT_FIXED = "nyfft = ntask_groups, nmany = nmany_"


def _restore_fft_interfaces(source: str) -> str:
    """Re-emit ``fft_interfaces`` specifics after ``fft_types`` is in scope.

    The pruner emits the empty ``INTERFACE invfft / fwfft`` blocks
    BEFORE ``MODULE fft_types`` / ``MODULE fft_param``, so the upstream
    specifics' ``USE fft_types`` forward references can't resolve.
    Solution: delete the empty block, then re-insert the upstream module
    body immediately after ``END MODULE fft_types``.

    :returns: rewritten source, or the original verbatim when the
        empty-interface pattern is already gone (future upstream pruner
        fix).
    """
    stripped, n1 = _FFT_INTERFACES_EMPTY_RE.subn("", source, count=1)
    if n1 == 0:
        return source
    out, n2 = re.subn(r"(END MODULE fft_types\s*\n)", r"\1" + _FFT_INTERFACES_FULL, stripped, count=1)
    if n2 == 0:
        raise RuntimeError("_restore_fft_interfaces: ``END MODULE fft_types`` anchor not "
                           "found; the QE checkpoint's module order may have changed.  "
                           "Inspect ast_v1_h_psi.f90 and update the anchor.")
    return out


def _fix_change_data_structure_nyfft(source: str) -> str:
    """Restore the dropped ``nyfft`` actual in ``change_data_structure``.

    The pruner left a ``CALL fft_type_init(..., nyfft = nyfft, ...)``
    whose ``nyfft`` local lost its declaration; under ``IMPLICIT NONE``
    both flang-new-21 and gfortran reject it (``No explicit type
    declared for 'nyfft'``).  The sibling ``"wave"`` ``fft_type_init``
    call in the same routine passes ``nyfft = ntask_groups``, which is
    the upstream value here too, so substitute it back.  The call lives
    behind the EXX data-structure switch and never runs on the no-op
    path -- this only unblocks the parse.  No-op once the pruner keeps
    the declaration.
    """
    return source.replace(_NYFFT_BAD, _NYFFT_FIXED)


def _preprocess(source: str) -> str:
    """Apply both parse-unblocking rewrites (order-independent)."""
    return _fix_change_data_structure_nyfft(_restore_fft_interfaces(source))


def _make_paw_flag_public(source: str) -> str:
    """Make ``paw_has_init_paw_fockrnl`` PUBLIC for the binding ``use``.

    The kernel reads the module SAVE flag ``paw_has_init_paw_fockrnl``
    (``MODULE paw_exx``, behind the ``okpaw`` gate -- never taken on the
    no-op path), so the generated binding sources it from the host via
    ``use paw_exx, only: <local> => paw_has_init_paw_fockrnl``.  Fortran
    forbids USE-importing a ``PRIVATE`` entity, so the import won't compile.

    Unlike the ``fft_interfaces`` restore, this is NOT a pruner artifact --
    the flag is ``PRIVATE`` in real QE too.  And ``PRIVATE`` is a
    source-level access attribute that does NOT survive into FIR (it does
    not change linkage), so the FIR-based bridge / binding pipeline cannot
    see it to degenerate the symbol automatically; the only place to resolve
    it is the source.  Promoting it to PUBLIC changes nothing semantically
    (the flag is never read on the no-op path) -- it only makes the host
    symbol USE-accessible to the wrapper.  No-op once it is already public.
    """
    return source.replace("LOGICAL, PRIVATE :: paw_has_init_paw_fockrnl", "LOGICAL, PUBLIC :: paw_has_init_paw_fockrnl")


# C-callable driver that initialises the QE module state (shared with the
# gfortran reference via ``init_h_psi_state_c``), then dispatches into the
# SDFG through its GENERATED Fortran binding ``h_psi_dace`` rather than the
# original kernel.  The binding marshals every USE'd module global (wvfct
# g2kin, scf vrs, the dffts descriptor, klist igk_k, ...) from the real host
# on entry / back on exit, so the SDFG sees the same no-op state the
# reference does.  ``h_psi`` takes no OPTIONAL ``becpsi`` (unlike
# ``vexx_bp_k_gpu``), so the driver is a plain forward of psi / hpsi.
_SDFG_DRIVER = r"""
subroutine run_h_psi_dace_c(lda, n, m, npol_in, psi, hpsi) &
    bind(c, name="run_h_psi_dace_c")
  ! ``, intrinsic`` selects the real iso_c_binding even though the QE
  ! checkpoint stubs a same-named module (which lacks ``c_int``).
  use, intrinsic :: iso_c_binding
  use kinds, only: dp
  use noncollin_module, only: npol
  use h_psi_dace_bindings, only: h_psi_dace, h_psi_dace_finalize
  implicit none
  integer(c_int), value :: lda, n, m, npol_in
  complex(dp), intent(inout) :: psi(lda*npol_in, m)
  complex(dp), intent(inout) :: hpsi(lda*npol_in, m)
  interface
    subroutine init_h_psi_state_c(lda, n, m, npol_in) &
        bind(c, name="init_h_psi_state_c")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: lda, n, m, npol_in
    end subroutine
  end interface
  call init_h_psi_state_c(lda, n, m, npol_in)
  call h_psi_dace(lda, n, m, psi, hpsi)
  call h_psi_dace_finalize()
end subroutine run_h_psi_dace_c
"""


def test_restore_and_nyfft_unblock_flang_parse(tmp_path):
    """The two rewrites make the checkpoint flang-parseable.

    The fixture without ``_restore_fft_interfaces`` triggers ``No
    specific subroutine of generic 'invfft' matches the actual
    arguments`` at every ``invfft`` / ``fwfft`` call site; without
    ``_fix_change_data_structure_nyfft`` it triggers ``No explicit type
    declared for 'nyfft'`` at line 4479.  With both applied flang
    processes the file to HLFIR with only the documented warnings
    (``DOUBLE PRECISION :: max`` intrinsic shadow and the BLAS / qvan2
    implicit-interface argument-kind mismatches), none of which is a
    parse error.  This pins the preprocessing's correctness
    independently of the bridge: when the full SDFG path also lands,
    ``test_h_psi_parses`` flips.
    """
    import subprocess
    src = _preprocess(_SRC.read_text())
    rewritten = tmp_path / "h_psi_rewritten.F90"
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


@pytest.mark.xfail(reason="bridge gap: hlfir-expand-vector-subscript-scatter rejects an "
                          "inlined vector-subscript scatter source type on the h_psi "
                          "USE-closure; flips to a clean build once that pass lands "
                          "(cf. the vexx_bp_k_gpu vcut_a gap that already closed).",
                   strict=False)
def test_h_psi_parses(tmp_path):
    """End-to-end SDFG build for ``h_psi``: the QE checkpoint parses,
    inlines, and lowers to a validated SDFG.

    Currently xfails inside the MLIR pass pipeline
    (``hlfir-expand-vector-subscript-scatter: unsupported source type``)
    on a deeply-inlined call site in the h_psi USE-closure; the parse
    itself (``test_restore_and_nyfft_unblock_flang_parse``) already
    passes.  When that downstream gap closes the build returns cleanly
    and this test flips to a pass."""
    src = _preprocess(_SRC.read_text())
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=_ENTRY, name="h_psi")
    sdfg.validate()
    assert sdfg is not None
    assert any('h_psi' in name for name in sdfg.arrays) or \
        'h_psi' in str(sdfg.label)


_CALLER = _HERE / "h_psi_caller.f90"


def _compile_reference(tmp_path):
    """Compile QE source + caller wrapper into a ctypes-loadable .so.

    Returns ``(ctypes.CDLL, init, run)`` where ``init`` and ``run`` are
    ready-to-call function objects.  The caller wrapper provides:

    * ``init_h_psi_state_c(lda, n, m, npol)`` -- one-shot module-state
      init for the controlled path (gamma_only=.false., noncolin=.false.,
      use_gpu=.false., real_space=.false., nkb=0, use_bgrp_in_hpsi=.false.,
      lda_plus_u=.false., scissor=.false., lelfield=.false.,
      exx_started=.false., ismeta=.false.), with g2kin(i)=i, vrs=0, and a
      flat identity smooth-FFT descriptor.
    * ``run_h_psi_c(lda, n, m, psi*, hpsi*)`` -- forwards to
      ``h_psi_module::h_psi``.
    * stubs for ``fwfft_y`` / ``invfft_y`` / ``f_tcpu`` / ``f_wall``.
      The FFT stubs ARE reached (vloc_psi_k_acc), but vrs=0 multiplies
      their roundtrip result by zero before it touches hpsi, so the
      local-potential contribution is exactly 0.

    ``-fallow-argument-mismatch`` downgrades the BLAS / qvan2 argument-kind
    mismatches (COMPLEX(8) passed to REAL(8)) from a hard error to a
    warning, matching flang's permissive handling.  Every mismatched call
    site sits behind a gate (``IF (okvan)`` / ``lda_plus_u`` / ...) and
    never executes on the no-op path.
    """
    import ctypes
    import shutil
    import subprocess

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required for the reference build")

    src = _preprocess(_SRC.read_text())
    src_path = tmp_path / "qe_ref.f90"
    src_path.write_text(src)
    libpath = tmp_path / "libhpsi_ref.so"
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        "-fallow-argument-mismatch",
        str(src_path),
        str(_CALLER), "-o",
        str(libpath)
    ],
                          cwd=str(tmp_path))
    lib = ctypes.CDLL(str(libpath))

    init = lib.init_h_psi_state_c
    init.argtypes = [ctypes.c_int] * 4
    init.restype = None

    run = lib.run_h_psi_c
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = None
    return lib, init, run


def _make_random_inputs(lda, npol, m, *, seed=0):
    """Deterministic complex(:,:) psi / hpsi pair for the wrapper signature.

    Returns ``(psi, hpsi_initial)`` Fortran-ordered ``complex128`` arrays
    of shape ``(lda*npol, m)`` seeded by ``np.random.default_rng(seed)``.
    Both buffers are populated; the caller can ``.copy(order='F')`` to
    keep a pre-call snapshot for after-vs-before comparisons.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    shape = (lda * npol, m)
    psi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    hpsi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    return psi, hpsi


def _expected_kinetic(psi, lda, n, m, npol):
    """Analytic no-op-path result: hpsi(i,j) = g2kin(i)*psi(i,j).

    On the controlled path (vrs=0 -> vloc contributes 0; nkb=0,
    meta/+U/scissor/exx/elfield all skipped) ``h_psi`` reduces to the
    bare kinetic term with the caller's ``g2kin(i) = i`` and
    noncolin=.false., so for the first ``n`` rows of each band
    ``hpsi(i,j) = i*psi(i,j)`` and the rest are zero.
    """
    import numpy as np
    exp = np.zeros_like(psi)
    for j in range(m):
        for i in range(lda * npol):
            if i < n:  # i is 0-based; g2kin(i+1)=i+1, gate i+1<=n
                exp[i, j] = (i + 1) * psi[i, j]
    return exp


def test_h_psi_reference_runs(tmp_path):
    """gfortran reference reaches the kernel and returns the kinetic term.

    With ``gamma_only=.false.``, ``noncolin=.false.``, ``use_gpu=.false.``,
    ``real_space=.false.``, ``nkb=0``, ``use_bgrp_in_hpsi=.false.`` (and
    lda_plus_u / scissor / lelfield / exx_started / ismeta all .false.),
    the kernel trace is: ``h_psi`` -> ``h_psi_`` (no band-group split) ->
    kinetic ``hpsi = g2kin*psi`` -> k-point ``vloc_psi_k_acc`` with
    ``vrs=0`` (contributes exactly 0) -> calbec / +U / meta / scissor /
    exx / elfield all skipped.  The expected post-call state is therefore
    ``hpsi(i,j) = g2kin(i)*psi(i,j)`` exactly (g2kin(i)=i), with no other
    floating-point operation touching it.

    This pins the caller wrapper / state-init / linker-stub harness
    independently of the SDFG build: if it ever stops passing, the QE
    fixture or wrapper changed shape, not the bridge.
    """
    import numpy as np
    _, init, run = _compile_reference(tmp_path)
    lda, n, m, npol = 4, 4, 1, 1
    init(lda, n, m, npol)
    psi, hpsi_in = _make_random_inputs(lda, npol, m)
    psi_snapshot = psi.copy(order="F")
    hpsi_out = hpsi_in.copy(order="F")
    run(lda, n, m, psi.ctypes.data, hpsi_out.ctypes.data)
    np.testing.assert_array_equal(hpsi_out, _expected_kinetic(psi_snapshot, lda, n, m, npol))
    # psi is INTENT(IN) -- the kernel must not perturb it.
    np.testing.assert_array_equal(psi, psi_snapshot)


@pytest.mark.xfail(reason="depends on test_h_psi_parses: the SDFG build currently xfails "
                          "in the MLIR pass pipeline (hlfir-expand-vector-subscript-"
                          "scatter), so the binding is never emitted; flips with it.",
                   strict=False)
def test_h_psi_numerical_correctness(tmp_path):
    """End-to-end numerical correctness for ``h_psi`` THROUGH the generated
    Fortran binding.

    The kernel USE-imports dozens of QE module globals (wvfct, scf,
    fft_base, klist, control_flags, ...); a direct DaCe call would have to
    hand-supply every module-global array plus the free symbols, so the
    e2e path goes through the emitted ``h_psi_dace`` binding, which
    marshals all that state from the real host on entry / back on exit --
    exactly how the kernel would run in production.

    Both sides see byte-identical seeded random complex ``psi`` / ``hpsi``
    (``numpy.random.default_rng``) and the SAME controlled module state
    from ``init_h_psi_state_c`` (gamma_only / noncolin / use_gpu /
    real_space = .false., nkb = 0, vrs = 0, g2kin(i) = i).  On that path
    the kernel reduces to the kinetic term ``hpsi = g2kin*psi`` (see
    ``test_h_psi_reference_runs``), so the SDFG-via-binding output must
    match the gfortran reference element-wise -- pinning that the
    inlined-dummy section-aliasing, the dffts descriptor marshalling, and
    the module-global copy-in/out all lower correctly end to end.
    """
    import ctypes

    import numpy as np

    from _util import build_sdfg
    from dace_fortran.bindings.build_fortran_library import build_fortran_library
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface

    lda, n, m, npol = 4, 4, 1, 1

    # --- gfortran reference (kinetic term on the no-op path) ---
    _, init, run = _compile_reference(tmp_path)
    init(lda, n, m, npol)
    psi_ref, hpsi_ref = _make_random_inputs(lda, npol, m)
    psi_dace = psi_ref.copy(order="F")
    hpsi_dace = hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)

    # --- SDFG-via-binding ---
    src = _make_paw_flag_public(_preprocess(_SRC.read_text()))
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name="h_psi", entry=_ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "h_psi"
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)

    driver_path = tmp_path / "driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        name="h_psi_lib",
        prelude_sources=[src_path],
        extra_sources=[_CALLER, driver_path],
        # The pruned ``qvan2`` / BLAS calls have implicit-interface
        # COMPLEX->REAL arg-kind mismatches behind ``IF(okvan)`` /
        # ``lda_plus_u`` -- never run on the no-op path; matches the
        # reference build's permissive flag.
        extra_flags=["-fallow-argument-mismatch"])
    dace_lib = lib.load()

    fn = dace_lib.run_h_psi_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
    fn(lda, n, m, npol, psi_dace.ctypes.data, hpsi_dace.ctypes.data)

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-12, atol=1e-12)
