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
# gfortran reference via ``init_vexx_bp_k_gpu_state_c``), then dispatches into
# the SDFG through its GENERATED Fortran binding ``vexx_bp_k_gpu_dace`` rather
# than the original kernel.  The binding marshals every USE'd module global
# (mp_exx all_start/all_end/ibands/..., the becxx AoS struct, ...) from the
# real host on entry / back on exit, so the SDFG sees the same no-op state the
# reference does.
_SDFG_DRIVER = r"""
subroutine run_vexx_dace_c(lda, n, m, npol_in, max_ibands_in, psi, hpsi) &
    bind(c, name="run_vexx_dace_c")
  ! ``, intrinsic`` selects the real iso_c_binding even though the QE
  ! checkpoint stubs a same-named module (which lacks ``c_int``).
  use, intrinsic :: iso_c_binding
  use kinds, only: dp
  use becmod, only: bec_type
  use us_exx, only: becxx
  use noncollin_module, only: npol
  use mp_exx, only: max_ibands
  use vexx_bp_k_gpu_dace_bindings, only: vexx_bp_k_gpu_dace, vexx_bp_k_gpu_dace_finalize
  implicit none
  integer(c_int), value :: lda, n, m, npol_in, max_ibands_in
  complex(dp), intent(inout) :: psi(lda*npol_in, max_ibands_in)
  complex(dp), intent(inout) :: hpsi(lda*npol_in, max_ibands_in)
  type(bec_type) :: becpsi
  interface
    subroutine init_vexx_bp_k_gpu_state_c(lda, n, m, npol_in, max_ibands_in) &
        bind(c, name="init_vexx_bp_k_gpu_state_c")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: lda, n, m, npol_in, max_ibands_in
    end subroutine
  end interface
  call init_vexx_bp_k_gpu_state_c(lda, n, m, npol_in, max_ibands_in)
  ! becxx is the module-global AoS struct the binding copies in via
  ! size(becxx); allocate a degenerate element so the copy is valid.
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(1))
  allocate(becxx(1)%k(1,1));  becxx(1)%k = (0.0_dp, 0.0_dp)
  ! becpsi is a required wrapper arg (the wrapper c_loc's its components).
  allocate(becpsi%r(1,1));    becpsi%r = 0.0_dp
  allocate(becpsi%k(1,1));    becpsi%k = (0.0_dp, 0.0_dp)
  allocate(becpsi%nc(1,1,1)); becpsi%nc = (0.0_dp, 0.0_dp)
  becpsi%nbnd = 1
  call vexx_bp_k_gpu_dace(lda, n, m, psi, hpsi, becpsi)
  call vexx_bp_k_gpu_dace_finalize()
end subroutine run_vexx_dace_c
"""


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


def test_vexx_bp_k_gpu_parses(tmp_path):
    """End-to-end SDFG build for ``vexx_bp_k_gpu``: the QE checkpoint
    parses, inlines, and lowers to a validated SDFG.  Previously xfailed on
    a ``KeyError: 'vcut_a'`` (module-level struct global not flattened);
    that downstream gap is now closed, so the build returns cleanly.  The
    numerical-correctness sibling stays gated until the reference compare
    lands."""
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


def test_vexx_bp_k_gpu_numerical_correctness(tmp_path):
    """End-to-end numerical correctness for ``vexx_bp_k_gpu`` THROUGH the
    generated Fortran binding.

    The kernel USE-imports dozens of QE module globals (mp_exx, exx_base,
    us_exx, ...); a direct DaCe call would have to hand-supply ~80
    module-global arrays plus hundreds of free symbols, so the e2e path
    goes through the emitted ``vexx_bp_k_gpu_dace`` binding, which marshals
    all that state from the real host on entry / back on exit -- exactly
    how the kernel would run in production.

    Both sides see byte-identical seeded random complex ``psi`` / ``hpsi``
    (``numpy.random.default_rng``) and the SAME no-op module state from
    ``init_vexx_bp_k_gpu_state_c`` (noncolin / okvan / okpaw = .false.,
    negrp = 1, nqs = 0, nibands = [0]).  On that path the kernel reduces to
    ``hpsi = hpsi_d`` (identity, see ``test_vexx_bp_k_gpu_reference_runs``),
    so the SDFG-via-binding output must match the gfortran reference
    element-wise -- pinning that the inlined-dummy section-aliasing, the
    becxx AoS-struct marshalling, and the module-global copy-in/out all
    lower correctly end to end.
    """
    import ctypes

    import numpy as np

    from _util import build_sdfg
    from dace_fortran.bindings.build_fortran_library import build_fortran_library
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface

    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1

    # --- gfortran reference (identity on the no-op path) ---
    # ``_compile_reference`` writes ``qe_ref.f90`` / ``libvexx_ref.so`` into
    # ``tmp_path`` directly; those names don't collide with the SDFG side's
    # ``qe.f90`` / ``sdfg`` / ``lib`` / ``driver.f90``.
    _, init, run = _compile_reference(tmp_path)
    init(lda, n, m, npol, max_ibands)
    psi_ref, hpsi_ref = _make_random_inputs(lda, npol, max_ibands)
    psi_dace = psi_ref.copy(order="F")
    hpsi_dace = hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)

    # --- SDFG-via-binding ---
    src = _make_paw_flag_public(_restore_fft_interfaces(_SRC.read_text()))
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name="vexx_bp_k_gpu", entry=_ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "vexx_bp_k_gpu"
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)

    driver_path = tmp_path / "driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        name="vexx_lib",
        prelude_sources=[src_path],
        extra_sources=[_CALLER, driver_path],
        # The pruned ``qvan2`` has an implicit-interface COMPLEX->REAL
        # arg-kind mismatch (qg) behind ``IF(okvan)`` -- never run on the
        # no-op path; matches the reference build's permissive flag.
        extra_flags=["-fallow-argument-mismatch"])
    dace_lib = lib.load()

    fn = dace_lib.run_vexx_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_void_p]
    fn(lda, n, m, npol, max_ibands, psi_dace.ctypes.data, hpsi_dace.ctypes.data)

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-12, atol=1e-12)
