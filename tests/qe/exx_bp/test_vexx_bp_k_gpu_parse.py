"""Parse-stress anchor for QE's exx_bp::vexx_bp_k_gpu GPU kernel.

ast_v1_vexx_bp_k_gpu.f90 is the pre-processed flat-Fortran checkpoint from
f2dace-qe-source's pruning pipeline for vexx_bp_k_gpu (single TU, all
USE-closure modules inlined, ~2k lines) -- the bridge-facing analogue of the
cloudsc/ICON full-source tests, scoped to one QE microkernel.

Two issues blocked a clean flang parse:

* The pruning pipeline emits empty INTERFACE invfft/fwfft blocks inside
  MODULE fft_interfaces, leaving the eight call sites (lines 1454/1455/1460/
  1529/1553/1598/1599/1605) with no specific subroutine to resolve against
  ("No specific subroutine of generic 'invfft' matches the actual arguments").
* MODULE fft_interfaces is emitted BEFORE MODULE fft_types/fft_param, so
  inlining the upstream specifics in place fails their USE forwards.

This test loader (_restore_fft_interfaces) deletes the empty fft_interfaces
block and re-emits it AFTER END MODULE fft_types so the USE statements
forward-resolve. The fixture file itself stays untouched.

The DOUBLE PRECISION :: max shadow at line 1404 is a warning under flang-21,
not an error.

Once the pruning pipeline emits the specifics inline, _restore_fft_interfaces
folds to a no-op and this test keeps passing.
"""

import re
from pathlib import Path

import pytest

import dace_fortran
from tests._util import flang_binary, flang_intrinsic_modules_path
from tests.qe._conforming import QVAN2_REWRITES, apply_rewrites

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ast_v1_vexx_bp_k_gpu.f90"
_ENTRY = "exx_bp::vexx_bp_k_gpu"


_FFT_INTERFACES_EMPTY_RE = re.compile(
    r"MODULE fft_interfaces\s*\n"
    r"  IMPLICIT NONE\s*\n"
    r"  INTERFACE invfft\s*\n  END INTERFACE\s*\n"
    r"  INTERFACE fwfft\s*\n  END INTERFACE\s*\n"
    r"END MODULE fft_interfaces\s*\n"
)

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
    """Re-emit fft_interfaces specifics after fft_types is in scope.

    Pruner emits empty INTERFACE invfft/fwfft blocks BEFORE MODULE fft_types/
    fft_param, so the upstream specifics' USE fft_types can't forward-resolve.
    Deletes the empty block, re-inserts the upstream body after END MODULE
    fft_types. Returns the source verbatim if the empty-interface pattern is already gone.
    """
    stripped, n1 = _FFT_INTERFACES_EMPTY_RE.subn("", source, count=1)
    if n1 == 0:
        return source
    out, n2 = re.subn(r"(END MODULE fft_types\s*\n)", r"\1" + _FFT_INTERFACES_FULL, stripped, count=1)
    if n2 == 0:
        raise RuntimeError(
            "_restore_fft_interfaces: ``END MODULE fft_types`` anchor not "
            "found; the QE checkpoint's module order may have changed.  "
            "Inspect ast_v1_vexx_bp_k_gpu.f90 and update the anchor."
        )
    return out


def _preprocess(source: str) -> str:
    """:func:`_restore_fft_interfaces` plus the ``qvan2`` COMPLEX-as-REAL rewrite, so gfortran needs no
    ``-fallow-argument-mismatch``."""
    return apply_rewrites(_restore_fft_interfaces(source), QVAN2_REWRITES)


def _make_paw_flag_public(source: str) -> str:
    """Make paw_has_init_paw_fockrnl PUBLIC for the binding use.

    The kernel reads module SAVE flag paw_has_init_paw_fockrnl (MODULE
    paw_exx, behind the okpaw gate -- never taken on the no-op path); the
    generated binding sources it via `use paw_exx, only: <local> =>
    paw_has_init_paw_fockrnl`, but Fortran forbids USE-importing PRIVATE.

    Not a pruner artifact -- PRIVATE in real QE too. PRIVATE doesn't survive
    into FIR (no linkage change), so the FIR-based pipeline can't resolve it
    automatically; only the source can. Promoting to PUBLIC is semantically a
    no-op (flag never read on the no-op path) and is itself a no-op once already public.
    """
    return source.replace("LOGICAL, PRIVATE :: paw_has_init_paw_fockrnl", "LOGICAL, PUBLIC :: paw_has_init_paw_fockrnl")


# C-callable driver: inits QE module state (shared with gfortran reference via
# init_vexx_bp_k_gpu_state_c), then dispatches into the SDFG via the GENERATED
# vexx_bp_k_gpu_dace binding, which marshals every USE'd module global from the
# real host on entry/exit so the SDFG sees the same no-op state as the reference.
_SDFG_DRIVER = r"""
subroutine run_vexx_dace_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in, psi, hpsi) &
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
  integer(c_int), value :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
  complex(dp), intent(inout) :: psi(lda*npol_in, max_ibands_in)
  complex(dp), intent(inout) :: hpsi(lda*npol_in, max_ibands_in)
  type(bec_type) :: becpsi
  interface
    subroutine init_vexx_bp_k_gpu_state_c(lda, n, m, npol_in, max_ibands_in) &
        bind(c, name="init_vexx_bp_k_gpu_state_c")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: lda, n, m, npol_in, max_ibands_in
    end subroutine
    subroutine init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in) &
        bind(c, name="init_vexx_bp_k_gpu_perf_state_c")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
    end subroutine
  end interface
  ! nqs_in == 0 selects the no-op state, anything else the real exchange path.
  if (nqs_in == 0) then
    call init_vexx_bp_k_gpu_state_c(lda, n, m, npol_in, max_ibands_in)
  else
    call init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in)
  end if
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
    """_restore_fft_interfaces rewrites the source to a flang-parseable form.

    Without the rewrite, every invfft/fwfft call site triggers "No specific
    subroutine of generic 'invfft' matches the actual arguments". The rewrite
    leaves only the documented DOUBLE PRECISION :: max shadow warning, not a
    parse error. Pins preprocessing correctness
    independently of the bridge.
    """
    import subprocess

    src = _preprocess(_SRC.read_text())
    rewritten = tmp_path / "vexx_bp_k_gpu_rewritten.F90"
    rewritten.write_text(src)
    out = tmp_path / "qe.hlfir"
    result = subprocess.run(
        [
            flang_binary(),
            "-fc1",
            "-fintrinsic-modules-path",
            flang_intrinsic_modules_path(),
            "-emit-hlfir",
            str(rewritten),
            "-o",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"flang rejected the rewritten source:\n{result.stderr[:2000]}"
    assert out.exists() and out.stat().st_size > 0, "flang did not produce a HLFIR output"


# Reproducer: module-level struct global vcut_a not flattened, KeyError in the bridge (55521e5)
def test_vexx_bp_k_gpu_parses(tmp_path):
    """End-to-end SDFG build for vexx_bp_k_gpu: QE checkpoint parses, inlines,
    and lowers to a validated SDFG. Previously xfailed on KeyError: 'vcut_a'
    (module-level struct global not flattened), now closed."""
    src = _preprocess(_SRC.read_text())
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=_ENTRY, name="vexx_bp_k_gpu")
    sdfg.validate()
    assert sdfg is not None
    assert any("vexx_bp_k_gpu" in name for name in sdfg.arrays) or "vexx_bp_k_gpu" in str(sdfg.label)


_CALLER = _HERE / "vexx_bp_k_gpu_caller.f90"


def _compile_reference(tmp_path):
    """Compile QE source + caller wrapper into a ctypes-loadable .so.

    Returns (ctypes.CDLL, init, run). The caller wrapper provides:
    * init_vexx_bp_k_gpu_state_c(lda, n, m, npol, max_ibands) -- one-shot
      no-op-path module-state init (noncolin/okvan/okpaw=.false., negrp=1,
      nqs=0, nibands=[0]).
    * run_vexx_bp_k_gpu_c(lda, n, m, psi*, hpsi*) -- forwards to
      exx_bp::vexx_bp_k_gpu with OPTIONAL becpsi omitted.
    * linker-only stubs for fwfft_y/invfft_y/f_tcpu/f_wall (never entered).
    """
    import ctypes
    import subprocess

    src = _preprocess(_SRC.read_text())
    src_path = tmp_path / "qe_ref.f90"
    src_path.write_text(src)
    libpath = tmp_path / "libvexx_ref.so"
    subprocess.check_call(
        [
            "gfortran",
            "-shared",
            "-fPIC",
            "-O0",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-ffree-line-length-none",
            str(src_path),
            str(_CALLER),
            "-o",
            str(libpath),
        ],
        cwd=str(tmp_path),
    )
    lib = ctypes.CDLL(str(libpath))

    init = lib.init_vexx_bp_k_gpu_state_c
    init.argtypes = [ctypes.c_int] * 5
    init.restype = None

    run = lib.run_vexx_bp_k_gpu_c
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = None
    return lib, init, run


def _make_random_inputs(lda, npol, max_ibands, *, seed=0):
    """Deterministic complex(:,:) psi/hpsi pair for the wrapper signature.

    Returns Fortran-ordered complex128 (psi, hpsi) of shape (lda*npol, max_ibands), seeded by default_rng(seed)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    shape = (lda * npol, max_ibands)
    psi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    hpsi = np.asfortranarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
    return psi, hpsi


def test_vexx_bp_k_gpu_reference_runs(tmp_path):
    """gfortran reference reaches the kernel and returns the identity hpsi.

    With noncolin/okvan/okpaw=.false., negrp=1, nqs=0, nibands=[0]: setup loop
    skips -> vexxmain skips -> result_sum no-ops (negrp==1) -> final
    accumulation skips (iexx_istart(1)==0) -> hpsi = hpsi_d (identity copy).
    So hpsi_out == hpsi_in exactly.

    Pins the caller wrapper/state-init/linker-stub harness independently of the SDFG build.
    """
    import numpy as np

    _, init, run = _compile_reference(tmp_path)
    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1
    init(lda, n, m, npol, max_ibands)
    psi, hpsi_in = _make_random_inputs(lda, npol, max_ibands)
    hpsi_out = hpsi_in.copy(order="F")
    run(lda, n, m, psi.ctypes.data, hpsi_out.ctypes.data)
    np.testing.assert_array_equal(hpsi_out, hpsi_in)


def build_and_run_vexx_sdfg(tmp_path, dims, psi, hpsi, sdfg_hook=None, name="vexx_bp_k_gpu"):
    """Build the vexx_bp_k_gpu SDFG, bind it through the generated Fortran binding and run it in place on psi/hpsi.

    ``dims`` is (lda, n, m, npol, max_ibands, nnr, nqs, jblock); nqs == 0 runs the no-op state, otherwise the real
    exchange-path state.  ``sdfg_hook(sdfg)`` runs on the built SDFG before the library is generated (the CPF driver
    canonicalizes there). ``name`` names the SDFG and so its library and binding: two builds in one process need two
    names, or the second binding resolves against the first build's already-loaded library.
    """
    import ctypes

    from dace_fortran.bindings.build_fortran_library import build_fortran_library
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from tests._util import build_sdfg

    src = _make_paw_flag_public(_preprocess(_SRC.read_text()))
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name=name, entry=_ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = name
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    if sdfg_hook is not None:
        sdfg_hook(sdfg)

    driver_path = tmp_path / "driver.f90"
    driver_path.write_text(_SDFG_DRIVER.replace("vexx_bp_k_gpu_dace", f"{name}_dace"))

    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        name="vexx_lib",
        prelude_sources=[src_path],
        extra_sources=[_CALLER, driver_path],
    )
    dace_lib = lib.load()

    fn = dace_lib.run_vexx_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 8 + [ctypes.c_void_p, ctypes.c_void_p]
    fn(*dims, psi.ctypes.data, hpsi.ctypes.data)


def run_exchange_reference(tmp_path, dims, psi, hpsi):
    """gfortran reference on the real exchange-path state, in place on psi/hpsi."""
    import ctypes

    lib, _, run = _compile_reference(tmp_path)
    init = lib.init_vexx_bp_k_gpu_perf_state_c
    init.argtypes = [ctypes.c_int] * 8
    init.restype = None
    init(*dims)
    lda, n, m = dims[:3]
    run(lda, n, m, psi.ctypes.data, hpsi.ctypes.data)


# Real exchange path, small: 3 bands in jblocks of 2 (two band blocks, the second partial), 2 q-points, 8-point grid.
EXCHANGE_DIMS = (4, 3, 3, 1, 3, 8, 2, 2)


def test_vexx_bp_k_gpu_exchange_reference_changes_hpsi(tmp_path):
    """On the real exchange path the gfortran reference adds a non-zero exchange term to every band of hpsi."""
    import numpy as np

    lda, n, m, npol, max_ibands = EXCHANGE_DIMS[:5]
    psi, hpsi_in = _make_random_inputs(lda, npol, max_ibands)
    hpsi = hpsi_in.copy(order="F")
    run_exchange_reference(tmp_path, EXCHANGE_DIMS, psi, hpsi)
    delta = np.abs(hpsi - hpsi_in)
    assert np.all(delta[:n, :m] > 1e-6), delta
    np.testing.assert_array_equal(hpsi[n:], hpsi_in[n:])


# Builds, compiles and binds the whole vexx_bp_k_gpu SDFG: ~4 min on a CI runner, past the lanes' 300 s default.
@pytest.mark.timeout(1200)
def test_vexx_bp_k_gpu_numerical_correctness(tmp_path):
    """End-to-end numerical correctness for vexx_bp_k_gpu THROUGH the
    generated Fortran binding, on the no-op state.

    Kernel USE-imports dozens of QE module globals; a direct DaCe call would
    need ~80 hand-supplied module-global arrays, so the e2e path goes through
    the emitted vexx_bp_k_gpu_dace binding, which marshals all that state from
    the real host on entry/exit -- as it would run in production.

    Both sides see identical seeded psi/hpsi and the same no-op module state
    (see test_vexx_bp_k_gpu_reference_runs), so output must match the
    gfortran reference element-wise -- pinning that inlined-dummy
    section-aliasing, becxx AoS marshalling, and module-global copy-in/out all lower correctly.
    """
    import numpy as np

    dims = (4, 4, 1, 1, 1, 1, 0, 1)
    lda, n, m, npol, max_ibands = dims[:5]

    # _compile_reference writes qe_ref.f90/libvexx_ref.so into tmp_path; no name collision with the SDFG side's files.
    _, init, run = _compile_reference(tmp_path)
    init(lda, n, m, npol, max_ibands)
    psi_ref, hpsi_ref = _make_random_inputs(lda, npol, max_ibands)
    psi_dace = psi_ref.copy(order="F")
    hpsi_dace = hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)

    build_and_run_vexx_sdfg(tmp_path, dims, psi_dace, hpsi_dace)

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.timeout(1200)
def test_vexx_bp_k_gpu_exchange_numerical_correctness(tmp_path):
    """The default codegen matches gfortran on the REAL exchange path: nqs=2 q-points, 3 bands with all-pairs
    exchange in two jblocks, band- and q-dependent exxbuff, x_occupation and Coulomb factors. Both sides run the same
    flat DFT for fwfft/invfft (the reference through flat_dft_stub, the SDFG through its FFT library nodes)."""
    import numpy as np

    lda, n, m, npol, max_ibands = EXCHANGE_DIMS[:5]
    psi_ref, hpsi_ref = _make_random_inputs(lda, npol, max_ibands)
    hpsi_in = hpsi_ref.copy(order="F")
    psi_dace = psi_ref.copy(order="F")
    hpsi_dace = hpsi_ref.copy(order="F")
    run_exchange_reference(tmp_path, EXCHANGE_DIMS, psi_ref, hpsi_ref)
    assert np.all(np.abs(hpsi_ref - hpsi_in)[:n, :m] > 1e-6), "reference did not take the exchange path"

    build_and_run_vexx_sdfg(tmp_path, EXCHANGE_DIMS, psi_dace, hpsi_dace, name="vexx_bp_k_gpu_exchange")

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
