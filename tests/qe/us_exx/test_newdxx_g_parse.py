"""Parse-stress + active-path anchor for QE's ``us_exx::newdxx_g`` kernel.

The fixture :file:`ast_v1_newdxx.f90` is the pre-processed flat-Fortran
checkpoint emitted by ``f2dace-qe-source``'s pruning pipeline for the
``newdxx_g`` entry point (single TU, all USE-closure modules inlined
into one file, ~490 lines).  It is the bridge-facing analogue of the
cloudsc / ICON full-source tests, scoped down to a single QE
microkernel, and a sibling of the ``exx_bp::vexx_bp_k_gpu`` /
``h_psi_module::h_psi`` checkpoints under ``../exx_bp`` / ``../h_psi``.

Unlike those siblings, this checkpoint needs NO source rewrites to parse
(no empty ``fft_interfaces`` block, no dropped declaration), so flang and
gfortran accept it verbatim (``test_newdxx_g_flang_parses`` pins that) and
the SDFG build lands cleanly (``test_newdxx_g_parses``).

This harness drives the ACTIVE ultrasoft path: ``okvan = .TRUE.`` with
``flag = 'c'`` (the complex / non-gamma case).  ``newdxx_g`` then runs its
full body -- the ``eigqts`` structure-factor phase, the ``auxvc = vc(nl)``
gather, and the blocked ``qgm``/``becphi_c`` contraction that accumulates
the augmentation into ``deexx`` -- rather than the degenerate
``okvan = .FALSE.`` early return.  The caller (``newdxx_g_caller.f90``)
sets up a small fixed deterministic problem (1 ultrasoft atom, nh=2
projectors, nkb=2, ngms=4) so the reference and the SDFG-via-binding see
byte-identical pseudopotential state; ``flag='c'`` requires
``gamma_only=.FALSE.`` (which also selects the ``becphi_c`` inner branch
and disables the ``gstart==2`` gamma correction).

The gfortran reference matches an independent numpy model of the
augmentation to machine precision (``test_newdxx_g_reference_runs``).  The
end-to-end SDFG path, however, currently mis-lowers the *complex*
``DOT_PRODUCT(aux2, aux1)``: Fortran conjugates the first argument
(``SUM(CONJG(aux2)*aux1)``), but the bridge emits the non-conjugated
``SUM(aux2*aux1)``.  Every other part of the 'c' path lowers correctly
(the ``flag``-folded branch select picks ``add_complex``; ``okvan`` /
``gamma_only`` marshal from the host; ``eigts``/``qgm``/``mill`` index
correctly), so the only residual is that conjugation.
``test_newdxx_g_numerical_correctness`` is therefore xfail until the
bridge conjugates complex ``DOT_PRODUCT`` -- at which point it flips to a
pass.  See ``newdxx_g_caller.f90`` for the C-callable driver harness.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ast_v1_newdxx.f90"
_ENTRY = "us_exx::newdxx_g"
_CALLER = _HERE / "newdxx_g_caller.f90"

# Fixed problem dimensions -- must match the PARAMETERs / ALLOCATEs in
# ``init_newdxx_g_state_c`` and the ``run_newdxx_g_c`` buffer shapes.
_NNR = 4
_NGMS = 4
_NKB = 2

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


# C-callable driver that initialises the QE module state (shared with the
# gfortran reference via ``init_newdxx_g_state_c``), then dispatches into the
# SDFG through its GENERATED Fortran binding ``newdxx_g_dace`` rather than the
# original kernel.  The binding marshals every USE'd module global (uspp
# okvan/nkb/ofsbeta/ijtoh, us_exx qgm/nij_type, gvect eigts*/mill, ions_base
# ityp/tau/nat, cell_base omega, control_flags gamma_only, the dffts
# descriptor struct, ...) from the real host on entry / back on exit, so the
# SDFG sees the same active 'c'-path state the reference does.  The generated
# wrapper DROPS the ``CHARACTER`` ``flag`` dummy (the bridge constant-folds the
# ``flag=='c'`` test and marshals only numeric / struct args), so the call
# omits it; ``becphi_r`` is OPTIONAL and omitted, ``becphi_c`` passed by
# keyword.
_SDFG_DRIVER = r"""
subroutine run_newdxx_g_dace_c(nnr, ngms, nkb_in, vc, deexx, becphi_c) &
    bind(c, name="run_newdxx_g_dace_c")
  ! ``, intrinsic`` selects the real iso_c_binding even though the QE
  ! checkpoint stubs a same-named module (which lacks ``c_int``).
  use, intrinsic :: iso_c_binding
  use kinds, only: dp
  use fft_types, only: fft_type_descriptor
  use newdxx_g_dace_bindings, only: newdxx_g_dace, newdxx_g_dace_finalize
  implicit none
  integer(c_int), value :: nnr, ngms, nkb_in
  complex(dp), intent(inout) :: vc(nnr)
  complex(dp), intent(inout) :: deexx(nkb_in)
  complex(dp), intent(in) :: becphi_c(nkb_in)
  type(fft_type_descriptor) :: dfftt
  real(dp) :: xk(3), xkq(3)
  integer :: ig
  interface
    subroutine init_newdxx_g_state_c() bind(c, name="init_newdxx_g_state_c")
    end subroutine
  end interface
  call init_newdxx_g_state_c()
  dfftt%nnr = nnr
  dfftt%ngm = ngms
  allocate(dfftt%nl(ngms))
  do ig = 1, ngms
    dfftt%nl(ig) = ig
  end do
  xk(:) = [0.1_dp, 0.2_dp, 0.3_dp]
  xkq(:) = [0.0_dp, 0.0_dp, 0.0_dp]
  call newdxx_g_dace(dfftt, vc, xkq, xk, deexx, becphi_c=becphi_c)
  call newdxx_g_dace_finalize()
  deallocate(dfftt%nl)
end subroutine run_newdxx_g_dace_c
"""


def test_newdxx_g_flang_parses(tmp_path):
    """The QE checkpoint emits HLFIR with no source rewrites.

    Unlike the ``vexx_bp_k_gpu`` / ``h_psi`` siblings (empty
    ``fft_interfaces`` block / dropped ``nyfft`` declaration), this fixture
    is flang-parseable verbatim.  flang processes the file with at most
    benign warnings, none of which is a parse error.  This pins the
    checkpoint's parseability independently of the bridge.
    """
    import subprocess
    out = tmp_path / "qe.hlfir"
    result = subprocess.run([
        "/usr/bin/flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
        str(_SRC), "-o",
        str(out)
    ],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, \
        f"flang rejected the checkpoint:\n{result.stderr[:2000]}"
    assert out.exists() and out.stat().st_size > 0, \
        "flang did not produce a HLFIR output"


def test_newdxx_g_parses(tmp_path):
    """End-to-end SDFG build for ``newdxx_g``: the QE checkpoint parses,
    inlines, and lowers to a validated SDFG.  This small self-contained
    kernel builds cleanly (no downstream gap), so the build returns a
    validated SDFG."""
    src = _SRC.read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=_ENTRY, name="newdxx_g")
    sdfg.validate()
    assert sdfg is not None
    assert any('newdxx_g' in name for name in sdfg.arrays) or \
        'newdxx_g' in str(sdfg.label)


def _compile_reference(tmp_path):
    """Compile QE source + caller wrapper into a ctypes-loadable .so.

    Returns ``(ctypes.CDLL, init, run)`` where ``init`` and ``run`` are
    ready-to-call function objects.  The caller wrapper provides:

    * ``init_newdxx_g_state_c()`` -- one-shot module-state init for the
      ACTIVE 'c' path (``okvan=.TRUE.``, ``gamma_only=.FALSE.``, 1 ultrasoft
      atom, nh=2, nkb=2, ngms=4, with eigts / qgm / mill / tau / ... filled
      deterministically).
    * ``run_newdxx_g_c(nnr, ngms, nkb, vc*, deexx*, becphi_c*)`` -- builds
      the ``fft_type_descriptor`` and forwards to ``us_exx::newdxx_g`` with
      ``flag='c'`` and the OPTIONAL ``becphi_r`` omitted.
    * stubs for ``f_tcpu`` / ``f_wall`` (the active path runs the clocks, so
      these resolve the EXTERNAL timer symbols and keep their totals finite).

    The fixture compiles verbatim (no ``-fallow-argument-mismatch`` needed).
    """
    import ctypes
    import shutil
    import subprocess

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required for the reference build")

    src_path = tmp_path / "qe_ref.f90"
    src_path.write_text(_SRC.read_text())
    libpath = tmp_path / "libnewdxx_ref.so"
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        str(src_path),
        str(_CALLER), "-o",
        str(libpath)
    ],
                          cwd=str(tmp_path))
    lib = ctypes.CDLL(str(libpath))

    init = lib.init_newdxx_g_state_c
    init.argtypes = []
    init.restype = None

    run = lib.run_newdxx_g_c
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = None
    return lib, init, run


def _make_random_inputs(*, seed=0):
    """Deterministic complex ``vc`` / ``deexx`` / ``becphi_c`` for the wrapper.

    Returns Fortran-ordered ``complex128`` arrays ``vc`` shape ``(NNR,)``,
    ``deexx`` shape ``(NKB,)`` (the in/out accumulator), and ``becphi_c``
    shape ``(NKB,)``, seeded by ``np.random.default_rng(seed)`` in that
    draw order.  The caller copies ``deexx`` to keep a pre-call snapshot.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    vc = np.asfortranarray(rng.standard_normal(_NNR) + 1j * rng.standard_normal(_NNR), dtype=np.complex128)
    deexx = np.asfortranarray(rng.standard_normal(_NKB) + 1j * rng.standard_normal(_NKB), dtype=np.complex128)
    becphi_c = np.asfortranarray(rng.standard_normal(_NKB) + 1j * rng.standard_normal(_NKB), dtype=np.complex128)
    return vc, deexx, becphi_c


def _expected_deexx_c(vc, deexx_in, becphi_c):
    """Independent numpy model of the ``flag='c'`` augmentation.

    Mirrors ``init_newdxx_g_state_c`` + ``run_newdxx_g_c`` + the ``newdxx_g``
    body for the complex / non-gamma path (1 atom, nh=2, ngms=4, nl(ig)=ig,
    mill(d,ig)=ig, nij=ofsbeta=0):

        eigqts   = exp(-i * tpi * sum((xk - xkq) * tau))
        auxvc    = vc(nl) = vc                       ! add_complex; fact=omega
        aux2(ig) = conj(auxvc(ig)) * eigqts
                   * eigts1(ig) * eigts2(ig) * eigts3(ig)
        for ih: aux1(ig) = sum_jh becphi_c(jh) * conj(qgm(ig, ijtoh(ih,jh)))
                deexx(ih) += omega * DOT_PRODUCT(aux2, aux1)

    with ``DOT_PRODUCT(a,b) = sum(conj(a)*b)`` (numpy ``vdot``).  Keep this
    in lockstep with the caller's hardcoded state.
    """
    import numpy as np
    omega = 2.0
    tpi = 2.0 * np.pi
    tau = np.array([0.5, 0.6, 0.7])
    xk = np.array([0.1, 0.2, 0.3])
    xkq = np.array([0.0, 0.0, 0.0])
    ig = np.arange(1, _NGMS + 1)
    eigqts = np.cos(tpi * np.sum((xk - xkq) * tau)) - 1j * np.sin(tpi * np.sum((xk - xkq) * tau))
    eigts1 = 1.0 + 0.10j * ig
    eigts2 = 0.9 + 0.20j * ig
    eigts3 = 0.8 + 0.30j * ig
    qgm = np.zeros((_NGMS, 4), dtype=np.complex128)
    for col in range(1, 5):
        qgm[:, col - 1] = (0.10 * ig + 0.01 * col) + 1j * (0.02 * ig - 0.03 * col)
    ijtoh = lambda ih, jh: (ih - 1) * 2 + jh

    auxvc = vc.copy()  # add_complex: auxvc(ig) = vc(nl(ig)) = vc(ig)
    aux2 = np.conj(auxvc) * eigqts * eigts1 * eigts2 * eigts3
    deexx = deexx_in.astype(np.complex128).copy()
    nh = 2
    for ih in range(1, nh + 1):
        aux1 = np.zeros(_NGMS, dtype=np.complex128)
        for jh in range(1, nh + 1):
            aux1 += becphi_c[jh - 1] * np.conj(qgm[:, ijtoh(ih, jh) - 1])
        deexx[ih - 1] += omega * np.vdot(aux2, aux1)  # vdot conjugates aux2
    return deexx


def test_newdxx_g_reference_runs(tmp_path):
    """gfortran reference runs the ACTIVE 'c' augmentation and matches numpy.

    With ``okvan=.TRUE.`` and ``flag='c'`` (``gamma_only=.FALSE.``) the kernel
    runs its full body: the ``eigqts`` phase, the ``auxvc=vc(nl)`` gather, and
    the blocked ``qgm``/``becphi_c`` contraction that accumulates into
    ``deexx``.  The post-call ``deexx`` must (a) differ from the input -- the
    active path ran, not the ``okvan=.FALSE.`` no-op -- and (b) equal
    ``_expected_deexx_c`` (an independent numpy model of the same arithmetic).

    This pins the caller wrapper / state-init / linker-stub harness AND the
    augmentation math, independently of the SDFG build.
    """
    import numpy as np
    _, init, run = _compile_reference(tmp_path)
    init()
    vc, deexx_in, becphi_c = _make_random_inputs()
    deexx_out = deexx_in.copy(order="F")
    run(_NNR, _NGMS, _NKB, vc.ctypes.data, deexx_out.ctypes.data, becphi_c.ctypes.data)
    # the active path actually moved deexx (vs. the okvan=.FALSE. identity)
    assert not np.allclose(deexx_out, deexx_in)
    np.testing.assert_allclose(deexx_out, _expected_deexx_c(vc, deexx_in, becphi_c), rtol=1e-12, atol=1e-12)


@pytest.mark.xfail(reason="bridge gap: complex DOT_PRODUCT(aux2, aux1) is lowered without "
                          "conjugating the first argument -- SUM(aux2*aux1) instead of "
                          "SUM(CONJG(aux2)*aux1).  Every other part of the flag='c' path "
                          "lowers correctly (branch select, okvan/gamma_only marshalling, "
                          "eigts/qgm/mill indexing); flips to a pass once complex "
                          "DOT_PRODUCT conjugates.",
                   strict=False)
def test_newdxx_g_numerical_correctness(tmp_path):
    """End-to-end numerical correctness for ``newdxx_g`` THROUGH the generated
    Fortran binding, on the active ``flag='c'`` path.

    The kernel USE-imports a dozen QE module globals (uspp, us_exx, gvect,
    ions_base, uspp_param, cell_base, control_flags) plus a
    ``fft_type_descriptor`` struct argument; a direct DaCe call would have to
    hand-supply all of them, so the e2e path goes through the emitted
    ``newdxx_g_dace`` binding, which marshals that state from the real host.

    Both sides see byte-identical seeded random ``vc`` / ``deexx`` /
    ``becphi_c`` and the SAME deterministic pseudopotential state from
    ``init_newdxx_g_state_c``.  The SDFG-via-binding ``deexx`` must match the
    gfortran reference element-wise.  It currently does NOT: the bridge
    lowers the complex ``DOT_PRODUCT`` without conjugating its first argument
    (confirmed to be the sole discrepancy -- the DaCe output equals the
    reference with the conjugation dropped), so this is xfail until that
    lowering is fixed.
    """
    import ctypes

    import numpy as np

    from _util import build_sdfg
    from dace_fortran.bindings.build_fortran_library import build_fortran_library
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface

    # --- gfortran reference (active 'c' augmentation) ---
    _, init, run = _compile_reference(tmp_path)
    init()
    vc_ref, deexx_ref, becphi_ref = _make_random_inputs()
    vc_dace = vc_ref.copy(order="F")
    deexx_dace = deexx_ref.copy(order="F")
    becphi_dace = becphi_ref.copy(order="F")
    run(_NNR, _NGMS, _NKB, vc_ref.ctypes.data, deexx_ref.ctypes.data, becphi_ref.ctypes.data)

    # --- SDFG-via-binding ---
    src = _SRC.read_text()
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name="newdxx_g", entry=_ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "newdxx_g"
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)

    driver_path = tmp_path / "driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        name="newdxx_lib",
        prelude_sources=[src_path],
        extra_sources=[_CALLER, driver_path])
    dace_lib = lib.load()

    fn = dace_lib.run_newdxx_g_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn(_NNR, _NGMS, _NKB, vc_dace.ctypes.data, deexx_dace.ctypes.data, becphi_dace.ctypes.data)

    np.testing.assert_allclose(deexx_dace, deexx_ref, rtol=1e-12, atol=1e-12)
