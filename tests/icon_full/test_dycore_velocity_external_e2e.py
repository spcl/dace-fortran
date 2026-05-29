"""Dycore + external velocity_tendencies E2E (xfail anchor pre-v2).

This is the velocity-scale, struct-shaped E2E of the dycore +
external-SDFG pattern.  The architecture proof at small scale
(``test_dycore_struct_ext_e2e.py`` -- ``state_t{u, v}`` + per-member
SoA) is generalised here to the actual ICON ``velocity_tendencies``
signature (five derived types -- ``t_nh_prog`` / ``t_patch`` /
``t_int_state`` / ``t_nh_metrics`` / ``t_nh_diag`` -- plus three
naked rank-3 arrays plus scalars).

Architecture under test:

  1. **Inner SDFG** (the velocity stand-in): built from
     ``velocity_full.f90`` with
     ``build_fortran_library(..., bind_c_shim=True)`` --
     ``libvelocity_inner_wrap.so`` exposes ``velocity_tendencies_c``
     with one ``c_ptr`` per leaf of the marshal expansion (the
     ``bind_c_shim`` emits the per-member C ABI for each derived
     type, matching what the outer's ``emit_call`` will forward).

  2. **Outer SDFG** (the dycore stand-in): a thin
     ``dycore_wrapper`` subroutine that takes the same arg list as
     ``velocity_tendencies`` and just calls it via an ``interface``
     block.  ``velocity_tendencies`` registers as
     ``keep_external(c_name='velocity_tendencies_c',
     libraries=[inner.so_path])`` with each derived-type arg
     declared ``Arg(kind='aos', c_abi='per_member_soa')`` -- the
     per-member SoA pointers the marshal expansion produces forward
     directly into ``velocity_tendencies_c``, no AoS struct buffer.

  3. **Caller**: drives the outer via standard
     ``build_fortran_library`` bindings (``dycore_wrapper_dace``
     entry); a flat-C-ABI shim derived from the proven
     ``run_velocity_flat_c`` swaps the target call from
     ``velocity_tendencies`` to ``dycore_wrapper_dace`` so the same
     ``ctypes`` driver runs both paths.

  4. **Reference**: the existing gfortran reference of the
     un-transformed ``velocity_tendencies`` + ``run_velocity_flat_c``.

**Why xfail today**: ``hlfir-marshal-external-structs`` v2.1
(``4de6c7a``) covers nested-record members but explicitly does not
cover *box / pointer / allocatable / dynamic-shape* members --
exactly what ``t_nh_prog`` / ``t_patch`` / etc. carry.  The bridge
build of the outer SDFG therefore raises in ``emit_call`` with the
diagnostic "``'aos' arg #0 has no marshalling group``" -- the
same boundary anchored by
``test_v2_aos_external_with_nested_struct``'s xfail-flipped peer
``test_v2_aos_external_diagnostic_mentions_inline_external``.  When
v2 box / allocatable expansion lands this test flips green; the
``xfail(strict=True)`` makes the flip visible.
"""
import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from icon_full._harness import _INIT_ARRAY_ORDER, _OUTPUT_NAMES, _allocate

from dace_fortran.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    build_fortran_library,
)
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_HERE = Path(__file__).resolve().parent
_VELOCITY_PATH = _HERE / "velocity_full.f90"
_CALLER_PATH = _HERE / "velocity_full_caller.f90"


# Dycore stand-in: a passthrough wrapper with the exact
# velocity_tendencies signature.  The bridge sees ``call
# velocity_tendencies(...)``; with the callee registered as
# ``keep_external``, ``hlfir-marshal-external-structs`` is asked to
# expand each derived-type arg into its per-member leaves, then
# ``emit_call`` emits the C call directly into
# ``velocity_tendencies_c`` exported by the inner SDFG.  The wrapper
# does nothing else so any numerical divergence vs the reference is
# attributable to the external boundary, not to outer-side work.
_DYCORE_WRAPPER_SRC = """
module mo_dycore_wrapper
contains
  subroutine dycore_wrapper(p_prog, p_patch, p_int, p_metrics, p_diag, &
                            z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                            ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    use mo_model_domain,   only: t_patch
    use mo_intp_data_strc, only: t_int_state
    use mo_nonhydro_types, only: t_nh_prog, t_nh_diag, t_nh_metrics
    use mo_velocity_advection, only: velocity_tendencies
    type(t_patch),     target, intent(in)    :: p_patch
    type(t_int_state), target, intent(in)    :: p_int
    type(t_nh_prog),           intent(inout) :: p_prog
    type(t_nh_metrics),        intent(inout) :: p_metrics
    type(t_nh_diag),           intent(inout) :: p_diag
    real(kind=8), dimension(:, :, :), intent(inout) :: z_w_concorr_me, z_kin_hor_e, z_vt_ie
    integer, intent(in) :: ntnd, istep
    logical, intent(in) :: lvn_only
    real(kind=8), intent(in) :: dtime, dt_linintp_ubc
    logical, intent(in) :: ldeepatmo
    call velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, &
                             z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                             ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
  end subroutine dycore_wrapper
end module mo_dycore_wrapper
"""


def _scalar(name, ftype, intent, stype=None):
    return OriginalArg(name=name, fortran_type=ftype, rank=0, shape=(), intent=intent,
                       struct_type=stype)


def _arr3(name, intent):
    return OriginalArg(name=name, fortran_type="real(8)", rank=3,
                       shape=(":", ":", ":"), intent=intent, struct_type=None)


# Same ``OriginalInterface`` shape ``test_velocity_full_bindings_e2e``
# uses for the inner velocity binding; reused here for both the inner
# (entry = ``velocity_tendencies``) and the outer (entry =
# ``dycore_wrapper`` -- same arg list, the passthrough wraps it).
def _velocity_iface(entry: str) -> OriginalInterface:
    return OriginalInterface(
        entry=entry,
        args=(
            _scalar("p_prog", "type(t_nh_prog)", "inout", "t_nh_prog"),
            _scalar("p_patch", "type(t_patch)", "in", "t_patch"),
            _scalar("p_int", "type(t_int_state)", "in", "t_int_state"),
            _scalar("p_metrics", "type(t_nh_metrics)", "inout", "t_nh_metrics"),
            _scalar("p_diag", "type(t_nh_diag)", "inout", "t_nh_diag"),
            _arr3("z_w_concorr_me", "inout"),
            _arr3("z_kin_hor_e", "inout"),
            _arr3("z_vt_ie", "inout"),
            _scalar("ntnd", "integer", "in"),
            _scalar("istep", "integer", "in"),
            _scalar("lvn_only", "logical", "in"),
            _scalar("dtime", "real(8)", "in"),
            _scalar("dt_linintp_ubc", "real(8)", "in"),
            _scalar("ldeepatmo", "logical", "in"),
        ),
        struct_types={},
        used_modules={
            "mo_model_domain": ("t_patch", ),
            "mo_intp_data_strc": ("t_int_state", ),
            "mo_nonhydro_types": ("t_nh_prog", "t_nh_metrics", "t_nh_diag"),
        },
        module_symbol_sources={
            "nproma": ("mo_parallel_config", "nproma"),
            "timers_level": ("mo_run_config", "timers_level"),
            "nrdmax": ("mo_vertical_grid", "nrdmax"),
            "nflatlev": ("mo_init_vgrid", "nflatlev"),
            "i_am_accel_node": ("mo_mpi", "i_am_accel_node"),
            "lextra_diffu": ("mo_nonhydrostatic_config", "lextra_diffu"),
            "lvert_nest": ("mo_run_config", "lvert_nest"),
            "timer_intp": ("mo_timer", "timer_intp"),
            "timer_solve_nh_veltend": ("mo_timer", "timer_solve_nh_veltend"),
        },
    )


def _make_sdfg_shim_for_outer(caller_src: str) -> str:
    """Derive the SDFG-side shim from the proven flat caller: rename
    ``run_velocity_flat_c`` -> ``run_velocity_flat_sdfg``, retarget the
    kernel call from ``velocity_tendencies`` to ``dycore_wrapper_dace``
    (the outer SDFG's binding entry), and add the finalize call.
    Mirrors ``_make_sdfg_driver`` in ``test_velocity_full_bindings_e2e``
    but targeting the dycore wrapper instead of the velocity binding."""
    m = re.search(r"(?is)(SUBROUTINE\s+run_velocity_flat_c\b.*?END\s+SUBROUTINE\s+run_velocity_flat_c)",
                  caller_src)
    if not m:
        raise RuntimeError("run_velocity_flat_c not found in caller source")
    shim = m.group(1)
    shim = shim.replace("run_velocity_flat_c", "run_velocity_flat_sdfg")
    shim = shim.replace(
        "USE mo_velocity_advection,  ONLY: velocity_tendencies",
        "USE dycore_wrapper_dace_bindings, ONLY: dycore_wrapper_dace, "
        "dycore_wrapper_dace_finalize",
    )
    shim = shim.replace("CALL velocity_tendencies(p_prog, p_patch",
                        "CALL dycore_wrapper_dace(p_prog, p_patch")
    shim = re.sub(
        r"(?i)\bEND\s+SUBROUTINE\s+run_velocity_flat_sdfg",
        "  CALL dycore_wrapper_dace_finalize()\nEND SUBROUTINE run_velocity_flat_sdfg",
        shim,
    )
    return shim


def _gfortran(out_so: Path, *sources, mod_dir: Path, link_so: Path | None = None):
    cmd = [
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off",
        "-ffree-line-length-none", f"-J{mod_dir}",
    ]
    cmd += [str(s) for s in sources]
    cmd += ["-o", str(out_so)]
    if link_so is not None:
        cmd += [f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"]
    subprocess.check_call(cmd, cwd=mod_dir)


def _run(lib, fn, dims, bufs, z_arrays):
    f = getattr(lib, fn)
    f.restype = None
    f.argtypes = ([ctypes.c_int] * 6 + [ctypes.c_int, ctypes.c_int] +
                  [ctypes.c_int8, ctypes.c_int8] +
                  [ctypes.c_double, ctypes.c_double] +
                  [ctypes.c_void_p, ctypes.c_void_p] +
                  [ctypes.c_int8, ctypes.c_int8, ctypes.c_int] +
                  [ctypes.c_void_p] * (len(_INIT_ARRAY_ORDER) + 3))
    nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v = dims
    nrdmax_in = np.full(10, nlev, dtype=np.int32, order='F')
    nflatlev_in = np.ones(10, dtype=np.int32, order='F')
    f(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, 1, 1, 0, 0, 60.0, 0.0,
      nrdmax_in.ctypes.data, nflatlev_in.ctypes.data, 0, 0, 0,
      *[bufs[k].ctypes.data for k in _INIT_ARRAY_ORDER],
      *[z.ctypes.data for z in z_arrays])


@pytest.mark.xfail(
    strict=True,
    reason="dycore + external velocity_tendencies SDFG.  The marshal-pass "
           "half is done -- the box / pointer / allocatable v2 expansion "
           "landed in ``452f41a``, and the bridge struct-member extractor "
           "now emits box-of-scalar-array dtypes after the box-unwrap.  "
           "What remains is the bind_c_shim's *nested-struct* support: "
           "ICON's t_patch has nested t_grid_cells / t_grid_edges / "
           "t_grid_vertices members, and ``_emit_struct_arg`` today walks "
           "only one level (top-level members).  Extending it to descend "
           "recursively into nested struct members (the bridge needs to "
           "also recursively populate ``struct_types`` for nested "
           "records) is the remaining gate.  Flip xfail when both halves "
           "of the bind_c_shim nested-struct walk land.",
)
def test_dycore_outer_calls_velocity_sdfg_via_c_abi(tmp_path: Path):
    """The dycore SDFG calls the standalone velocity_tendencies SDFG
    over the C ABI, with each derived-type arg crossing via
    per-member SoA pointers (``Arg(kind='aos',
    c_abi='per_member_soa')``).  Random inputs from the existing
    velocity harness; reference is the gfortran-compiled
    velocity_tendencies + run_velocity_flat_c driver.  Numerical
    comparison element-by-element on every output array."""
    # ---- 1. Inner velocity SDFG with bind_c_shim ----
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    velocity_src = _VELOCITY_PATH.read_text()
    inner_sdfg = build_sdfg(velocity_src, inner_sdfg_dir,
                            name="velocity_tendencies",
                            entry="_QMmo_velocity_advectionPvelocity_tendencies").build()
    inner_sdfg.name = "velocity_tendencies"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    # Bridge-derived ``OriginalInterface`` -- carries the
    # ``struct_types`` member layouts the bind_c_shim emitter needs
    # (populated since c7b1f41).  The hand-authored ``_velocity_iface``
    # below is for the outer-side ``build_fortran_library`` call where
    # ``module_symbol_sources`` matters.
    inner_iface = build_auto_interface(inner_sdfg._fortran_interface_raw,
                                       "velocity_tendencies")
    inner_plan = FlattenPlan.from_dict(inner_sdfg._flatten_plan_raw or {})
    inner_lib = build_fortran_library(
        inner_sdfg,
        iface=inner_iface,
        plan=inner_plan,
        out_dir=str(inner_dir / "lib"),
        name="velocity_inner_wrap",
        # The bind_c_shim ``use``s the ICON type modules
        # (``mo_nonhydro_types`` etc.).  Build velocity_full.f90 as a
        # prelude so its ``.mod`` files land in the bind dir before
        # the shim source needs them.
        prelude_sources=[_VELOCITY_PATH],
        bind_c_shim=True,
    )
    assert inner_lib.bind_c_shim_f90 is not None

    # ---- 2. Register velocity_tendencies as a per_member_soa external ----
    # ``c_name='velocity_tendencies_c'`` is the bind_c_shim entry on
    # the inner; each derived-type arg crosses as per-member SoA so
    # the outer's emit_call forwards the marshal-expanded leaves
    # verbatim into ``velocity_tendencies_c(...)``.
    keep_external(
        "velocity_tendencies",
        c_name="velocity_tendencies_c",
        args=(
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_prog
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),     # p_patch
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),     # p_int
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_metrics
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_diag
            Arg(kind="array", dtype="float64", intent="inout"),        # z_w_concorr_me
            Arg(kind="array", dtype="float64", intent="inout"),        # z_kin_hor_e
            Arg(kind="array", dtype="float64", intent="inout"),        # z_vt_ie
            Arg(kind="scalar", dtype="int32", intent="in"),            # ntnd
            Arg(kind="scalar", dtype="int32", intent="in"),            # istep
            Arg(kind="scalar", dtype="bool", intent="in"),             # lvn_only
            Arg(kind="scalar", dtype="float64", intent="in"),          # dtime
            Arg(kind="scalar", dtype="float64", intent="in"),          # dt_linintp_ubc
            Arg(kind="scalar", dtype="bool", intent="in"),             # ldeepatmo
        ),
        libraries=(str(inner_lib.so_path), ),
    )
    try:
        # ---- 3. Outer dycore wrapper SDFG ----
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_src = velocity_src + _DYCORE_WRAPPER_SRC
        outer_sdfg = build_sdfg(outer_src, outer_sdfg_dir,
                                name="dycore_wrapper",
                                entry="_QMmo_dycore_wrapperPdycore_wrapper").build()
        outer_sdfg.name = "dycore_wrapper"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = _velocity_iface("dycore_wrapper")
        outer_plan = FlattenPlan.from_dict(outer_sdfg._flatten_plan_raw or {})
        outer_lib = build_fortran_library(
            outer_sdfg,
            iface=outer_iface,
            plan=outer_plan,
            out_dir=str(outer_dir / "lib"),
            name="dycore_wrapper",
            prelude_sources=[_VELOCITY_PATH, _CALLER_PATH],
            extra_sources=[outer_dir / "sdfg_shim.f90"],
            bind_c_shim=False,
        )
    finally:
        clear_external_registry()

    # ---- 4. Write the SDFG shim retargeting run_velocity_flat_c
    #         -> run_velocity_flat_sdfg + dycore_wrapper_dace ----
    sdfg_shim = outer_dir / "sdfg_shim.f90"
    sdfg_shim.write_text(_make_sdfg_shim_for_outer(_CALLER_PATH.read_text()))

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    # ---- 5. Reference: gfortran-link velocity_full + caller ----
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_so = ref_dir / "libvelocity_ref.so"
    _gfortran(ref_so, _VELOCITY_PATH, _CALLER_PATH, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # ---- 6. Random inputs via the existing harness allocator ----
    nproma, nlev, nblks_c, nblks_e, nblks_v = 8, 6, 4, 4, 4
    nlevp1 = nlev + 1
    dims = (nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v)
    bufs_ref = _allocate(*dims)
    init = ref_lib.init_inputs_random_c
    init.restype = None
    init.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * len(_INIT_ARRAY_ORDER)
    init(42, *dims, *[bufs_ref[k].ctypes.data for k in _INIT_ARRAY_ORDER])
    bufs_sdfg = {k: v.copy(order='F') for k, v in bufs_ref.items()}

    zshape = ((nproma, nlev, nblks_e), (nproma, nlev, nblks_e),
              (nproma, nlevp1, nblks_e))
    z_ref = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]
    z_sdfg = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]

    # ---- 7. Run both paths and compare on every output ----
    _run(ref_lib, "run_velocity_flat_c", dims, bufs_ref, z_ref)
    _run(sdfg_so, "run_velocity_flat_sdfg", dims, bufs_sdfg, z_sdfg)

    extras = dict(zip(('z_w_concorr_me', 'z_kin_hor_e', 'z_vt_ie'),
                      zip(z_sdfg, z_ref)))
    for nm in _OUTPUT_NAMES:
        sd, rf = extras[nm] if nm in extras else (bufs_sdfg[nm], bufs_ref[nm])
        np.testing.assert_allclose(sd, rf, rtol=1e-10, atol=1e-10,
                                   equal_nan=True,
                                   err_msg=f"output {nm!r} diverged")
