"""Module-level array-of-structs (AoS) global with an allocatable component,
marshalled through the generated Fortran binding.

QE ``us_exx`` ``TYPE(bec_type), ALLOCATABLE :: becxx(:)`` shape, reduced to a
minimal kernel: a section ``becxx(i)%k(:, jb)`` passed to an inlined callee
(the ``vexx_bp_k_gpu`` pattern that leaked ``addusxx_r_becphi`` as a program
arg).

One test per pattern: READ (copy-in only), WRITE (copy-in + copy-out),
ALLOC-IN (kernel allocates the component; binding skips copy-in).  Each
builds the SDFG, generates the binding, links against the module + a driver,
and compares to a plain-gfortran reference.
"""
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# --- READ: out(:) += becxx(1)%k(:, jb), via an inlined callee ---------------
_SRC_READ = """
module bec_read_mod
  implicit none
  type :: bec_type
    real(8), allocatable :: k(:, :)
  end type bec_type
  type(bec_type), allocatable :: becxx(:)
contains
  subroutine accum(len, x, out)
    integer, intent(in) :: len
    real(8), intent(in) :: x(len)
    real(8), intent(inout) :: out(len)
    integer :: i
    do i = 1, len
      out(i) = out(i) + x(i)
    end do
  end subroutine accum

  subroutine read_aos(n, jb, out)
    integer, intent(in) :: n, jb
    real(8), intent(inout) :: out(n)
    call accum(n, becxx(1) % k(:, jb), out)
  end subroutine read_aos
end module bec_read_mod
"""
_ENTRY_READ = "bec_read_mod::read_aos"

# C-callable drivers: set up module-global AoS ``becxx``, call the SDFG binding or the
# plain reference, return ``out``.
_DRIVER_READ = """
subroutine run_read_aos(n, jb, nelem, k0, k1, kvals, out) bind(c, name='run_read_aos')
  use iso_c_binding
  use bec_read_mod, only: becxx
  use read_aos_dace_bindings, only: read_aos_dace, read_aos_dace_finalize
  implicit none
  integer(c_int), value :: n, jb, nelem, k0, k1
  real(c_double), intent(in) :: kvals(k0, k1)
  real(c_double), intent(inout) :: out(n)
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(nelem))
  allocate(becxx(1) % k(k0, k1))
  becxx(1) % k = kvals
  call read_aos_dace(n, jb, out)
  call read_aos_dace_finalize()
end subroutine run_read_aos
"""

_REF_DRIVER_READ = """
subroutine run_read_aos_ref(n, jb, nelem, k0, k1, kvals, out) bind(c, name='run_read_aos_ref')
  use iso_c_binding
  use bec_read_mod, only: becxx, read_aos
  implicit none
  integer(c_int), value :: n, jb, nelem, k0, k1
  real(c_double), intent(in) :: kvals(k0, k1)
  real(c_double), intent(inout) :: out(n)
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(nelem))
  allocate(becxx(1) % k(k0, k1))
  becxx(1) % k = kvals
  call read_aos(n, jb, out)
end subroutine run_read_aos_ref
"""


def _compile_so(out_so, *sources, mod_dir, link_so=None):
    import subprocess
    cmd = [
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        f"-J{mod_dir}"
    ]
    cmd += [str(s) for s in sources]
    cmd += ["-o", str(out_so)]
    if link_so is not None:
        cmd += [f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"]
    subprocess.check_call(cmd, cwd=mod_dir)


def _build_module(src, name, entry, tmp_path):
    """Build the SDFG, emit the Fortran binding, return ``(builder, sdfg, so_path, binding_path)``."""
    from _util import build_sdfg
    from dace_fortran.bindings import emit_bindings, FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(src, sdfg_dir, name=name, entry=entry)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    binding = tmp_path / f"{name}_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(binding), tuple(getattr(compiled, "_sig", None) or ()))
    return builder, sdfg, so_path, binding


def _link_pair(src, name, sdfg_driver, ref_driver, so_path, binding, tmp_path):
    """Compile the SDFG-via-binding lib and the plain-gfortran reference lib.  Returns
    ``(sdfg_lib, ref_lib)``."""
    import ctypes

    mod_src = tmp_path / f"{name}_mod.f90"
    mod_src.write_text(src)

    sdfg_build = tmp_path / "sdfg_build"
    sdfg_build.mkdir(parents=True, exist_ok=True)
    drv = tmp_path / "driver.f90"
    drv.write_text(sdfg_driver)
    sdfg_so = sdfg_build / f"{name}_sdfg.so"
    _compile_so(sdfg_so, mod_src, binding, drv, mod_dir=sdfg_build, link_so=so_path)

    ref_build = tmp_path / "ref_build"
    ref_build.mkdir(parents=True, exist_ok=True)
    ref_drv = tmp_path / "ref_driver.f90"
    ref_drv.write_text(ref_driver)
    ref_so = ref_build / f"{name}_ref.so"
    _compile_so(ref_so, mod_src, ref_drv, mod_dir=ref_build)

    return ctypes.CDLL(str(sdfg_so)), ctypes.CDLL(str(ref_so))


def _var_meta(builder, fortran_name):
    """The bridge ``VarInfo`` for ``fortran_name``."""
    for v in builder.module.get_variables():
        if getattr(v, "fortran_name", "") == fortran_name:
            return v
    raise AssertionError(f"no variable {fortran_name!r} in extracted SDFG")


def test_read_aos_module_global_e2e(tmp_path):
    """Module-global AoS component read through an inlined callee: binding packs
    AoS->SoA into ``becxx_k`` and must match the gfortran reference."""
    import ctypes
    import shutil

    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required")

    builder, _sdfg, so_path, binding = _build_module(_SRC_READ, "read_aos", _ENTRY_READ, tmp_path)

    # The read-only AoS component must NOT be marked written (copy-in only).
    meta = _var_meta(builder, "becxx_k")
    assert meta.aos_origin_struct == "becxx"
    assert meta.aos_member_path == "k"
    assert meta.aos_outer_rank == 1
    assert meta.is_written is False
    assert meta.global_alloc_inside is False

    sdfg_lib, ref_lib = _link_pair(_SRC_READ, "read_aos", _DRIVER_READ, _REF_DRIVER_READ, so_path, binding, tmp_path)

    n, jb, nelem, k0, k1 = 5, 2, 3, 5, 4
    rng = np.random.default_rng(0)
    kvals = np.asfortranarray(rng.standard_normal((k0, k1)))
    out0 = np.asfortranarray(rng.standard_normal(n))

    def _run(fn):
        out = out0.copy(order="F")
        fn(ctypes.c_int(n), ctypes.c_int(jb), ctypes.c_int(nelem), ctypes.c_int(k0), ctypes.c_int(k1),
           kvals.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
        return out

    out_sdfg = _run(sdfg_lib.run_read_aos)
    out_ref = _run(ref_lib.run_read_aos_ref)
    # Closed form: out += becxx(1)%k(:, jb).
    np.testing.assert_allclose(out_ref, out0 + kvals[:, jb - 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)


# --- WRITE: becxx(1)%k(:, jb) = src(:), via an inlined callee ---------------
# Kernel STORES into the AoS component; bridge must flag ``becxx_k`` written so the
# binding packs the SoA buffer back into the host AoS on exit (copy-OUT).
_SRC_WRITE = """
module bec_write_mod
  implicit none
  type :: bec_type
    real(8), allocatable :: k(:, :)
  end type bec_type
  type(bec_type), allocatable :: becxx(:)
contains
  subroutine store(len, x, out)
    integer, intent(in) :: len
    real(8), intent(in) :: x(len)
    real(8), intent(out) :: out(len)
    integer :: i
    do i = 1, len
      out(i) = x(i)
    end do
  end subroutine store

  subroutine write_aos(n, jb, src)
    integer, intent(in) :: n, jb
    real(8), intent(in) :: src(n)
    call store(n, src, becxx(1) % k(:, jb))
  end subroutine write_aos
end module bec_write_mod
"""
_ENTRY_WRITE = "bec_write_mod::write_aos"

_DRIVER_WRITE = """
subroutine run_write_aos(n, jb, nelem, k0, k1, src, kout) bind(c, name='run_write_aos')
  use iso_c_binding
  use bec_write_mod, only: becxx
  use write_aos_dace_bindings, only: write_aos_dace, write_aos_dace_finalize
  implicit none
  integer(c_int), value :: n, jb, nelem, k0, k1
  real(c_double), intent(in) :: src(n)
  real(c_double), intent(out) :: kout(k0, k1)
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(nelem))
  allocate(becxx(1) % k(k0, k1))
  becxx(1) % k = -1.0d0
  call write_aos_dace(n, jb, src)
  call write_aos_dace_finalize()
  kout = becxx(1) % k
end subroutine run_write_aos
"""

_REF_DRIVER_WRITE = """
subroutine run_write_aos_ref(n, jb, nelem, k0, k1, src, kout) bind(c, name='run_write_aos_ref')
  use iso_c_binding
  use bec_write_mod, only: becxx, write_aos
  implicit none
  integer(c_int), value :: n, jb, nelem, k0, k1
  real(c_double), intent(in) :: src(n)
  real(c_double), intent(out) :: kout(k0, k1)
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(nelem))
  allocate(becxx(1) % k(k0, k1))
  becxx(1) % k = -1.0d0
  call write_aos(n, jb, src)
  kout = becxx(1) % k
end subroutine run_write_aos_ref
"""


def test_write_aos_module_global_e2e(tmp_path):
    """Module-global AoS component WRITTEN by the kernel: binding adds the SoA->AoS
    copy-OUT loop; host ``becxx(1)%k`` must hold the kernel's result."""
    import ctypes
    import shutil

    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required")

    builder, _sdfg, so_path, binding = _build_module(_SRC_WRITE, "write_aos", _ENTRY_WRITE, tmp_path)

    # The store through the inlined dummy must surface as a written AoS arg.
    meta = _var_meta(builder, "becxx_k")
    assert meta.aos_origin_struct == "becxx"
    assert meta.is_written is True
    assert meta.global_alloc_inside is False
    # The generated binding must emit the copy-OUT, not just a deallocate.
    assert "copy-out" in binding.read_text()

    sdfg_lib, ref_lib = _link_pair(_SRC_WRITE, "write_aos", _DRIVER_WRITE, _REF_DRIVER_WRITE, so_path, binding,
                                   tmp_path)

    n, jb, nelem, k0, k1 = 5, 2, 3, 5, 4
    rng = np.random.default_rng(1)
    src = np.asfortranarray(rng.standard_normal(n))

    def _run(fn):
        kout = np.asfortranarray(np.zeros((k0, k1)))
        fn(ctypes.c_int(n), ctypes.c_int(jb), ctypes.c_int(nelem), ctypes.c_int(k0), ctypes.c_int(k1),
           src.ctypes.data_as(ctypes.c_void_p), kout.ctypes.data_as(ctypes.c_void_p))
        return kout

    k_sdfg = _run(sdfg_lib.run_write_aos)
    k_ref = _run(ref_lib.run_write_aos_ref)
    # Closed form: column jb becomes src, every other column stays -1.
    expected = -np.ones((k0, k1))
    expected[:, jb - 1] = src
    np.testing.assert_allclose(k_ref, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(k_sdfg, k_ref, rtol=1e-12, atol=1e-12)


# --- ALLOC-INSIDE: kernel allocates a module-global array, binding writes it back.
# Host global is UNALLOCATED on entry; binding must skip copy-in (UB to read it),
# let the kernel's own allocate provide the buffer, then assign back on exit.
_SRC_ALLOC = """
module g_alloc_mod
  implicit none
  real(8), allocatable :: gbuf(:)
contains
  subroutine make_g(n, val)
    integer, intent(in) :: n
    real(8), intent(in) :: val
    integer :: i
    if (.not. allocated(gbuf)) allocate(gbuf(n))
    do i = 1, n
      gbuf(i) = val * i
    end do
  end subroutine make_g
end module g_alloc_mod
"""
_ENTRY_ALLOC = "g_alloc_mod::make_g"

_DRIVER_ALLOC = """
subroutine run_make_g(n, val, gout) bind(c, name='run_make_g')
  use iso_c_binding
  use g_alloc_mod, only: gbuf
  use make_g_dace_bindings, only: make_g_dace, make_g_dace_finalize
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: val
  real(c_double), intent(out) :: gout(n)
  if (allocated(gbuf)) deallocate(gbuf)
  call make_g_dace(n, val)
  call make_g_dace_finalize()
  gout = gbuf
end subroutine run_make_g
"""

_REF_DRIVER_ALLOC = """
subroutine run_make_g_ref(n, val, gout) bind(c, name='run_make_g_ref')
  use iso_c_binding
  use g_alloc_mod, only: gbuf, make_g
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: val
  real(c_double), intent(out) :: gout(n)
  if (allocated(gbuf)) deallocate(gbuf)
  call make_g(n, val)
  gout = gbuf
end subroutine run_make_g_ref
"""


def test_alloc_inside_module_global_e2e(tmp_path):
    """Module-global allocatable the kernel ALLOCATEs itself: binding flags
    ``global_alloc_inside``, skips copy-in, writes the result back on exit."""
    import ctypes
    import shutil

    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required")

    builder, _sdfg, so_path, binding = _build_module(_SRC_ALLOC, "make_g", _ENTRY_ALLOC, tmp_path)

    meta = _var_meta(builder, "gbuf")
    assert meta.intent == "inout"
    assert meta.is_written is True
    assert meta.global_alloc_inside is True
    # Binding must NOT copy the (unallocated) host global in, but MUST write back.
    text = binding.read_text()
    assert "kernel-allocated, no copy-in" in text
    assert "gbuf__mod = gbuf" in text

    sdfg_lib, ref_lib = _link_pair(_SRC_ALLOC, "make_g", _DRIVER_ALLOC, _REF_DRIVER_ALLOC, so_path, binding, tmp_path)

    n, val = 6, 2.5

    def _run(fn):
        gout = np.asfortranarray(np.zeros(n))
        fn(ctypes.c_int(n), ctypes.c_double(val), gout.ctypes.data_as(ctypes.c_void_p))
        return gout

    g_sdfg = _run(sdfg_lib.run_make_g)
    g_ref = _run(ref_lib.run_make_g_ref)
    np.testing.assert_allclose(g_ref, val * np.arange(1, n + 1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_sdfg, g_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# PRESENCE: kernel branches on ALLOCATED/ASSOCIATED of a module global. Flang folds
# both to a ``<g>_allocated`` FREE symbol; the binding must source it from the REAL
# host state (not the defensive copy-in buffer, which would always look "present").
# Each test runs PRESENT and ABSENT through the binding vs a plain reference.
# ---------------------------------------------------------------------------
_SRC_ALLOC_PRESENT = """
module pres_alloc_mod
  implicit none
  real(8), allocatable :: gbuf(:)
contains
  subroutine sum_if_present(n, r)
    integer, intent(in) :: n
    real(8), intent(out) :: r
    integer :: i
    if (allocated(gbuf)) then
      r = 0.0d0
      do i = 1, n
        r = r + gbuf(i)
      end do
    else
      r = -1.0d0
    end if
  end subroutine sum_if_present
end module pres_alloc_mod
"""
_ENTRY_ALLOC_PRESENT = "pres_alloc_mod::sum_if_present"

_DRIVER_ALLOC_PRESENT = """
subroutine run_alloc_present(n, present, nelem, vals, r) bind(c, name='run_alloc_present')
  use iso_c_binding
  use pres_alloc_mod, only: gbuf
  use sum_if_present_dace_bindings, only: sum_if_present_dace, sum_if_present_dace_finalize
  implicit none
  integer(c_int), value :: n, present, nelem
  real(c_double), intent(in) :: vals(nelem)
  real(c_double), intent(out) :: r
  if (allocated(gbuf)) deallocate(gbuf)
  if (present /= 0) then
    allocate(gbuf(nelem))
    gbuf = vals
  end if
  call sum_if_present_dace(n, r)
  call sum_if_present_dace_finalize()
end subroutine run_alloc_present
"""

_REF_DRIVER_ALLOC_PRESENT = """
subroutine run_alloc_present_ref(n, present, nelem, vals, r) bind(c, name='run_alloc_present_ref')
  use iso_c_binding
  use pres_alloc_mod, only: gbuf, sum_if_present
  implicit none
  integer(c_int), value :: n, present, nelem
  real(c_double), intent(in) :: vals(nelem)
  real(c_double), intent(out) :: r
  if (allocated(gbuf)) deallocate(gbuf)
  if (present /= 0) then
    allocate(gbuf(nelem))
    gbuf = vals
  end if
  call sum_if_present(n, r)
end subroutine run_alloc_present_ref
"""


def test_allocated_module_global_presence_e2e(tmp_path):
    """Kernel branches on ``ALLOCATED(gbuf)``: binding sources ``gbuf_allocated`` from
    ``allocated(gbuf__mod)``; an unallocated host drives the ``else`` branch (``r=-1``)."""
    import ctypes
    import shutil

    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required")

    builder, _sdfg, so_path, binding = _build_module(_SRC_ALLOC_PRESENT, "sum_if_present", _ENTRY_ALLOC_PRESENT,
                                                     tmp_path)

    # The presence symbol is sourced from the REAL host, not left as a TODO.
    text = binding.read_text()
    assert "gbuf_allocated = int(merge(1, 0, allocated(" in text, \
        f"presence symbol not sourced from host allocated():\n{text}"

    sdfg_lib, ref_lib = _link_pair(_SRC_ALLOC_PRESENT, "sum_if_present", _DRIVER_ALLOC_PRESENT,
                                   _REF_DRIVER_ALLOC_PRESENT, so_path, binding, tmp_path)

    n, nelem = 3, 3
    vals = np.asfortranarray(np.array([1.0, 2.0, 4.0]))

    def _run(fn, present):
        r = np.asfortranarray(np.array([0.0]))
        fn(ctypes.c_int(n), ctypes.c_int(present), ctypes.c_int(nelem), vals.ctypes.data_as(ctypes.c_void_p),
           r.ctypes.data_as(ctypes.c_void_p))
        return r[0]

    for present, expect in ((1, 7.0), (0, -1.0)):  # present -> sum=7; absent -> -1
        r_ref = _run(ref_lib.run_alloc_present_ref, present)
        r_sdfg = _run(sdfg_lib.run_alloc_present, present)
        assert r_ref == expect, f"reference wrong for present={present}: {r_ref}"
        assert r_sdfg == r_ref, f"SDFG-via-binding != reference for present={present}"


_SRC_ASSOC_PRESENT = """
module pres_ptr_mod
  implicit none
  real(8), dimension(:), pointer :: gptr => null()
contains
  subroutine sum_if_assoc(n, r)
    integer, intent(in) :: n
    real(8), intent(out) :: r
    integer :: i
    if (associated(gptr)) then
      r = 0.0d0
      do i = 1, n
        r = r + gptr(i)
      end do
    else
      r = -2.0d0
    end if
  end subroutine sum_if_assoc
end module pres_ptr_mod
"""
_ENTRY_ASSOC_PRESENT = "pres_ptr_mod::sum_if_assoc"

_DRIVER_ASSOC_PRESENT = """
subroutine run_assoc_present(n, present, nelem, vals, r) bind(c, name='run_assoc_present')
  use iso_c_binding
  use pres_ptr_mod, only: gptr
  use sum_if_assoc_dace_bindings, only: sum_if_assoc_dace, sum_if_assoc_dace_finalize
  implicit none
  integer(c_int), value :: n, present, nelem
  real(c_double), intent(in) :: vals(nelem)
  real(c_double), intent(out) :: r
  if (associated(gptr)) then
    deallocate(gptr)
    nullify(gptr)
  end if
  if (present /= 0) then
    allocate(gptr(nelem))
    gptr = vals
  end if
  call sum_if_assoc_dace(n, r)
  call sum_if_assoc_dace_finalize()
end subroutine run_assoc_present
"""

_REF_DRIVER_ASSOC_PRESENT = """
subroutine run_assoc_present_ref(n, present, nelem, vals, r) bind(c, name='run_assoc_present_ref')
  use iso_c_binding
  use pres_ptr_mod, only: gptr, sum_if_assoc
  implicit none
  integer(c_int), value :: n, present, nelem
  real(c_double), intent(in) :: vals(nelem)
  real(c_double), intent(out) :: r
  if (associated(gptr)) then
    deallocate(gptr)
    nullify(gptr)
  end if
  if (present /= 0) then
    allocate(gptr(nelem))
    gptr = vals
  end if
  call sum_if_assoc(n, r)
end subroutine run_assoc_present_ref
"""


def test_associated_pointer_module_global_presence_e2e(tmp_path):
    """Kernel branches on ``ASSOCIATED(gptr)``: binding sources ``gptr_allocated`` from
    ``associated(gptr__mod)``; an unassociated host drives the ``else`` branch (``r=-2``)."""
    import ctypes
    import shutil

    import numpy as np

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required")

    builder, _sdfg, so_path, binding = _build_module(_SRC_ASSOC_PRESENT, "sum_if_assoc", _ENTRY_ASSOC_PRESENT, tmp_path)

    text = binding.read_text()
    assert "gptr_allocated = int(merge(1, 0, associated(" in text, \
        f"presence symbol not sourced from host associated():\n{text}"

    sdfg_lib, ref_lib = _link_pair(_SRC_ASSOC_PRESENT, "sum_if_assoc", _DRIVER_ASSOC_PRESENT, _REF_DRIVER_ASSOC_PRESENT,
                                   so_path, binding, tmp_path)

    n, nelem = 3, 3
    vals = np.asfortranarray(np.array([1.0, 2.0, 4.0]))

    def _run(fn, present):
        r = np.asfortranarray(np.array([0.0]))
        fn(ctypes.c_int(n), ctypes.c_int(present), ctypes.c_int(nelem), vals.ctypes.data_as(ctypes.c_void_p),
           r.ctypes.data_as(ctypes.c_void_p))
        return r[0]

    for present, expect in ((1, 7.0), (0, -2.0)):  # present -> sum=7; absent -> -2
        r_ref = _run(ref_lib.run_assoc_present_ref, present)
        r_sdfg = _run(sdfg_lib.run_assoc_present, present)
        assert r_ref == expect, f"reference wrong for present={present}: {r_ref}"
        assert r_sdfg == r_ref, f"SDFG-via-binding != reference for present={present}"
