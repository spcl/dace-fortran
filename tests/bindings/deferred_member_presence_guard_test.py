"""Presence guards for deferred-storage (POINTER / ALLOCATABLE) struct
members in the emitted bindings.

An absent member's descriptor bounds are undefined -- ``c_loc`` / ``size``
on it read garbage, and gfortran's ``internal_pack`` at the unguarded
``c_f_pointer`` alias site then smashes the stack.  This is exactly how the
ICON Held-Suarez run died on the first ``velocity_tendencies_dace`` call:
``t_nh_diag``'s ``ddt_ua_* / ddt_va_*`` tendency POINTERs are disassociated
in that configuration, and the wrapper marshalled all of them anyway
(217 ``internal_pack`` sites over the alias block; the stomped frames held
0x20-filled "extents" read from adjacent CHARACTER heap data).

Fix under test, end to end:

* the bridge records each member's deferred-storage class
  (``FortranMemberInfo.alloc`` from ``box<heap|ptr<...>>``),
* the emitter wraps every marshal of such a member -- alias
  ``c_f_pointer``, extent / offset symbol population, copy loops -- in
  ``associated(...)`` / ``allocated(...)``,
* the ABSENT branch bounds-remaps the flat POINTER onto a length-1
  ``presence_scratch_<dtype>`` target (so the SDFG call's
  ``c_loc(<flat>)`` stays defined) and zeroes the extent symbols.

``test_emitted_guard_text`` pins the emission (deterministic RED without
the fix); the e2e tests run both member states against a gfortran
reference of the same kernel.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.fortran_interface import build_auto_interface
from tests.bindings.struct_bindings_e2e_test import _build_reference_lib, _build_sdfg_lib

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_TYPES_SRC = """
module mo_opt_state
  use iso_c_binding
  implicit none
  integer, parameter :: N = 6
  type :: t_s
     real(c_double), allocatable :: base(:)
     real(c_double), pointer     :: opt(:) => null()
     logical                     :: use_opt = .false.
  end type t_s
end module mo_opt_state
"""

_KERNEL_SRC = """
module kern_opt_mod
contains
subroutine kern_opt(st)
  use mo_opt_state
  use iso_c_binding
  implicit none
  type(t_s), intent(inout) :: st
  integer :: i
  if (st%use_opt) then
     do i = 1, N
        st%base(i) = st%base(i) + st%opt(i)
     end do
  else
     do i = 1, N
        st%base(i) = st%base(i) + 1.0_c_double
     end do
  end if
end subroutine kern_opt
end module kern_opt_mod
"""

_SRC = _TYPES_SRC + _KERNEL_SRC

# Drivers: ``mode`` selects the member state.  mode=0 leaves ``st%opt``
# DISASSOCIATED (the ICON Held-Suarez shape); mode=1 associates it.
_DRIVER_BODY = """
  use iso_c_binding
  use mo_opt_state, only: t_s, N
  {use_line}
  implicit none
  real(c_double), intent(inout) :: base_ptr(N)
  real(c_double), intent(in)    :: opt_ptr(N)
  integer(c_int), intent(in), value :: mode
  type(t_s), target :: st
  allocate(st%base(N))
  st%base = base_ptr
  if (mode == 1) then
     allocate(st%opt(N))
     st%opt = opt_ptr
     st%use_opt = .true.
  end if
  call {call_name}(st)
  base_ptr = st%base
  if (associated(st%opt)) deallocate(st%opt)
  deallocate(st%base)
"""

_DRIVER = ("subroutine run_opt(base_ptr, opt_ptr, mode) bind(c, name='run_opt')" +
           _DRIVER_BODY.format(use_line="use kern_opt_dace_bindings", call_name="kern_opt_dace") +
           "  call kern_opt_dace_finalize()\nend subroutine run_opt\n")

_REF_DRIVER = ("subroutine run_opt_ref(base_ptr, opt_ptr, mode) bind(c, name='run_opt_ref')" +
               _DRIVER_BODY.format(use_line="use kern_opt_mod, only: kern_opt", call_name="kern_opt") +
               "end subroutine run_opt_ref\n")


def test_emitted_guard_text(tmp_path: Path):
    """The wrapper must presence-guard every marshal of the deferred
    members: ``associated(st%opt)`` around the alias + extents with a
    scratch-remap ELSE, ``allocated(st%base)`` for the allocatable one."""
    sdfg_dir = tmp_path / "sdfg"
    builder = build_sdfg(_SRC, sdfg_dir, name="kern_opt", entry="kern_opt_mod::kern_opt")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "kern_opt"
    iface = build_auto_interface(sdfg._fortran_interface_raw, "kern_opt")

    st = iface.struct_types["t_s"]
    allocs = {m.name.lower(): m.alloc for m in st.members}
    assert allocs["opt"] == "pointer", allocs
    assert allocs["base"] == "allocatable", allocs
    assert allocs["use_opt"] == "", allocs

    out = tmp_path / "kern_opt_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(out))
    text = out.read_text()
    assert "if (associated(st%opt)) then" in text, text
    assert "if (allocated(st%base)) then" in text, text
    assert "presence_scratch_float64" in text, text
    # The absent branch rebinds the flat onto the scratch target.
    assert "st_opt(1:1) => presence_scratch_float64" in text, text


def test_e2e_disassociated_pointer_member(tmp_path: Path):
    """mode=0: ``st%opt`` stays disassociated.  Pre-fix the wrapper
    marshals its garbage descriptor (UB / stack smash); post-fix the
    guarded wrapper runs the kernel's absent branch and matches the
    reference exactly."""
    sdfg_lib = _build_sdfg_lib(tmp_path,
                               kernel_src=_SRC,
                               types_src=_TYPES_SRC,
                               name="kern_opt",
                               entry="kern_opt_mod::kern_opt",
                               driver_src=_DRIVER)
    ref_lib = _build_reference_lib(tmp_path,
                                   types_src=_TYPES_SRC,
                                   kernel_src=_KERNEL_SRC,
                                   ref_driver_src=_REF_DRIVER,
                                   name="kern_opt")
    for fn in (sdfg_lib.run_opt, ref_lib.run_opt_ref):
        fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        fn.restype = None

    rng = np.random.default_rng(43)
    base_init = np.asfortranarray(rng.standard_normal(6))
    opt_vals = np.asfortranarray(rng.standard_normal(6))

    base_ref = base_init.copy(order="F")
    ref_lib.run_opt_ref(base_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        opt_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 0)
    base_sdfg = base_init.copy(order="F")
    sdfg_lib.run_opt(base_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     opt_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 0)
    np.testing.assert_array_equal(base_sdfg, base_ref)
    np.testing.assert_array_equal(base_ref, base_init + 1.0)


def test_e2e_associated_pointer_member(tmp_path: Path):
    """mode=1: ``st%opt`` associated -- the guard's PRESENT branch must
    behave exactly like the unguarded alias did."""
    sdfg_lib = _build_sdfg_lib(tmp_path,
                               kernel_src=_SRC,
                               types_src=_TYPES_SRC,
                               name="kern_opt2",
                               entry="kern_opt_mod::kern_opt",
                               driver_src=_DRIVER.replace("run_opt",
                                                          "run_opt2").replace("kern_opt_dace", "kern_opt2_dace"))
    ref_lib = _build_reference_lib(tmp_path,
                                   types_src=_TYPES_SRC,
                                   kernel_src=_KERNEL_SRC,
                                   ref_driver_src=_REF_DRIVER,
                                   name="kern_opt2")
    for fn in (sdfg_lib.run_opt2, ref_lib.run_opt_ref):
        fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        fn.restype = None

    rng = np.random.default_rng(47)
    base_init = np.asfortranarray(rng.standard_normal(6))
    opt_vals = np.asfortranarray(rng.standard_normal(6))

    base_ref = base_init.copy(order="F")
    ref_lib.run_opt_ref(base_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        opt_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
    base_sdfg = base_init.copy(order="F")
    sdfg_lib.run_opt2(base_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      opt_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
    np.testing.assert_array_equal(base_sdfg, base_ref)
    np.testing.assert_array_equal(base_ref, base_init + opt_vals)
