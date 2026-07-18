"""End-to-end test for ``dace_fortran.external.inline_external``.

Builds a callee SDFG, a caller that CALLs it as a registered external, then
inlines the callee into the caller.  Verifies the ExternalCall node is
replaced by a :class:`dace.sdfg.nodes.NestedSDFG` and the inlined caller
matches a gfortran/f2py reference numerically.
"""

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang
from dace_fortran.external import (Arg, apply_external_functions, clear_external_registry, inline_external,
                                   keep_external)
from dace_fortran.external_functions import ExternalFunction

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_inline_external_swaps_libnode_for_nested_sdfg(tmp_path):
    """Inline a separately-built callee SDFG at the caller's external call site."""
    callee_src = """
module add_one_mod
contains
subroutine add_one(arr, n)
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: arr(n)
  integer :: i
  do i = 1, n
    arr(i) = arr(i) + 1.0d0
  end do
end subroutine add_one
end module add_one_mod
"""
    caller_src = """
module caller_mod
contains
subroutine caller(arr, n)
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: arr(n)
  interface
    subroutine add_one(arr, n) bind(c, name="add_one")
      use iso_c_binding
      integer(c_int), value :: n
      real(c_double), intent(inout) :: arr(n)
    end subroutine
  end interface
  call add_one(arr, n)
end subroutine caller
end module caller_mod
"""
    # Build the callee with an EMPTY registry -- its own build must not mark it external.
    clear_external_registry()
    callee_sdfg = build_sdfg(callee_src, tmp_path / "callee", name="add_one", entry="add_one_mod::add_one").build()
    # Register so the caller build emits an ExternalCall (else hlfir-inline-all lowers the bind(c) interface away).
    clear_external_registry()
    try:
        # Declare add_one external via the unified policy; arg plan derived from the HLFIR call site.
        apply_external_functions([ExternalFunction("add_one")])
        caller_sdfg = build_sdfg(caller_src, tmp_path / "caller", name="caller", entry="caller_mod::caller").build()
    finally:
        clear_external_registry()

    # Library output is named after the entry procedure; _util.build_sdfg appends a per-test suffix, so check the prefix.
    assert callee_sdfg.name.startswith("add_one")
    assert caller_sdfg.name.startswith("caller")

    # Pre-condition: the caller SDFG carries an ExternalCall library node for add_one.
    from dace_fortran.external import ExternalCall
    ext_sites = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, ExternalCall) and n.c_name == "add_one"
    ]
    assert len(ext_sites) == 1, (f"expected exactly one ExternalCall for add_one before inline, "
                                 f"got {len(ext_sites)}")

    # Re-declare so the lookup inside inline_external resolves (only c_name
    # matters -- connector order comes from the callee SDFG's arglist, not args).
    apply_external_functions([ExternalFunction("add_one")])
    try:
        replaced = inline_external(caller_sdfg, "add_one", callee_sdfg=callee_sdfg)
    finally:
        clear_external_registry()
    assert replaced == 1

    # Post-condition: the ExternalCall is gone and a NestedSDFG wraps the callee SDFG.
    ext_sites_after = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, ExternalCall) and n.c_name == "add_one"
    ]
    assert ext_sites_after == []
    from dace.sdfg.nodes import NestedSDFG
    nested_sites = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, NestedSDFG) and n.sdfg is callee_sdfg
    ]
    assert len(nested_sites) == 1

    # Functional: the inlined caller should match gfortran on the same source.
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64, order='F')
    arr_sdfg = arr.copy(order='F')
    caller_sdfg.validate()
    caller_sdfg(arr=arr_sdfg, n=np.int32(4))
    np.testing.assert_allclose(arr_sdfg, arr + 1.0, rtol=0, atol=0)


def test_inline_external_aos_struct_callee(tmp_path):
    """Callee takes a derived-type ``t_vec``.  The caller's bridge expands the
    struct arg into per-member SoA connectors (``_a0``=s%u, ``_a1``=s%v) via
    ``hlfir-marshal-external-structs``; the standalone callee SDFG flattens the
    same dummy via ``hlfir-flatten-structs``, so the two signatures agree by
    position.  ``inline_external`` swaps the ExternalCall for a NestedSDFG; the
    inlined caller then runs pure SoA dataflow with no AoS packing at the boundary.
    """
    # Plain (non-bind(c)) module subroutine; hlfir-flatten-structs flattens the struct dummy into per-member leaves (s_u, s_v).
    callee_src = """
module aos_mod
  implicit none
  type t_vec
    real(kind=8) :: u(4)
    real(kind=8) :: v(4)
  end type
contains
  subroutine add_vec(s)
    type(t_vec), intent(inout) :: s
    integer :: i
    do i = 1, 4
      s%u(i) = s%u(i) + s%v(i)
    end do
  end subroutine
end module
"""
    # Caller USEs the same module + CALLs add_vec; Arg(kind="aos") makes
    # hlfir-marshal-external-structs expand the struct arg into per-member SoA connectors (_a0=u, _a1=v).
    caller_src = """
module caller_mod
  use aos_mod
  implicit none
contains
subroutine caller(s)
  implicit none
  type(t_vec), intent(inout) :: s
  call add_vec(s)
end subroutine caller
end module caller_mod
"""
    callee_ext_name = "_QMaos_modPadd_vec"
    clear_external_registry()
    callee_sdfg = build_sdfg(callee_src, tmp_path / "callee", name="add_vec", entry=callee_ext_name).build()
    clear_external_registry()
    # Stage aos_mod source in the caller's scratch dir so merge_used_modules
    # resolves USE aos_mod -- needed even though keep_external keeps add_vec
    # external, since the call site must still resolve type(t_vec) to declare the dummy.
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir(parents=True, exist_ok=True)
    (caller_dir / "aos_mod.f90").write_text(callee_src)
    try:
        keep_external(callee_ext_name, args=(Arg(kind="aos", intent="inout"), ))
        caller_sdfg = build_sdfg(caller_src, caller_dir, name="caller", entry="caller_mod::caller").build()
    finally:
        clear_external_registry()

    from dace_fortran.external import ExternalCall
    ext_sites = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, ExternalCall) and n.c_name == callee_ext_name
    ]
    assert len(ext_sites) == 1
    ext_node = ext_sites[0]
    # Marshal pass expanded the single struct arg into per-member connectors -- _a0 and _a1 here.
    assert set(ext_node.in_connectors) == {"_a0", "_a1"}
    assert set(ext_node.out_connectors) == {"_a0_o", "_a1_o"}

    # Confirm callee SDFG flattened to the same two leaves so the arglist matches by position.
    callee_args = list(callee_sdfg.arglist().keys())
    assert len(callee_args) == 2, (f"callee expected to flatten its struct dummy to 2 leaves, got "
                                   f"{callee_args}")

    keep_external(callee_ext_name, args=(Arg(kind="aos", intent="inout"), ))
    try:
        replaced = inline_external(caller_sdfg, callee_ext_name, callee_sdfg=callee_sdfg)
    finally:
        clear_external_registry()
    assert replaced == 1

    ext_after = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, ExternalCall) and n.c_name == callee_ext_name
    ]
    assert ext_after == []
    from dace.sdfg.nodes import NestedSDFG
    nested_after = [
        n for state in caller_sdfg.all_states() for n in state.nodes()
        if isinstance(n, NestedSDFG) and n.sdfg is callee_sdfg
    ]
    assert len(nested_after) == 1

    caller_sdfg.validate()
    u = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64, order='F')
    v = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64, order='F')
    expected = u + v
    u_sdfg = u.copy(order='F')
    v_sdfg = v.copy(order='F')
    caller_sdfg(s_u=u_sdfg, s_v=v_sdfg)
    np.testing.assert_allclose(u_sdfg, expected, rtol=0, atol=0)
    np.testing.assert_allclose(v_sdfg, v, rtol=0, atol=0)
