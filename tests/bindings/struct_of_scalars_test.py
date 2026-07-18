"""DT-of-scalar-constants flattening: a derived-type dummy with scalar members (cloudsc's
YDCST/YDTHF/YDECLDP pattern) must flatten to per-member scalar SDFG args via hlfir-flatten-structs,
with cst%rg reads lowered to the flat cst_rg scalar. Passing this means the bridge can consume
upstream cloudsc's derived-type bundles directly, without manual ASSOCIATE flattening."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import FlattenPlan

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_consts
  use iso_c_binding
  implicit none
  type :: t_consts
     real(c_double) :: rg
     real(c_double) :: rd
     real(c_double) :: rcpd
  end type t_consts
end module mo_consts

module kernel_dt_const_mod
contains
subroutine kernel_dt_const(cst, n, out)
  use mo_consts
  use iso_c_binding
  implicit none
  type(t_consts), intent(in) :: cst
  integer, intent(in) :: n
  real(c_double), intent(out) :: out(n)
  integer :: i
  do i = 1, n
     out(i) = (cst%rg / cst%rcpd) * cst%rd * real(i, c_double)
  end do
end subroutine kernel_dt_const
end module kernel_dt_const_mod
"""


def test_dt_of_scalar_constants_flattens_per_member(tmp_path):
    """type(t_consts) with three scalar members flattens to three flat scalar SDFG args
    (FlattenEntry shape/dtype + arglist names)."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_SRC, sdfg_dir, name="kernel_dt_const", entry="kernel_dt_const_mod::kernel_dt_const")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()

    # FlattenPlan has one entry per scalar member.
    entries_by_outer = {e.outer_expr: e for e in plan.entries}
    assert "cst%rg" in entries_by_outer, f"missing cst%rg entry: {list(entries_by_outer.keys())}"
    assert "cst%rd" in entries_by_outer
    assert "cst%rcpd" in entries_by_outer

    for member, e in entries_by_outer.items():
        assert e.recipe.rank == 0, f"{member}: scalar member should be rank 0, got {e.recipe.rank}"
        assert e.recipe.scratch_dtype == "float64", f"{member}: dtype {e.recipe.scratch_dtype}"
        assert e.recipe.flat_names == (member.replace("%", "_"), ), e.recipe.flat_names
        assert e.recipe.read_exprs == (member, ), e.recipe.read_exprs

    # SDFG arglist carries the flat per-member names.
    arglist = list(sdfg.arglist().keys())
    for flat in ("cst_rg", "cst_rd", "cst_rcpd"):
        assert flat in arglist, f"{flat} missing from SDFG arglist: {arglist}"


def test_dt_of_scalar_constants_numerical(tmp_path):
    """Bridge SDFG must produce (rg/rcpd)*rd*i per element bit-for-bit. Reference is plain NumPy
    (not f2py): f2py's crackfortran maps type(t_consts) dummies to 'void' and crashes on lookup."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="kernel_dt_const", entry="kernel_dt_const_mod::kernel_dt_const").build()

    rg = 9.80665
    rd = 287.0597
    rcpd = 1004.709
    n = 8

    # Reference: same left-to-right evaluation Fortran would do.
    out_ref = np.empty(n, dtype=np.float64, order="F")
    for i in range(1, n + 1):
        out_ref[i - 1] = (rg / rcpd) * rd * float(i)

    # bridge surfaces scalar struct members as length-1 Array(1,) rather than Scalar; route accordingly.
    from dace.data import Scalar
    arglist = sdfg.arglist()

    def _route(name, value):
        desc = arglist.get(name)
        if desc is None or isinstance(desc, Scalar):
            return value
        return np.array([value], dtype=np.float64)

    out_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(
        cst_rg=_route("cst_rg", rg),
        cst_rd=_route("cst_rd", rd),
        cst_rcpd=_route("cst_rcpd", rcpd),
        n=n,
        out=out_sdfg,
    )
    np.testing.assert_array_equal(out_sdfg, out_ref)
