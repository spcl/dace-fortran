"""Source-level body inlining of a named subprogram resolves the ICON halo
``sync_patch_array`` ``typ``-ladder pointer rebind.

``sync_patch_array(typ, p_patch, arr)`` rebinds ``p_pat`` to one of
``p_patch%comm_pat_c`` / ``comm_pat_e`` selected by a runtime ``typ``; the bridge's
View model cannot lower a runtime-selected rebind (``hlfir-rewrite-pointer-assigns``
rejects it as an interleaved rebind).  But every call site passes a COMPILE-TIME
constant ``typ`` -- so inlining the wrapper into its caller at the SOURCE level lets
the existing ``const_eval`` + ``prune_branches`` fold the ladder to a single
``p_pat => p_patch%comm_pat_c`` rebind, which lowers as an ordinary single source.
"""
import pytest

from dace_fortran.inliner.ast_desugaring import optimizations, pruning
from dace_fortran.inliner.ast_desugaring.specialize_at_source import (inline_named_functions, inline_named_subprograms)
from dace_fortran.inliner.ast_desugaring.monomorphize import parse_program

from _util import build_sdfg, have_flang

_LADDER_SRC = """
module mo_comm
  implicit none
  type :: t_comm_pattern_orig
    integer :: n
  end type
  type :: t_patch
    type(t_comm_pattern_orig), pointer :: comm_pat_c
    type(t_comm_pattern_orig), pointer :: comm_pat_e
  end type
contains
  subroutine exchange_data(p_pat, recv)
    type(t_comm_pattern_orig), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer :: i
    do i = 1, p_pat%n
      recv(i) = recv(i) * 2.0d0
    end do
  end subroutine
  subroutine sync_patch_array(typ, p_patch, arr)
    integer, intent(in) :: typ
    type(t_patch), intent(in), target :: p_patch
    real(8), intent(inout) :: arr(:)
    type(t_comm_pattern_orig), pointer :: p_pat
    if (typ == 1) then
      p_pat => p_patch%comm_pat_c
    else if (typ == 2) then
      p_pat => p_patch%comm_pat_e
    end if
    call exchange_data(p_pat, arr)
  end subroutine
end module

subroutine dycore_step(p_patch, hnew, n)
  use mo_comm
  implicit none
  type(t_patch), intent(in), target :: p_patch
  real(8), intent(inout) :: hnew(:)
  integer, intent(in) :: n
  integer :: i
  do i = 2, n - 1
    hnew(i) = hnew(i) + 1.0d0
  end do
  call sync_patch_array(1, p_patch, hnew)
end subroutine
"""


def _inline_and_fold(src, targets):
    prog = parse_program(src)
    n = inline_named_subprograms(prog, targets)
    prog = optimizations.const_eval_nodes(prog)
    prog = pruning.prune_branches(prog)
    return prog, n


def test_inline_folds_typ_ladder_to_single_rebind():
    """After inlining + the existing const-fold/prune, the runtime ``typ`` ladder
    is gone and exactly the ``typ==1`` arm's rebind survives in the caller."""
    prog, n = _inline_and_fold(_LADDER_SRC, ["sync_patch_array"])
    assert n == 1
    body = str(prog)
    caller = body.split("SUBROUTINE dycore_step")[1].split("END SUBROUTINE dycore_step")[0]
    # the wrapper's body is now in the caller, its local renamed and declared
    assert "=> p_patch % comm_pat_c" in caller
    # the ladder collapsed: no surviving IF on the (constant-folded) typ, and the
    # typ==2 arm is gone
    assert "comm_pat_e" not in caller
    assert "1 == 1" not in caller and "1 == 2" not in caller
    # the wrapper call is gone; the inner exchange call survives (inlined body)
    assert "sync_patch_array" not in caller
    assert "exchange_data" in caller


def test_keyword_and_positional_actuals_bind_correctly():
    """Positional + keyword actuals map to the right dummies on inline."""
    src = """
module m
  implicit none
contains
  subroutine inner(a, b, c)
    integer, intent(in) :: a, b
    real(8), intent(inout) :: c(:)
    c(a) = c(b)
  end subroutine
end module
subroutine outer(x)
  use m
  implicit none
  real(8), intent(inout) :: x(:)
  call inner(2, c=x, b=5)
end subroutine
"""
    prog, n = _inline_and_fold(src, ["inner"])
    assert n == 1
    caller = str(prog).split("SUBROUTINE outer")[1]
    # a=2 (positional), b=5 (keyword), c=x (keyword)
    assert "x(2) = x(5)" in caller.replace(" ", " ")
    assert "CALL inner" not in caller


#: The real ICON ``sync_patch_array_mult`` shape: a 2-level wrapper chain
#: (``f3din`` -> ``mixprec``) that forwards OPTIONAL fields ``f2``/``f3``, with the
#: ``typ`` ladder and an unconditional ``SIZE(fN)`` accumulation inside a runtime
#: MPI guard.  The caller omits ``f3`` (an optional) and passes a constant ``typ``.
_MULT_SRC = """
module mo_comm
  implicit none
  type :: t_comm_pattern_orig
    integer :: n
  end type
  type :: t_patch
    type(t_comm_pattern_orig), pointer :: comm_pat_c
    type(t_comm_pattern_orig), pointer :: comm_pat_e
  end type
  logical :: do_mpi = .true.
contains
  subroutine exchange_data_mult(p_pat, ndim, recv1, recv2, recv3)
    type(t_comm_pattern_orig), intent(in) :: p_pat
    integer, intent(in) :: ndim
    real(8), intent(inout) :: recv1(:,:)
    real(8), optional, intent(inout) :: recv2(:,:), recv3(:,:)
    integer :: i
    do i = 1, p_pat%n
      recv1(i,1) = recv1(i,1) * 2.0d0
    end do
  end subroutine
  subroutine sync_mult_mixprec(typ, p_patch, nfields, f1, f2, f3)
    integer, intent(in) :: typ, nfields
    type(t_patch), intent(in), target :: p_patch
    real(8), intent(inout) :: f1(:,:)
    real(8), optional, intent(inout) :: f2(:,:), f3(:,:)
    type(t_comm_pattern_orig), pointer :: p_pat
    integer :: ndim
    if (typ == 1) then
      p_pat => p_patch%comm_pat_c
    else if (typ == 2) then
      p_pat => p_patch%comm_pat_e
    end if
    if (do_mpi) then
      ndim = 0
      ndim = ndim + size(f1, 2)
      ndim = ndim + size(f2, 2)
      ndim = ndim + size(f3, 2)
      call exchange_data_mult(p_pat, ndim, recv1=f1, recv2=f2, recv3=f3)
    end if
  end subroutine
  subroutine sync_mult_f3din(typ, p_patch, nfields, f1, f2, f3)
    integer, intent(in) :: typ, nfields
    type(t_patch), intent(in), target :: p_patch
    real(8), intent(inout) :: f1(:,:)
    real(8), optional, intent(inout) :: f2(:,:), f3(:,:)
    call sync_mult_mixprec(typ=typ, p_patch=p_patch, nfields=nfields, f1=f1, f2=f2, f3=f3)
  end subroutine
end module

subroutine dycore_step(p_patch, a, b, n)
  use mo_comm
  implicit none
  type(t_patch), intent(in), target :: p_patch
  real(8), intent(inout) :: a(:,:), b(:,:)
  integer, intent(in) :: n
  call sync_mult_f3din(2, p_patch, 2, f1=a, f2=b)
end subroutine
"""


def test_two_level_chain_with_optionals_folds():
    """The 2-level wrapper chain flattens fully: the ``typ`` ladder folds to the
    single live arm, the OMITTED optional ``f3`` becomes ``SIZE(f3)->0`` and its
    forwarded ``recv3=f3`` actual is dropped -- no absent dummy survives."""
    prog, n = _inline_and_fold(_MULT_SRC, ["sync_mult_f3din", "sync_mult_mixprec"])
    assert n == 2, "both wrapper levels must inline"
    caller = str(prog).split("SUBROUTINE dycore_step")[1].split("END SUBROUTINE dycore_step")[0]
    assert "=> p_patch % comm_pat_e" in caller  # typ==2 arm
    assert "comm_pat_c" not in caller  # typ==1 arm folded away
    assert "f2dace_absent" not in caller  # no leaked absent marker
    assert "recv3" not in caller.lower()  # absent field's actual dropped
    assert "exchange_data_mult" in caller.lower()


def test_call_passing_positional_absent_optional_is_dropped():
    """A wrapper whose body calls a debug routine passing an OMITTED optional
    POSITIONALLY (``check(typ, f3)`` with ``f3`` absent) -- which cannot be dropped
    as a keyword actual -- is dropped as a whole statement (dead in this
    specialization), so the wrapper still inlines cleanly.  This is the ICON
    ``check_patch_array_3d_dp(typ, p_patch, f3dinN_dp, ...)`` shape."""
    src = """
module m
  implicit none
contains
  subroutine check(typ, fld)
    integer, intent(in) :: typ
    real(8), intent(in) :: fld(:)
    if (fld(typ) > 0.0d0) continue
  end subroutine
  subroutine wrap(typ, f1, f2)
    integer, intent(in) :: typ
    real(8), intent(inout) :: f1(:)
    real(8), optional, intent(in) :: f2(:)
    call check(typ, f1)
    call check(typ, f2)
    f1(1) = f1(1) + 1.0d0
  end subroutine
end module
subroutine drv(a)
  use m
  implicit none
  real(8), intent(inout) :: a(:)
  call wrap(1, a)
end subroutine
"""
    prog, n = _inline_and_fold(src, ["wrap"])
    assert n == 1
    caller = str(prog).split("SUBROUTINE drv")[1].split("END SUBROUTINE drv")[0]
    assert "f2dace_absent" not in caller  # absent f2 did not leak
    assert "CALL check(1, a)" in caller.replace("  ", " ")  # present-field call kept
    assert caller.lower().count("call check") == 1  # absent-field call dropped
    assert "a(1) = a(1) + 1.0" in caller.replace("  ", " ")


#: The single-field ICON path: ``sync_patch_array_3d`` passes the comm pattern as
#: ``comm_pat_of_type(p_patch, typ)`` -- a FUNCTION (pointer result) with the same
#: ``typ`` ladder, called inside another call's argument list (a ``Part_Ref``).
_FUNC_SRC = """
module mo_comm
  implicit none
  type :: t_comm_pattern_orig
    integer :: n
  end type
  type :: t_patch
    type(t_comm_pattern_orig), pointer :: comm_pat_c
    type(t_comm_pattern_orig), pointer :: comm_pat_e
  end type
contains
  function comm_pat_of_type(p_patch, typ) result(p_pat)
    integer, intent(in) :: typ
    type(t_patch), intent(in), target :: p_patch
    type(t_comm_pattern_orig), pointer :: p_pat
    if (typ == 1) then
      p_pat => p_patch%comm_pat_c
    else if (typ == 2) then
      p_pat => p_patch%comm_pat_e
    end if
  end function
  subroutine exchange_data_r3d(p_pat, recv)
    type(t_comm_pattern_orig), intent(in) :: p_pat
    real(8), intent(inout) :: recv(:)
    integer :: i
    do i = 1, p_pat%n
      recv(i) = recv(i) * 2.0d0
    end do
  end subroutine
  subroutine sync_patch_array_3d(typ, p_patch, arr)
    integer, intent(in) :: typ
    type(t_patch), intent(in), target :: p_patch
    real(8), intent(inout) :: arr(:)
    call exchange_data_r3d(comm_pat_of_type(p_patch, typ), arr)
  end subroutine
end module

subroutine dycore_step(p_patch, hnew)
  use mo_comm
  implicit none
  type(t_patch), intent(in), target :: p_patch
  real(8), intent(inout) :: hnew(:)
  call sync_patch_array_3d(2, p_patch, hnew)
end subroutine
"""


def test_keyword_name_not_substituted_when_forwarding_same_named_dummy():
    """A wrapper forwarding a dummy to a NON-target call by keyword
    (``leaf(typ=typ, lacc=.TRUE.)``) where the keyword name equals the dummy name:
    inlining the wrapper must substitute only the VALUE (right of ``=``), never the
    keyword (left) -- else the call becomes ``leaf(2 = 2, ...)`` and fails to parse.
    This is the exact shape of ICON's ``sync_patch_array_mult_f3din_dp`` forwarding
    ``typ``/``lacc``/``opt_varname`` to ``..._mixprec``.
    """
    src = """
module m
  implicit none
contains
  subroutine leaf(typ, lacc, x)
    integer, intent(in) :: typ
    logical, intent(in) :: lacc
    real(8), intent(inout) :: x(:)
    if (lacc) x(typ) = x(typ) + 1.0d0
  end subroutine
  subroutine wrap(typ, x)
    integer, intent(in) :: typ
    real(8), intent(inout) :: x(:)
    call leaf(typ=typ, lacc=.true., x=x)
  end subroutine
end module
subroutine drv(x)
  use m
  implicit none
  real(8), intent(inout) :: x(:)
  call wrap(2, x)
end subroutine
"""
    prog, n = _inline_and_fold(src, ["wrap"])  # leaf is NOT a target -> its call survives
    assert n == 1
    caller = str(prog).split("SUBROUTINE drv")[1].split("END SUBROUTINE drv")[0]
    assert "leaf(typ=2" in caller.replace(" ", "")  # keyword preserved, value folded
    assert "2 = 2" not in caller and ".TRUE. = .TRUE." not in caller.upper()


def test_function_result_inline_folds_ladder():
    """``comm_pat_of_type`` (a pointer-result FUNCTION used as an actual argument)
    hoists into the caller: the body assigns a result temp, the ladder folds to the
    constant-``typ`` arm, and the call argument becomes that single-source temp."""
    prog = parse_program(_FUNC_SRC)
    inline_named_subprograms(prog, ["sync_patch_array_3d"])
    nf = inline_named_functions(prog, ["comm_pat_of_type"])
    assert nf >= 1
    prog = optimizations.const_eval_nodes(prog)
    prog = pruning.prune_branches(prog)
    caller = str(prog).split("SUBROUTINE dycore_step")[1].split("END SUBROUTINE dycore_step")[0]
    assert "=> p_patch % comm_pat_e" in caller  # typ==2 arm, single source
    assert "comm_pat_c" not in caller
    assert "comm_pat_of_type" not in caller  # the function call is gone
    assert "exchange_data_r3d" in caller.lower()


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_function_result_inline_lowers(tmp_path):
    """The function-inlined single-field path lowers to an SDFG."""
    prog = parse_program(_FUNC_SRC)
    inline_named_subprograms(prog, ["sync_patch_array_3d"])
    inline_named_functions(prog, ["comm_pat_of_type"])
    prog = optimizations.const_eval_nodes(prog)
    prog = pruning.prune_branches(prog)
    sdfg = build_sdfg(str(prog), tmp_path / "sdfg", name="func_inl", entry="dycore_step").build()
    assert sdfg.number_of_nodes() >= 1


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_two_level_chain_with_optionals_lowers(tmp_path):
    """The flattened mult chain lowers to an SDFG (single-source rebind)."""
    prog, _ = _inline_and_fold(_MULT_SRC, ["sync_mult_f3din", "sync_mult_mixprec"])
    sdfg = build_sdfg(str(prog), tmp_path / "sdfg", name="mult_inl", entry="dycore_step").build()
    assert sdfg.number_of_nodes() >= 1


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_inlined_ladder_lowers_to_sdfg(tmp_path):
    """End-to-end: the inlined+folded source lowers to an SDFG (the raw ladder does
    not -- it is rejected as an interleaved rebind)."""
    prog, _ = _inline_and_fold(_LADDER_SRC, ["sync_patch_array"])
    sdfg = build_sdfg(str(prog), tmp_path / "sdfg", name="ladder_inl", entry="dycore_step").build()
    assert sdfg.number_of_nodes() >= 1


# --------------------------------------------------------------------------
# Forwarded caller-optional: PRESENT() must NOT fold when the actual is the
# caller's own OPTIONAL dummy (its presence is a runtime property).
# --------------------------------------------------------------------------

# ``outer`` forwards its OWN optional ``opt_start_level`` positionally into
# ``inner``'s optional dummy, exactly like ICON's
# ``div_oce_3D_mlevels`` -> ``div_oce_3D_mlevels_onTriangles``.  ``inner``
# defaults ``start_level`` to 1 when the optional is absent.  Specializing
# ``inner`` into ``outer`` must KEEP the ``IF (PRESENT(...)) ... ELSE
# start_level = 1`` guard: folding it ``.TRUE.`` deletes the default, and the
# absent scalar then reads 0 -> vertical loop starts at level 0 -> out-of-bounds
# read with silently wrong results.
_FORWARDED_OPTIONAL_SRC = """
module mo_fwd
  implicit none
contains
  subroutine inner(field, nlev, opt_start_level)
    real(8), intent(inout) :: field(:)
    integer, intent(in) :: nlev
    integer, intent(in), optional :: opt_start_level
    integer :: start_level, k
    if (present(opt_start_level)) then
      start_level = opt_start_level
    else
      start_level = 1
    end if
    do k = start_level, nlev
      field(k) = field(k) + 1.0d0
    end do
  end subroutine
  subroutine outer(field, nlev, opt_start_level)
    real(8), intent(inout) :: field(:)
    integer, intent(in) :: nlev
    integer, intent(in), optional :: opt_start_level
    call inner(field, nlev, opt_start_level)
  end subroutine
end module

subroutine root(field, nlev)
  use mo_fwd
  implicit none
  real(8), intent(inout) :: field(:)
  integer, intent(in) :: nlev
  ! Omits opt_start_level entirely -> absent all the way down.
  call outer(field, nlev)
end subroutine
"""


def test_forwarded_caller_optional_keeps_present_guard():
    """A dummy bound to the caller's own optional keeps its ``PRESENT`` guard
    when inlined -- its presence is only known at the caller's runtime."""
    prog = parse_program(_FORWARDED_OPTIONAL_SRC)
    inline_named_subprograms(prog, ["inner"])
    outer = str(prog).split("SUBROUTINE outer")[1].split("END SUBROUTINE outer")[0].upper()
    # The guard survives: both the runtime PRESENT test and the ELSE default.
    assert "PRESENT(OPT_START_LEVEL)" in outer
    assert "= 1" in outer
    # It must NOT have folded to a constant-true branch that drops the default.
    assert "IF (.TRUE.)" not in outer.replace(" ", "")


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
def test_forwarded_optional_absent_defaults_to_one_e2e(tmp_path):
    """End-to-end: with the optional omitted at the ROOT, both inlining levels
    fold to ``start_level = 1``, so every element is written.  A ``.TRUE.`` misfold
    at the ``outer -> inner`` boundary would instead read the absent scalar as 0 and
    start the loop at level 0 -- out of bounds (the ICON ocean level-0 bug)."""
    import numpy as np
    prog = parse_program(_FORWARDED_OPTIONAL_SRC)
    # Inline inner into outer (guard preserved by the fix), then outer into root
    # (where the optional is statically absent -> guard folds to the ELSE default).
    inline_named_subprograms(prog, ["inner", "outer"])
    prog = optimizations.const_eval_nodes(prog)
    prog = pruning.prune_branches(prog)
    sdfg = build_sdfg(str(prog), tmp_path / "sdfg", name="fwd_opt", entry="root").build()
    n = 8
    field = np.zeros(n, dtype=np.float64, order="F")
    sdfg(field=field, nlev=n)
    # start_level defaulted to 1 -> all n elements incremented, none left at 0.
    np.testing.assert_allclose(field, np.ones(n), rtol=1e-12, atol=1e-12)
