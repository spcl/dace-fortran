"""Unit test for the ``mo_solve_nh_diff`` differential helpers.

The ICON binding-swap integration runs the stock Fortran ``solve_nh`` and the
SDFG ``solve_nh_dace_icon`` on the SAME input and compares the mutable state
BIT-FOR-BIT (:file:`mo_solve_nh_diff.f90`).  Three properties must hold for that
to be sound, and this test pins them against gfortran with a small driver:

  * **deep copy is independent BOTH ways** -- ``clone_state_indep_prog`` /
    ``clone_prepadv_indep`` must allocate FRESH targets (the fields are
    ``POINTER``, so a shallow ``dst = src`` would alias the same storage and
    the reference run would clobber the SDFG run).  Mutating the original must
    NOT change the clone, and mutating the clone must NOT change the original.

  * **non-pointer scalars survive the clone** -- ``diag%max_vcfl_dyn`` is a
    MAX-accumulator the velocity callback carries across substeps; it reaches
    the clone via the shallow ``dst = src`` value copy, and ``clone_diag_indep``
    takes the diag INTENT(INOUT) precisely so that copy is not default-reset
    (an INTENT(OUT) diag zeroed it and flipped the ``*_is_associated`` guards
    -- the regression this pins).

  * **compare is bit-exact and counts exactly** -- each ``compare_*`` reports 0
    differing elements for identical state and the exact perturbation count
    after single-element mutations (no tolerance).

``free_state_clone`` / ``free_prepadv_clone`` releasing everything the clones
allocated is pinned by running the same driver under valgrind when available
(pointer allocations are not auto-finalized, so a missed member leaks and a
double-free aborts).

The real ``t_nh_state`` / ``t_prepare_adv`` are replaced with minimal modules of
the same names carrying only the fields the helpers touch
(:file:`_solve_nh_min_types.py`), so the test needs just gfortran (no flang /
SDFG / ICON build).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from icon.full._solve_nh_min_types import MIN_STATE_TYPES_F90

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")

_HERE = Path(__file__).resolve().parent
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"

_DRIVER = """\
program test_diff
  use mo_nonhydro_types
  use mo_prepadv_types
  use mo_solve_nh_diff
  implicit none
  type(t_nh_state)    :: st, clone
  type(t_prepare_adv) :: pa, pa_clone
  integer :: nnow, nnew, ndiff, i, j, k
  real(kind=8) :: v

  nnow = 1
  nnew = 2
  allocate(st%prog(2))
  do i = 1, 2
    allocate(st%prog(i)%vn(3,4,5), st%prog(i)%w(3,4,5), st%prog(i)%rho(3,4,5), &
             st%prog(i)%exner(3,4,5), st%prog(i)%theta_v(3,4,5))
    do k = 1, 5
      do j = 1, 4
        v = real(100*i + 10*j + k, kind=8) * 1.5d0
        st%prog(i)%vn(:,j,k)      = v
        st%prog(i)%w(:,j,k)       = v + 0.25d0
        st%prog(i)%rho(:,j,k)     = v + 0.5d0
        st%prog(i)%exner(:,j,k)   = v + 0.75d0
        st%prog(i)%theta_v(:,j,k) = v + 0.125d0
      end do
    end do
  end do
  ! Allocate a rank-3 (vt) and a rank-4 (ddt_vn_apc_pc) diag field so the deep
  ! copy exercises clone_ptr3 AND clone_ptr4; the rest stay unassociated (the
  ! => NULL() default), which the clone / compare helpers no-op on.
  allocate(st%diag%vt(3,4,5))
  allocate(st%diag%ddt_vn_apc_pc(3,4,5,3))
  st%diag%vt = 7.0d0
  st%diag%ddt_vn_apc_pc = -3.0d0
  ! Non-pointer scalar set BEFORE the clone: must reach the clone through the
  ! shallow value copy (an INTENT(OUT) diag in clone_diag_indep default-reset
  ! it to 0 -- the regression pinned in step (2)).
  st%diag%max_vcfl_dyn = 2.25d0
  ! prep_adv accumulators (all four POINTER fields, non-1 lower bound on one of
  ! them so the lbound/ubound-preserving allocate is exercised).
  allocate(pa%mass_flx_me(3,4,5), pa%mass_flx_ic(3,4,5), pa%vol_flx_ic(3,4,5))
  allocate(pa%vn_traj(0:2,4,5))
  pa%mass_flx_me = 1.0d0
  pa%mass_flx_ic = 2.0d0
  pa%vol_flx_ic  = 3.0d0
  pa%vn_traj     = 4.0d0

  call clone_state_indep_prog(st, clone)
  call clone_prepadv_indep(pa, pa_clone)

  ! (1) a fresh clone is bit-exact on prog(nnew), prog(nnow), diag and prep_adv.
  call compare_prog_nnew(st, clone, nnew, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone prog(nnew) is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_prog_nnew(st, clone, nnow, 'init:nnow', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone prog(nnow) is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_diag(st%diag, clone%diag, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone diag is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_prepadv(pa, pa_clone, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh prep_adv clone is not bit-exact, ndiff=', ndiff
    stop 1
  end if

  ! (2) the shallow value copy carried the scalar accumulator into the clone.
  if (clone%diag%max_vcfl_dyn /= 2.25d0) then
    print *, 'FAIL: max_vcfl_dyn lost in clone (INTENT(OUT) default-reset?), got', clone%diag%max_vcfl_dyn
    stop 1
  end if

  ! (3) the clone owns independent PROG storage: mutate the original, the clone
  !     must not move, and the compare must SEE exactly the perturbation --
  !     only at the mutated time level.
  st%prog(nnew)%vn(1,1,1) = st%prog(nnew)%vn(1,1,1) + 1.0d0
  if (clone%prog(nnew)%vn(1,1,1) == st%prog(nnew)%vn(1,1,1)) then
    print *, 'FAIL: clone aliases the original prog (shallow pointer copy)'
    stop 1
  end if
  call compare_prog_nnew(st, clone, nnew, 'mutated', ndiff)
  if (ndiff /= 1) then
    print *, 'FAIL: compare_prog_nnew should report exactly 1 differing element, ndiff=', ndiff
    stop 1
  end if
  call compare_prog_nnew(st, clone, nnow, 'mutated:nnow', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: nnew mutation leaked into the nnow compare, ndiff=', ndiff
    stop 1
  end if

  ! (4) independence holds the OTHER way too: mutate the CLONE's nnow level,
  !     the original must not move (the reference run writing its copy must
  !     never reach the SDFG run's state).
  clone%prog(nnow)%rho(2,3,4) = clone%prog(nnow)%rho(2,3,4) - 1.0d0
  if (st%prog(nnow)%rho(2,3,4) == clone%prog(nnow)%rho(2,3,4)) then
    print *, 'FAIL: original aliases the clone prog (shallow pointer copy)'
    stop 1
  end if
  call compare_prog_nnew(st, clone, nnow, 'mutated-clone:nnow', ndiff)
  if (ndiff /= 1) then
    print *, 'FAIL: clone-side nnow mutation should count exactly 1, ndiff=', ndiff
    stop 1
  end if

  ! (5) the clone owns independent DIAG storage too: mutate a rank-3 and a
  !     rank-4 diag field on the original; the clone must not move and
  !     compare_diag must report exactly the two perturbations; a third shows
  !     up once the scalar accumulator diverges as well.
  st%diag%vt(2,2,2) = st%diag%vt(2,2,2) + 1.0d0
  st%diag%ddt_vn_apc_pc(1,1,1,1) = st%diag%ddt_vn_apc_pc(1,1,1,1) - 1.0d0
  if (clone%diag%vt(2,2,2) == st%diag%vt(2,2,2)) then
    print *, 'FAIL: clone aliases the original diag (shallow pointer copy)'
    stop 1
  end if
  call compare_diag(st%diag, clone%diag, 'mutated', ndiff)
  if (ndiff /= 2) then
    print *, 'FAIL: compare_diag should report exactly 2 differing elements, ndiff=', ndiff
    stop 1
  end if
  st%diag%max_vcfl_dyn = 4.5d0
  call compare_diag(st%diag, clone%diag, 'mutated+scalar', ndiff)
  if (ndiff /= 3) then
    print *, 'FAIL: scalar max_vcfl_dyn divergence not counted, ndiff=', ndiff
    stop 1
  end if

  ! (6) prep_adv independence both ways + exact counting (vn_traj's clone must
  !     keep the 0-based lower bound for the element compare to line up).
  pa%mass_flx_me(1,2,3) = pa%mass_flx_me(1,2,3) + 1.0d0
  if (pa_clone%mass_flx_me(1,2,3) == pa%mass_flx_me(1,2,3)) then
    print *, 'FAIL: prep_adv clone aliases the original (shallow pointer copy)'
    stop 1
  end if
  pa_clone%vn_traj(0,1,1) = pa_clone%vn_traj(0,1,1) - 1.0d0
  if (pa%vn_traj(0,1,1) == pa_clone%vn_traj(0,1,1)) then
    print *, 'FAIL: prep_adv original aliases the clone (shallow pointer copy)'
    stop 1
  end if
  call compare_prepadv(pa, pa_clone, 'mutated', ndiff)
  if (ndiff /= 2) then
    print *, 'FAIL: compare_prepadv should report exactly 2 differing elements, ndiff=', ndiff
    stop 1
  end if

  call free_state_clone(clone)
  call free_prepadv_clone(pa_clone)
  ! Release the ORIGINALS too: the valgrind lane must end with EVERY block
  ! freed, so a clone helper that aliased instead of copied turns into a
  ! double-free here, and a clone member the free_* helpers miss stays behind
  ! as a definite leak.
  do i = 1, 2
    deallocate(st%prog(i)%vn, st%prog(i)%w, st%prog(i)%rho, st%prog(i)%exner, st%prog(i)%theta_v)
  end do
  deallocate(st%prog)
  deallocate(st%diag%vt, st%diag%ddt_vn_apc_pc)
  deallocate(pa%mass_flx_me, pa%mass_flx_ic, pa%vol_flx_ic, pa%vn_traj)
  print *, 'PASS'
end program test_diff
"""


@pytest.fixture(scope="module")
def diff_exe(tmp_path_factory) -> Path:
    """Compile the stand-in types + helpers + driver once for both lanes."""
    build = tmp_path_factory.mktemp("solve_nh_diff")
    (build / "min_types.f90").write_text(MIN_STATE_TYPES_F90)
    (build / "driver.f90").write_text(_DRIVER)
    shutil.copy(_DIFF_F90, build / "mo_solve_nh_diff.f90")

    exe = build / "test_diff"
    compile = subprocess.run([
        "gfortran", "-ffree-line-length-none", "-fcheck=all", "-g", "min_types.f90", "mo_solve_nh_diff.f90",
        "driver.f90", "-o",
        str(exe)
    ],
                             cwd=str(build),
                             capture_output=True,
                             text=True)
    assert compile.returncode == 0, f"mo_solve_nh_diff.f90 did not compile:\n{compile.stderr}"
    return exe


def test_solve_nh_diff_deepcopy_and_compare(diff_exe: Path):
    """Clone independence (both directions), scalar preservation, exact
    bit-level diff counts across prog / diag / prep_adv."""
    run = subprocess.run([str(diff_exe)], cwd=str(diff_exe.parent), capture_output=True, text=True)
    assert run.returncode == 0 and "PASS" in run.stdout, \
        f"differential helper driver failed:\nstdout={run.stdout}\nstderr={run.stderr}"


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="valgrind not on PATH")
def test_solve_nh_diff_frees_cleanly(diff_exe: Path):
    """free_state_clone / free_prepadv_clone release every fresh target the
    clones allocated: no definite leak, no double-free / invalid access.
    (gfortran's runtime keeps still-reachable I/O buffers, so only definite
    leaks are errors.)"""
    run = subprocess.run(
        ["valgrind", "--error-exitcode=42", "--leak-check=full", "--errors-for-leak-kinds=definite",
         str(diff_exe)],
        cwd=str(diff_exe.parent),
        capture_output=True,
        text=True)
    assert run.returncode == 0 and "PASS" in run.stdout, \
        f"driver leaked or corrupted memory under valgrind:\nstdout={run.stdout}\nstderr={run.stderr}"
