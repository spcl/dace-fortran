# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical-correctness tests for static-vtable monomorphisation: each
monomorphised kernel must lower to an SDFG and match the original polymorphic program run
through gfortran's real dispatch, across all four rewrite primitives (local, component,
clone, retype). Driven with ``stack_slots=True`` -- once dispatch is gone, the bridge can't
lower an allocatable derived-type scalar, so the SDFG form uses plain stack slots.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang
from dace_fortran.build import build_sdfg
from dace_fortran.inliner.ast_desugaring.monomorphize import analyze, parse_program
from dace_fortran.inliner.ast_desugaring.monomorphize_rewrite import (clone_shared_interposers,
                                                                      monomorphize_component_dispatch,
                                                                      monomorphize_local_dispatch, retype_to_concrete)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# --- LOCAL dispatch: run(sel, x) allocates one of two arms and dispatches -------
_LOCAL_SRC = """
module m
  type, abstract :: base
  contains
    procedure(apply_i), deferred :: apply
  end type
  abstract interface
    subroutine apply_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: apply => gmres_apply
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: apply => cg_apply
  end type
contains
  subroutine gmres_apply(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0 + 1.0
  end subroutine
  subroutine cg_apply(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0 - 2.0
  end subroutine
  subroutine run(sel, x)
    integer, intent(in) :: sel
    real, intent(inout) :: x
    class(base), allocatable :: s
    if (sel == 1) then
      allocate(t_gmres :: s)
    else
      allocate(t_cg :: s)
    end if
    call s%apply(x)
  end subroutine
end module
"""

# --- COMPONENT dispatch: a container holds the polymorphic backend; factory + dispatch in one SDFG-able entry. ---
_COMPONENT_SRC = """
module m
  type, abstract :: base
  contains
    procedure(solve_i), deferred :: solve
  end type
  abstract interface
    subroutine solve_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: solve => gmres_solve
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: solve => cg_solve
  end type
  type :: container
    class(base), allocatable :: act
  contains
    procedure :: setup => container_setup
    procedure :: run   => container_run
  end type
contains
  subroutine gmres_solve(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0 + 1.0
  end subroutine
  subroutine cg_solve(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0 - 2.0
  end subroutine
  subroutine container_setup(this, sel)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
  end subroutine
  subroutine container_run(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%solve(x)
  end subroutine
  subroutine run(sel, x)
    integer, intent(in) :: sel
    real, intent(inout) :: x
    type(container) :: c
    call c%setup(sel)
    call c%run(x)
  end subroutine
end module
"""

# --- CLONE: slot's dispatched binding is a NON-deferred shared interposer (base_run) that dispatches this%doit; clone specialises it per arm. ---
_CLONE_SRC = """
module m
  type, abstract :: base
  contains
    procedure :: run => base_run
    procedure(doit_i), deferred :: doit
  end type
  abstract interface
    subroutine doit_i(this, x)
      import base
      class(base), intent(inout) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(base) :: t_gmres
  contains
    procedure :: doit => gmres_doit
  end type
  type, extends(base) :: t_cg
  contains
    procedure :: doit => cg_doit
  end type
  type :: container
    class(base), allocatable :: act
  contains
    procedure :: setup => container_setup
    procedure :: go    => container_go
  end type
contains
  subroutine base_run(this, x)
    class(base), intent(inout) :: this
    real, intent(inout) :: x
    call this%doit(x)
  end subroutine
  subroutine gmres_doit(this, x)
    class(t_gmres), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 2.0 + 1.0
  end subroutine
  subroutine cg_doit(this, x)
    class(t_cg), intent(inout) :: this
    real, intent(inout) :: x
    x = x * 3.0 - 2.0
  end subroutine
  subroutine container_setup(this, sel)
    class(container), intent(inout) :: this
    integer, intent(in) :: sel
    if (sel == 1) then
      allocate(t_gmres :: this%act)
    else
      allocate(t_cg :: this%act)
    end if
  end subroutine
  subroutine container_go(this, x)
    class(container), intent(inout) :: this
    real, intent(inout) :: x
    call this%act%run(x)
  end subroutine
  subroutine run(sel, x)
    integer, intent(in) :: sel
    real, intent(inout) :: x
    type(container) :: c
    call c%setup(sel)
    call c%go(x)
  end subroutine
end module
"""

# --- RETYPE: axis pinned to one concrete type at the call site (pointer component), specialised by retyping CLASS(base) -> TYPE(concrete). ---
_RETYPE_SRC = """
module m
  type, abstract :: t_transfer
  contains
    procedure(into_i), deferred :: into
  end type
  abstract interface
    subroutine into_i(this, x)
      import t_transfer
      class(t_transfer), intent(in) :: this
      real, intent(inout) :: x
    end subroutine
  end interface
  type, extends(t_transfer) :: t_trivial
  contains
    procedure :: into => trivial_into
  end type
  type, extends(t_transfer) :: t_subset
  contains
    procedure :: into => subset_into
  end type
  type :: backend
    class(t_transfer), pointer :: trans => null()
  contains
    procedure :: use => backend_use
  end type
contains
  subroutine trivial_into(this, x)
    class(t_trivial), intent(in) :: this
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
  subroutine subset_into(this, x)
    class(t_subset), intent(in) :: this
    real, intent(inout) :: x
    x = x + 2.0
  end subroutine
  subroutine backend_use(this, x)
    class(backend), intent(inout) :: this
    real, intent(inout) :: x
    call this%trans%into(x)
  end subroutine
  subroutine run(x)
    real, intent(inout) :: x
    type(backend) :: b
    type(t_trivial), target :: tr
    b%trans => tr
    call b%use(x)
  end subroutine
end module
"""

#: bind(c) shims over the module's run -- two shapes: ladder fixtures take (sel, x); retype fixture takes (x).
_REF_CALLER_SELX = """
subroutine run_c(sel, x) bind(c, name='run_c')
  use iso_c_binding
  use m
  integer(c_int), intent(in) :: sel
  real(c_float), intent(inout) :: x
  call run(sel, x)
end subroutine
"""
_REF_CALLER_X = """
subroutine run_c(x) bind(c, name='run_c')
  use iso_c_binding
  use m
  real(c_float), intent(inout) :: x
  call run(x)
end subroutine
"""


def _gfortran_ref(work: Path, module_src: str, caller_src: str) -> ctypes.CDLL:
    """Compile the original polymorphic program + bind(c) caller into a .so; returns the ctypes handle (the real-virtual-dispatch reference)."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "mod.f90").write_text(module_src)
    (work / "caller.f90").write_text(caller_src)
    so = work / "libref.so"
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        "mod.f90", "caller.f90", "-o",
        str(so)
    ],
                          cwd=str(work))
    return ctypes.CDLL(str(so))


def _build(tmp_path: Path, src: str, rewrite, name: str):
    """Monomorphise ``src`` in place via ``rewrite(prog)``, lower to an SDFG, compile."""
    prog = parse_program(src)
    rewrite(prog)
    return build_sdfg(str(prog), entry="m::run", name=name, out_dir=str(tmp_path / name)).compile()


def _check_selx(tmp_path: Path, src: str, rewrite, name: str):
    """Ladder fixture: compare SDFG vs gfortran on random ``x`` for both arms."""
    csdfg = _build(tmp_path, src, rewrite, name)
    ref = _gfortran_ref(tmp_path / f"{name}_ref", src, _REF_CALLER_SELX)
    rng = np.random.default_rng(1234)
    for sel in (1, 2):
        for _ in range(5):
            xv = float(rng.uniform(-10.0, 10.0))
            expected = ctypes.c_float(xv)
            ref.run_c(ctypes.byref(ctypes.c_int(sel)), ctypes.byref(expected))
            x = np.array([xv], dtype=np.float32)
            csdfg(sel=np.int32(sel), x=x)
            assert abs(float(x[0]) - expected.value) < 1e-5, \
                f"{name} sel={sel} x0={xv}: SDFG {x[0]} != Fortran {expected.value}"


def _plan(prog):
    return analyze(prog)[0]


def test_local_dispatch_sdfg_matches_fortran(tmp_path: Path):
    _check_selx(tmp_path, _LOCAL_SRC, lambda prog: monomorphize_local_dispatch(prog, _plan(prog), stack_slots=True),
                "mono_local")


def test_component_dispatch_sdfg_matches_fortran(tmp_path: Path):
    _check_selx(tmp_path, _COMPONENT_SRC,
                lambda prog: monomorphize_component_dispatch(prog, _plan(prog), stack_slots=True), "mono_component")


def _rewrite_clone(prog):
    plan = _plan(prog)
    monomorphize_component_dispatch(prog, plan, stack_slots=True)
    clone_shared_interposers(prog, plan)


def test_clone_interposer_sdfg_matches_fortran(tmp_path: Path):
    _check_selx(tmp_path, _CLONE_SRC, _rewrite_clone, "mono_clone")


def test_retype_sdfg_matches_fortran(tmp_path: Path):
    csdfg = _build(tmp_path, _RETYPE_SRC, lambda prog: retype_to_concrete(prog, "t_transfer", "t_trivial"),
                   "mono_retype")
    ref = _gfortran_ref(tmp_path / "mono_retype_ref", _RETYPE_SRC, _REF_CALLER_X)
    rng = np.random.default_rng(99)
    for _ in range(8):
        xv = float(rng.uniform(-10.0, 10.0))
        expected = ctypes.c_float(xv)
        ref.run_c(ctypes.byref(expected))
        x = np.array([xv], dtype=np.float32)
        csdfg(x=x)
        assert abs(float(x[0]) - expected.value) < 1e-5, f"retype x0={xv}: SDFG {x[0]} != Fortran {expected.value}"
