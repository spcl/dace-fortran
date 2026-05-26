# HLFIR -> DaCe Fortran frontend

Turn a Fortran kernel into an optimisable [DaCe](https://github.com/spcl/dace)
SDFG and get back a Fortran-callable shared library that preserves the
caller's original interface.

```
 kernel.f90
     |  (0) preprocess + flang -fc1 -emit-hlfir
     v
 kernel.hlfir            MLIR (HLFIR dialect) -- flang already did all the
     |  (1-3) C++ bridge   parsing, name binding, type inference, intrinsic
     v                     lowering, inlining, and normalisation
 normalised single-TU IR
     |  (4) walk -> DaCe
     v
 dace.SDFG  --(you optimise it)-->  any DaCe transformation
     |  (5) emit binding
     v
 <entry>_bindings.f90  +  lib<entry>.so   <- a real Fortran caller links here
```

## Why this exists

Flang already parses Fortran, binds names, infers types, and lowers
elemental intrinsics. Re-implementing any of that in Python duplicates
work and drifts against the standard. So this frontend consumes
Flang's already-elaborated **HLFIR** (an MLIR dialect), normalises it
into one narrow IR shape, walks it into an SDFG, and regenerates a
Fortran wrapper around the optimised result.

The job splits into six steps, each strengthening one invariant so the
Python SDFG walker only ever sees a single, predictable IR shape:

| Step | What it does |
| --- | --- |
| **(0) Preprocess** | Flang-friendly Fortran-text rewrites (`USE`-merge, OMP/ACC strip, integer-power expansion), then `flang-new-21 -fc1 -emit-hlfir`. |
| **(1) Parse** | Load the `.hlfir` into one MLIR `ModuleOp`; snapshot the caller-visible dummy list + derived-type layouts as a `FortranInterface`. |
| **(2) Inline + link** | `hlfir-inline-all` folds the whole call tree into the pinned entry; `symbol-dce` drops dead siblings; multi-file builds fail loudly on any surviving cross-TU `fir.call`. |
| **(3) Normalise** | A fixed chain of HLFIR rewrites (select-case lowering, AoS->SoA flattening, vector-subscript expansion, polymorphism rejection, shape propagation, ...) leaves one canonical IR shape. |
| **(4) Build SDFG** | Walk HLFIR into a tiny AST, emit the SDFG, run post-generation cleanup, then pin a `FrozenSignature` snapshot of the arg list. |
| **(5) Emit binding** | Generate `<entry>_bindings.f90`: a ref-counted Fortran module that re-presents the caller's original signature and, per struct member, picks a zero-copy alias or a deep copy. |

After step (4) you hold a normal `dace.SDFG`. Apply any DaCe
transformation you like; step (5) re-checks the frozen signature against
the live SDFG before emitting the binding, so a transformation that
re-ordered arguments or changed dtypes raises `SignatureDriftError`
instead of silently shipping a wrong wrapper.

## Quick start

```python
import dace_fortran

# 1. Fortran source -> SDFG.  ``entry`` is the mangled flang symbol:
#    ``_QPname`` for a free subroutine, ``_QM<mod>Pname`` for a
#    module procedure.  Omit it only when the source has exactly one
#    procedure (it is then auto-resolved).
sdfg = dace_fortran.build_sdfg(open("kernel.f90").read(), entry="_QPkernel")

# 2. Optimise.  Any DaCe transformation is fair game.
sdfg.simplify()

# 3. Emit + link a Fortran-callable library.  ``build_fortran_library``
#    re-verifies the frozen signature, writes ``<entry>_bindings.f90``,
#    and gfortran-links it against the compiled SDFG ``.so``.
from dace_fortran.bindings import build_fortran_library, SignatureDriftError
try:
    lib = build_fortran_library(sdfg, iface, plan, out_dir="build",
                                extra_sources=["caller.f90"], name="kernel")
except SignatureDriftError as e:
    raise SystemExit(f"a transformation invalidated the binding: {e}")

cdll = lib.load()                       # -> ctypes.CDLL
print("binding:", lib.bindings_f90, "library:", lib.so_path)
```

If the SDFG needs an OpenMP runtime, supply it at run time
(`LD_PRELOAD=<libgomp/libomp> python ...`); the library never hard-codes
one. The lower-level sequence (`emit_bindings` -> `frozen.verify_against(sdfg)`
-> gfortran link) is still supported and is exactly what
`build_fortran_library` consolidates.

## The three input tiers

Three entry points, ordered by how much the bridge does for you. All
return a built, validated `dace.SDFG` with `sdfg._frozen_signature`
attached. `import dace_fortran` is lazy (the C++ bridge builds on first
use).

### Tier 1 -- inline source

One self-contained string. No external deps, no build system.

```python
src = """
subroutine add_array(a, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  integer :: i
  do i = 1, n
    a(i) = a(i) + 1.0d0
  end do
end subroutine add_array
"""
sdfg = dace_fortran.build_sdfg(src, entry="_QPadd_array")
a = np.asfortranarray(np.zeros(8, dtype=np.float64))
sdfg(a=a, n=8)
assert (a == 1.0).all()
```

### Tier 2 -- multi-file project

A driver plus the modules it `USE`s, in any order. The file defining
`entry` is the root; the rest are merged into one translation unit via
`merge_used_modules` so flang sees a single self-contained TU.

```python
sdfg = dace_fortran.build_sdfg_from_files(
    ["driver.f90", "math_utils.f90"], entry="_QPcompute_tendencies")
```

### Tier 3 -- a real CMake / Autotools project

Tiers 1 and 2 drive flang internally, which does not scale to codebases
with hundreds of modules and real `netcdf` / `hdf5` / `yaxt` externals.
Those projects already have a build system that knows the right include
paths and cpp defines -- tier 3 reuses it. The whole contract is: **(a)
get a `compile_commands.json` from your build, (b) one Python call.**

```bash
# CMake / Ninja: one configure flag drops the database into the build dir.
cmake -S src/ -B build/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/

# Autotools / plain Make (ICON's shape): wrap the build in `bear`,
# which intercepts the compiler exec() calls.
./configure ...
bear -- make                  # writes ./compile_commands.json
```

```python
sdfg = dace_fortran.build_sdfg_from_project(
    "build/compile_commands.json",
    entry="_QMmod_jacobiPjacobi2d_update",      # mangled flang symbol
    stubs=["mpi_stub.f90", "netcdf_stub.f90"])  # see below
```

`stubs` are flang-buildable stand-ins for modules flang ships no `.mod`
for (`mpi` / `netcdf` / `hdf5` / ...): a small module declaring the names
the project `USE`s, compiled before the project TUs so those `USE` lines
resolve. Omit when there are no such externals.

To emit once and lower several entries (or to inspect the intermediate
`.hlfir` files), use the two explicit steps that
`build_sdfg_from_project` wraps:

```bash
python -m dace_fortran.emit_hlfir build/compile_commands.json \
    --out build/hlfir --stub mpi_stub.f90 --stub netcdf_stub.f90
```
```python
sdfg = dace_fortran.build_sdfg_from_hlfir(
    "build/hlfir", entry="_QMmod_jacobiPjacobi2d_update")
```

`tests/prebuilt_hlfir/` ships one worked project per capture route, each
with a plain build that knows nothing about HLFIR or the bridge:

| Project | Build | Externals | Demonstrates |
|---|---|---|---|
| `jacobi/` | autotools + `bear -- make` | MPI + netCDF (2 stubs) | ICON-shape build; entry stays MPI-free even though a sibling `USE`s MPI |
| `csr_spmv/` | cmake export flag | none | minimal happy path |

**Inlining scope.** Inlining is intra-TU: flang emits one `.hlfir` per
translation unit and the bridge consumes one. A procedure `USE`d from a
different TU stays an external symbol reference in the SDFG -- the right
contract for halo exchanges or I/O routines you *want* left external.

### Calling a separately-compiled external

A kernel that `CALL`s a separately-compiled `bind(c)` function declares
that callee's signature out-of-band; the bridge lowers the `CALL` to an
`ExternalCall` library node and links the callee's `.so` into the SDFG
`.so` with an rpath (so it stays self-contained -- no `LD_PRELOAD`).

```python
from dace_fortran import Arg, build_sdfg, register_external, ExternalSignature
register_external("foo", ExternalSignature(
    c_name="foo",
    args=[Arg("array", "float64", "inout"),   # inout default; a missed write is a silent bug
          Arg("scalar", "int32", "in")],
    libraries=["/abs/path/libfoo.so"]))
```

`intent` defaults to `inout` (a missed write into an opaque external is a
silent correctness bug; a missed read is only a missed optimisation).
`keep_external(name, ...)` is the shorter form when the defaults are
fine. MPI (`MPI_Send` / `Recv` / `Isend` / `Irecv` / `Wait`, including a
non-default communicator) is recognised automatically and lowered to
`dace.libraries.mpi` nodes -- no registration needed. References:
`tests/external_call/test_external_call.py`,
`tests/external_call/test_keep_external.py`.

## Worked example: a Quantum Espresso kernel (complex AXPY)

`tests/qe_loopnests/qe_e4_zaxpy.f90` is the hot loop of QE's `zaxpy`
(BLAS-shaped `Y = a*X + Y` on `complex(8)` data), one of the
SC26-Layout-AD experiment kernels:

```fortran
subroutine kernel(n, a, x, y)
  implicit none
  integer,    intent(in)    :: n
  ! ``a`` is a length-1 array, not a plain ``complex(8)`` scalar, to
  ! dodge a DaCe-core gap: ctypes on Python 3.12 has no
  ! ``c_double_complex``, so a by-value complex128 truncates the
  ! imaginary part.  Length-1 arrays use the pointer ABI (bit-identical).
  complex(8), intent(in)    :: a(1)
  complex(8), intent(in)    :: x(n)
  complex(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a(1) * x(i) + y(i)
  end do
end subroutine kernel
```

### (a) Build the SDFG and emit the binding

The entry symbol is `_QPkernel` (a free subroutine). Build the SDFG,
then describe the caller-facing interface with an `OriginalInterface`
and emit the binding. This kernel has no derived types, so its
`FlattenPlan` is empty (a pure pass-through):

```python
import dace_fortran
from dace_fortran.bindings import (OriginalArg, OriginalInterface,
                                   FlattenPlan, emit_bindings)

src = open("tests/qe_loopnests/qe_e4_zaxpy.f90").read()
sdfg = dace_fortran.build_sdfg(src, entry="_QPkernel", name="kernel")
# sdfg.arglist() -> {'a', 'x', 'y'} arrays + free symbol 'n'

iface = OriginalInterface(entry="kernel", args=(
    OriginalArg(name="n", fortran_type="integer(c_int)",    rank=0,            intent="in"),
    OriginalArg(name="a", fortran_type="complex(c_double)", rank=1, shape=("1",), intent="in"),
    OriginalArg(name="x", fortran_type="complex(c_double)", rank=1, shape=("n",), intent="in"),
    OriginalArg(name="y", fortran_type="complex(c_double)", rank=1, shape=("n",), intent="inout"),
))
emit_bindings(sdfg._frozen_signature, iface,
              FlattenPlan.from_dict({"entries": []}), "kernel_bindings.f90")
```

For an end-to-end SDFG-vs-f2py numerical check of this exact kernel see
`tests/qe_loopnests/test_sdfg_equivalence.py::test_e4_zaxpy`, which uses
the `tests/_util.build_sdfg` test funnel (a thin wrapper over the real
`make_builder`).

### (b) The generated binding

A flat kernel like this takes the **zero-copy alias path**: every array
is passed straight through by address (`c_loc`), scalars go by value,
and a ref-counted handle keeps one DaCe state per process. This is the
real `kernel_bindings.f90` the step above produces (trimmed to the
illustrative parts):

```fortran
module kernel_dace_bindings
  use iso_c_binding
  implicit none

  ! ---------------- C ABI interface to the compiled SDFG ----------------
  interface
    function dace_init_kernel() bind(c, name='__dace_init_kernel') result(h)
      import :: c_ptr, c_int
      type(c_ptr) :: h
    end function

    subroutine dace_program_kernel(h, a, x, y, n) bind(c, name='__program_kernel')
      import
      type(c_ptr), value :: h
      type(c_ptr), value :: a
      type(c_ptr), value :: x
      type(c_ptr), value :: y
      integer(c_int), value :: n         ! free symbol, by value
    end subroutine

    function dace_exit_kernel(h) bind(c, name='__dace_exit_kernel') result(err)
      import :: c_ptr, c_int
      type(c_ptr), value :: h
      integer(c_int) :: err
    end function
  end interface

  type(c_ptr), save :: dace_handle = c_null_ptr
  integer,     save :: init_count  = 0

contains

  subroutine kernel_dace(n, a, x, y)     ! <- caller's ORIGINAL signature
    integer(c_int),    intent(in),    target :: n
    complex(c_double), intent(in),    target :: a(1)
    complex(c_double), intent(in),    target :: x(n)
    complex(c_double), intent(inout), target :: y(n)
    integer(c_int) :: dace_err

    ! Ref-counted init -- first call allocates the state, the rest reuse it.
    if (init_count == 0) dace_handle = dace_init_kernel()
    init_count = init_count + 1

    ! Every array passes straight through by address; n goes by value.
    call dace_program_kernel(dace_handle, c_loc(a), c_loc(x), c_loc(y), n)
  end subroutine kernel_dace

  subroutine kernel_dace_finalize()
    integer(c_int) :: err
    if (init_count > 0) then
      init_count = init_count - 1
      if (init_count == 0) then
        err = dace_exit_kernel(dace_handle)
        dace_handle = c_null_ptr
      end if
    end if
  end subroutine kernel_dace_finalize
end module kernel_dace_bindings
```

The caller still writes `call kernel_dace(n, a, x, y)` exactly as it
called the original kernel; the wrapper forwards by address to the
compiled SDFG entry `__program_kernel`.

## Flattening derived types

DaCe has no native array-of-structs, so `hlfir-flatten-structs` (step
(3)) decomposes every derived-type dummy into one flat companion array
per leaf member (`type(t){a, b}` -> companions `t_a`, `t_b`), and the
binding reconnects the caller's struct to those companions. Whether that
reconnection is a **zero-copy alias** or a **deep copy** depends on the
member's memory layout: a whole contiguous array member aliases
(`c_f_pointer`); an interleaved or jagged member must be copied.
`tests/bindings/struct_bindings_e2e_test.py` covers every shape.

### A struct that needs a deep copy

Not every derived type can be aliased; this case
(`test_e2e_array_of_scalar_structs_deepcopy`) *requires* a copy:

```fortran
module mo_pt
  use iso_c_binding
  implicit none
  integer, parameter :: N = 6
  type :: point
     real(c_double) :: x, y, z, w
  end type point
end module mo_pt

subroutine kern_aos(pts)
  use mo_pt
  implicit none
  type(point), intent(inout) :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = pts(i)%x + pts(i)%y * pts(i)%z - pts(i)%w
  end do
end subroutine kern_aos
```

### Why it can't alias

Step (3)'s `hlfir-flatten-structs` turns the array-of-structs into four
struct-of-arrays companions (`pts_x`, `pts_y`, `pts_z`, `pts_w`), and the
SDFG kernel reads/writes those flat arrays. But in the caller's memory
the four members are **interleaved** -- `pts(1)%x, pts(1)%y, pts(1)%z,
pts(1)%w, pts(2)%x, ...` -- so `pts_x` is a *strided* view (stride = the
struct size), not contiguous storage. A zero-copy `c_f_pointer` alias
needs contiguous memory, which a strided AoS member does not provide.
The binding therefore allocates contiguous companions and **scatters the
members in / gathers them back out** with `do` loops. (When the members
are whole contiguous arrays -- e.g. `type(t_fields){a(:,:), b(:,:)}` --
each member *is* contiguous and the binding aliases it zero-copy via
`c_f_pointer`; that is the other path in the same test file.)

### The generated binding (copy-in / copy-out)

This is the real wrapper the bridge emitted for `kern_aos`
(`aliasable=False` for every member -> `render_copy_in_loop` /
`render_copy_out_loop`):

```fortran
! AUTO-GENERATED by dace.frontend.hlfir.bindings -- do not edit.
module kern_aos_dace_bindings
  use iso_c_binding
  use mo_pt, only: point, N
  implicit none
  private
  public :: kern_aos_dace, kern_aos_dace_finalize

  interface
    function dace_init_kern_aos() bind(c, name='__dace_init_kern_aos') result(h)
      import :: c_ptr, c_int
      type(c_ptr) :: h
    end function
    subroutine dace_program_kern_aos(h, pts_w, pts_x, pts_y, pts_z) &
        bind(c, name='__program_kern_aos')
      import
      type(c_ptr), value :: h
      type(c_ptr), value :: pts_w, pts_x, pts_y, pts_z
    end subroutine
    function dace_exit_kern_aos(h) bind(c, name='__dace_exit_kern_aos') result(err)
      import :: c_ptr, c_int
      type(c_ptr), value :: h
      integer(c_int) :: err
    end function
  end interface

  type(c_ptr), save :: dace_handle = c_null_ptr
  integer,     save :: init_count  = 0

contains

  subroutine kern_aos_dace(pts)          ! <- caller's ORIGINAL AoS signature
    type(point), intent(inout), target :: pts(N)

    ! Contiguous SoA companions allocated per call (NOT aliases).
    real(c_double), allocatable, target :: pts_x(:), pts_y(:), pts_z(:), pts_w(:)
    integer(c_int) :: i1
    integer(c_int) :: dace_err

    ! ----- Copy-in: scatter the interleaved AoS into the SoA companions -----
    allocate(pts_x(size(pts, dim=1)))
    do i1 = 1, size(pts, dim=1)
      pts_x(i1) = pts(i1)%x
    end do
    allocate(pts_y(size(pts, dim=1)))
    do i1 = 1, size(pts, dim=1)
      pts_y(i1) = pts(i1)%y
    end do
    ! ... pts_z, pts_w likewise ...

    if (init_count == 0) dace_handle = dace_init_kern_aos()
    init_count = init_count + 1

    call dace_program_kern_aos(dace_handle, &
      c_loc(pts_w), c_loc(pts_x), c_loc(pts_y), c_loc(pts_z))

    ! ----- Copy-out: gather the SoA companions back into the AoS -----
    do i1 = 1, size(pts, dim=1)
      pts(i1)%x = pts_x(i1)
    end do
    deallocate(pts_x)
    ! ... pts_y, pts_z, pts_w likewise ...
  end subroutine kern_aos_dace

  subroutine kern_aos_dace_finalize()
    integer(c_int) :: err
    if (init_count > 0) then
      init_count = init_count - 1
      if (init_count == 0) then
        err = dace_exit_kern_aos(dace_handle)
        dace_handle = c_null_ptr
      end if
    end if
  end subroutine kern_aos_dace_finalize
end module kern_aos_dace_bindings
```

(The real file repeats the `pts_z` / `pts_w` loops in full; trimmed here.)
An `intent(in)` struct array gets copy-in but no copy-out; an
`intent(out)` one gets copy-out only.

### Edge case: jagged allocatable members

The hardest flatten shape is an array-of-structs whose member is
`allocatable` and a **different length in each instance**
(`test_e2e_array_of_jagged_alloc_structs_deepcopy`):

```fortran
type :: bag
   real(c_double), allocatable :: w(:)   ! per-instance length differs
end type bag
! ... type(bag), intent(inout) :: a(NB), each a(i)%w allocated separately
do i = 1, NB
   do j = 1, size(a(i)%w)                ! per-instance extent
      a(i)%w(j) = a(i)%w(j) * 2.0_c_double
   end do
end do
```

There is no single contiguous companion shape for a jagged member, so
the pass packs it into an **ELLPACK companion** `a_w(NB, cap)` whose
inner extent `cap_a_w` is a runtime symbol -- the maximum member length
over all instances. The binding computes that maximum, allocates the
zero-padded buffer, scatters each instance's live region in, calls the
SDFG, and gathers only each live region back. Crucially, `size(a(i)%w)`
inside the kernel (the inner loop bound) flattens to that same
`cap_a_w` symbol, so the SDFG and the binding agree on the buffer width:

```fortran
  subroutine kern_jag_dace(a)
    type(bag), intent(inout), target :: a(NB)
    real(c_double), allocatable, target :: a_w(:, :)
    integer(c_long_long) :: cap_a_w        ! int64 runtime cap (by value)
    integer(c_int) :: i1

    ! ----- cap = max_i size(a(i)%w), guarded against unallocated rows -----
    cap_a_w = 0
    do i1 = 1, size(a, dim=1)
      if (allocated(a(i1)%w)) then
        if (size(a(i1)%w) > cap_a_w) cap_a_w = size(a(i1)%w)
      end if
    end do
    if (cap_a_w == 0) cap_a_w = 1          ! empty-batch sentinel

    allocate(a_w(size(a, dim=1), cap_a_w))
    a_w = 0                                ! padding rows stay zero
    do i1 = 1, size(a, dim=1)              ! pack-in: live region only
      if (allocated(a(i1)%w)) a_w(i1, 1:size(a(i1)%w)) = a(i1)%w
    end do

    if (init_count == 0) dace_handle = dace_init_kern_jag()
    init_count = init_count + 1
    call dace_program_kern_jag(dace_handle, c_loc(a_w), cap_a_w)

    do i1 = 1, size(a, dim=1)              ! pack-out: live region only
      if (allocated(a(i1)%w)) a(i1)%w = a_w(i1, 1:size(a(i1)%w))
    end do
    deallocate(a_w)
  end subroutine kern_jag_dace
```

The SDFG loops to `cap_a_w` for every row; the zero padding makes the
extra iterations harmless for elementwise kernels, and the pack-out
copies back only each instance's live `1:size` slice. Kernels that
*reduce* over the padding (a product, a count) would see the zeros, so
this packing is correct for elementwise and sum-like members only.

## Pipeline detail

### Preprocess rewrites -- step (0)

`dace_fortran.preprocess` holds text rewrites that must run before flang
(they change what flang accepts, or what arithmetic each backend may
pick). All are SED-style regex transforms with shared comment/string
awareness (`_scan_line`), not a Fortran parser, so each is deliberately
narrow:

- **`merge_used_modules`** -- inlines every externally-`USE`-d module's
  source so flang sees one self-contained TU. Pass-through for
  self-contained input; only genuine multi-file projects activate it.
- **`strip_openmp_directives`** (unconditional) -- drops `!$OMP` / `!$ACC`
  / `!$` sentinels, the ICON `#include "omp_definitions.inc"`, and
  `#ifdef _OPENMP` / `_OPENACC` blocks, so accelerator-annotated legacy
  code is consumable without `-fopenmp`.
- **`rewrite_integer_powers`** (unconditional) -- expands an
  integer-valued REAL-literal power (`x**2.0` -> `(x*x)`): algebraically
  exact and removes a backend-dependent `pow(x, 2.0)` vs `x*x` rounding
  difference against the gfortran reference. Bare-integer `x**2` is left
  to flang; genuine fractional `**0.5` stays `pow()`; a base containing a
  call is left alone (duplicating it would invoke it twice).
- **`preprocess_fortran`** (opt-in, `preprocess=True`) -- rewrites
  `IF (intvar)` -> `IF (intvar /= 0)` for INTEGER scalars, which
  flang-new-21 rejects (only LOGICAL is a legal IF condition). Off by
  default so clean source is untouched.

### HLFIR normalisation passes -- steps (2)+(3)

The `DEFAULT_PIPELINE` (in `builder/__init__.py`), in order. Step (2)'s
`hlfir-inline-all` is one entry in this same table; it is called out in
the top-level flow only because inlining is the conceptual hinge between
raw HLFIR and normalised single-TU IR.

| Pass | Purpose |
| --- | --- |
| `lower-fir-select-case` | `fir.select_case` -> `cf.cond_br` **before** inlining (the inliner's block-operand remap segfaults on a callee containing select-case). |
| `lift-cf-to-scf` | Structurise callees first: fold early `RETURN` / in-callee CFG into single-block `scf.if` so `hlfir-inline-all` cannot corrupt a structured region at the call site. |
| `hlfir-inline-all` | Splice every callee body into the pinned entry. |
| `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elemental / scalar-arg procedures. |
| `hlfir-expand-vector-subscript-gather` | `hlfir.associate` of an `hlfir.elemental` (Flang's noncontiguous-slice gather temp) -> explicit `fir.alloca` + gather `do` loop. |
| `hlfir-expand-vector-subscript-scatter` | `hlfir.region_assign` with an `hlfir.elemental_addr` destination (`d(cols) = source`) -> explicit scatter `do` loop. |
| `symbol-dce` | Drop private callee bodies once inlined. |
| `fir-polymorphic-op` | Statically devirtualise resolvable `fir.dispatch` / `fir.select_type`. |
| `hlfir-reject-polymorphism` | Loud-fail on any surviving virtual dispatch / `SELECT TYPE` / `fir.box_tdesc` -- the bridge supports CLASS-as-monomorphic-box only. |
| `hlfir-rewrite-sequence-association` | Collapse sequence-association adapters (a scalar array element passed where an explicit-shape array is expected) into a section designate of the parent. |
| `hlfir-lift-alloc-array-of-records` | Lift `type(t), allocatable :: f(:)` struct members (ICON's `p_patch%pprog(jg)`) into top-level companions with a leading runtime-extent dim, before flatten. |
| `hlfir-flatten-structs` | AoS -> SoA; emits the `hlfir.flatten_plan` attribute. Peels `fir.class<T>` so monomorphic CLASS flattens like TYPE. |
| `hlfir-rewrite-pointer-assigns` | Collapse `ptr => target` rebinds under a strict-no-aliasing assumption (each pointer read/write becomes an access to the target's storage). Warns per rewrite. |
| `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols. |
| `hlfir-lift-reduction-operands` | Lift an inline reduction operand (`out = max(x, MAXVAL(slice))`) into a preceding scalar-temp assign so `buildExpr` need not render a reduction inside a tasklet expression. |
| `hlfir-default-intent` | Intent-less dummies default to `intent_inout`. |
| `lift-cf-to-scf` | Raw-CFG loops (`DO WHILE`, `DO...EXIT`) -> `scf.while` + `scf.if`. |
| `sccp,canonicalize,cse` | Fold + simplify + dedupe after every rewrite exposed its constants. |

Multi-file builds additionally run `hlfir-verify-no-unresolved-calls`,
failing loudly on any `fir.call` that survives the inliner outside the
Flang-runtime / libm / C-stdlib allowlist. Inlining requires the
`fir` / `func` / `LLVM` `DialectInlinerInterface` extensions, attached
once by the bridge constructor.

### SDFG emission -- step (4)

`bridge/extract_vars.cpp` classifies every `hlfir.declare` as `array`,
`symbol`, or `scalar` (see *Mechanisms*). `bridge/extract_ast.cpp`
dispatches into per-responsibility translation units under `bridge/ast/`
(`expressions.cpp`, `assigns.cpp`, `elementals.cpp`, `control_flow.cpp`,
`dispatch.cpp`). The walker produces a recursive `ASTNode` tree covering
`loop` / `while` / `conditional` / `assign` / `copy` / `memset` /
`libcall` / `reduce` / `break` / `return`.

**Loop bounds and IF conditions are hoisted to symbols.** Every
non-trivial loop bound or branch guard is materialised as an SDFG symbol
on a state-change before the block (`loopbegin_<N>` / `loopend_<N>` /
`if_cond_<N>`), so the `LoopRegion` / `ConditionalBlock` references only
the symbol. This keeps the emitters small, funnels indirect-array reads
(`row_ptr[i+1] - 1`) through the symbol-staging machinery, and gives the
SSA loop-iter pass a uniform input shape.

`builder/SDFGBuilder` emits the SDFG, runs the post-generation cleanup
below, then snapshots `sdfg.arglist()` + free symbols into a
`FrozenSignature` pinned on the SDFG.

### Post-generation cleanup -- between (4) and (5)

Run over the freshly built SDFG, in order, **before** the
`FrozenSignature` snapshot (so the binding emitter sees the post-cleanup
signature):

1. **`SSALoopIterators`** -- renames each `LoopRegion.loop_variable` to a
   globally-unique `_it_<N>` and propagates the rename through the body;
   adds a reconstruction state re-asserting `<original_var> = <loop_end>`
   so downstream code reading the un-renamed name is correct. The bridge
   thus emits each loop with the source iter name (`jk`, `je`, ...) and
   lets this pass uniquify it -- no `iter_map` plumbing in the emitters.
2. **`replace_length_one_arrays_with_scalars`** (`transient_only=True`)
   -- rewrites local 1-element transients (loop accumulators) to true
   `Scalar`s, stripping leftover `[0]` subscripts. Signature scalars
   follow the I/O convention below; recurses into nested SDFGs.
3. **`IntegerizePowerExponents`** -- retypes integer-valued float `**`
   exponents (`base**2.0`) to `int` so codegen routes them through
   `dace::math::ipow` (repeated multiply, bit-identical to a Fortran
   reference) instead of libm `pow`. Genuine fractional exponents are
   left untouched.

**Loop-iterator validation.** SDFG validation rejects writing a
`LoopRegion.loop_variable` from an interstate-edge assignment inside its
own region; the `LoopRegion` owns the iterator update via `init_expr` /
`update_expr`.

## Design notes / mechanisms

These are the load-bearing modelling decisions.

**Symbol vs scalar classification.** A Fortran integer is a *symbol* iff
it is a `DO` induction variable, an array shape extent, a `DO` bound, an
`hlfir.designate` index, or feeds a control-flow condition -- everything
else integer is a *scalar*. Writes to symbols become interstate-edge
assignments; writes to scalars become tasklets. Only symbols may appear
as array indices.

**lbound handling.** Every descriptor carries `shape_symbols` +
`lower_bounds`; `access.build_memlet_index` folds the lbound offset once
at subset-build time and leaves DaCe's descriptor `offset` at zero, so
downstream transformations reason about one convention.

**Assumed-shape alias re-basing.** When `hlfir-inline-all` splices an
`arr(:)` callee into a caller whose actual has custom bounds
(`x(-2:2)`), flang emits a second aliasing `hlfir.declare`. The bridge
skips the alias in `extract_vars`, follows it in `traceToDecl`, and
re-bases each access index by `outer_lbound - inner_lbound`.

**ELEMENTAL inlining.** Flang lowers each elemental call to
`fir.do_loop { hlfir.designate per-arg; fir.call scalar_body }`. After
inlining, `hlfir-fold-element-aliases` folds the per-element alias
declares, so the SDFG builder sees a hand-written per-element loop.

**OPTIONAL dummies.** `fir.is_present %x` becomes an `i32` scalar
`<name>_present` on the signature; the existing if/else lowering reads it
like any condition. Intent-less optionals default to `intent_in`.

**AoS<->SoA flattening.** `hlfir-flatten-structs` hoists every struct
member as its own top-level dummy and stamps the `hlfir.flatten_plan`
recipe that step (5) consumes to restore the caller's AoS view (see the
deep-copy example above for the alias-vs-copy decision).

**Section reductions.** Whole-array `SUM` / `PRODUCT` / `ANY` / `ALL`
lower to DaCe's `standard.Reduce`. Section reductions
(`ANY(mask(lo:hi, jk))`) synthesise an init + a `kind="loop"` AST,
because DaCe's `Reduce` cannot express a dynamic-section input directly.

**Sibling-assign RAW hazards.** Multiple assigns in one `fir.do_loop`
body targeting the same non-transient storage would race in one state;
the emitter detects read-write name overlap across siblings and
serialises them into a chain of states. Non-overlapping siblings still
share a state.

**Signature freezing.** `codegen.generate_code` verifies
`sdfg._frozen_signature` before emitting the C++ header; drift raises
`SignatureDriftError`. Transformations mutate SDFGs freely -- a header
that disagrees with the emitted Fortran binding cannot ship.

### Allocatable buffer tracking (the unifying model)

`ALLOCATABLE` arrays under arbitrary `ALLOCATE` / `DEALLOCATE` /
conditional-allocate patterns are all handled by **one** abstraction:
buffer reaching-definitions. This is implemented in `extract_vars.cpp`
(`groupAllocSites`) and `ast/dispatch.cpp` (`bindAllocSite`); the
mixed-pattern cases below are live tests in
`tests/conditional_alloc_test.py`.

What the Fortran standard enforces (the semantics modelled):

- **One name, one current buffer.** `a` refers to at most one allocated
  buffer at a time; `ALLOCATE` of an already-allocated `a` is an error.
- **Allocation status persists across control flow.** `ALLOCATE` inside
  a taken `IF` branch leaves `a` allocated after the `IF`.
- **Referencing an unallocated allocatable is prohibited** (the program
  guarantees it is allocated wherever it reads). So the bridge never has
  to *prove* allocation -- it models "the current buffer at each point"
  and trusts the program is conforming. Where Fortran would leave `a`
  unallocated on some path, the bridge may harmlessly over-allocate.

The unifying rule: model each `ALLOCATE` site as a **buffer definition**
and each `DEALLOCATE` as a **kill**, then compute which sites *reach*
each point. Two sites belong to the **same DaCe transient** iff their
buffers can reach a common point as alternatives (the two arms of an
`IF` that both stay live to the join). Sites never simultaneously
reaching are **distinct transients** (sequential re-allocation: one dies
before the next is born). Formally: build a merge relation over sites
(`s ~ t` if both are in the reaching set at some join/use) and take the
union-find equivalence classes -- **each class is one transient**.

| Pattern | Reaching at the post-IF use | Transients |
|---|---|---|
| `IF/ELSE` both alloc | {s0, s1} | 1 (branch-dependent extent) |
| `IF/ELSEIF/ELSE` | {s0..s3} | 1 |
| single-branch | {s0} | 1 (concrete extent) |
| sequential `A;dealloc;A` | s0 dies before s1 | 2 |
| chain x4 | each dies before next | 4 |
| conditional + realloc | {s0,s1} at join; {s2} after | 2 |
| realloc-chain inside `IF` | {s1,s2} at join; s0 freed | 2 |

For a class whose sites differ in extent, the shape is a path-dependent
extent symbol `<buf>_d<i>` (a PHI of the per-site extents): each site
assigns the symbol on its own path; the assignments merge at the join.
Classes are named in first-definition order -- class 0 keeps the base
name `a`, the rest get `a_alloc1`, `a_alloc2`, ... -- and the bridge's
alias map routes reads/writes to the current class buffer as it walks.
At an `IF` join both branches set the same merged-class buffer, so
post-join reads route correctly with no extra join handling.

The algorithm is a structured recursive walk over `scf`/`fir.if`
regions (no general iterative dataflow needed): track the reaching set,
reset it at each `ALLOCATE`/`DEALLOCATE`, union it at uses with more than
one reaching site, and union the then/else reaching sets at each `IF`.

*Out of scope:* `MOVE_ALLOC` and allocatable-assignment auto-realloc
(`a = expr`) are a separate lowering; buffer-reuse/aliasing optimisation
(giving merged classes the same storage) is a DaCe-core transformation,
not a frontend concern.

## Data artefacts

The structured records that flow between steps -- the frontend's stable
contract surface. New features extend these; they do not invent parallel
channels.

| Artefact | Produced at | Consumed at | Role |
| --- | --- | --- | --- |
| `FortranInterface` | (1) snapshot | (5) emit | Caller-facing dummy list + derived-type layouts |
| `FlattenPlan` (MLIR attr) | (3) flatten-structs | (5) emit | Per-dummy AoS->SoA recipe (`flat_names`, `read_exprs`, `shape_exprs`, `aliasable`, ...) |
| `VarInfo[]` | (4) extract_vars | (4) SDFGBuilder | Classification + shape + intent per variable |
| `ASTNode` tree | (4) extract_ast | (4) SDFGBuilder | Normalised CFG + assigns + library-op references |
| `FrozenSignature` | (4) end of build | codegen, (5) emit | SDFG arg-list snapshot -- drift-checked at codegen |

## Components

```
dace_fortran/
|-- bridge/            C++ -- HLFIR parser + classifier + walker (nanobind)
|   |-- bridge.cpp           MLIRContext, pass pipeline, Python exports
|   |-- extract_vars.cpp     hlfir.declare -> VarInfo[]; groupAllocSites
|   |-- extract_ast.cpp      entry point; calls into ast/dispatch.cpp
|   |-- trace_utils.cpp      SSA tracing + alias helpers + depth limits
|   \-- ast/                 AST extraction split per responsibility
|       |-- expressions.cpp    buildExpr, buildIndexExpr, lowerIsPresent
|       |-- assigns.cpp        buildAssignNode, copy/memset/libcall, sections
|       |-- elementals.cpp     reductions, elemental walks, select-case chains
|       |-- control_flow.cpp   cmp predicates, buildBoolExpr, scf.while/if walkers
|       \-- dispatch.cpp       top-level walker; bindAllocSite
|-- passes/            C++ -- HLFIR -> HLFIR rewrites (see the pipeline table)
|-- builder/           Python -- SDFG emission (step 4)
|   |-- __init__.py          SDFGBuilder, DEFAULT_PIPELINE, _emit dispatch
|   |-- context.py           per-build state
|   |-- descriptors.py       add_descriptors, DTYPE mapping
|   |-- access.py            build_memlet_index, indirect-symbol lifting
|   |-- emit_tasklet.py      per-occurrence tasklet + scalar assign
|   |-- emit_cfg.py          assign / loop / while / conditional
|   \-- emit_library.py      copy / memset / libcall / reduce / break / return
|-- intrinsics/        Python -- Fortran intrinsic registry
|-- bindings/          Python -- Fortran wrapper emitter (step 5)
|   |-- frozen_signature.py  FrozenArg + FrozenSignature + drift check
|   |-- fortran_interface.py OriginalInterface (outer surface)
|   |-- flatten_plan.py      FlattenPlan + FlattenRecipe + to/from_dict
|   |-- block_builders.py    per-Fortran-section emitters
|   |-- loop_copy.py         alias vs deep-copy renderers
|   |-- emit_bindings.py     -> <entry>_bindings.f90
|   \-- build_fortran_library.py  emit + drift-verify + gfortran link
|-- build.py           public entry: build_sdfg / _from_files / _from_hlfir / _from_project
|-- emit_hlfir.py      tier-3 helper (compile_commands.json -> .hlfir)
|-- external.py        register_external / keep_external (ExternalCall libnode)
|-- preprocess.py      Fortran-text rewrites (step 0)
|-- build_bridge.py    one-time CMake build of the C++ bridge
|-- hlfir_to_sdfg.py   back-compat shim re-exporting from builder/
\-- integer_power_exponents.py  post-generation float-exponent retype
```

## Extending the frontend

| If you're adding... | Change here | Then cover in |
| --- | --- | --- |
| a new `math.*` intrinsic | `ast/expressions.cpp` `unary_math` / `binary_math` tables | `tests/elemwise_intrinsics_test.py` |
| a new reducer | `ast/dispatch.cpp::kRedTable` (+ `ast/assigns.cpp::buildSectionReduceAssign`) | `tests/reduce_intrinsics_test.py` |
| a new CFG op | `ast/dispatch.cpp::buildAST` dispatch + `builder/__init__.py::_EMIT_DISPATCH` + an `emit_cfg.py` emitter | ports from `baseline_*_test.py` |
| a new binding layout rule | `bindings/loop_copy.py` + a new `FlattenRecipe` field | `tests/bindings/emit_bindings_test.py` |
| a new HLFIR pass | a file in `passes/`, register in `Passes.cpp`, slot into `DEFAULT_PIPELINE` | `tests/<pass>_test.py` |

## Supported / not supported

[OK] supported, [!] planned (tracked in xfails), [X] never (out of scope).

### Types

| Feature | Status | Notes |
|---|---|---|
| `INTEGER(1/2/4/8)` | [OK] | -> `int8/16/32/64` |
| `REAL(4/8)` | [OK] | -> `float32/64` |
| `LOGICAL(1/4/8)` | [OK] | surfaced as `uint8/int32/int64` for the f2py ABI |
| `COMPLEX(4/8)` | [OK] | arrays only -- scalar by-value is a DaCe-core gap |
| `CHARACTER(*)` | [X] | string handling out of scope |
| Derived type, flat members | [OK] | `hlfir-flatten-structs` |
| Derived type, nested | [OK] | path-flattened name `base_m1_m2_leaf` |
| Array-of-struct with array members (`A(N)%w(M,M)`) | [OK] | `A(i)%w(j,k)` -> `A_w(i,j,k)` |
| Whole-member access on AoS (`A(i)%w = ...`) | [OK] | triplet section `A_w(i, 1:M:1, ...)` |
| Cross-subroutine struct args (incl. AoS) | [OK] | per-member block args + flatten recipe |
| Derived type, allocatable members | [OK] | flat allocatable companion + per-allocate-site rename |
| AoS + allocatable, uniform constant inner size | [OK] | static companion `A_w(N, M)`; alloc chain erased |
| AoS + allocatable as SDFG-boundary dummy | [OK] | padding-to-max with a runtime cap symbol |
| AoS + allocatable, kernel-internal first alloc (`intent(out)`) | [!] | needs an HLFIR shape-discovery pre-pass |
| Jagged AoS, two allocatable members of differing lengths | [!] | two-cap-symbol shape not yet exercised |
| Derived type with parametric array dim from a struct field | [!] | 1 xfail |
| Circular type definitions (recursion through a pointer chain) | [X] | out of scope |

### Control flow

| Feature | Status | Notes |
|---|---|---|
| `DO`, `DO WHILE`, `DO CONCURRENT` | [OK] | LoopRegion + scf.while |
| `IF` / `ELSE IF` / `ELSE` | [OK] | scf.if |
| `SELECT CASE` | [OK] | `lower-fir-select-case` |
| `EXIT`, `CYCLE` | [OK] | |
| Statement functions (`f(x) = ...`) | [!] | 1 xfail |
| `GOTO` | [X] | unstructured GOTO doesn't lift to scf |
| `SELECT TYPE` | [X] | requires runtime type discrimination |

### Allocatable

| Feature | Status | Notes |
|---|---|---|
| `ALLOCATE` / `DEALLOCATE` (local + dummy) | [OK] | buffer reaching-definitions model |
| Conditional `IF/ELSE` allocate (branch extent) | [OK] | merged class, `<buf>_d<i>` PHI symbol |
| Sequential realloc chain | [OK] | distinct transients per epoch |
| Conditional + realloc, realloc-chain inside `IF` | [OK] | mixed classes |
| `MOVE_ALLOC`, allocatable assignment `a = expr` | [X] | separate lowering |

### Subprograms / linkage

| Feature | Status | Notes |
|---|---|---|
| Module-contained `SUBROUTINE` / `FUNCTION` | [OK] | inlined by `hlfir-inline-all` |
| Internal subprograms (`CONTAINS`) | [OK] | |
| `INTERFACE` blocks, `USE`, `USE ... ONLY:` | [OK] | resolved at flang time |
| `OPTIONAL` dummy + `PRESENT` | [OK] | folded statically post-inline |
| `POINTER` (`ptr => target` rebind) | [OK] | collapsed under a strict-no-alias assumption (warns); true aliasing is unsupported |
| Separately-compiled `bind(c)` external | [OK] | `register_external` / `keep_external` |
| MPI send/recv (incl. non-default communicator) | [OK] | `dace.libraries.mpi`, recognised automatically |
| `EXTERNAL` statements | [X] | use modules instead |
| BLAS/LAPACK via `EXTERNAL` | [X] | use module-contained or DaCe libnodes |

### Slicing / array ops

| Feature | Status | Notes |
|---|---|---|
| Contiguous slice `a(i:j, k:l)` | [OK] | |
| Whole-array assign `a = b` | [OK] | `hlfir.elemental` + emit_library |
| Elementwise intrinsics on real / complex | [OK] | sin/cos/exp/sqrt/... |
| Reductions (sum/product/min/max/all/any/count/minval/maxval) | [OK] | |
| BLAS/LAPACK (matmul, transpose) | [OK] | dense -> libnode, strided -> explicit `do` loop |
| Noncontiguous gather `a(idx, :)` -- rank-1, constant extent | [OK] | gather-expand pass |
| Noncontiguous gather -- rank-2+ (`d(cols2, cols)`) | [!] | 5 xfails -- pass bails with a clear error |
| Noncontiguous slice -- symbolic extent | [X] | DaCe can't express runtime-sized symbol arrays |
| Noncontiguous scatter -- `intent(out)` write-back | [!] | not yet modelled |
| `ASSOCIATE` block | [!] | relative indexing only |

### Codegen targets

| Feature | Status | Notes |
|---|---|---|
| CPU C++ tasklets | [OK] | |
| GPU CUDA | [X] | would need OpenACC-style shim emission |
| Native `!$OMP` directives | [X] | DaCe handles parallelism itself |
| COARRAY | [X] | |

## Non-goals

- Re-parsing Fortran in Python. Flang is authoritative.
- GPU target bindings (would need OpenACC shim emission).
- Fortran SIMD / COARRAY semantics.
- Cross-kernel fusion across TU boundaries -- inline-all handles intra-TU
  fusion; cross-TU is the binding emitter's problem.

## Testing

Every supported construct has a seeded numerical test against
gfortran / f2py. Binding-specific tests live in `tests/bindings/`; the QE
loopnest kernels are in `tests/qe_loopnests/`; the ICON velocity-advection
loopnests in `tests/icon_loopnests/`. All executable-Fortran tests
compile with `gfortran` (Ubuntu's `flang-new-21` ships without
`libflang_rt`, so flang is emit-HLFIR-only).

```bash
# Main sweep -- excludes the multi-rank mpi-marked tests.
python3 -m pytest -n 4 -m "not mpi" tests/

# Multi-rank MPI tests; --oversubscribe lets the 4-rank tests run on
# laptops with fewer than 4 cores (the flag CI uses).
mpirun --oversubscribe -n 4 python3 -m pytest -m mpi -p no:cacheprovider tests/

# Dump built SDFGs for inspection.
__DACE_HLFIR_GEN_TEST_SDFGS=1        python3 -m pytest tests/
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/
```
