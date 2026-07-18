# Adjacent Fortran patterns to ``AoS-of-pointer-records``

``LiftAosPointerRecords`` currently recognises one shape:

```fortran
TYPE t
  REAL(wp), POINTER :: m1(:), m2(:,:), ...
END TYPE
TYPE(t) :: arr(N)         ! all members POINTER, static N
arr(c1)%m => target_c1    ! constant outer index
... = arr(idx)%m(i, j)    ! constant OR runtime idx
```

F2018 §4.5 (Derived types), §5.3 (Type declarations), §7.5 (Pointers), §8.5 (Array constructors) admit several adjacent shapes sharing the same lowering challenge — catalogued here so the next iteration of this pass (or a sibling pass) can absorb them coherently.

## 1. AoS-of-allocatable-records (already handled elsewhere)

```fortran
TYPE t
  REAL(wp), ALLOCATABLE :: m(:)
END TYPE
TYPE(t) :: arr(N)
ALLOCATE (arr(c1)%m(K))
```

Covered by ``hlfir-flatten-structs`` Phase 5c (AoS + allocatable). Boundary clarity only — not in scope.

## 2. Pointer-to-AoS-of-records

```fortran
TYPE t
  REAL(wp) :: m(:,:)
END TYPE
TYPE(t), POINTER :: arr_ptr(:)
arr_ptr => static_target_array
... = arr_ptr(c)%m(i, j)
```

Outer indirection: the AoS itself is behind a pointer. ``hlfir-rewrite-pointer-assigns`` collapses the outer rebind to its target; the resulting whole-array pointer alias becomes a top-level array, handled by ``hlfir-flatten-structs`` Phase 2 (AoS with regular array members). No new pass needed.

## 3. AoS where members are themselves AoS-of-pointer-records

```fortran
TYPE inner
  REAL(wp), POINTER :: x(:,:)
END TYPE
TYPE outer
  TYPE(inner) :: a(M)
END TYPE
TYPE(outer) :: q(N)
q(c1)%a(c2)%x => target
```

Two levels of struct indirection plus the pointer alias. Same lift conceptually, but the concat array has rank 4 (N x M x inner_shape). ``LiftAosPointerRecords`` doesn't recurse today -- add when a workload needs it.

## 4. Polymorphic AoS

```fortran
TYPE, ABSTRACT :: t_base
END TYPE
TYPE, EXTENDS(t_base) :: t_real
  REAL(wp), POINTER :: x(:,:)
END TYPE
CLASS(t_base) :: arr(N)     ! polymorphic
SELECT TYPE (arr(c))
  TYPE IS (t_real); arr(c)%x => ...
END SELECT
```

Bridge already rejects ``CLASS`` via ``hlfir-reject-polymorphism``. Out of scope until DaCe can model type-discriminated dispatch.

## 5. Mixed-kind members (some POINTER, some flat / allocatable)

```fortran
TYPE t
  REAL(wp), POINTER :: m_ptr(:)
  REAL(wp)         :: m_flat(M)
END TYPE
TYPE(t) :: arr(N)
```

Not matched: ``matchCandidate`` requires every member to be box-of-pointer. Mixed records need a per-member strategy -- pointer members via ``LiftAosPointerRecords``, flat members via ``hlfir-flatten-structs`` Phase 2. Extendable to accept a non-empty pointer-typed subset (let flatten handle the rest), but the access-chain rewrite then has to coexist with flatten's rewrite -- not trivial.

## 6. Coarray + AoS

```fortran
TYPE(t), POINTER :: arr(:)[*]
arr(c)%m[image_id] => target
```

Coarrays + remote dereference: well outside DaCe's data-parallel model. Reject loud.

## 7. F2018 ``PARAMETERIZED`` derived types

```fortran
TYPE :: t(K)
  INTEGER, KIND :: K
  REAL(K), POINTER :: x(:)
END TYPE
```

KIND parameter contributes to type identity -- matcher sees ``t(8)``/``t(16)`` as different record types; a workload using both lands as two separate AoS candidates. Same recognition logic, no extra work.

## 8. Function results returning a TYPE with pointer members

```fortran
FUNCTION make() RESULT(r)
  TYPE(t) :: r
  r%x => some_array
END FUNCTION
```

Function-result temporary lives across the call boundary. Inliner expands ``r`` into caller scope; storing the result into an AoS slot is a rebind-to-rebind chain that collapses through normal trace. Caller-side handling unchanged.

## 9. PROCEDURE pointer members

```fortran
TYPE :: t
  PROCEDURE(iface_t), POINTER, NOPASS :: f
END TYPE
```

Function-pointer slots, not data-pointer slots. Bridge does not currently support function pointers; rejected by ``RejectPolymorphism`` / surrounding declare-walk. Out of scope.

## 10. Pointer remap (lower bound != 1)

```fortran
arr(c)%m(0:N-1) => target_array(1:N)
```

Pointer rebind carries a lower-bound remap. ``RewritePointerAssigns`` already bail-loud-guards this (``BOUNDS REMAP`` in its preflight). Lift pass should similarly refuse to materialise a concat array when any rebind sets a non-default lower bound on the pointer slot -- otherwise ``q(c)%m(i)`` would silently shift.
