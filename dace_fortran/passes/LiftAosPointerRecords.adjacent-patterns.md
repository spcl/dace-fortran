# Adjacent Fortran patterns to ``AoS-of-pointer-records``

The ``LiftAosPointerRecords`` pass currently recognises one shape:

```fortran
TYPE t
  REAL(wp), POINTER :: m1(:), m2(:,:), ...
END TYPE
TYPE(t) :: arr(N)         ! all members POINTER, static N
arr(c1)%m => target_c1    ! constant outer index
... = arr(idx)%m(i, j)    ! constant OR runtime idx
```

Per F2018 sections 4.5 (Derived types), 5.3 (Type declarations),
7.5 (Pointers), and 8.5 (Array constructors), the language admits
several adjacent shapes that share most of the same lowering
challenge.  Documenting them here so the next iteration of this
pass (or a sibling pass) can absorb them coherently.

## 1. AoS-of-allocatable-records (already handled elsewhere)

```fortran
TYPE t
  REAL(wp), ALLOCATABLE :: m(:)
END TYPE
TYPE(t) :: arr(N)
ALLOCATE (arr(c1)%m(K))
```

Covered by ``hlfir-flatten-structs`` Phase 5c (AoS + allocatable).
Listed here for boundary clarity; not in our scope.

## 2. Pointer-to-AoS-of-records

```fortran
TYPE t
  REAL(wp) :: m(:,:)
END TYPE
TYPE(t), POINTER :: arr_ptr(:)
arr_ptr => static_target_array
... = arr_ptr(c)%m(i, j)
```

Outer indirection level: the AoS itself is behind a pointer.
``hlfir-rewrite-pointer-assigns`` already collapses the outer
rebind to its target; the resulting whole-array pointer alias
becomes a top-level array, which ``hlfir-flatten-structs`` then
handles via Phase 2 (AoS with regular array members).  No new
pass needed.

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

Two levels of struct indirection plus the pointer alias.  Conceptually
the same lift but the concat array has rank 4 (N x M x inner_shape).
``LiftAosPointerRecords`` doesn't recurse today -- add when a
workload needs it.

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

The bridge already rejects ``CLASS`` via
``hlfir-reject-polymorphism``.  Out of scope until DaCe can model
type-discriminated dispatch.

## 5. Mixed-kind members (some POINTER, some flat / allocatable)

```fortran
TYPE t
  REAL(wp), POINTER :: m_ptr(:)
  REAL(wp)         :: m_flat(M)
END TYPE
TYPE(t) :: arr(N)
```

Not currently matched: ``matchCandidate`` requires every member to
be a box-of-pointer.  Mixed records would need a per-member
strategy -- pointer members go through ``LiftAosPointerRecords``,
flat members continue to ``hlfir-flatten-structs`` Phase 2.  The
pass can be extended to accept candidates whose pointer-typed
members form a non-empty subset of the record (let flatten handle
the rest), but the access-chain rewrite then has to coexist with
flatten's rewrite -- not trivial.

## 6. Coarray + AoS

```fortran
TYPE(t), POINTER :: arr(:)[*]
arr(c)%m[image_id] => target
```

Coarrays + remote dereference: well outside DaCe's data-parallel
model.  Reject loud.

## 7. F2018 ``PARAMETERIZED`` derived types

```fortran
TYPE :: t(K)
  INTEGER, KIND :: K
  REAL(K), POINTER :: x(:)
END TYPE
```

The KIND parameter contributes to the type identity; the matcher
sees ``t(8)`` and ``t(16)`` as different record types.  A workload
that uses both would currently land as two separate AoS candidates.
Same recognition logic, no extra work.

## 8. Function results returning a TYPE with pointer members

```fortran
FUNCTION make() RESULT(r)
  TYPE(t) :: r
  r%x => some_array
END FUNCTION
```

The function-result temporary lives across the call boundary.  The
inliner expands ``r`` into the caller scope; if the result is then
stored into an AoS slot, that's just a rebind-to-rebind chain that
collapses through normal trace.  Caller-side handling is the same.

## 9. PROCEDURE pointer members

```fortran
TYPE :: t
  PROCEDURE(iface_t), POINTER, NOPASS :: f
END TYPE
```

Function-pointer slots, not data-pointer slots.  The bridge does
not currently support function pointers; rejected by
``RejectPolymorphism`` / surrounding declare-walk.  Out of scope.

## 10. Pointer remap (lower bound != 1)

```fortran
arr(c)%m(0:N-1) => target_array(1:N)
```

The pointer rebind carries a lower-bound remap.  ``RewritePointerAssigns``
already documents this case as a bail-loud guard
(``BOUNDS REMAP`` in its preflight).  The lift pass should similarly
refuse to materialise a concat array when any rebind sets a
non-default lower bound on the pointer slot -- otherwise the access
``q(c)%m(i)`` would silently shift.
