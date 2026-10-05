// ============================================================================
// ragged_aos.h  --  runtime-sized ALLOCATABLE members of an array of records.
// ============================================================================
// ``type(t) :: a(N)`` with ``real, allocatable :: w(:, ...)`` whose elements
// are allocated in the kernel with RUNTIME extents
// (``ALLOCATE(a(i) % w(cnt + 1, m))``).  Flattened ELLPACK-style:
//
//   * ``<a>_<w>(outer..., cap_1, ..., cap_r)``  --  one padded companion, sized
//     once before the first ALLOCATE by re-evaluating every site's extents over
//     its enclosing loops (``cap_d`` = max over all executions);
//   * ``<a>_<w>_len(outer..., r)``  --  each element's live extents, written at
//     its ALLOCATE (extent + 1, ``0`` while unallocated); whole-member reads, ``SIZE`` and
//     ``ALLOCATED`` read it.
//
// An extent is boundable when it is side-effect-free arithmetic over values
// available before the loops (loop indices included), or a counter of the
// ``x = 0; DO i = lo, hi; [IF (c)] x = x + 1; END DO`` idiom right before the
// ALLOCATE, which is bounded by the counting loop's trip count.  Anything else
// is left alone (the bridge then rejects the ALLOCATE).
// ============================================================================

#pragma once

#include <string>

#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/StringRef.h"

namespace hlfir_bridge {

/// Whether every ``ALLOCATE(decl(i...) % memName(...))`` site can be flattened by
/// :func:`flattenRaggedAosMember`.
bool raggedAosMemberFlattenable(hlfir::DeclareOp decl, llvm::StringRef memName);

/// Flattens ``decl``'s allocatable member ``memName`` into the padded companion
/// ``flatName`` and the extent table ``flatName + "_len"``; rewrites every access.
/// Requires :func:`raggedAosMemberFlattenable`.
void flattenRaggedAosMember(hlfir::DeclareOp decl, llvm::StringRef memName, const std::string& flatName);

/// Once every member of the array of records ``decl`` is flattened, drops its allocation runtime calls and the
/// ``ALLOCATED`` guards around them; does nothing while any element is still accessed.
void dropUnusedRecordArray(hlfir::DeclareOp decl);

}  // namespace hlfir_bridge
