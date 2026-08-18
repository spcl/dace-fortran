# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct-replacement intrinsics (SIZE/LBOUND/UBOUND/BIT_SIZE/PRESENT/
ALLOCATED).  Phase 4 -- not yet implemented; empty set so callers can query
without special-casing the "not yet implemented" state.  Will mirror
``dace/frontend/fortran/intrinsics/direct_replacements.py`` (legacy frontend).
"""

DIRECT_INTRINSICS: set[str] = set()
