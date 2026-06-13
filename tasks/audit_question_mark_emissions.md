# Audit: ``?`` emission sites in the bridge

## Goal

Per user request: ``I dont think emitting a "?" can be ever correct,
maybe we should replace all of them with descriptive runtime errors
(e.g. corresponding not implemented?)``

Replace silent ``?`` fallbacks across the bridge with descriptive
errors that name the offending IR op / location.  Surfaces missing
cases at the earliest point with full context, instead of letting
``?`` propagate through expression strings to surface as opaque
downstream ``NameError`` / ``ValueError`` / ``KeyError`` after a
20-state walk.

## Audit (this session)

Total ``?`` emission sites in the bridge:

  * ``expressions.cpp``       --  4 sites
  * ``elementals.cpp``        -- 10 sites + ``push_back({"?", "?"})``
  * ``assigns.cpp``           --  6 sites
  * ``control_flow.cpp``      --  ~7 sites
  * ``extract_vars.cpp``      --  6 ``push_back("?")`` sites

## Categories

The ``?`` returns fall into two categories with different correct-fix
shapes:

### (A) "?" as expected sentinel for caller-checked fallback

Used where a higher-level handler tries multiple alternatives:

  * Recursion depth limit
    (``if (d > limits::kBuildExprDepth) return "?";``).
  * Null defining op
    (``if (!def) return "?";``).
  * Inner-elemental block walk fallback at
    ``materialiseElementalForLibcall`` / ``materialiseElementalToTransient``
    (``std::string body = "?";`` -- caller checks ``body == "?"`` and
    falls back to ``{{}, {}}``).
  * ``buildBoolExpr`` returning ``?`` to signal "this isn't a boolean
    expression" so caller falls back to ``buildExpr``.

These are legitimate sentinel uses.  Migration path: introduce a
``std::optional<std::string>`` or a small ``Result<std::string,
NotHandled>`` type so the success / "try-another-path" distinction
is explicit at the call site.

### (B) "?" as silent unhandled-op fallthrough

Surfaces in the END-OF-FUNCTION fallthrough of ``buildExpr``
(``expressions.cpp:1600``), ``buildIndexExpr``
(``assigns.cpp:469``), and similar.  This category is what the user
is concerned about -- the ``?`` lands in the tasklet body and
downstream errors are opaque.

This session: added unconditional stderr logging at
``expressions.cpp:1600`` so the op-name + location appear in
``DACE_FORTRAN_DEBUG_BUILDEXPR=0`` builds too (previous behaviour was
opt-in via env var).  Pinpoints the missing handler at the C++
source point without breaking the (A)-category sentinel protocol.

## Migration plan (next session)

1. **Introduce ``BuildExprResult``** -- a sum type expressing the
   three outcomes:  ``Ok(std::string)``,  ``Sentinel`` (caller
   should try an alternative), and  ``UnhandledOp(op_name, location)``
   (definitional bug; caller should propagate up to an exception).

2. **Migrate buildExpr's END fallthrough** to throw a C++ exception
   on ``UnhandledOp`` -- the nanobind binding surfaces it as a
   Python ``RuntimeError`` with the op-name in the message.
   ``materialiseElementalForLibcall`` and similar update their
   ``body == "?"`` check to handle the ``Sentinel`` case explicitly.

3. **Audit each ``?`` site** in the survey list above and re-classify
   into (A) or (B):
     * (A) sites get ``Sentinel`` returns.
     * (B) sites get explicit ``throw std::runtime_error`` (or
       ``throw NotImplementedError`` via a custom exception type).

4. **Probe tests**: for each (B) -> ``throw`` migration, write a
   minimal kernel that hits the unhandled op and ``pytest.raises``
   against the descriptive error message.

5. **Remove the stderr log** added this session -- the throws make
   it redundant.

## See also

- ``tasks/audit_no_question_mark_in_elementals.md`` -- the per-op
  audit catalogue of likely-NOT-yet-handled buildExpr gaps.
- ``project_qe_kernel_session_20260610.md`` -- session record where
  multiple ``?`` -> downstream errors surfaced over many turns.
