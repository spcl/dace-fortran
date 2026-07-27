# Audit: No ``?`` placeholders inside elementals or pure functions

## Goal

Ensure ``buildExpr``/``buildIndexExpr`` never return ``?`` for an expression that ends up inside the body of an ``hlfir.elemental`` (or any pure-function context where every operand should be expressible as a closed-form tasklet RHS).

Surfaced this session by QE's ``rcut = 0.5 * MINVAL(SQRT(SUM(a ** 2, 1)))`` -- the SQRT elemental's body had ``math.sqrt %x`` where ``%x = hlfir.apply %sum_with_dim`` and ``buildExpr`` couldn't trace the apply (returned ``?``), so the tasklet body landed as ``_out__mask_0 = sqrt(?)`` and validation failed.

Fixed this session for the SUM/MINVAL/MAXVAL/PRODUCT dim-reduction case (commit on this branch). The audit catches NEXT-session latent issues sharing the same shape but a different source op.

Already-confirmed sources of ``?`` in elemental contexts (this session):
- ``hlfir.apply`` of an ``hlfir.sum``/``hlfir.minval``/``hlfir.maxval``/``hlfir.product`` with ``dim`` operand -- **FIXED** (``materialiseElementalToTransient`` pre-walks for dim-reductions).
- ``hlfir.apply`` of an ``hlfir.matmul``/``hlfir.transpose``/``hlfir.matmul_transpose``/``hlfir.dot_product``/``hlfir.cshift``/``hlfir.count``/``hlfir.minloc``/``hlfir.maxloc`` -- **FIXED** (``findApplies`` in ``control_flow.cpp:289+``).

## Coverage re-verification (2026-07-27, [b742bd6b dface-fortran])

Enumerated every dispatch site in ``buildExpr`` (``expressions.cpp``, 65 ``dyn_cast`` + name-map
``bin_ops``@881 / ``minmax_ops``@1001). The "likely-not-handled" list below is PARTLY STALE:

- HANDLED now (dyn_cast present): ``hlfir.elemental``, ``hlfir.as_expr``, ``hlfir.concat``,
  ``hlfir.apply``, ``hlfir.all``/``any``, ``arith.select``, ``scf.if``, ``fir.do_loop``,
  ``fir.embox``/``rebox``/``box_addr``/``unboxchar``/``emboxchar``, ``fir.extract_value``/``insert_value``,
  ``fir.is_present``, ``fir.addrof``, ``fir.call``. arith/math fully mapped (add/sub/mul/div/rem/
  shift/cmp/min/max/neg + sqrt/exp/trig/erf/floor/ceil/copysign/...).
- GENUINELY ABSENT from the expression path (candidates for the h_psi residual):
  ``fir.iterate_while``, ``scf.index_switch``, ``fir.allocmem``.
- ⚠ These three are CONTROL-FLOW-ish and only surface in the deeply-inlined h_psi body:
  isolated tiny probes (SELECT CASE→scalar, allocatable temp, do-while) all build CLEAN
  (``scratchpad/probe_buildexpr.py``) — they do NOT reproduce a bare ``?``. So confirming which
  op actually blocks h_psi REQUIRES the real ground-truth run (paused: box saturated by another
  session's continuous ``tests/transformations`` sweeps). Queued command:
  ``PY13 + PYTHONPATH=/home/primrose/Work/d-face  pytest tests/qe/h_psi/test_h_psi_parse.py::test_h_psi_parses --runxfail -x``
  → capture stderr ``[buildExpr unhandled-op] op=…`` + the downstream ``?``-propagation traceback.

Likely-NOT-yet-handled (audit these next session):
- ``hlfir.apply`` of an ``hlfir.assoc``/``hlfir.as_expr`` (hlfir conversion ops between value/box/expr).
- ``hlfir.apply`` of an ``hlfir.parent_comp`` (parent type-extension component access).
- ``hlfir.char_length``/``hlfir.set_length``/``hlfir.concat`` inside an elemental body (string ops).
- ``fir.do_loop``/``scf.if`` results consumed inside an elemental body (Flang emits these for some inline conditional expressions).
- ``hlfir.elemental`` inside an elemental body where the inner elemental's body itself yields a ``?`` chain.
- ``fir.alloca``-backed scratch scalars whose name doesn't resolve through ``allocaSynthName`` because the scratch is written inside a loop (``__al_<n>`` naming path).
- ``fir.box_addr`` of an ``fir.box`` that aliases a non-named storage.
- ``fir.coordinate_of`` on a struct with no surrounding designate.
- ``arith.select`` whose true/false values are themselves complex expressions ``buildExpr`` can't resolve.
- ``math.atan2``/other 2-arg math ops with non-trivial operands.

## Defensive checks to add

1. **Compile-time assert in ``materialiseElementalToTransient``**: after ``buildExpr`` returns, scan for a bare ``?`` token in the body string and raise a NotImplementedError with the source op name + IR location. Today the raise is in ``emit_tasklet.py:198+`` -- too late to give the surfacing op name in C++ form; raising at materialisation time gives us ``op->getName()`` directly.
2. **Compile-time assert in ``buildElementalAssign``**: same as above for the inline assign path (not just the reduction-over-elemental Mode-C path).
3. **Audit walk in ``buildExpr``**: when the op-name doesn't match any known handler AND the surrounding context is an ``hlfir.elemental`` (or ``func.func`` whose body has no side effects), emit a one-line diagnostic on stderr listing the op name. Gated by ``DACE_FORTRAN_DEBUG_BUILDEXPR=1`` so production builds stay silent.
4. **Surfacing tests**: probe each of the "likely-NOT-yet-handled" list above with a minimal Fortran kernel that triggers it; xfail each one with a clear reason naming the gap. When the gap closes, the xfail auto-flips to PASS and the regression is locked in.

## Triage order (next session)

1. Surface each gap with a minimal Fortran probe (xfailed).
2. Fix the dispatch-routing gap (the SUM-of-LOG-of-SUM-dim shape -- see ``tests/sqrt_of_dim_reduction_test.py::test_sum_of_log_of_sum_dim`` xfail). Mode-C dispatch in ``dispatch.cpp:2353+`` should also pre-walk the source elemental's body for dim-reductions.
3. Walk the QE source past the next ``?`` placeholder (the SQRT fix closes the ``vcut_spheric_get`` one; the next surfacing op is likely the ``vcut_get`` ``DBLE(l * l - m * m)`` chain in the spherical harmonics).

## See also

- ``project_qe_kernel_status.md`` for the cumulative QE progress chain.
- ``tasks/qe_e2e_guide.md`` for the end-to-end QE lifecycle.
- ``tests/sqrt_of_dim_reduction_test.py`` -- the regression probes closing the dim-reduction shape this session.
