"""``_Ctx``  --  per-region emission context.

Tracks the "current" SDFG state, pending scalar assignments that need
flushing as tasklets, and the active DO-loop iterator renames.
"""

from dace import InterstateEdge, SDFG


class _Ctx:
    """Tracks the current state and pending scalar assignments."""

    def __init__(self, sdfg: SDFG, builder):
        self.sdfg = sdfg
        self.builder = builder
        self.cur = None
        self.pending = []
        # Active DO-loop iterator renames (Fortran name -> unique DaCe name).
        # Populated by ``emit_loop`` for the duration of each loop body so
        # downstream emitters (``emit_cond`` / ``emit_tasklet``) can
        # substitute iterators referenced in conditions or RHS expressions.
        self.iter_map = {}

    def ensure(self, region=None):
        """Make ``self.cur`` a writable ``SDFGState``: create the start
        state when empty, or wire a fresh successor past a non-state
        control-flow block.  ``region`` defaults to the SDFG."""
        # ``not self.cur`` misfires the same way ``region or self.sdfg`` did:
        # SDFGState / LoopRegion define __len__ that returns 0 when empty,
        # so a freshly-created state is treated as falsy even though we
        # want to keep emitting into it.  Use explicit None checks.
        from dace.sdfg.state import SDFGState
        if self.cur is None:
            r = self.sdfg if region is None else region
            # First state added to an otherwise-empty control-flow region
            # must be marked as the starting block, otherwise DaCe's
            # validator raises "Ambiguous or undefined starting block".
            is_start = (len(r.nodes()) == 0)
            self.cur = r.add_state(f"s_{self.builder.nid()}", is_start_block=is_start)
            return
        # After a ConditionalBlock (or any non-SDFGState control-flow block
        # like a LoopRegion), the next emitter needs a fresh successor state
        # wired from that block so tasklets / memlets have somewhere to land.
        if not isinstance(self.cur, SDFGState):
            r = self.sdfg if region is None else region
            succ = r.add_state(f"s_{self.builder.nid()}")
            r.add_edge(self.cur, succ, InterstateEdge())
            self.cur = succ

    def flush(self, builder, region=None):
        """Emit any pending scalar assignments into the current state."""
        if not self.pending:
            return
        r = self.sdfg if region is None else region
        self.ensure(r)
        for target, value in self.pending:
            builder.emit_scalar_assign(self.cur, target, value)
        self.pending.clear()

    def flush_and_ensure(self, builder, region=None):
        """Flush pending scalar assignments, then guarantee a writable
        current state and return it.  Callers never ``ensure`` without
        first ``flush``-ing; this enforces that order in one place.

        :returns: ``self.cur`` (the now-writable current state).
        """
        self.flush(builder, region)
        self.ensure(region)
        return self.cur

    def new_state(self, builder, region=None, label=None):
        """Flush pending assignments, then open a fresh successor state."""
        self.flush(builder, region)
        r = self.sdfg if region is None else region
        s = r.add_state(label or f"s_{self.builder.nid()}")
        if self.cur is not None:
            r.add_edge(self.cur, s, InterstateEdge())
        self.cur = s
        return s
