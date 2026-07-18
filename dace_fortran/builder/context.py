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
        # DO-loop iterator renames (Fortran name -> unique DaCe name); populated
        # by ``emit_loop`` so downstream emitters can substitute iterators in
        # conditions/RHS expressions.
        self.iter_map = {}
        # isend/irecv posts per request array (base name -> N); ``emit_mpi`` uses
        # this as the MPI_Waitall count when the Fortran count arg renders "?".
        self.mpi_req_posts = {}

    def ensure(self, region=None):
        """Make ``self.cur`` a writable ``SDFGState``: create the start
        state when empty, or wire a fresh successor past a non-state
        control-flow block.  ``region`` defaults to the SDFG."""
        # Explicit None check: SDFGState/LoopRegion define __len__ returning 0
        # when empty, so ``not self.cur`` would misfire on a fresh state.
        from dace.sdfg.state import SDFGState
        if self.cur is None:
            r = self.sdfg if region is None else region
            # First state in an empty region must set start_block, or DaCe's validator errors.
            is_start = (len(r.nodes()) == 0)
            self.cur = r.add_state(f"s_{self.builder.nid()}", is_start_block=is_start)
            return
        # Non-SDFGState control-flow block (ConditionalBlock, LoopRegion, ...) needs
        # a fresh successor state wired after it for further tasklets/memlets.
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
        """Flush pending assignments, then guarantee and return a writable
        current state.  Enforces flush-before-ensure ordering in one place."""
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
