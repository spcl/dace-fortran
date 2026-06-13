"""Fortran-side distributed-data descriptors.

The Fortran caller boundary differs from DaCe's default MPI assumption
in one place that matters for codegen: DaCe's stock
:class:`dace.data.distributed.ProcessGrid` hardcodes ``MPI_COMM_WORLD``
as the parent communicator of every Cartesian-topology process grid it
builds at ``__dace_init`` time.  The Fortran caller, however, typically
hands the SDFG a sub-communicator created by ``MPI_Comm_split`` /
``MPI_Cart_sub`` / similar -- the integer handle for that
sub-communicator is what the user's program actually wants to use as
the parent.

This module exports :class:`FortranProcessGrid`, a subclass of
:class:`dace.data.distributed.ProcessGrid` whose ``init_code()`` reads
the parent communicator from an SDFG symbol (typically populated by the
bindings layer from an :func:`MPI_Comm_f2c` conversion of the Fortran
integer handle).  The ``exit_code()`` is inherited unchanged -- the
Cartesian sub-grid created here is destroyed the same way DaCe's
stock pgrid is.
"""
import dace.dtypes as dtypes
from dace.data.distributed import ProcessGrid
from dace.properties import Property, make_properties


@make_properties
class FortranProcessGrid(ProcessGrid):
    """Process grid whose parent communicator is supplied by the
    Fortran caller via an SDFG symbol of type ``opaque(MPI_Comm)``.

    Differs from the stock :class:`ProcessGrid` in exactly one place:
    the ``MPI_Cart_create`` call's first argument is the symbol named
    by :attr:`parent_comm_symbol` (read at ``__dace_init`` time from
    the C MPI_Comm produced by ``MPI_Comm_f2c`` in the bindings
    wrapper) instead of ``MPI_COMM_WORLD``.

    All other process-grid semantics are inherited.  Sub-grids
    (``is_subgrid=True``) ignore :attr:`parent_comm_symbol` -- the
    parent comm comes from the named ``parent_grid``, same as the
    stock class.

    :ivar parent_comm_symbol: Name of the SDFG symbol carrying the
        C ``MPI_Comm`` (an :class:`opaque <dace.dtypes.opaque>` of
        ``MPI_Comm``) that should parent this top-level grid.  Used
        only when ``is_subgrid`` is false.
    """

    parent_comm_symbol = Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="SDFG symbol name carrying the C MPI_Comm to use as the "
             "parent of MPI_Cart_create (top-level grid only).  When "
             "None, falls back to the stock ProcessGrid behaviour "
             "(MPI_COMM_WORLD).",
    )

    def __init__(self,
                 name,
                 shape,
                 parent_comm_symbol=None,
                 exact_grid=None,
                 root=0):
        super().__init__(name=name, is_subgrid=False, shape=shape,
                         parent_grid=None, color=None,
                         exact_grid=exact_grid, root=root)
        self.parent_comm_symbol = parent_comm_symbol

    def init_code(self) -> str:
        """Emit the MPI initialisation lines that build the process
        grid + group + rank/size/coords cache in ``__state``.  The C
        identifier for the parent communicator is
        :attr:`parent_comm_symbol` when set, else ``MPI_COMM_WORLD``
        (the stock behaviour)."""
        parent = self.parent_comm_symbol or "MPI_COMM_WORLD"
        ndim = len(self.shape)
        dims_assigns = "\n".join(
            f"__state->{self.name}_dims[{i}] = {s};" for i, s in enumerate(self.shape))
        return f"""
            {dims_assigns}
            int {self.name}_periods[{ndim}] = {{0}};
            MPI_Cart_create({parent}, {ndim}, __state->{self.name}_dims, {self.name}_periods, 0, &__state->{self.name});
            if (__state->{self.name} != MPI_COMM_NULL) {{
                MPI_Comm_group(__state->{self.name}, &__state->{self.name}_group);
                MPI_Comm_rank(__state->{self.name}, &__state->{self.name}_rank);
                MPI_Comm_size(__state->{self.name}, &__state->{self.name}_size);
                MPI_Cart_coords(__state->{self.name}, __state->{self.name}_rank, {ndim}, __state->{self.name}_coords);
                __state->{self.name}_valid = true;
            }} else {{
                __state->{self.name}_group = MPI_GROUP_NULL;
                __state->{self.name}_rank = MPI_PROC_NULL;
                __state->{self.name}_size = 0;
                __state->{self.name}_valid = false;
            }}
        """
