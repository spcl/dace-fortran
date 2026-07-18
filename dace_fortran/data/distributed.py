"""Fortran-side distributed-data descriptors.

DaCe's stock ``ProcessGrid`` hardcodes ``MPI_COMM_WORLD`` as the parent
communicator; the Fortran caller typically hands the SDFG a sub-communicator
(``MPI_Comm_split``/``MPI_Cart_sub``) instead.  ``FortranProcessGrid`` reads
the parent communicator from an SDFG symbol (populated by the bindings layer
via ``MPI_Comm_f2c``).  ``exit_code()`` is inherited unchanged.
"""
import dace.dtypes as dtypes
from dace.data.distributed import ProcessGrid
from dace.properties import Property, make_properties


@make_properties
class FortranProcessGrid(ProcessGrid):
    """Process grid whose parent communicator is supplied by the Fortran
    caller via an SDFG symbol of type ``opaque(MPI_Comm)``, instead of the
    stock class's hardcoded ``MPI_COMM_WORLD``.  Sub-grids
    (``is_subgrid=True``) ignore :attr:`parent_comm_symbol` -- their parent
    comm comes from ``parent_grid``, same as the stock class."""

    parent_comm_symbol = Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="SDFG symbol name carrying the C MPI_Comm to use as the "
        "parent of MPI_Cart_create (top-level grid only).  When "
        "None, falls back to the stock ProcessGrid behaviour "
        "(MPI_COMM_WORLD).",
    )

    def __init__(self, name, shape, parent_comm_symbol=None, exact_grid=None, root=0):
        super().__init__(name=name,
                         is_subgrid=False,
                         shape=shape,
                         parent_grid=None,
                         color=None,
                         exact_grid=exact_grid,
                         root=root)
        self.parent_comm_symbol = parent_comm_symbol

    def init_code(self) -> str:
        """Emit the MPI init lines building the process grid + group +
        rank/size/coords cache in ``__state``; parent communicator is
        ``parent_comm_symbol`` when set, else ``MPI_COMM_WORLD``."""
        parent = self.parent_comm_symbol or "MPI_COMM_WORLD"
        ndim = len(self.shape)
        dims_assigns = "\n".join(f"__state->{self.name}_dims[{i}] = {s};" for i, s in enumerate(self.shape))
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
