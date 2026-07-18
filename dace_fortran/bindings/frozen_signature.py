"""Frozen SDFG signature -- snapshotted at build time, verified at codegen.

Captured when the kernel SDFG leaves ``SDFGBuilder.build()`` and pinned
on the SDFG (``sdfg._frozen_signature``).  The binding emitter uses this
snapshot, not the live SDFG, so later transformations can't silently
invalidate a generated wrapper.

Drift gate lives in ``build_fortran_library``: before emit/link it calls
``fs.verify_against(sdfg)``, raising ``SignatureDriftError`` on
divergence.  dace-core stays vanilla -- the contract is dace-fortran-only.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


class SignatureDriftError(RuntimeError):
    """Raised when the live SDFG's arglist / free_symbols disagrees with
    a ``FrozenSignature`` attached to it."""


@dataclass(frozen=True)
class FrozenArg:
    """One argument in the frozen signature.

    sdfg_name: name DaCe sees, may differ from fortran_name after struct
        flattening (``st%u`` -> ``st_u``).
    kind: 'array'|'scalar'|'symbol'|'mpi_comm' (integer communicator,
        wrapper converts via MPI_Comm_f2c).
    from_struct_member: original Fortran expr (``st%u``) if extracted by
        hlfir-flatten-structs, else None.
    layout: 'same' (alias via c_loc) | 'complex_split' | 'transpose' --
        binding emitter picks its copy strategy off this tag.
    is_written: True if this is a module-scope global the kernel WRITES;
        binding copies the final value back to the host module var.
    """

    fortran_name: str
    sdfg_name: str
    kind: str
    dtype: str
    rank: int
    shape: Tuple[str, ...] = field(default_factory=tuple)
    intent: str = ''
    from_struct_member: Optional[str] = None
    layout: str = 'same'
    is_written: bool = False
    # Provenance for a flattened component of a MODULE-LEVEL array-of-structs
    # global (QE ``becxx(ikq)%k``, TYPE(bec_type) ALLOCATABLE).  This arg is
    # the SoA image (``becxx_k``); binding sources it via an AoS<->SoA copy
    # loop instead of a direct assign.  Empty for ordinary args.
    #   aos_origin_mod/struct -- owning module / AoS global name.
    #   aos_member_path       -- '%'-joined component path.
    #   aos_outer_rank        -- leading record-array (element) dim count.
    #   global_alloc_inside   -- kernel ALLOCATEs the component: binding
    #                            allocates the host global, skips copy-in.
    aos_origin_mod: str = ''
    aos_origin_struct: str = ''
    aos_member_path: str = ''
    aos_outer_rank: int = 0
    global_alloc_inside: bool = False
    aos_struct_pointer: bool = False
    aos_member_pointer: bool = False
    # Module-origin global storage class: copy-in guarded with allocated()/
    # associated() so an unallocated host isn't read; both false = static.
    module_origin_allocatable: bool = False
    module_origin_pointer: bool = False

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (``shape`` tuple becomes a list)."""
        d = asdict(self)
        d['shape'] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenArg":
        """Rebuild from a :meth:`to_dict` mapping (list back to tuple)."""
        d = dict(d)
        d['shape'] = tuple(d.get('shape', []))
        return cls(**d)


@dataclass(frozen=True)
class FrozenSignature:
    """Full snapshot of one entry subroutine's SDFG signature.

    ``args`` order matches ``__program_<entry>``'s C params: data args
    sorted, then scalars, then free symbols (DaCe's generate_headers order).
    """

    entry: str  # 'compute_tendencies'
    mangled: str  # '_QPcompute_tendencies'
    args: Tuple[FrozenArg, ...]
    free_symbols: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1
    # Auto-detected module-global provenance for SDFG names that aren't
    # outer dummies.  Maps sdfg_name -> (module, entity).  Binding emitter
    # merges with hand-authored OriginalInterface.module_symbol_sources
    # (explicit map wins on conflict).
    module_symbol_origins: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Integer communicator dummy the wrapper feeds (via MPI_Comm_f2c +
    # MPI_Comm_size) into __user_comm/__user_comm_size at dace_init_<entry>
    # time.  None if no runtime MPI comm.  Set from emit_mpi's
    # _fortran_user_comm_source sidecar.
    user_comm_source: Optional[str] = None

    # ----- I/O ---------------------------------------------------------

    def to_json(self, path: str):
        """Write the snapshot to ``path`` as indented JSON."""
        with open(path, 'w') as fh:
            json.dump(
                {
                    'entry': self.entry,
                    'mangled': self.mangled,
                    'args': [a.to_dict() for a in self.args],
                    'free_symbols': list(self.free_symbols),
                    'schema_version': self.schema_version,
                    'module_symbol_origins': {
                        k: list(v)
                        for k, v in self.module_symbol_origins.items()
                    },
                    'user_comm_source': self.user_comm_source,
                },
                fh,
                indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FrozenSignature":
        """Load a snapshot previously written by :meth:`to_json`."""
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            entry=d['entry'],
            mangled=d['mangled'],
            args=tuple(FrozenArg.from_dict(a) for a in d['args']),
            free_symbols=tuple(d.get('free_symbols', [])),
            schema_version=d.get('schema_version', 1),
            module_symbol_origins={
                k: tuple(v)
                for k, v in d.get('module_symbol_origins', {}).items()
            },
            user_comm_source=d.get('user_comm_source'),
        )

    # ----- Drift check -------------------------------------------------

    def verify_against(self, sdfg):
        """Compare live ``sdfg.arglist()`` + free symbols against this
        snapshot; raise ``SignatureDriftError`` on divergence (arg name
        set/order, dtype per arg, free-symbol set).

        Doesn't check dimensionality past order/dtype -- symbolic shapes
        may canonicalise; codegen catches concrete mismatches later.
        """
        live_arglist = sdfg.arglist()
        live_fs = set(str(s) for s in sdfg.free_symbols)
        snap_fs = set(self.free_symbols)

        # arglist() folds free symbols into the arg list; the snapshot
        # models them separately, so validate as two partitions here.
        live_names = [k for k in live_arglist if k not in live_fs]
        snap_names = [a.sdfg_name for a in self.args if a.sdfg_name not in snap_fs]
        if live_names != snap_names:
            raise SignatureDriftError(f"signature drift on {self.entry!r}: "
                                      f"expected args {snap_names}, got {live_names}")

        # dtype per data/scalar arg -- skip free-symbol args (checked
        # below) and any snapshot arg the live arglist no longer carries.
        for a in self.args:
            if a.sdfg_name in snap_fs or a.sdfg_name not in live_arglist:
                continue
            live_dtype = _dtype_string(live_arglist[a.sdfg_name])
            if live_dtype != a.dtype:
                raise SignatureDriftError(f"signature drift on {self.entry!r}: arg {a.sdfg_name!r} "
                                          f"dtype {a.dtype!r} in snapshot but {live_dtype!r} now")

        if live_fs != snap_fs:
            raise SignatureDriftError(f"signature drift on {self.entry!r}: "
                                      f"expected free symbols {sorted(snap_fs)}, got {sorted(live_fs)}")


def _dtype_string(desc) -> str:
    """Stringify a DaCe data descriptor's dtype for comparison."""
    import dace

    t = getattr(desc, 'dtype', None)
    if t is None:
        return '?'
    if isinstance(t, dace.dtypes.opaque):
        # opaque.to_string() is unimplemented here; ctype is its identity.
        return t.ctype
    # typeclass instances have to_string; fall back to repr otherwise.
    return getattr(t, 'to_string', lambda: str(t))()
