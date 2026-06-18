"""Frozen SDFG signature  --  snapshotted at build time, verified at codegen.

At the moment the kernel SDFG leaves ``SDFGBuilder.build()``, we
capture its argument list + free symbols into a ``FrozenSignature``
and pin it on the SDFG (``sdfg._frozen_signature = fs``).  The
binding emitter downstream uses this snapshot, not the live SDFG, so
transformations that mutate the SDFG can't silently invalidate a
generated ``.f90`` wrapper.

The drift gate lives in the dace-fortran ``build_fortran_library``
entrypoint: before the binding is emitted/linked it calls
``fs.verify_against(sdfg)``.  Any drift from the snapshot raises
``SignatureDriftError``.  dace-core ``compile`` / ``generate_code``
stay vanilla -- the contract is dace-fortran-only, not baked into
DaCe codegen.
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

    Fields:
        fortran_name: name as declared in the user's Fortran source.
        sdfg_name:    name DaCe sees (may differ after struct
                      flattening  --  e.g. ``st%u`` becomes ``st_u``).
        kind:         ``'array'`` | ``'scalar'`` | ``'symbol'`` |
                      ``'mpi_comm'`` (a Fortran integer communicator
                      the wrapper converts via ``MPI_Comm_f2c``).
        dtype:        ``'float64'`` | ``'int32'`` | ``'complex128'`` | ...
        rank:         tensor rank (0 for scalars).
        shape:        symbolic extents in Fortran symbols.  Empty tuple
                      for scalars / symbols.
        intent:       ``'in'`` | ``'out'`` | ``'inout'`` | ``''``.
        from_struct_member: when this arg was extracted from a struct
                      dummy by ``hlfir-flatten-structs``, the original
                      Fortran expression (``st%u``).  ``None`` otherwise.
        layout:       ``'same'`` (caller + callee share layout  --  alias
                      via ``c_loc``) | ``'complex_split'`` (Fortran
                      complex split into two reals) | ``'transpose'`` /
                      similar.  The binding emitter picks its copy
                      strategy off this tag.
        is_written:   true when this arg is a module-scope global the
                      kernel WRITES (host-shared inout state).  The
                      binding writes its final value back to the host
                      module variable on exit (copy-out), so the update
                      is visible to the caller -- not just copied in.
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
    # Marshalling provenance for a flattened component of a MODULE-LEVEL
    # array-of-structs global (QE ``us_exx`` ``TYPE(bec_type),ALLOCATABLE::
    # becxx(:)``, accessed ``becxx(ikq)%k``).  This arg is the SoA image
    # (``becxx_k``, shape [element-dims..., member-dims...]); the binding
    # sources it from the host struct with an AoS<->SoA copy loop
    # (``do i; becxx_k(i,:,:) = becxx(i)%k; end do``) instead of a direct
    # ``x = x__mod`` assign.  Empty for ordinary args.
    #   aos_origin_mod    -- module owning the global (``us_exx``)
    #   aos_origin_struct -- the AoS global's name (``becxx``)
    #   aos_member_path   -- ``%``-joined component path (``k`` / ``a%b``)
    #   aos_outer_rank    -- number of leading record-array (element) dims
    #   global_alloc_inside -- kernel ALLOCATEs the component: binding
    #                          allocates the host global before copy-out and
    #                          skips copy-in (host has no data yet).
    aos_origin_mod: str = ''
    aos_origin_struct: str = ''
    aos_member_path: str = ''
    aos_outer_rank: int = 0
    global_alloc_inside: bool = False
    aos_struct_pointer: bool = False
    aos_member_pointer: bool = False

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (``shape`` tuple becomes a list)."""
        d = asdict(self)
        # shape round-trips as a list in JSON; rebuild as tuple on load.
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

    ``args`` is ordered to match the generated C function
    ``__program_<entry>``'s parameter order (data args sorted, then
    scalars, then free symbols  --  the order DaCe's
    ``generate_headers`` emits).
    """

    entry: str  # 'compute_tendencies'
    mangled: str  # '_QPcompute_tendencies'
    args: Tuple[FrozenArg, ...]
    free_symbols: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1
    # Auto-detected Fortran module-global provenance for SDFG names
    # that are NOT outer dummies  --  free symbols (a scalar module
    # global lifted into a shape / bound) and module-global args
    # (the bridge ``intent=inout`` lift).  Maps ``sdfg_name ->
    # (module, entity)``.  The binding emitter merges this with any
    # hand-authored ``OriginalInterface.module_symbol_sources`` (the
    # explicit map wins on conflict) so no hand-authored list is
    # required for kernels the bridge can resolve on its own.
    module_symbol_origins: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Fortran ``integer`` communicator dummy whose value the bindings
    # wrapper feeds (via ``MPI_Comm_f2c`` + ``MPI_Comm_size``) into the
    # SDFG-side ``__user_comm`` / ``__user_comm_size`` symbols at
    # ``dace_init_<entry>`` time.  ``None`` when the kernel has no MPI
    # calls on a runtime user communicator.  Set by
    # :meth:`SDFGBuilder._attach_frozen_signature` from the
    # ``_fortran_user_comm_source`` sidecar that ``emit_mpi``
    # stashes on the SDFG.
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
        """Compare the live ``sdfg.arglist()`` + free-symbol set against
        this snapshot.  Raise ``SignatureDriftError`` on any divergence.

        Checks:
        - Same set of argument names.
        - Same order of argument names.
        - Same dtype per argument.
        - Same set of free symbols.

        We DON'T check dimensionality invariants past order/dtype since
        symbolic shapes may canonicalise; downstream codegen will catch
        concrete mismatches when it assembles memlets.
        """
        live_arglist = sdfg.arglist()
        live_fs = set(str(s) for s in sdfg.free_symbols)
        snap_fs = set(self.free_symbols)

        # The current ``SDFG.arglist()`` folds free symbols into the
        # argument list.  The frozen signature models the data/scalar
        # argument contract and the free-symbol set separately, so
        # validate them as two partitions: compare argument *names*
        # excluding free symbols here, and the free-symbol set on its
        # own below.
        live_names = [k for k in live_arglist if k not in live_fs]
        snap_names = [a.sdfg_name for a in self.args if a.sdfg_name not in snap_fs]
        if live_names != snap_names:
            raise SignatureDriftError(f"signature drift on {self.entry!r}: "
                                      f"expected args {snap_names}, got {live_names}")

        # dtype per data/scalar arg  --  guard against silent type
        # change.  Skip free-symbol args (validated by the set check)
        # and any snapshot arg the live arglist no longer carries.
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
        # ``opaque.to_string()`` is unimplemented in this dace (no
        # ``typename``); the ctype (``MPI_Comm`` / ...) is its identity.
        return t.ctype
    # dace.typeclass instances have a ``to_string``  --  fall back to repr.
    return getattr(t, 'to_string', lambda: str(t))()
