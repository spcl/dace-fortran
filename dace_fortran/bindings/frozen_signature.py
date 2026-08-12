# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen SDFG signature -- snapshotted at build time, verified at codegen.

Captured when the kernel SDFG leaves ``SDFGBuilder.build()`` and pinned
on the SDFG (``sdfg._frozen_signature``).  The binding emitter uses this
snapshot, not the live SDFG, so later transformations can't silently
invalidate a generated wrapper.

Drift gate lives in ``build_fortran_library``: before emit/link it calls
``fs.verify_against(sdfg)``, raising ``SignatureDriftError`` on
divergence.  dace-core contributes only the opaque ``SDFG.frontend_metadata``
dict this rides in and never reads it -- the contract is dace-fortran-only.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional, Tuple

# Where a Fortran caller's buffers live unless a pass says otherwise.  Anything a
# transformation moves into a ``GPU_*`` storage is a device relocation the binding has
# to bridge; everything else is still reachable through the pointer the caller passed.
HOST_STORAGE = 'CPU_Heap'
DEVICE_STORAGE_PREFIX = 'GPU_'


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
    # Where the buffer lives.  ``storage`` is the CALLER's location, frozen at build
    # time and never rewritten -- a Fortran dummy is host memory, which is why the
    # default is a host storage rather than empty.  Offloading relocates the kernel's
    # copy, and :func:`refreeze` records that in ``device_storage``; the binding then
    # brackets the call in OpenACC data clauses instead of handing the host pointer
    # straight through.  Back on the host, ``device_storage`` clears again.
    storage: str = HOST_STORAGE
    device_storage: str = ''

    @property
    def acc_data_clause(self) -> str:
        """``copyin`` / ``copyout`` / ``copy`` for this arg, '' if it never leaves the host.

        Direction is the Fortran intent: an ``in`` dummy the kernel doesn't write needs
        only the host->device leg, an ``out`` dummy only the device->host one, anything
        else both.
        """
        if self.kind != 'array' or not self.device_storage or self.device_storage == self.storage:
            return ''
        intent = self.intent.lower()
        if intent == 'in' and not self.is_written:
            return 'copyin'
        if intent == 'out':
            return 'copyout'
        return 'copy'

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

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (every tuple becomes a list).

        Also the on-SDFG storage form -- see :func:`attach_to_sdfg`.
        """
        return {
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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenSignature":
        """Rebuild from a :meth:`to_dict` mapping (lists back to tuples)."""
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

    def to_json(self, path: str):
        """Write the snapshot to ``path`` as indented JSON."""
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FrozenSignature":
        """Load a snapshot previously written by :meth:`to_json`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

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


# ----- On-SDFG storage ----------------------------------------------------
#
# The snapshot lives inside ``sdfg.frontend_metadata``, a dace Property, so it
# survives ``sdfg.save()`` / ``SDFG.from_file()``.  It is still reached through
# the historical ``sdfg._frozen_signature`` name, installed below as a
# descriptor on ``SDFG``; before this it was a plain Python attribute, which
# every serialization round-trip silently dropped.

SDFG_METADATA_KEY = 'frozen_signature'
_CACHE_ATTR = '_frozen_signature_cache'

# Argument kinds an optimization pass is allowed to delete; see :func:`refreeze`.
_MAY_SHRINK = frozenset({'scalar', 'symbol'})


def get_frozen_signature(sdfg) -> Optional["FrozenSignature"]:
    """Deserialise the snapshot stored on ``sdfg``; None if it carries none."""
    raw = getattr(sdfg, 'frontend_metadata', {}).get(SDFG_METADATA_KEY)
    if raw is None:
        return None
    # Hand back the same object while the stored dict is untouched, so repeated
    # reads don't rebuild several hundred FrozenArgs apiece.
    cached = sdfg.__dict__.get(_CACHE_ATTR)
    if cached is not None and cached[0] is raw:
        return cached[1]
    frozen = FrozenSignature.from_dict(raw)
    sdfg.__dict__[_CACHE_ATTR] = (raw, frozen)
    return frozen


def attach_to_sdfg(sdfg, frozen: Optional["FrozenSignature"]):
    """Store ``frozen`` on ``sdfg`` in serialized form; None clears it."""
    if frozen is None:
        sdfg.frontend_metadata.pop(SDFG_METADATA_KEY, None)
        sdfg.__dict__.pop(_CACHE_ATTR, None)
        return
    raw = frozen.to_dict()
    sdfg.frontend_metadata[SDFG_METADATA_KEY] = raw
    sdfg.__dict__[_CACHE_ATTR] = (raw, frozen)


def _install_sdfg_accessor():
    """Make ``sdfg._frozen_signature`` a view onto ``sdfg.frontend_metadata``."""
    from dace.sdfg import SDFG

    if isinstance(SDFG.__dict__.get('_frozen_signature'), property):
        return
    if 'frontend_metadata' not in getattr(SDFG, '__properties__', {}):
        raise RuntimeError("this dace has no SDFG.frontend_metadata property, so a frozen "
                           "signature could not survive save/load; update dace")
    SDFG._frozen_signature = property(get_frozen_signature, attach_to_sdfg)


_install_sdfg_accessor()


def refreeze(sdfg) -> "FrozenSignature":
    """Re-snapshot after a DELIBERATE transformation of the built SDFG (e.g. an optimization
    pipeline run between ``build()`` and ``build_fortran_library``), so the bindings regenerate
    against the live signature instead of tripping the drift check.

    Contract -- optimization must not change the BUFFER side of the Fortran-facing ABI:

    * array (and MPI communicator) args must match the original snapshot exactly -- names,
      order, dtypes.  The caller's buffers are the ABI, so one going missing means the
      wrapper would stop passing memory the kernel still expects, or vice versa;
    * scalar args and the free-symbol set may SHRINK, which is what specialization does:
      folding a value to a constant deletes its argument, and the binding simply derives
      and passes fewer. A scalar the kernel no longer reads costs the caller nothing;
    * anything NEW is refused, symbol or scalar -- the binding has no value derivation
      for an argument that was not in the Fortran interface.

    An array's STORAGE may change, which is how offloading shows up here: the caller-side
    location stays pinned in ``FrozenArg.storage`` and the kernel's new one is recorded in
    ``device_storage``, so the binding can bracket the call in OpenACC data clauses rather
    than pass a host pointer to a device kernel.

    Returns the new snapshot and attaches it to ``sdfg._frozen_signature``.
    """
    frozen: FrozenSignature = getattr(sdfg, "_frozen_signature", None)
    if frozen is None:
        raise RuntimeError(f"refreeze: SDFG {sdfg.name!r} carries no _frozen_signature; "
                           "it must come from SDFGBuilder.build()")
    live_arglist = sdfg.arglist()
    live_fs = set(str(s) for s in sdfg.free_symbols)
    snap_fs = set(frozen.free_symbols)

    added = sorted(live_fs - snap_fs)
    if added:
        raise SignatureDriftError(f"refreeze on {frozen.entry!r}: optimization introduced free "
                                  f"symbols {added} the binding cannot derive values for")

    dropped_buffers = [
        a.sdfg_name for a in frozen.args if a.kind not in _MAY_SHRINK and a.sdfg_name not in live_arglist
    ]
    if dropped_buffers:
        raise SignatureDriftError(f"refreeze on {frozen.entry!r}: optimization removed "
                                  f"args {dropped_buffers}; only scalars and free symbols may shrink")

    def _survives(a: FrozenArg) -> bool:
        # Anything gone from both the arglist and the free symbols was folded to a constant:
        # drop it so the regenerated binding stops deriving and passing it.  Only scalars and
        # symbols reach here -- the rest are refused above.
        return a.sdfg_name in live_arglist or a.sdfg_name in live_fs

    def _relocate(a: FrozenArg) -> FrozenArg:
        # Host-side storage churn (Register, Pinned) still hands the caller's pointer
        # straight through, so only a move into device memory is worth recording.
        live = getattr(sdfg.arrays.get(a.sdfg_name), 'storage', None)
        name = live.name if live is not None else ''
        return replace(a, device_storage=name if name.startswith(DEVICE_STORAGE_PREFIX) else '')

    new = replace(
        frozen,
        args=tuple(_relocate(a) for a in frozen.args if _survives(a)),
        free_symbols=tuple(s for s in frozen.free_symbols if s in live_fs),
    )
    # Full re-validation (arg partition, per-arg dtypes, symbol set) against the live SDFG.
    new.verify_against(sdfg)
    sdfg._frozen_signature = new
    return new


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
