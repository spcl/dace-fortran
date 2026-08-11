# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenACC transfer planning for the generated ICON binding wrapper.

ICON runs its dycore with OpenACC on the GPU; an SDFG lowered by this
project is CPU-only today.  A wrapper that hands ICON's device-resident
arguments straight to a host SDFG reads stale host memory, so the
wrapper has to stage them across the boundary.

Which arguments ICON keeps on the device is a *parse-time* fact, read
out of the OpenACC clauses in the ICON source by the
``<routine>.acc_residency.json`` sidecar::

    {"routine": ..., "source": ...,
     "args": {"<argname>": {"residency": "device"|"host",
                            "clause": ..., "evidence": ...}},
     "unclassified": [...]}

Which side the SDFG expects them on is a *build-time* fact, read off
the SDFG's own data descriptors.  Crossing the two gives exactly four
outcomes:

===============  =============  ==========================================
sidecar          SDFG storage   emitted
===============  =============  ==========================================
``device``       host           ``!$ACC UPDATE HOST`` in, ``UPDATE DEVICE``
                                back out for arguments the SDFG writes
``device``       GPU            nothing -- the device pointer is passed
                                through an ``!$ACC HOST_DATA USE_DEVICE``
                                region (the drift rule, for a future
                                dace-gpu SDFG)
``host``         host           nothing (the CPU-only status quo)
``host``         GPU            :class:`AccResidencyError`
===============  =============  ==========================================

Anything the crossing cannot decide -- an argument the sidecar never
mentions, or one the extractor parked in ``unclassified`` while the
routine does have device arguments -- is also an
:class:`AccResidencyError`.  There is deliberately no "assume host"
fallback: guessing wrong produces a silently wrong answer at runtime
rather than a failed build.

Scalar dummies are exempt from all of it.  They reach the SDFG by value
(or as an SDFG symbol), so there is no buffer to stage and no residency
to get wrong -- and ICON's own directives rarely name them, which is why
the extractor reports ``velocity_tendencies``' six scalars as
``unclassified``.  Array/pointer dummies get no such pass.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import dace

_GPU_STORAGE = (
    dace.dtypes.StorageType.GPU_Global,
    dace.dtypes.StorageType.GPU_Shared,
)

DEVICE = "device"
HOST = "host"

# ICON launches its dycore kernels on ASYNC(1); putting our copies on the
# same queue orders them after ICON's in-flight work with no extra fence.
ACC_QUEUE = 1


class AccResidencyError(Exception):
    """A sidecar/SDFG residency crossing with no defined emission."""


@dataclass(frozen=True)
class AccResidency:
    """Parsed ``<routine>.acc_residency.json`` sidecar.

    ``refs`` (additive, may be empty for old sidecars) retains the full entity
    reference string(s) the deciding clause named per argument -- e.g.
    ``("p_diag%vt(:,:,jb)",)`` -- while ``args`` stays keyed by the base name.
    """

    routine: str
    source: str
    args: Mapping[str, str]
    unclassified: Tuple[str, ...] = ()
    refs: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def has_device_args(self) -> bool:
        """True iff at least one argument was classified device-resident."""
        return any(r == DEVICE for r in self.args.values())

    @classmethod
    def from_dict(cls, raw: Mapping) -> "AccResidency":
        """Build from the sidecar mapping, rejecting unknown residencies."""
        args = {}
        refs = {}
        for name, entry in (raw.get("args") or {}).items():
            residency = (entry or {}).get("residency")
            if residency not in (DEVICE, HOST):
                raise AccResidencyError(f"acc residency sidecar: argument {name!r} has residency "
                                        f"{residency!r}; expected {DEVICE!r} or {HOST!r}")
            args[str(name)] = residency
            entry_refs = (entry or {}).get("refs") or ([(entry or {}).get("ref")] if (entry or {}).get("ref") else [])
            if entry_refs:
                refs[str(name)] = tuple(str(r) for r in entry_refs)
        return cls(
            routine=str(raw.get("routine", "")),
            source=str(raw.get("source", "")),
            args=args,
            unclassified=tuple(str(a) for a in (raw.get("unclassified") or ())),
            refs=refs,
        )

    @classmethod
    def from_file(cls, path) -> "AccResidency":
        """Read and parse the sidecar at ``path``."""
        path = Path(path)
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise AccResidencyError(f"acc residency sidecar not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise AccResidencyError(f"acc residency sidecar {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(raw)


@dataclass(frozen=True)
class AccTransferPlan:
    """The three emission slots of the wrapper, in wrapper-argument order."""

    update_host: Tuple[str, ...] = ()
    update_device: Tuple[str, ...] = ()
    use_device: Tuple[str, ...] = ()
    storage: Mapping[str, str] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """True iff the wrapper needs any OpenACC directive at all."""
        return bool(self.update_host or self.update_device or self.use_device)


def sdfg_containers_for_arg(sdfg: dace.SDFG, arg: str) -> Tuple[str, ...]:
    """Non-transient SDFG data containers backing wrapper argument ``arg``.

    ``hlfir-flatten-structs`` unpacks a derived-type dummy into one flat
    container per reachable member, named ``<dummy>_<member path>``, and
    leaves plain array dummies under their own name.  Both are recovered
    by the same prefix rule.  Sorted, so callers are deterministic.
    """
    prefix = arg + "_"
    return tuple(sorted(n for n, desc in sdfg.arrays.items() if not desc.transient and (n == arg or n.startswith(prefix))))


def is_scalar_arg(sdfg: dace.SDFG, arg: str, containers: Sequence[str]) -> bool:
    """True iff ``arg`` reaches the SDFG as a value rather than a buffer.

    Either it was promoted to an SDFG symbol (no container at all) or
    every container backing it is a :class:`dace.data.Scalar`.
    """
    if not containers:
        return arg in sdfg.symbols or arg in sdfg.free_symbols
    return all(isinstance(sdfg.arrays[n], dace.data.Scalar) for n in containers)


def _arg_storage(sdfg: dace.SDFG, arg: str, containers: Sequence[str]) -> str:
    """Classify ``arg`` as ``host`` or ``device`` storage from the SDFG."""
    if not containers:
        raise AccResidencyError(f"argument {arg!r} has no SDFG data container and is not an SDFG "
                                f"symbol; its storage side cannot be determined")
    sides = {(DEVICE if sdfg.arrays[n].storage in _GPU_STORAGE else HOST) for n in containers}
    if len(sides) > 1:
        raise AccResidencyError(f"argument {arg!r} maps to SDFG containers with mixed host/GPU storage: " +
                                ", ".join(f"{n}={sdfg.arrays[n].storage.name}" for n in containers))
    return sides.pop()


def _written_args(sdfg: dace.SDFG, containers: Mapping[str, Tuple[str, ...]]) -> frozenset:
    """Wrapper arguments the SDFG writes, via ``SDFG.read_and_write_sets``."""
    _, write_set = sdfg.read_and_write_sets()
    return frozenset(arg for arg, names in containers.items() if write_set & set(names))


#: Array-section / argument-list parentheses of an entity ref, innermost first.
_PAREN_GROUP_RE = re.compile(r"\([^()]*\)")


def _ref_to_flat_name(ref: str) -> str:
    """Normalise a sidecar entity ref to its flat-binding-name form.

    ``hlfir-flatten-structs`` names a flattened member ``<dummy>_<member path>``
    with ``%`` -> ``_``; array-section subscripts select elements, not members,
    so they are dropped: ``p_diag%vt(:,:,jb)`` -> ``p_diag_vt``.
    """
    text = re.sub(r"\s+", "", ref)
    while _PAREN_GROUP_RE.search(text):
        text = _PAREN_GROUP_RE.sub("", text)
    return text.replace("%", "_")


def validate_acc_mappability(sdfg: dace.SDFG, residency: AccResidency, arg_order: Iterable[str]) -> None:
    """Pre-plan invariant: every device-resident sidecar entity the routine's
    interface exposes must resolve, through the flatten plan frozen into the
    SDFG's data descriptors, to exactly one flat binding argument.

    Data ICON keeps on the GPU is staged per flat binding buffer; an entity
    with no flat counterpart (a pruned member, a pointer the flattener could
    not unpack) has no buffer to stage through, so it must fail loudly here --
    a pointer on the GPU must be mappable.  A whole-struct entity resolving to
    several flats is one uniquely-resolved leaf per flat, which is fine.
    Host-resident and unmentioned arguments keep their existing behaviour.
    """
    problems = []
    for arg in arg_order:
        if residency.args.get(arg) != DEVICE:
            continue
        containers = sdfg_containers_for_arg(sdfg, arg)
        if is_scalar_arg(sdfg, arg, containers):
            continue
        if not containers:
            problems.append(f"{arg!r} is device-resident but has no flat binding argument in the "
                            f"SDFG (pointer on GPU must be mappable)")
            continue
        for ref in residency.refs.get(arg) or (arg, ):
            flat = _ref_to_flat_name(ref)
            exact = [n for n in containers if n == flat]
            prefixed = [n for n in containers if n.startswith(flat + "_")]
            if exact:
                continue
            if len(prefixed) == 0:
                problems.append(f"device-resident entity {ref!r} of argument {arg!r} does not resolve "
                                f"to any flat binding argument (candidates: "
                                f"{', '.join(containers)}); pointer on GPU must be mappable")
    if problems:
        raise AccResidencyError(f"acc residency sidecar for {residency.routine or '<unknown>'} fails the "
                                f"mappability invariant -- " + "; ".join(problems))


def plan_acc_transfers(sdfg: dace.SDFG, residency: AccResidency, arg_order: Iterable[str]) -> AccTransferPlan:
    """Cross the sidecar with the SDFG and return what the wrapper must emit.

    ``arg_order`` is the wrapper's declared dummy order and fixes the
    order of every emitted directive, so two runs over the same inputs
    produce byte-identical Fortran.  Scalar dummies are dropped from the
    crossing entirely (see the module docstring); every remaining
    argument must be decidable or the call raises.
    """
    arg_order = tuple(arg_order)
    unclassified = set(residency.unclassified)

    containers = {a: sdfg_containers_for_arg(sdfg, a) for a in arg_order}
    buffers = tuple(a for a in arg_order if not is_scalar_arg(sdfg, a, containers[a]))

    offenders = []
    missing = [a for a in buffers if a not in residency.args and a not in unclassified]
    if missing:
        offenders.append("not present in the sidecar: " + ", ".join(missing))
    if residency.has_device_args:
        stuck = [a for a in buffers if a in unclassified]
        if stuck:
            offenders.append("left unclassified by the extractor while the routine has device-resident "
                             "arguments: " + ", ".join(stuck))
    if offenders:
        raise AccResidencyError(f"acc residency sidecar for {residency.routine or '<unknown>'} cannot be "
                                f"applied -- " + "; ".join(offenders))

    validate_acc_mappability(sdfg, residency, arg_order)

    storage = {a: _arg_storage(sdfg, a, containers[a]) for a in buffers}
    written = _written_args(sdfg, {a: containers[a] for a in buffers})

    update_host, update_device, use_device, bad = [], [], [], []
    for arg in buffers:
        side = residency.args.get(arg, HOST)
        if side == DEVICE and storage[arg] == HOST:
            update_host.append(arg)
            if arg in written:
                update_device.append(arg)
        elif side == DEVICE and storage[arg] == DEVICE:
            use_device.append(arg)
        elif side == HOST and storage[arg] == DEVICE:
            bad.append(f"{arg} (host-resident in ICON, GPU storage in the SDFG)")
    if bad:
        raise AccResidencyError(f"acc residency sidecar for {residency.routine or '<unknown>'} cannot be "
                                f"applied -- no transfer is defined for: " + "; ".join(bad))

    return AccTransferPlan(
        update_host=tuple(update_host),
        update_device=tuple(update_device),
        use_device=tuple(use_device),
        storage=storage,
    )


def _directive(keyword: str, names: Sequence[str], indent: str) -> list:
    """One async directive per name, so no line needs a continuation marker."""
    return [f"{indent}!$ACC {keyword}({name}) ASYNC({ACC_QUEUE})" for name in names]


def render_pre_call(plan: AccTransferPlan, indent: str = "  ") -> list:
    """Directive lines between the entry timer read and the SDFG call."""
    return _directive("UPDATE HOST", plan.update_host, indent)


def render_post_call(plan: AccTransferPlan, indent: str = "  ") -> list:
    """Directive lines between the SDFG call and the exit sync."""
    return _directive("UPDATE DEVICE", plan.update_device, indent)


def render_host_data_open(plan: AccTransferPlan, indent: str = "  ") -> list:
    """``HOST_DATA USE_DEVICE`` opener for the drift-rule arguments."""
    if not plan.use_device:
        return []
    return [f"{indent}!$ACC HOST_DATA USE_DEVICE({', '.join(plan.use_device)})"]


def render_host_data_close(plan: AccTransferPlan, indent: str = "  ") -> list:
    """Closer matching :func:`render_host_data_open`."""
    if not plan.use_device:
        return []
    return [f"{indent}!$ACC END HOST_DATA"]


def render_sync(plan: AccTransferPlan, indent: str = "  ") -> list:
    """The sync line that must precede each ``SYSTEM_CLOCK`` read.

    ICON launches its kernels on ``ASYNC(1)`` and our copies join the
    same queue (``ACC_QUEUE``), so ``WAIT(1)`` alone drains everything
    that matters: without it the entry read charges us ICON's
    outstanding work and the exit read returns before the copy-back has
    landed.  Other queues never carry our traffic, so they are not
    drained.
    """
    return [f"{indent}!$ACC WAIT({ACC_QUEUE})"] if plan.active else []
