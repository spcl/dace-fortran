#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""LLVM/flang discovery shared by the bridge build and the flang drivers.

The bridge supports every major in :data:`SUPPORTED_LLVM_VERSIONS`.  ``$LLVM_VERSION``
pins one; otherwise the first one actually installed wins.  Stdlib-only on purpose --
:mod:`dace_fortran.emit_hlfir` and :mod:`dace_fortran.flang_codebase` import this
without dragging in the compiled bridge.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

#: LLVM major versions the bridge builds against, in preference order.
SUPPORTED_LLVM_VERSIONS = ("21", "22")

#: Unversioned binary names; they only pin a major once ``--version`` confirms it.
_UNVERSIONED = ("flang-new", "flang")

_FLANG_VERSION_RE = re.compile(r"flang(?:-new)? version (\d+)")


def requested_version() -> str:
    """The LLVM major pinned by ``$LLVM_VERSION``, or ``''`` to auto-detect."""
    return os.environ.get("LLVM_VERSION", "").strip()


def candidate_versions() -> Sequence[str]:
    """Majors to probe, in order: the pinned one alone, else all supported ones."""
    pinned = requested_version()
    return (pinned, ) if pinned else SUPPORTED_LLVM_VERSIONS


def flang_names(version: Optional[str] = None) -> List[str]:
    """flang binary names to probe, most specific first.

    Debian/Ubuntu ship ``flang-new-<major>`` / ``flang-<major>``; spack and upstream
    installs ship only unversioned ``flang-new`` / ``flang``, so those come last and
    are version-checked by :func:`flang_major` before being accepted.
    """
    if version is None:
        names = [f"{stem}-{v}" for v in candidate_versions() for stem in _UNVERSIONED]
    else:
        names = [f"{stem}-{version}" for stem in _UNVERSIONED]
    return names + list(_UNVERSIONED)


def flang_major(binary: str) -> str:
    """The LLVM major a flang binary reports, or ``''`` if it is not LLVM flang."""
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = _FLANG_VERSION_RE.search(proc.stdout)
    return match.group(1) if match else ""


def find_flang(version: Optional[str] = None) -> Optional[str]:
    """Path to a supported flang on ``$PATH``, or ``None``.

    ``$FC`` wins when it points at an LLVM flang of an acceptable major.
    """
    wanted = [version] if version else list(candidate_versions())
    override = os.environ.get("FC", "").strip()
    probes = [override] if override else []
    probes += flang_names(version)
    for name in probes:
        found = shutil.which(name)
        if not found:
            continue
        if os.path.basename(name) in _UNVERSIONED or name == override:
            if flang_major(found) not in wanted:
                continue
        return found
    return None


def require_flang(version: Optional[str] = None) -> str:
    """:func:`find_flang`, or raise naming every binary and major that was tried."""
    found = find_flang(version)
    if found is None:
        majors = "/".join([version] if version else list(candidate_versions()))
        raise RuntimeError(f"No LLVM flang for LLVM {majors} on PATH (tried "
                           f"{', '.join(flang_names(version))}); install LLVM/Flang "
                           f"{majors} to use the HLFIR frontend, or point $FC at one.")
    return found


def llvm_prefix(flang_binary: str) -> Path:
    """Install prefix of a flang binary, resolving through the usual ``bin/`` symlink."""
    return Path(flang_binary).resolve().parent.parent


def intrinsic_modules_path(flang_binary: str) -> Path:
    """``-fintrinsic-modules-path`` for a flang binary: ``<prefix>/include/flang``.

    Debian's ``/usr/lib/llvm-<major>`` and spack/upstream prefixes share this layout,
    so deriving it beats hardcoding either one.
    """
    return llvm_prefix(flang_binary) / "include" / "flang"
