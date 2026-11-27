"""Disk memo for deterministic test by-products (f2py reference builds, static-analysis verdicts).

An entry is keyed by a digest of everything that determines it (source text, flags, toolchain versions), so a hit
returns exactly what a rebuild would produce, and an edited input or an upgraded compiler misses. The root is
``$DACE_FORTRAN_TEST_CACHE`` (``off`` disables the cache), else ``~/.cache/dace_fortran_tests``.
"""

import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

_OFF = "off"


def _root() -> Path | None:
    root = os.environ.get("DACE_FORTRAN_TEST_CACHE", "")
    return None if root == _OFF else Path(root) if root else Path.home() / ".cache" / "dace_fortran_tests"


@cache
def tool_version(tool: str) -> str:
    """First line of ``<tool> --version``; part of every key a compiler's output depends on."""
    done = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
    return done.stdout.splitlines()[0] if done.stdout else done.stderr.strip()


def digest(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        h.update(len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def entry(kind: str, key: str) -> Path | None:
    """The file stored under (``kind``, ``key``), or ``None`` on a miss or with the cache off."""
    root = _root()
    path = None if root is None else root / kind / key
    return path if path is not None and path.is_file() else None


def store(kind: str, key: str, source: Path) -> None:
    """Publish ``source`` under (``kind``, ``key``); the rename is atomic, so a concurrent reader never sees a partial file."""
    root = _root()
    if root is None:
        return
    dest = root / kind / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f"{key}.{os.getpid()}")
    shutil.copyfile(source, staging)
    os.replace(staging, dest)
