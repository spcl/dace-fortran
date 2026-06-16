# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Test helpers for the fparser inliner tests.

Ported from the upstream DaCe ``tests/fortran/fortran_test_helper.py``,
trimmed to the surface the inliner tests use (``SourceCodeBuilder`` +
``parse_and_improve``) and re-pointed at the dace-fortran package
(``dace_fortran.fparser_inliner`` / ``dace_fortran.inliner``).
"""
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from os import path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple

import pytest

from dace_fortran.fparser_inliner import parse_and_improve  # noqa: F401  (re-exported for tests)


def _have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


@dataclass
class SourceCodeBuilder:
    """Helper to assemble multi-file Fortran sources for frontend tests.

    Mirrors the upstream builder: ``add_file`` (name auto-inferred from
    the first program / module / function / subroutine), an optional
    ``check_with_gfortran`` compile gate, and ``get`` to retrieve the
    ``{name: content}`` mapping.
    """
    sources: Dict[str, str] = field(default_factory=dict)

    def add_file(self, content: str, name: Optional[str] = None):
        """Add source file contents in the order you'd pass them to ``gfortran``."""
        if not name:
            name = SourceCodeBuilder._identify_name(content)
        name, ext = name.rsplit('.', 1) if '.' in name else (name, 'f90')
        key, counter = f"{name}.{ext}", 0
        while key in self.sources:
            key, counter = f"{name}_{counter}.{ext}", counter + 1
        self.sources[key] = content
        return self

    def check_with_gfortran(self):
        """Assert that it all compiles with ``gfortran`` (skips if gfortran absent)."""
        if not _have_gfortran():
            pytest.skip("gfortran not on PATH")
        with TemporaryDirectory() as td:
            for fname, content in self.sources.items():
                with open(path.join(td, fname), 'w') as f:
                    f.write(content)
            cmd = ['gfortran', '-Wall', '-shared', '-fPIC', '-ffree-line-length-none', *self.sources.keys()]
            try:
                subprocess.run(cmd, cwd=td, capture_output=True).check_returncode()
                return self
            except subprocess.CalledProcessError as e:
                print("Fortran compilation failed!")
                print(e.stderr.decode())
                raise e

    def get(self) -> Tuple[Dict[str, str], Optional[str]]:
        """Get a dictionary mapping file names to their content + the ``main`` source."""
        main = self.sources.get('main.f90')
        return self.sources, main

    @staticmethod
    def _identify_name(content: str) -> str:
        PPAT = re.compile(r"^.*\bprogram\b\s*\b(?P<prog>[a-zA-Z0-9_]*)\b.*$", re.I | re.M | re.S)
        if PPAT.match(content):
            return PPAT.search(content).group('prog') or 'main'
        MPAT = re.compile(r"^.*\bmodule\b\s*\b(?P<mod>[a-zA-Z0-9_]+)\b.*$", re.I | re.M | re.S)
        if MPAT.match(content):
            return MPAT.search(content).group('mod')
        FPAT = re.compile(r"^.*\bfunction\b\s*\b(?P<fn>[a-zA-Z0-9_]+)\b.*$", re.I | re.M | re.S)
        if FPAT.match(content):
            return FPAT.search(content).group('fn')
        SPAT = re.compile(r"^.*\bsubroutine\b\s*\b(?P<subr>[a-zA-Z0-9_]+)\b.*$", re.I | re.M | re.S)
        if SPAT.match(content):
            return SPAT.search(content).group('subr')
        raise ValueError(f"Could not find any identifiable object in the content:\n{content}")
