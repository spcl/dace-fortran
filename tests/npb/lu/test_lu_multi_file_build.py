"""NAS Parallel Benchmark LU -- multi-file SDFG build.

The LU benchmark (`NPB3.4 -- NPB3.4-OMP/LU
<https://github.com/llnl/NPB/tree/master/NPB3.4/NPB3.4-OMP/LU>`_) is
an SSOR-based solver for a synthetic CFD problem.  This test drives
its Modern-Fortran port through the bridge as a **multi-file
project**:

  * ``lu.F90`` -- the benchmark proper, a single ``MODULE lu`` with
    13 ``CONTAIN``ed subroutines (``dolu``, ``domain``, ``setcoeff``,
    ``setbv``, ``setiv``, ``erhs``, ``ssor``, ``rhs``, ``l2norm``,
    ``jacld``, ``blts``, ``buts``, ``jacu``).  Module-level scope
    holds the geometry / coefficient state common to NPB
    (``isiz1=isiz2=isiz3=33``, ``c1..c5``, ``dxi/deta/dzeta``, ...).
    Only ``dolu`` is ``PUBLIC``.
  * ``useapplu.F90`` -- a 10-line driver module that ``USE lu, ONLY:
    dolu`` and exposes ``call_dolu`` as the SDFG entry.

The bridge sees both files via :func:`dace_fortran.build_sdfg_from_files`,
which stages them and runs ``merge_used_modules`` so flang lowers one
self-contained translation unit.  Same multi-file shape ICON's
cross-module kernels will use, just on the NPB instead.

This test is the **first** of the NPB suite under ``tests/npb/``.  New
benchmarks (BT / CG / EP / FT / MG / SP) drop into sibling folders on
the same pattern -- see ``tests/npb/README.md``.
"""
import json
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files


_HERE = Path(__file__).resolve().parent

# Mangled flang symbol for ``useapplu::call_dolu`` -- the driver entry.
# ``_QM<module>P<procedure>`` is flang's name-mangling form for a
# CONTAIN-ed subroutine inside a Fortran 90 module.
_ENTRY = "call_dolu"

_LU_SOURCES = [_HERE / "lu.F90", _HERE / "useapplu.F90"]

# LU compute kernels we expect to find referenced somewhere in the
# built SDFG.  The driver inlines ``dolu`` which sequences these in
# order; if none survive the multi-file merge, lu.F90's body was
# dropped silently.
_LU_KERNELS = ("ssor", "rhs", "jacld", "jacu", "blts", "buts", "erhs")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.mark.long
def test_lu_multi_file_builds(tmp_path):
    """The bridge ingests ``[lu.F90, useapplu.F90]`` and emits an SDFG
    rooted at ``useapplu::call_dolu``."""
    sdfg = build_sdfg_from_files(
        _LU_SOURCES,
        entry=_ENTRY,
        name="npb_lu",
        out_dir=tmp_path / "build",
    )
    sdfg.validate()
    # At least one LU compute kernel must show up in the serialised
    # SDFG (state / node labels, NestedSDFG names, ...) -- a silent
    # body-drop from the multi-file merge would otherwise produce a
    # benign-looking but empty SDFG.  ``sdfg.to_json()`` returns the
    # canonical dict; stringify it once for a substring scan.
    sdfg_text = json.dumps(sdfg.to_json()).lower()
    assert any(k in sdfg_text for k in _LU_KERNELS), (
        f"built SDFG does not reference any of {_LU_KERNELS}; the "
        "multi-file merge likely dropped lu.F90's body.")
