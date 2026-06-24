"""``input -> single TU`` extraction gate for the ICON-O ocean kernels.

This chat owns the *first* pipeline stage: each numerically critical ocean
kernel must extract from the real ICON source into a self-contained,
gfortran-compiling ``.f90`` (checked into this folder).  Lowering that
single TU to a DaCe SDFG is a separate concern handled elsewhere.

For each kernel this test regenerates the single TU via the fparser
inliner (merge closure -> cpp pre-pass -> prune -> ``gfortran
-fsyntax-only``) and asserts it (a) compiles and (b) is byte-identical to
the committed ``*_single_tu.f90`` artifact -- so the committed input the
SDFG stage builds never silently drifts from the source.

Slow (the merged closure is ~137k lines) and memory-heavy (the fparser
parse peaks near 9 GB), so it is marked ``long`` and serialised onto one
xdist worker; each extraction runs in its own memory-capped subprocess.
Gated on flang-new-21 + OpenMPI + the icon-model submodule.
"""
from pathlib import Path

import pytest

from icon.ocean._ocean_harness import (HAVE_FLANG, HAVE_OPENMPI, KERNELS,
                                       SINGLE_TU_ARTIFACTS, extract_single_tu, have_icon_ocean)

_HERE = Path(__file__).resolve().parent
_ARTIFACT_FILE = {a[0]: a[1] for a in SINGLE_TU_ARTIFACTS}

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (HAVE_FLANG and HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI"),
    pytest.mark.skipif(not have_icon_ocean(),
                       reason="icon-model ocean source not checked out; run "
                       "`git submodule update --init --recursive tests/icon/full/icon-model`"),
]


@pytest.mark.xdist_group("ocean_fparser")
@pytest.mark.parametrize("key,source,entry", [k[:3] for k in KERNELS], ids=[k[0] for k in KERNELS])
def test_extract_compiles_and_matches_committed(tmp_path, key, source, entry):
    """Extract one kernel into a compiling single TU and check it against the
    committed artifact."""
    res = extract_single_tu(source, entry, tmp_path / key)
    assert res["passed"], \
        f"{key}: extraction did not produce a compiling single TU.\n{res['output'][-4000:]}"
    # The closure merges to ~137k lines; pruning to the kernel must shrink it
    # by orders of magnitude.
    assert res["tu_lines"] is not None and res["tu_lines"] < 50_000, \
        f"{key}: pruned TU is {res['tu_lines']} lines -- pruning did not converge"
    # Drift guard: the freshly extracted TU must equal the committed artifact,
    # so the SDFG-lowering stage builds exactly the source validated here.
    committed = _HERE / _ARTIFACT_FILE[key]
    assert committed.is_file(), \
        f"{key}: no committed artifact {committed.name}; save the extracted TU into this folder"
    assert Path(res["tu_path"]).read_text() == committed.read_text(), \
        f"{key}: extracted TU drifted from committed {committed.name}; regenerate it"
