"""input -> single TU extraction gate for the ICON-O ocean kernels.

Each numerically critical ocean kernel must extract from the real ICON source into a
self-contained, gfortran-compiling .f90 (checked into this folder); lowering to a DaCe SDFG
is a separate concern handled elsewhere.

Regenerates the single TU via the fparser inliner (merge closure -> cpp pre-pass -> prune ->
gfortran -fsyntax-only) and asserts it compiles AND is byte-identical to the committed
*_single_tu.f90 artifact, so the SDFG stage's input never silently drifts from the source.

Slow (~137k-line merged closure) and memory-heavy (fparser peaks near 9GB): marked long,
serialised onto one xdist worker, each extraction in its own memory-capped subprocess.
"""
from pathlib import Path

import pytest

from icon.ocean._ocean_harness import (HAVE_FLANG, HAVE_OPENMPI, KERNELS, SINGLE_TU_ARTIFACTS, extract_single_tu,
                                       have_icon_ocean)

_HERE = Path(__file__).resolve().parent
_SOURCE = {k[0]: k[1] for k in KERNELS}

#: One case per ``(key, halo_mode, filename, entry)`` -- the non-solver kernels
#: in "external" only, the free-surface solver in BOTH halo modes.
_CASES = SINGLE_TU_ARTIFACTS

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (HAVE_FLANG and HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI"),
    pytest.mark.skipif(not have_icon_ocean(),
                       reason="icon-model ocean source not checked out; run "
                       "`git submodule update --init --recursive tests/icon/full/icon-model`"),
]


@pytest.mark.xdist_group("ocean_fparser")
@pytest.mark.parametrize("key,halo_mode,filename,entry", _CASES, ids=[f"{c[0]}-{c[1]}" for c in _CASES])
def test_extract_compiles_and_matches_committed(tmp_path, key, halo_mode, filename, entry):
    """Extract one (kernel, halo mode) into a compiling single TU and check it against the
    committed artifact -- both external (callback boundary) and inlined (MPI-only) modes."""
    res = extract_single_tu(_SOURCE[key], entry, tmp_path / f"{key}_{halo_mode}", halo_mode=halo_mode)
    assert res["passed"], \
        f"{key}[{halo_mode}]: extraction did not produce a compiling single TU.\n{res['output'][-4000:]}"
    # Closure merges to ~137k lines; pruning to the kernel must shrink it by orders of magnitude.
    assert res["tu_lines"] is not None and res["tu_lines"] < 50_000, \
        f"{key}[{halo_mode}]: pruned TU is {res['tu_lines']} lines -- pruning did not converge"
    # Drift guard: the freshly extracted TU must equal the committed artifact.
    committed = _HERE / filename
    assert committed.is_file(), \
        f"{key}[{halo_mode}]: no committed artifact {committed.name}; save the extracted TU into this folder"
    assert Path(res["tu_path"]).read_text() == committed.read_text(), \
        f"{key}[{halo_mode}]: extracted TU drifted from committed {committed.name}; regenerate it"
