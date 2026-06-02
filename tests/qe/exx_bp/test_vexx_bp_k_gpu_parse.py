"""Parse-stress anchor for QE's ``exx_bp::vexx_bp_k_gpu`` GPU kernel.

The fixture :file:`ast_v1_vexx_bp_k_gpu.f90` is the pre-processed
flat-Fortran checkpoint emitted by ``f2dace-qe-source``'s pruning pipeline
for the ``vexx_bp_k_gpu`` entry point (single TU, all USE-closure modules
inlined into one file, ~2k lines). It is the bridge-facing analogue of
the cloudsc / ICON full-source tests, scoped down to a single QE
microkernel.

The current bridge cannot ingest this file end-to-end: flang itself
rejects four call sites against the ``fft_interfaces`` generic
``fwfft`` / ``invfft`` interfaces because the pruning pipeline produced
``INTERFACE`` blocks without matching specific subroutines for the
``'Wave'`` / ``'Rho'`` mode + optional ``howmany=`` argument shape
that ``vexx_bp_k_gpu`` calls them with. There is also a local
``DOUBLE PRECISION :: max`` declaration that shadows the intrinsic
``max`` (a known QE idiom the bridge does not yet normalise).

The test therefore xfails strict on parse: an upstream bridge change
that lets the file build (e.g. specific-resolution synthesis for
unresolved generics, or intrinsic-shadow normalisation) will flip this
to XPASS and surface the structural / numerical checks underneath as
the next gap to close.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ast_v1_vexx_bp_k_gpu.f90"
_ENTRY = "exx_bp::vexx_bp_k_gpu"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.mark.xfail(strict=True,
                   reason="QE vexx_bp_k_gpu pre-processed checkpoint has unresolved "
                          "fft_interfaces generic + DOUBLE PRECISION :: max intrinsic shadow; "
                          "flang rejects at parse time. Anchored as a bridge gap target.")
def test_vexx_bp_k_gpu_parses(tmp_path):
    """Probe the bridge against the QE EXX kernel parse.

    Currently xfails because of the documented flang-side generic
    resolution gap. Once the bridge synthesises specific subroutines
    for unresolved generics (or the pruning pipeline emits them), the
    parse will succeed and this test flips.
    """
    src = _SRC.read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"),
                                   entry=_ENTRY,
                                   name="vexx_bp_k_gpu")
    sdfg.validate()
    assert sdfg is not None
    # Conservative parse-anchor assertion -- structural-only -- since the
    # numerical reference path through f2py would require the QE runtime
    # (FFTW, MPI, ScaLAPACK) which is not in scope here.
    assert any('vexx_bp_k_gpu' in name for name in sdfg.arrays) or \
        'vexx_bp_k_gpu' in str(sdfg.label)
