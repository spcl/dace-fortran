"""Shared fixtures for the full-CLOUDSC test family: strict-FP DaCe compiler
override so the SDFG side matches gfortran's IEEE arithmetic (else copy-pasted per module).
"""

import dace
import pytest


@pytest.fixture
def _strict_fp_cpu_args():
    """Match gfortran's FP semantics on the SDFG side.

    DaCe's default -O3 -march=native -ffast-math lets gcc fuse/reassociate FP
    ops, diverging from strict-IEEE gfortran. Drop -ffast-math, O3->O0, disable
    FMA contraction; restore flags after. Flag set is the flang-portable core
    (see CLOUDSC_F90FLAGS).
    """
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set(
        'compiler',
        'cpu',
        'args',
        value='-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off '
        '-Wno-unused-parameter -Wno-unused-label',
    )
    try:
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=prev)
