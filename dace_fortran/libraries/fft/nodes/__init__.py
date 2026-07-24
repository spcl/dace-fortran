# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace_fortran.libraries.fft.nodes.fft_interpolate import FFTInterpolate

# The QE fftcall lowering (emit_library.py) maps fwfft/invfft to DaCe's own DFT library nodes;
# re-export them here so the single ``dace_fortran.libraries.fft.nodes`` import resolves FFT/IFFT
# alongside the Fortran-specific FFTInterpolate.
from dace.libraries.fft.nodes import FFT, IFFT
