"""SC26 ``Xor64Rng`` reproduced in numpy for the Quantum-ESPRESSO tests.

The SC26 layout artifacts seed every kernel from one xorshift64 stream
so the AoS and SoA variants see identical numbers.  This is a verbatim
port of that scheme (``Experiments/common/prng.h``): ``splitmix64`` to
seed the state, then xorshift64 per draw with a 53-bit mantissa.  The
exact seed is unimportant for a correctness test -- only that both the
SDFG and the numpy reference draw from the same stream.
"""
import numpy as np

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """One ``splitmix64`` step -- seeds the xorshift64 state."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def xor64_uniform01(n: int, seed: int = 42) -> np.ndarray:
    """``n`` draws of the ``Xor64Rng.uniform01()`` stream: ``splitmix64``
    to seed the state, then xorshift64 (``^=<<13; ^=>>7; ^=<<17``) per
    draw with mantissa ``(next>>11)/2**53``."""
    state = _splitmix64(seed) or seed
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        state ^= (state << 13) & _MASK64
        state ^= state >> 7
        state ^= (state << 17) & _MASK64
        state &= _MASK64
        out[i] = (state >> 11) / float(1 << 53)
    return out


def complex_stream(n: int, seed: int) -> np.ndarray:
    """``n`` complex draws (re / im interleaved from one stream)."""
    flat = xor64_uniform01(2 * n, seed)
    return (flat[0::2] + 1j * flat[1::2]).astype(np.complex128)
