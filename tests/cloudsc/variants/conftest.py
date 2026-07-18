"""Shared fixtures for the CLOUDSC-GPU test family.

Reuses full-CLOUDSC's strict-IEEE DaCe C++ compiler-flag fixture (import registers it here).
"""

from cloudsc.full.conftest import _strict_fp_cpu_args  # noqa: F401
