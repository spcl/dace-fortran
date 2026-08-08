# Overlay of builtin's `gcc` that makes `+nvptx` build against CUDA 13.
#
# Builtin's nvptx_install() configures the accelerator compiler with neither
# --with-arch nor --with-multilib-list, so gcc/config.gcc's own nvptx default
# (sm_52) wins and the accelerator builds a single misa-sm_52 lane.  CUDA 13
# ptxas rejects every ISA below sm_80, so configure-target-libgcc dies with
# "ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'".
#
# The patch moves both config.gcc nvptx defaults up -- with_arch to sm_80 and
# nvptx_multilibs_default to sm_89 -- giving TM_MULTILIB_CONFIG "sm_80 sm_89".
# (configure.ac defaults with_multilib_list to "default" when the flag is not
# passed, so nvptx_multilibs_default is the knob a stock build actually hits.)
# Patching config.gcc rather than overriding nvptx_install() is deliberate:
# builtin registers nvptx_install with @run_before("configure"), and spack dedups
# run_before callbacks by function identity -- a subclass override registers
# as a *second* callback, so the sm_52 version would still run.
#
# Upstream gap: builtin's `gcc` has no cuda_arch plumbing for the offload
# compiler at all.  Worth reporting to spack-packages; drop this overlay if
# that lands.

from spack_repo.builtin.packages.gcc.package import Gcc as BuiltinGcc

from spack.package import *


class Gcc(BuiltinGcc):
    __doc__ = BuiltinGcc.__doc__

    # GCC 16 has no sm_90 ISA: nvptx-sm.def tops out at sm_89, and nvptx.opt
    # march-maps sm_90, sm_90a and sm_100 all onto misa=sm_89.  sm_89 is the
    # Hopper lane -- ptxas runs a `.target sm_XY` image on any sm_MN, MN >= XY.
    patch("nvptx-sm80-sm90.patch", when="@16:+nvptx")
