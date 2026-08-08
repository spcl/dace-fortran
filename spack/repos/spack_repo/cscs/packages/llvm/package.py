# Overlay of builtin's `llvm` that actually builds the OpenMP NVPTX device runtime.
#
# Builtin's `+libomptarget` gives you the host plugin but no device bitcode, so
# `clang -fopenmp --offload-arch=sm_80` dies with "no library 'libomptarget-nvptx.bc'
# found".  Every knob that used to fix this is gone in LLVM 22:
#
#   * `offload/DeviceRTL/` no longer exists -- the device runtime moved to
#     `openmp/device/` (present in release/22.x, absent in release/21.x and older).
#   * `LIBOMPTARGET_DEVICE_ARCHITECTURES` was removed; the DeviceRTL is now generic IR
#     and always covers AMDGPU + NVPTX (openmp/docs/ReleaseNotes.rst:24-27).  There is
#     no compute-capability input and no arch suffix on the bitcode filename.
#   * `LIBOMPTARGET_NVPTX_ENABLE_BCLIB` is dead -- it survives only as stale prose in
#     openmp/README.rst:270.  Builtin still sets it (llvm/package.py:972 and :1199) and
#     nothing reads it.
#
# What gates the device runtime now is openmp/CMakeLists.txt:166-170, an if/**else**:
# one openmp configure builds EITHER the host runtime OR `device/`, never both, chosen
# by the target triple.  So the device runtime needs a *second* cross runtimes build,
# and that is exactly what builtin never asks for: it sets no `LLVM_RUNTIME_TARGETS`
# and no per-target `RUNTIMES_<triple>_*` (its only RUNTIMES_* is the global
# `RUNTIMES_CMAKE_ARGS` at llvm/package.py:1168).
#
# Two details make the pair of defines below load-bearing:
#
#   * `default` must stay in LLVM_RUNTIME_TARGETS.  llvm/runtimes/CMakeLists.txt:600-608
#     calls runtime_default_target() only `if("default" IN_LIST LLVM_RUNTIME_TARGETS)`
#     and otherwise replaces the host runtimes build with empty custom targets -- i.e.
#     omitting it would silently drop libomp, compiler-rt, libcxx and libunwind.
#     runtime_default_target() is also the only consumer of RUNTIMES_CMAKE_ARGS
#     (:286), so builtin's `--gcc-install-dir` flags keep reaching the host runtimes.
#   * RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES must be pinned.
#     runtime_register_target() first appends the *whole* host LLVM_ENABLE_RUNTIMES to
#     the sub-build (:386-387) and only then lets `RUNTIMES_<name>_*` variables
#     override it (:391-405); without the pin the nvptx sub-build would try to
#     cross-build compiler-rt/libcxx/libcxxabi/libunwind/offload for nvptx.  Note it
#     does *not* receive RUNTIMES_CMAKE_ARGS (:418-433), so the host's
#     `--gcc-install-dir` never leaks into the device compile.
#
# No cuda dependency is added on purpose: nothing on this path calls find_package(CUDA),
# nvcc or ptxas -- verified with `clang -###`, neither ptxas nor nvlink is invoked.  A
# cuda dep would only drag builtin's host-compiler conflict table back in.
#
# The sting in the tail is that the two installed files are looked up in *different*
# default directories.  `tools::addOpenMPDeviceRTL` searches LIBRARY_PATH, the host
# toolchain's file paths and `<bindir>/../lib/<offload triple>`, so the .bc is found
# where cmake puts it -- but clang appends `-lompdevice` unconditionally for
# OpenMP+NVPTX (Clang.cpp:9250-9253) and `lib/<triple>` is not on *that* -L list, so a
# stock-path install alone still fails with "unable to find library -lompdevice".
# Hence cscs_link_ompdevice() below.  The .bc is deliberately left alone: prefix/lib is
# not on its search list.
#
# Upstream gap: spack's `llvm` has no notion of a runtimes cross-build for offload
# targets at all.  Worth reporting to spack-packages; drop this overlay if that lands.

import os

from spack_repo.builtin.packages.llvm.package import Llvm as BuiltinLlvm

from spack.package import *


class Llvm(BuiltinLlvm):
    __doc__ = BuiltinLlvm.__doc__

    #: Triple of the extra runtimes build that produces the OpenMP device runtime.
    #: Must be spelled exactly this way -- it is both the cmake cache-variable infix and
    #: the `lib/<triple>` install directory clang looks the bitcode up in.
    nvptx_runtime_triple = "nvptx64-nvidia-cuda"

    # openmp=project puts openmp in LLVM_ENABLE_PROJECTS, and llvm/runtimes/CMakeLists.txt
    # only walks LLVM_ENABLE_RUNTIMES -- there is no runtimes cross-build to attach an
    # nvptx target to, so the device runtime cannot be produced at all.  Scoped to the
    # specs this overlay targets so it never blocks an unrelated llvm build.
    for _t in ("nvptx", "all"):
        conflicts(
            "openmp=project",
            when="@22: +libomptarget targets={0}".format(_t),
            msg="llvm@22: +libomptarget with an nvptx target needs openmp=runtime: the "
            "OpenMP device runtime is built by a second LLVM_RUNTIME_TARGETS cross-build, "
            "which only exists for LLVM_ENABLE_RUNTIMES entries",
        )
    del _t

    @property
    def needs_nvptx_device_runtime(self):
        """True for the specs whose openmp/device sub-build this overlay adds."""
        spec = self.spec
        if not spec.satisfies("@22: +libomptarget openmp=runtime"):
            return False
        targets = spec.variants["targets"].value
        return "nvptx" in targets or "all" in targets

    def cmake_args(self):
        # Builtin defines cmake_args on the *package* class (llvm/package.py:921) and the
        # recipe has no CMakeBuilder, so spack's generated Adapter forwards here and a
        # plain zero-arg super() is the whole story.  Deliberately no CMakeBuilder
        # subclass in this module: it would win get_builder_class() and bypass both this
        # override and builtin's.
        args = super().cmake_args()

        if self.needs_nvptx_device_runtime:
            triple = self.nvptx_runtime_triple
            args.extend(
                [
                    # 'default' must stay first: it is the existing host runtimes build.
                    self.define("LLVM_RUNTIME_TARGETS", ["default", triple]),
                    self.define("RUNTIMES_{0}_LLVM_ENABLE_RUNTIMES".format(triple), "openmp"),
                ]
            )

        return args

    # A uniquely named callback on purpose.  Re-decorating builtin's post_install would
    # register a *second* callback (spack dedups by function identity), and overriding it
    # without the decorator would leave builtin's function object registered.  Derived
    # callbacks run before base ones, so this lands before builtin's post_install -- which
    # is fine, nothing there touches prefix/lib.
    @run_after("install")
    def cscs_link_ompdevice(self):
        """Expose lib/<triple>/libompdevice.a as lib/libompdevice.a for `-lompdevice`."""
        archive = "libompdevice.a"
        relative_target = join_path(self.nvptx_runtime_triple, archive)
        source = join_path(self.prefix.lib, relative_target)
        link = join_path(self.prefix.lib, archive)

        # No-op for every spec that did not build the device runtime, and idempotent.
        if not os.path.exists(source) or os.path.lexists(link):
            return

        # Relative so the link survives spack's prefix relocation.
        symlink(relative_target, link)
