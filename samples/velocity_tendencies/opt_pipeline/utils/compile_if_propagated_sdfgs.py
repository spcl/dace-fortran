"""Compile a set of propagated SDFGs (4 specialised variants) into an
executable / shared library.

Ported from ``icon-artifacts/velocity/utils/compile_if_propagated_sdfgs.py``
with the per-stage sed/regex source patches and OpenACC-stream rewrites
removed. Those patches fixed up DaCe-generated output for stages 2-9 of the
old pipeline; the baseline ("stage 0") SDFGs produced by
``generate_baselines.py`` come out of DaCe in a shape that ``main_per.cu``
already calls directly, so no post-generation text munging is needed.

What stayed: backend detection, flag / compiler selection, concurrent
compile-and-link, and the codegen-then-link structure.
"""

import os

# Pin ``__DACE_NO_SYNC=1`` BEFORE importing ``dace`` so the codegen's
# ``cudaStreamSynchronize`` / ``cudaDeviceSynchronize`` /
# ``cudaEventSynchronize`` emission sites (gated on
# ``dace.codegen.common.no_sync_emission()``) all elide. The velocity
# pipeline uses explicit per-state sync tasklets in
# ``OffloadVelocityToGPU`` instead. Callers that want sync emission
# back can set ``__DACE_NO_SYNC=0`` before the first import.
os.environ.setdefault("__DACE_NO_SYNC", "1")

import dace
import sys
import typing
from dace.sdfg import infer_types


AMD = (
    os.getenv("__HIP_PLATFORM_AMD__", "0") == "1"
    or os.getenv("HIP_PLATFORM_AMD", "0") == "1"
)

USE_NVHPC = os.getenv("_USE_NVHPC", "0").lower() in ("1", "true", "yes")


# ─── File path helpers ────────────────────────────────────────────────
# DaCe emits .cpp by default for both host and device translation units;
# nvcc is told to treat them as CUDA with ``-x cu`` at compile time.
def _cpu_src(build_loc: str, name: str) -> str:
    return f"{build_loc}/src/cpu/{name}.cpp"


def _gpu_src(build_loc: str, name: str) -> str:
    if AMD:
        return f"{build_loc}/src/cuda/hip/{name}_cuda.cpp"
    # DaCe's CUDA codegen writes the device translation unit as ``.cu``
    # by default (the f2dace fork rewrote it to ``.cpp``; we use
    # upstream, so keep the native extension and let ``_pick_compiler``
    # route it to nvcc).
    return f"{build_loc}/src/cuda/{name}_cuda.cu"


def _header(build_loc: str, name: str) -> str:
    return f"{build_loc}/include/{name}.h"


# ─── Build helpers ────────────────────────────────────────────────────
# nvcc handles .cu (CUDA kernel code); CXX (g++ / hipcc on AMD) handles .cpp.
# ``CXX`` env var wins so callers can swap the host compiler without
# touching the code.
def _cxx() -> str:
    return os.getenv("CXX") or ("hipcc" if AMD else "g++")


def _nvcc() -> str:
    return "hipcc" if AMD else "nvcc"


def _pick_compiler(src: str) -> str:
    return _nvcc() if src.endswith(".cu") else _cxx()


def _get_link_compiler(gpu: bool) -> str:
    """When ``gpu=True`` we always link with nvcc/hipcc. DaCe emits GPU
    runtime calls (``cudaMalloc``, ``DACE_GPU_CHECK``, stream/context
    management) into both the host ``.cpp`` and the kernel ``.cu`` --
    both need to go through the GPU toolchain, and the link command
    carries ``-arch=sm_*`` which only nvcc understands."""
    if not gpu:
        return _cxx()
    return "hipcc" if AMD else "nvcc"


def _get_flags(gpu: bool, release: bool, lib: bool, debuginfo: bool) -> str:
    if gpu and AMD:
        omp_flag = "-fopenmp"
    elif gpu:
        omp_flag = "-Xcompiler=-fopenmp"
    else:
        omp_flag = "-fopenmp"

    if gpu and AMD:
        arch = "gfx942"
        common = f"--offload-arch={arch} -std=c++20 -DNDEBUG"
        if release:
            flags = (
                f"{common} -Wall -Wextra {omp_flag} "
                f"--offload-arch={arch} "
                "-mllvm -amdgpu-early-inline-all=true "
                "-munsafe-fp-atomics "
                "-ffp-contract=fast "
                "-fPIC -O3 -ffast-math "
                "-Wno-unused-parameter -Wno-ignored-attributes -Wno-unused-result"
            )
        else:
            flags = (
                f"{common} -O0 -g -ggdb -fPIC -Wall -Wextra {omp_flag} "
                "-Wno-unused-parameter -Wno-ignored-attributes -Wno-unused-result "
                "-DDACE_VELOCITY_DEBUG"
            )
        if debuginfo:
            flags += " -g"
        if lib:
            flags += " -DNO_SERDE -fPIC -shared"
    elif gpu:
        gencode_num = os.getenv("GENCODE_NUMBER", "90a")
        if gencode_num == "0":
            raise ValueError("Set GENCODE_NUMBER (e.g. 90)")
        nvhpc = "-ccbin=nvc++ " if USE_NVHPC else ""
        suppress = (
            "--diag-suppress 68 --diag-suppress 550 --diag-suppress 20208 "
            "--diag-suppress 1835 --diag-suppress 177 --diag-suppress 20012 "
            "--diag-suppress 1098"
        )
        xcompiler_warns = (
            "-Xcompiler=-Wconversion -Xcompiler=-Wsign-conversion "
            "-Xcompiler=-Wfloat-conversion -Xcompiler=-Wno-unknown-pragmas "
            "-Xcompiler=-faligned-new"
        ) if not USE_NVHPC else ""
        debugflag = "-lineinfo" if debuginfo else ""
        arch_flag = f"-arch=sm_{gencode_num}"
        # nvcc 13.3 hard-errors on a host gcc newer than 15 and this tree is pinned to
        # gcc@16.1.0. ``samples/env.sh`` already carries the override for the dacecache
        # path (CUDAFLAGS / DACE_compiler_cuda_args); this build shells out to nvcc
        # itself and reads neither.
        suppress = f"-allow-unsupported-compiler {suppress}" if not USE_NVHPC else suppress

        if release:
            flags = (
                f"{nvhpc}{suppress} {xcompiler_warns} -DNDEBUG -Xcompiler=-DNDEBUG "
                f"-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-O3 {omp_flag} "
                f"--expt-relaxed-constexpr {arch_flag} "
                f"--use_fast_math -O3 {debugflag} --ftz=true "
                f"--prec-div=false --prec-sqrt=false --fmad=true "
                f"-Xptxas=-O3 -Xptxas=-v -Xcompiler=-march=native "
                f"-Xcompiler=-mtune=native --restrict"
            )
        else:
            # Debug: IEEE-compliant fp math. ``--fmad=false`` disables
            # fused multiply-add (otherwise ``a*b+c`` rounds once; CPU
            # does two rounds), ``--prec-div/--prec-sqrt`` force IEEE
            # division/sqrt, ``--ftz=false`` keeps denormals. Asserts
            # stay on (no ``-DNDEBUG``).
            flags = (
                f"{suppress} {xcompiler_warns} "
                f"-Xcompiler=-Wall -Xcompiler=-Wextra {omp_flag} "
                f"--expt-relaxed-constexpr {arch_flag} "
                f"-O0 -Xcompiler=-g -g -Xcompiler=-O0 -G {debugflag} "
                f"--fmad=false --prec-div=true --prec-sqrt=true --ftz=false "
                f"-DDACE_VELOCITY_DEBUG -Xcompiler=-DDACE_VELOCITY_DEBUG"
            )
        if lib:
            flags += " -DNO_SERDE -std=c++17 -Xcompiler=-fPIC --compiler-options '-fPIC' --shared"
        else:
            flags += " -std=c++20"
    else:
        warns = (
            "-Wconversion -Wno-sign-conversion -Wfloat-conversion "
            "-Wno-unknown-pragmas -faligned-new"
        ) if not USE_NVHPC else ""
        debugflag = "-g" if debuginfo else ""
        if release:
            flags = (
                f"{warns} {debugflag} -std=c++20 -Wall -Wextra "
                f"-Wno-unused-parameter -Wno-unused-variable -O3 -DNDEBUG "
                f"{omp_flag}"
            )
        else:
            flags = (
                f"{warns} -DDACE_VELOCITY_DEBUG -std=c++20 -Wall -Wextra "
                f"-Wno-unused-parameter -Wno-unused-variable "
                f"-Wno-unknown-pragmas -O0 -g -ggdb {debugflag} "
                f"{omp_flag}"
            )

    if AMD:
        flags += " -D__HIP_PLATFORM_AMD__=1"
        flags += " -DHIP_PLATFORM_AMD=1"

    return flags


import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def _compile_and_link(
    sources: typing.List[str],
    includes: str,
    flags: str,
    output: str,
    gpu: bool,
    lib: bool,
    jobs: int = os.cpu_count(),
    sdfg_names: typing.Optional[typing.List[str]] = None,
):
    compile_flags = flags.replace("--shared", "").replace("-shared", "")
    # nvcc-only flags that g++ doesn't understand. When routing a .cpp
    # file through g++, the GPU flag set carries nvcc dialect that
    # needs stripping / unwrapping:
    #   * ``--diag-suppress N``           → drop
    #   * ``-arch=sm_XX``                 → drop
    #   * ``--expt-relaxed-constexpr``    → drop
    #   * ``--use_fast_math`` etc.        → drop
    #   * ``-Xcompiler=X`` / ``-Xptxas=`` → unwrap to ``X`` / drop
    #   * ``-ccbin=...``                  → drop
    import re as _re
    _NVCC_ONLY_TOKENS = _re.compile(
        r"(?:--diag-suppress\s+\S+|"
        r"-allow-unsupported-compiler|"
        r"-arch=sm_\w+|"
        r"--expt-relaxed-constexpr|"
        r"--use_fast_math|"
        r"--ftz=\S+|--prec-div=\S+|--prec-sqrt=\S+|--fmad=\S+|"
        r"-Xptxas=\S+|"
        r"--restrict|"
        r"-lineinfo|"
        r"-G(?=\s|$)|"
        r"-ccbin=\S+)"
    )
    _XCOMPILER = _re.compile(r"-Xcompiler=(\S+)")

    def _flags_for_cxx(f: str) -> str:
        f = _NVCC_ONLY_TOKENS.sub("", f)
        f = _XCOMPILER.sub(r"\1", f)
        return " ".join(f.split())

    objects = []

    # OpenACC Fortran (the velocity_tendencies reference and its serde bridge). Compiled first and
    # strictly in the order given: nvfortran emits a .mod per module and a parallel batch would
    # race on the USE dependencies. nvfortran is the only ACC-offload Fortran compiler here.
    fortran_sources = [s for s in sources if s.endswith((".f90", ".F90"))]
    if fortran_sources:
        gencode_num = os.getenv("GENCODE_NUMBER", "90a").rstrip("a")
        moddir = "codegen/fmod"
        os.makedirs(moddir, exist_ok=True)
        acc_target = "gpu" if gpu else "multicore"
        # fastmath / flushz / fma mirror the DaCe side's --use_fast_math --ftz=true --fmad=true
        # --prec-div=false --prec-sqrt=false, so neither engine is timed under a different
        # floating-point contract than the other.
        gpu_opts = f"cc{gencode_num},fastmath,flushz,fma" if gpu else f"cc{gencode_num}"
        fflags = os.getenv(
            "VT_ACC_FFLAGS",
            f"-O3 -acc={acc_target} -gpu={gpu_opts} -Minfo=accel -module {moddir}",
        )
        for src in fortran_sources:
            # Objects go beside the .mod files, not next to the sources: one of these lives in the
            # parent sample directory and would drop a stray .o into the tracked tree.
            obj = os.path.join(moddir, os.path.basename(os.path.splitext(src)[0]) + ".o")
            cmd = f"nvfortran -c {src} {fflags} -o {obj}"
            print(f"  [FC nvfortran] {src}  ({fflags})")
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if ret.stderr:
                print(ret.stderr, end="", file=sys.stderr)
            if ret.returncode != 0:
                raise RuntimeError(f"FAILED: {src}")
            objects.append(obj)
        sources = [s for s in sources if s not in fortran_sources]

    def compile_one(src):
        obj = os.path.splitext(src)[0] + ".o"
        obj_dir = os.path.dirname(obj)
        if obj_dir:
            os.makedirs(obj_dir, exist_ok=True)
        # In GPU mode, DaCe puts CUDA runtime calls
        # (``cudaMalloc``, ``DACE_GPU_CHECK``, stream/context refs)
        # into the host .cpp as well -- so that file must go through
        # nvcc with ``-x cu`` so nvcc treats it as CUDA (without that
        # flag nvcc delegates .cpp to the host compiler, which doesn't
        # know the CUDA runtime).
        xlang = ""
        if gpu and src.endswith(".cpp"):
            cc = _nvcc()
            xlang = "-x cu"
        else:
            cc = _pick_compiler(src)
        per_flags = (compile_flags if cc.endswith("nvcc") or cc.endswith("hipcc")
                     else _flags_for_cxx(compile_flags))
        cmd = f"{cc} {xlang} -c {src} {includes} {per_flags} -o {obj}"
        ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return src, obj, cc, ret

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(compile_one, src): src for src in sources}
        for fut in as_completed(futures):
            src, obj, cc, ret = fut.result()
            print(f"  [CC {cc}] {src}")
            if ret.stdout:
                print(ret.stdout, end="")
            if ret.stderr:
                print(ret.stderr, end="", file=sys.stderr)
            if ret.returncode != 0:
                raise RuntimeError(f"FAILED: {src}")
            objects.append(obj)

    link_cc = _get_link_compiler(gpu)

    import re as _re
    arch_match = _re.search(r"-arch=sm_\w+", compile_flags)
    arch_flag = arch_match.group(0) if arch_match else ""

    # Per-variant partial-link + symbol localization. Each variant's
    # ``.cpp.o`` and ``_cuda.o`` carry duplicate definitions of DaCe-
    # internal CUDA helpers (``__dace_init_cuda``,
    # ``__dace_runkernel_*``, per-map kernel mangled names, …). To
    # link N variants into a single executable we ``ld -r`` each
    # variant's pair into one ``<sdfg>_merged.o`` and then run
    # ``objcopy --keep-global-symbol`` to keep only the four public
    # names DaCe exposes per SDFG (``__dace_init_<sdfg>``,
    # ``__dace_exit_<sdfg>``, ``__program_<sdfg>``,
    # ``__program_<sdfg>_internal``). Everything else becomes
    # local-only and the multi-variant link cleans up.
    if gpu and sdfg_names:
        objects = _localize_internal_symbols(objects, sdfg_names)

    # With OpenACC Fortran in the mix nvfortran has to drive the link: nvcc never runs nvfortran's
    # device-registration step, so its __fatbinwrap_* for the ACC translation unit goes unresolved.
    # -Mnomain because the entry point is C++, -c++libs for the DaCe host code, -cuda for cudart.
    if fortran_sources:
        gpu_arg = os.getenv("GENCODE_NUMBER", "90a").rstrip("a")
        link_cmd = (f"nvfortran {' '.join(objects)} -acc={'gpu' if gpu else 'multicore'} "
                    f"-gpu=cc{gpu_arg} -cuda -c++libs -mp -Mnomain -o {output}")
        print(f"  [LD nvfortran] {output}")
        if subprocess.run(link_cmd, shell=True).returncode != 0:
            raise RuntimeError(f"FAILED: {link_cmd}")
        return

    link_flags = ""
    if lib:
        link_flags = "-shared" if (AMD or link_cc == _cxx()) else "--shared -Xcompiler=-fPIC"

    if "nvcc" in link_cc:
        link_flags += " -Xcompiler=-fopenmp "
        # The link needs the same host driver as the compiles: objects built by nvc++ pull in
        # nvhpc runtime symbols (__nv_init_env, __c_mset4, __kmpc_*) that g++ cannot resolve.
        if USE_NVHPC:
            link_flags += " -ccbin=nvc++ "
        # OpenACC Fortran objects need the ACC runtime and libnvf pulled in through the host
        # driver; without -acc the device-side data clauses link but never initialise.
        if fortran_sources:
            gpu_arg = os.getenv("GENCODE_NUMBER", "90a").rstrip("a")
            link_flags += f" -Xcompiler=-acc -Xcompiler=-gpu=cc{gpu_arg} -Xlinker=-lnvf "
    else:
        link_flags += " -fopenmp "

    link_cmd = f"{link_cc} {' '.join(objects)} {arch_flag} {link_flags} -o {output}"
    print(f"  [LD {link_cc}] {output}")
    ret = subprocess.run(link_cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"FAILED: {link_cmd}")


def _localize_internal_symbols(
    objects: typing.List[str], sdfg_names: typing.List[str],
) -> typing.List[str]:
    """Per-SDFG: partial-link the ``.cpp.o`` + ``_cuda.o`` into
    ``<sdfg>_merged.o``, then run ``objcopy --localize-symbol`` to
    downgrade just the symbols that COLLIDE across variants. Symbols
    that aren't multi-defined keep their original visibility, so the
    linker's section-gc doesn't discard NVTX init / fatbin / nvcc
    auxiliary sections that aren't actually duplicated.

    Returns a new ``objects`` list with each variant's pair replaced
    by the merged object."""
    if len(sdfg_names) <= 1:
        return list(objects)

    # Step 1: ld -r every variant's pair.
    merged_paths: typing.Dict[str, str] = {}
    for sdfg_name in sdfg_names:
        cpu_obj = next(
            (o for o in objects
             if o.endswith(f'/{sdfg_name}.o') and '/src/cpu/' in o),
            None)
        cuda_obj = next(
            (o for o in objects
             if o.endswith(f'/{sdfg_name}_cuda.o') and '/src/cuda/' in o),
            None)
        if cpu_obj is None or cuda_obj is None:
            continue
        merged = os.path.join(os.path.dirname(cpu_obj),
                               f'{sdfg_name}_merged.o')
        ld_cmd = f"ld -r {cpu_obj} {cuda_obj} -o {merged}"
        ret = subprocess.run(ld_cmd, shell=True)
        if ret.returncode != 0:
            raise RuntimeError(f"FAILED: {ld_cmd}")
        merged_paths[sdfg_name] = merged

    # Step 2: collect the set of GLOBAL ``T`` (text) symbols defined in
    # each merged object. Anything that appears in more than one of
    # them is a collision -- localize it in every merged object.
    import collections as _collections
    sym_count = _collections.Counter()
    sym_per_obj: typing.Dict[str, typing.Set[str]] = {}
    for sdfg_name, merged in merged_paths.items():
        defined = set()
        nm = subprocess.run(f'nm --defined-only {merged}', shell=True,
                             capture_output=True, text=True)
        for line in (nm.stdout or '').splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) >= 3 and parts[1] in ('T', 'D', 'B', 'R'):
                defined.add(parts[2])
        sym_per_obj[sdfg_name] = defined
        for s in defined:
            sym_count[s] += 1

    # Symbols that appear in 2+ merged objects.
    colliders = {s for s, c in sym_count.items() if c >= 2}
    # The 4 public per-SDFG entry points are unique by construction
    # (they include the SDFG name); they can't be in colliders, so
    # nothing to do for those.
    if not colliders:
        out = list(objects)
        for sdfg_name, merged in merged_paths.items():
            cpu_obj = next(o for o in objects
                            if o.endswith(f'/{sdfg_name}.o')
                            and '/src/cpu/' in o)
            cuda_obj = next(o for o in objects
                            if o.endswith(f'/{sdfg_name}_cuda.o')
                            and '/src/cuda/' in o)
            out = [o for o in out if o != cpu_obj and o != cuda_obj]
            out.append(merged)
        return out

    # Step 3: objcopy --localize-symbol per colliding symbol per
    # merged object. Localizing in EVERY variant means each variant
    # carries a private copy that resolves only intra-object;
    # cross-object references already bound through ``ld -r`` stay
    # internal too. Public per-SDFG entries (``__dace_init_<name>``,
    # ``__program_<name>``, ...) aren't colliders so stay global.
    for sdfg_name, merged in merged_paths.items():
        # Only localize symbols actually defined in THIS merged
        # object (objcopy errors on missing names).
        local_set = colliders & sym_per_obj[sdfg_name]
        if not local_set:
            continue
        # Pass via -L=name; objcopy's CLI accepts repeated --localize-
        # symbol flags or a --localize-symbols=file. Use a temp file
        # for cleanliness.
        listfile = merged + '.localize.txt'
        with open(listfile, 'w') as f:
            for s in sorted(local_set):
                f.write(s + '\n')
        objcopy_cmd = f"objcopy --localize-symbols={listfile} {merged}"
        ret = subprocess.run(objcopy_cmd, shell=True)
        if ret.returncode != 0:
            raise RuntimeError(f"FAILED: {objcopy_cmd}")

    # Step 4: build the replacement object list.
    out = list(objects)
    for sdfg_name, merged in merged_paths.items():
        cpu_obj = next(o for o in objects
                        if o.endswith(f'/{sdfg_name}.o')
                        and '/src/cpu/' in o)
        cuda_obj = next(o for o in objects
                        if o.endswith(f'/{sdfg_name}_cuda.o')
                        and '/src/cuda/' in o)
        out = [o for o in out if o != cpu_obj and o != cuda_obj]
        out.append(merged)
    return out


# ─── Main entry point ────────────────────────────────────────────────
def compile_if_propagated_sdfgs(
    sdfgs: typing.List[dace.SDFG],
    gpu: bool,
    release: bool,
    generate_code: bool,
    lib: bool,
    main_name: typing.Optional[str],
    stage: int,  # unused; kept for API compat with pipeline callers
    debuginfo: bool,
    extra_sources: typing.Optional[typing.Iterable[str]] = None,
    extra_include_dirs: typing.Optional[typing.Iterable[str]] = None,
    extra_defines: typing.Optional[typing.Iterable[str]] = None,
    output: typing.Optional[str] = None,
    post_codegen_hook: typing.Optional[
        typing.Callable[[typing.List[dace.SDFG]], None]
    ] = None,
):
    """Codegen each SDFG and compile + link with ``main_name``.

    ``stage`` is kept in the signature for parity with the icon-artifacts
    pipeline; baseline SDFGs are treated as ``stage == 0`` and do not trigger
    any of the source-patch fix-ups that later stages required.

    ``post_codegen_hook`` runs after all SDFG codegen and before the
    compile step. Use it to regenerate driver-dependent artifacts that
    depend on the freshly-emitted variant headers -- e.g. the call-site
    macro in ``include/call_velocity.h`` (the signature can drift between
    branches, so regenerating here keeps ``main.cpp`` in sync).
    """
    # Compute the flags ONCE so the DaCe codegen path and the outer
    # ``_compile_and_link`` invocation see identical compile arguments.
    # Drift between the two used to manifest as numerics differing
    # between DaCe-built ``sdfg.compile()`` artefacts and the merged
    # multi-variant binary -- because nvcc/hipcc was invoked with
    # different ``--fmad`` / ``--use_fast_math`` / ``--offload-arch``
    # settings on each path.
    flags = _get_flags(gpu, release, lib, debuginfo)

    # -1 = default-stream mode: DaCe creates no streams and every
    # kernel launch / async copy takes nullptr.
    dace.Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    # Pin the legacy CUDA codegen. The new-gpu-codegen-dev branch ships
    # an ``experimental`` codegen that's the default but is producing
    # wrong numerics on the velocity SDFGs; ``legacy`` is the stable
    # path. Guarded for older DaCe branches that don't have the key.
    try:
        dace.Config.set("compiler", "cuda", "implementation", value="legacy")
    except (KeyError, ValueError):
        pass

    # Block dimensions: caller can override via ``_TBLOCK_DIMS`` env (the
    # platform-setup scripts use ``32,4,1`` on Beverin / MI300A and
    # ``32,16,1`` on Daint / GH200). Default matches the historical
    # ``32,16,1`` so behavior is unchanged when the env is unset.
    block_dims = os.getenv("_TBLOCK_DIMS", "32,16,1")
    dace.Config.set("compiler", "cuda", "default_block_size", value=block_dims)

    if gpu and AMD:
        dace.Config.set("compiler", "cuda", "backend", value="hip")
        dace.Config.set("compiler", "cuda", "path", value="/opt/rocm")
        dace.Config.set("compiler", "cuda", "hip_arch", value="gfx942")
        dace.Config.set("compiler", "cuda", "hip_args", value=flags)
    elif gpu:
        # NVIDIA: feed the same flag string to DaCe's CUDA codegen.
        dace.Config.set("compiler", "cuda", "args", value=flags)

    # CPU compile path (host main / non-GPU SDFGs) goes through
    # ``compiler.cpu.args``. Compute via _get_flags(gpu=False) so the
    # debuginfo / release / lib branches stay in sync with the GPU
    # path's matching switch.
    cpu_flags = _get_flags(False, release, lib, debuginfo)
    dace.Config.set("compiler", "cpu", "args", value=cpu_flags)

    sources: typing.List[str] = list(extra_sources) if extra_sources is not None else []
    headers = [f"-I{d}" for d in (extra_include_dirs or ["include"])]
    headers += [f"-D{d}" for d in (extra_defines or [])]

    from dace.codegen import codegen, compiler

    for sdfg in sdfgs:
        name = sdfg.name
        build_loc = sdfg.build_folder

        if generate_code:
            try:
                sdfg.fill_scope_connectors()
                infer_types.infer_connector_types(sdfg)
                infer_types.set_default_schedule_and_storage_types(sdfg, None)
                sdfg.expand_library_nodes()
                infer_types.infer_connector_types(sdfg)
                infer_types.set_default_schedule_and_storage_types(sdfg, None)
                program_objects = codegen.generate_code(sdfg, validate=False)
            except Exception:
                fpath = os.path.join("_dacegraphs", "failing.sdfgz")
                os.makedirs("_dacegraphs", exist_ok=True)
                sdfg.save(fpath, compress=True)
                print(f"Failing SDFG saved: {os.path.abspath(fpath)}")
                raise

            compiler.generate_program_folder(sdfg, program_objects, build_loc)

        cpu_file = _cpu_src(build_loc, name)
        gpu_file = _gpu_src(build_loc, name)

        sources.append(cpu_file)
        if gpu:
            # DaCe emits the device-side code into ``src/cuda/*_cuda.cu``
            # whenever any node in the SDFG has a GPU schedule. Stage 4
            # is where we first turn on GPU schedules, so the .cu must
            # join the compile pool from stage 4 onwards. (The old
            # ``stage >= 5`` gate predated moving the offload step
            # earlier.)
            sources.append(gpu_file)

    if post_codegen_hook is not None:
        post_codegen_hook(sdfgs)

    if main_name is not None:
        sources.append(main_name)
    elif not lib:
        sources.append("main_gpu.cpp" if gpu else "main.cpp")

    # ``flags`` already computed at the top of this function and pushed
    # into dace.Config so the codegen path uses identical arguments.
    dace_include = os.path.dirname(dace.__file__) + "/runtime/include/"
    includes = " ".join(
        [f"-I{sdfg.build_folder}/include" for sdfg in sdfgs]
        + [f"-I{dace_include}"]
        + headers
    )

    if output is None:
        if lib:
            output = "libvelocity_gpu.so" if gpu else "libvelocity_cpu.so"
        else:
            output = "velocity_gpu" if gpu else "velocity_cpu"

    print(f"\nBackend: {'HIP (AMD)' if AMD else 'CUDA (NVIDIA)' if gpu else 'CPU'}")
    print(f"CXX: {_cxx()}  nvcc: {_nvcc() if gpu else '(unused)'}")
    print(f"Output: {output}")
    print(f"Sources: {len(sources)} files\n")

    _compile_and_link(sources, includes, flags, output, gpu, lib,
                      sdfg_names=[s.name for s in sdfgs])
