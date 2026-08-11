---
name: cuda-quality
description: >-
  Rigorous quality-check workflow for a CUDA file, and correctness rules for writing
  new device code. Use whenever the user says "check this CUDA", "quality-check the
  .cu", "review this kernel", "lint this CUDA", "run the CUDA checks", "sanitize this
  kernel", "run compute-sanitizer", "racecheck this", "is this kernel clean", or
  "write modern CUDA"/"modernize this kernel". Runs clang-format, a warnings-as-errors
  nvcc compile, clang-tidy, plus RUNS under all four compute-sanitizer tools
  (memcheck, racecheck, initcheck, synccheck) — seven gates, warnings are errors,
  clean == zero diagnostics from all of them.
---

# cuda-quality

Two jobs: (A) QUALITY-CHECK an existing CUDA file through seven gates; (B) enforce
correct device code when WRITING CUDA. `<file>.cu` is the placeholder for the target
throughout — swap in the real path. Every command is copy-pasteable.

Host-side C++ in a `.cu` is still C++: `cpp-quality` Section B applies to it
unchanged. This skill covers what is specific to device code and the CUDA runtime.

## Golden rule

**All seven gates run. Warnings are errors. A clean pass = zero diagnostics from
every tool + a clean run under memcheck, racecheck, initcheck AND synccheck.** Do
not report "looks good" until all seven are green, and never call a kernel correct
on the strength of a matching numeric result alone — a kernel that never launched
and a kernel that raced both produce a plausible-looking array.

⛔ **An unchecked CUDA API call is a defect by itself, reported like any other
finding.** Device failures are silent by construction: the API returns a code
nobody reads, the kernel does not run, and the output buffer keeps whatever it
held. Section B.1 is the rule; the sanitizer gates are how you catch what it misses.

⛔ **First decide which kind of CUDA this is — it changes the clang-tidy set AND
whether Section B applies:**

- **Agent SELF-WRITTEN code** (the default): the full gates below, AND the writing
  rules in Section B are in force.

- **Machine-GENERATED outside code** (emitted by a tool the agent does not author —
  e.g. DaCe codegen in `.dacecache/<name>/src/cuda/*.cu`): treat as OPAQUE. Narrow
  clang-tidy to `clang-analyzer-*`, skip the format gate, and do NOT restyle it —
  fix its generator instead. **The compute-sanitizer runs are the real gate**, and
  they are the whole point: generated device code is exactly where an unchecked
  allocation or a missing event turns into a silent wrong answer. Section B is then
  a checklist for reviewing the GENERATOR, not the emitted file.

## A. The seven gates (run in this order)

### 0. Prerequisite — build with line info
Every gate after 2 wants source attribution. Build the binary under test with:
```bash
nvcc -std=c++20 -arch=native -lineinfo -g -O2 <file>.cu -o /tmp/cudaq_bin
```
`-lineinfo` (cheap, keeps optimization) is what makes a sanitizer report name a
line. Use `-G` (full device debug) only when a report is still unattributable —
it disables optimization and can make a race disappear.

### 1. clang-format (format first, in place)
```bash
if find "$(dirname <file>.cu)" -name .clang-format | grep -q .; then
  clang-format -i --style=file <file>.cu
else
  clang-format -i --style='{BasedOnStyle: LLVM, Standard: c++20, ColumnLimit: 120}' <file>.cu
fi
```

### 2. nvcc — warnings as errors, host and device
```bash
nvcc -std=c++20 -arch=native -lineinfo \
  -Werror all-warnings \
  -Xptxas=-Werror -Xptxas=-warn-spills -Xptxas=-warn-lmem-usage \
  -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wconversion -Xcompiler=-Wsign-conversion \
  -c <file>.cu -o /dev/null
```
`-Werror all-warnings` is the nvcc front end; `-Xptxas=-Werror` is the device
back end — they are different compilers and each has its own warning set, so both
are needed. `-warn-spills` catches register spills to local memory, which are a
correctness-adjacent perf cliff worth seeing. One `-Xcompiler` per flag: nvcc
splits the comma-separated form on commas.

Add `--expt-relaxed-constexpr` if device code calls host `constexpr` functions
(`std::numeric_limits` and friends), and `-arch=sm_XX` explicitly when the build
machine's GPU is not the target.

### 3. clang-tidy (COMPREHENSIVE for hand-written code)
clang parses CUDA natively; it does not need nvcc.
```bash
clang-tidy \
  --checks='-*,bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,portability-*,readability-*,clang-analyzer-*' \
  --header-filter='.*' --warnings-as-errors='*' \
  <file>.cu -- -x cuda --cuda-gpu-arch=<same sm as gate 0> \
  --cuda-path="$(dirname "$(dirname "$(command -v nvcc)")")" -std=c++20 -Wall -Wextra
```
Analyze for the arch you are building for, not a pinned one, or an arch-specific
finding is missed. If clang cannot parse the toolkit's headers (common when nvcc is
much newer than clang, which carries its own table of known CUDA versions),
`--cuda-host-only` is the documented fallback — but measured against nvcc 13.1 with
clang 21 it does NOT clear the errors either, and then the gate is DEFERRED. Either
way say in the report that device code got no clang-tidy coverage.

GENERATED-code variant:
```bash
clang-tidy --checks='-*,clang-analyzer-*' --header-filter='$^' -p <build-dir> <generated>.cu
```

### 4. compute-sanitizer memcheck — build and RUN
The CUDA equivalent of ASan. Out-of-bounds and misaligned device accesses, plus
leaks and API errors.
```bash
compute-sanitizer --tool memcheck --leak-check full --report-api-errors all \
  --error-exitcode 1 /tmp/cudaq_bin
```
`--error-exitcode 1` is REQUIRED: without it compute-sanitizer exits 0 even when it
reported errors, so a CI gate silently passes. `--report-api-errors all` is what
surfaces the failed `cudaMalloc` nobody checked.

### 5. compute-sanitizer racecheck — shared-memory races
```bash
compute-sanitizer --tool racecheck --racecheck-report all \
  --error-exitcode 1 /tmp/cudaq_bin
```
Catches shared-memory hazards a missing or divergent `__syncthreads()` leaves
behind. `--racecheck-report all` includes hazards it considers benign — read
them, do not filter them out by default.

### 6. compute-sanitizer initcheck — uninitialized global memory
```bash
compute-sanitizer --tool initcheck --track-unused-memory yes \
  --error-exitcode 1 /tmp/cudaq_bin
```
This is the gate that catches "the kernel never ran": reading device memory that
nothing wrote is exactly what a skipped launch or a no-op library call leaves.

### 7. compute-sanitizer synccheck — invalid synchronization
```bash
compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/cudaq_bin
```
Illegal `__syncthreads()` / `__syncwarp()` / cooperative-groups usage —
barriers not reached by every thread that must reach them.

#### Running the sanitizers on a kernel loaded by another process
When the kernel is `dlopen`ed into a host (a Python test, a generated DaCe
library), sanitize the host process and let it follow children:
```bash
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 1 \
  python -m pytest -q path/to/test.py
```
`--target-processes all` is mandatory under pytest, which forks. Add
`--force-blocking-launches yes` (or `CUDA_LAUNCH_BLOCKING=1`) when a report points
at a launch site instead of the faulting kernel — async launches otherwise
attribute the error to whatever call happened to be next.

**Report** each gate's status. Only "clean" when all seven pass with zero output.
State explicitly if a gate was DEFERRED (no GPU on the box, clang cannot parse the
toolkit headers) — never skip one silently, and never substitute "the numbers
matched" for a sanitizer run.

## B. Writing correct CUDA

### B.1 Error handling — the non-negotiable part
- **Every CUDA runtime and driver call is checked.** No exceptions, including
  `cudaFree` and the ones in teardown paths. Use one macro:
  ```cpp
  #define CUDA_CHECK(expr)                                                            \
      do {                                                                            \
          const cudaError_t status_ = (expr);                                         \
          if (status_ != cudaSuccess) {                                               \
              std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,      \
                           #expr, cudaGetErrorString(status_));                       \
              std::abort();                                                           \
          }                                                                           \
      } while (false)
  ```
  A project with its own macro (DaCe's `DACE_GPU_CHECK`) uses that one. An
  allocation whose result is never inspected is the single most common source of
  silent wrong answers on a loaded multi-tenant GPU.
- **After every kernel launch, check twice.** `cudaGetLastError()` immediately
  (bad launch configuration — too many threads, too much shared memory) and the
  next synchronization point (execution errors: illegal address, assertion).
  A launch that fails configuration validation never runs and reports nothing.
- **Most CUDA errors are sticky.** Once one occurs the context is poisoned and
  every later call in that process returns the same error. Never swallow one to
  "keep going"; the first swallowed error makes every later result meaningless.
- ⛔ **CUB and Thrust: `d_temp_storage == nullptr` means "only tell me the size".**
  The two-call protocol is a trap when either half is unchecked:
  ```cpp
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr, bytes, /*...*/));  // query
  void *storage = nullptr;
  CUDA_CHECK(cudaMalloc(&storage, std::max<size_t>(bytes, 1)));             // never 0
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(storage, bytes, /*...*/));  // work
  ```
  Three things must all hold. If the query fails, `bytes` stays 0. If `bytes` is 0,
  `cudaMalloc(&p, 0)` hands back a **null pointer with `cudaSuccess`** — hence the
  `max(bytes, 1)`. And if the allocation fails, `storage` stays null. In all three
  cases the second call sees a null pointer, quietly re-runs the size query, and
  **does no reduction at all** — leaving the output exactly as it was found. Fresh
  device memory reads as zeros, so the symptom is a plausible all-zero result with
  no error anywhere. Never let that pointer be null at the work call.
- Same shape in any library with a workspace protocol: cuBLAS, cuSPARSE, cuDNN,
  cuTENSOR. Check the workspace query, check the allocation, check the call.

### B.2 Streams and synchronization
- **A non-blocking stream does not synchronize with the legacy default stream.**
  `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` opts out of the implicit
  legacy-null-stream serialization. Code that mixes such a stream with `nullptr`
  (or with `cudaStreamPerThread`) needs explicit events; the null stream buys
  nothing there.
- Cross-stream dependencies are expressed with `cudaEventRecord` +
  `cudaStreamWaitEvent`, never with a bare `cudaDeviceSynchronize()` sprinkled to
  make a bug go away.
- ⛔⭐ **Never special-case the default stream in your own bookkeeping.** Wherever a
  code path says "this is `nullptr`, not a real stream, skip it", that skip is where
  the missing synchronization will be. Measured 2026-08-05 in DaCe: work pinned to
  the null stream was excluded from the set of streams synchronized at the end of a
  state, so nothing ordered it against the next state or against the host reading
  the result. It survived only because `cudaFree` synchronizes and async copies to
  pageable memory stage; pooled or pinned memory removes both. Render every stream
  through ONE function that maps "which stream" to its expression, `nullptr`
  included, and let it be passed to work and synchronized like any other.
- If a pass reassigns which stream an operation runs on, re-derive the events after
  it: two operations on the same stream need no event, and moving one later drops a
  dependency that was implicit in stream order. (In the case above this turned out
  NOT to be the bug — the rewrite moved whole connected components, so no edge ever
  straddled two streams. Check which of the two you actually have before fixing.)
- Read a device result only after synchronizing the stream that produced it
  (`cudaStreamSynchronize`), or after an event recorded on it.
- `cudaMemcpyAsync` is only genuinely async from **pinned** host memory
  (`cudaHostAlloc`/`cudaMallocHost`). From pageable memory it stages through a
  driver buffer, which hides ordering bugs on one machine and exposes them on
  another.
- Per-thread default stream (`--default-stream per-thread`) changes the semantics
  of every `nullptr` stream in the TU. Pick one convention per project.

### B.3 Device code
- **`warpSize` is 32 on every NVIDIA GPU, but do not hard-code warp-synchronous
  behaviour.** Since Volta, threads in a warp can diverge and reconverge
  independently: any exchange between lanes needs an explicit `__shfl_*_sync`,
  `__ballot_sync`, `__any_sync` with a correct mask, or `__syncwarp()`. Code that
  relies on lockstep lanes is broken on sm_70+ even when it appears to work.
- `__syncthreads()` must be reached by **every** thread of the block. A barrier
  inside a divergent `if` is undefined behaviour — synccheck finds it.
- Prefer cooperative groups (`cg::thread_block`, `cg::tiled_partition`) over raw
  masks when the partitioning is not trivially constant.
- **Do not promote to double by accident.** `x * 2.0` in a `float` kernel drags the
  whole expression through FP64, which on a consumer GPU is 1/64 rate. Write `2.0f`,
  and use `-Xcompiler=-Wdouble-promotion` plus `__fmaf_rn`/`fmaf` where the fused
  form is intended.
- Grid-stride loops over one-element-per-thread indexing, so a kernel is correct
  for any launch geometry.
- `__restrict__` on non-aliasing pointer parameters, `const` on read-only ones —
  this is what enables `__ldg`/read-only-cache paths.
- `__launch_bounds__(maxThreads, minBlocksPerSM)` when the launch geometry is
  known: it bounds register allocation and prevents spills (gate 2 warns on them).
- Shared memory: declare sizes explicitly, and for dynamic `extern __shared__`
  remember there is exactly ONE such array — carve multiple buffers out of it by
  offset, with alignment respected.
- Use `cuda::std::` / libcu++ (`cuda/std/atomic`, `cuda/std/limits`) in device code
  rather than host-only `std::`.
- Atomics: `atomicAdd` on `double` needs sm_60+; `cuda::atomic_ref` with an
  explicit memory order and scope (`cuda::thread_scope_block`/`device`) says what
  is actually meant. Floating-point atomics are non-deterministic in ordering —
  never use them where bitwise reproducibility is required.
- Bounds-check every global write against the real extent, not against the launch
  geometry, whenever the grid is rounded up.

### B.4 Build and portability
- `-arch=native` resolves at compile time on the build machine; it silently falls
  back to a default when detection fails. For anything shipped, list the
  architectures explicitly (`-gencode arch=compute_80,code=sm_80 ...` plus a PTX
  fallback `code=compute_80`) — a fatbin with no compatible image makes
  `cudaGetDeviceCount` report **no CUDA-capable device**, which reads as a broken
  driver rather than a build problem.
- Host and device halves of a program must be compiled with the same C++ standard
  and the same header-affecting flags; they share the same headers, and disagreeing
  is an ODR violation. CMake does NOT propagate `CMAKE_CXX_FLAGS` to nvcc.
- `-Xptxas -v` to read register counts, spill stores/loads and shared-memory usage
  when tuning; `cuobjdump -sass` / `nvdisasm` to see what actually got generated.
- Profiling belongs to `ncu` (kernel counters) and `nsys` (timeline), not to
  hand-rolled `cudaEvent` timing sprinkled through the kernel.

- **Comments: zero fluff, ratio <= 0.2.** At most 1 comment line per 5 lines of code, per
  file and per block. Explain only the non-obvious **why**, never the **what**; never restate
  signatures or types; single line where possible. A long rationale over a short definition
  belongs in the commit message, not the source — leave one line saying which.

After writing or modernizing, run all seven gates in section A on the result.
