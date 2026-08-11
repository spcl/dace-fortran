---
name: hip-quality
description: >-
  Rigorous quality-check workflow for a HIP/ROCm file, and correctness rules for
  writing new AMD device code. Use whenever the user says "check this HIP", "quality-check
  the .hip", "review this HIP kernel", "lint this HIP", "run the HIP checks", "sanitize
  this HIP kernel", "run ROCm ASan", "is this HIP clean", or "write modern HIP"/"modernize
  this HIP kernel". Runs clang-format, a warnings-as-errors hipcc compile, clang-tidy,
  plus RUNS under ROCm device AddressSanitizer and a serialized-dispatch pass — five
  gates, warnings are errors, clean == zero diagnostics from all of them.
---

# hip-quality

Two jobs: (A) QUALITY-CHECK an existing HIP file through five gates; (B) enforce
correct device code when WRITING HIP. `<file>.hip` is the placeholder for the target
throughout (`.cpp` compiled with hipcc works the same) — swap in the real path.

Host-side C++ in a `.hip` is still C++: `cpp-quality` Section B applies to it
unchanged. Read `cuda-quality` alongside this: Section B.1 (error handling) and B.2
(streams) transfer one-for-one with the names changed, and are not repeated in full
here.

## Golden rule

**All five gates run. Warnings are errors. A clean pass = zero diagnostics from
every tool + a clean device-ASan run + a clean serialized-dispatch run.** Do not
report "looks good" until all five are green, and never call a kernel correct on the
strength of a matching numeric result alone.

⛔ **An unchecked HIP API call is a defect by itself.** Same reasoning as CUDA:
the call returns a code nobody reads, the kernel does not run, and the buffer keeps
whatever it held.

⛔ **ROCm's tooling is NOT a feature-for-feature match for compute-sanitizer. Say so
in the report rather than implying equivalent coverage.** What exists:

| CUDA tool | ROCm equivalent | Status |
|---|---|---|
| `compute-sanitizer --tool memcheck` | device AddressSanitizer (`-fsanitize=address`) | real, needs an xnack-capable GPU |
| `compute-sanitizer --tool racecheck` | — | **no equivalent**; review shared-memory (LDS) sync by hand |
| `compute-sanitizer --tool initcheck` | — | **no equivalent**; poison buffers yourself (below) |
| `compute-sanitizer --tool synccheck` | — | **no equivalent**; review barrier uniformity by hand |
| `CUDA_LAUNCH_BLOCKING=1` | `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` | real |
| `cuda-gdb` | `rocgdb` | real |
| `ncu` / `nsys` | `rocprofv3` | real |

The three missing tools are why Section B.3's barrier and LDS rules are checked by
reading the code, and why gate 5 exists.

⛔ **First decide which kind of HIP this is** — self-written (full gates + Section B)
versus machine-generated (`clang-analyzer-*` only, no restyling, sanitizer runs are
the real gate, and Section B becomes a review checklist for the GENERATOR). Same
split as `cpp-quality` and `cuda-quality`.

## A. The five gates (run in this order)

### 0. Prerequisite — know the target architecture
```bash
rocminfo | grep -m4 gfx           # or: rocm_agent_enumerator
```
Everything below needs the real `gfx` target. Device ASan additionally needs the
`xnack+` variant of it (e.g. `gfx90a:xnack+`, `gfx942:xnack+`); on a GPU without
xnack support, gate 4 is DEFERRED and must be reported as such.

### 1. clang-format (format first, in place)
```bash
if find "$(dirname <file>.hip)" -name .clang-format | grep -q .; then
  clang-format -i --style=file <file>.hip
else
  clang-format -i --style='{BasedOnStyle: LLVM, Standard: c++20, ColumnLimit: 120}' <file>.hip
fi
```

### 2. hipcc — warnings as errors
```bash
hipcc -std=c++20 --offload-arch=gfx90a -g -O2 \
  -Wall -Wextra -Wconversion -Wsign-conversion -Wdouble-promotion -Werror \
  -c <file>.hip -o /dev/null
```
hipcc is a single clang driver: there is **no `-Xcompiler`**, host and device flags
go on the one command line. `-Werror` therefore covers both halves at once, unlike
nvcc where the front end and ptxas need separate flags.

### 3. clang-tidy (COMPREHENSIVE for hand-written code)
```bash
clang-tidy \
  --checks='-*,bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,portability-*,readability-*,clang-analyzer-*' \
  --header-filter='.*' --warnings-as-errors='*' \
  <file>.hip -- -x hip --offload-arch=<gfx> -nogpulib -std=c++20 -Wall -Wextra
```
⛔ `-nogpulib` is what makes this run at all on a packaged ROCm. Without it clang
fails with "cannot find ROCm device library", and neither `--rocm-path=/opt/rocm`
(the usual guess — `hipconfig --rocmpath` answers `/usr` there) nor the hipconfig
value fixes it, because system clang looks in neither place. clang-tidy does not
link, so the device bitcode is irrelevant to a lint pass; measured 0 errors with
`-nogpulib` against 3 with the `/opt/rocm` pin, on ROCm 7.1 with clang 21.
hipcc IS clang, so clang-tidy parses HIP without special handling. If the
`--offload-arch` device pass is too slow or trips on headers, add `--cuda-host-only`
and report that device code got no clang-tidy coverage.

GENERATED-code variant:
```bash
clang-tidy --checks='-*,clang-analyzer-*' --header-filter='$^' -p <build-dir> <generated>.hip
```

### 4. ROCm device AddressSanitizer — build and RUN
The closest thing to memcheck: instruments host AND device allocations, catching
out-of-bounds and use-after-free on global and (on recent ROCm) LDS memory.
```bash
hipcc -std=c++20 --offload-arch=gfx90a:xnack+ -fsanitize=address -shared-libasan \
  -g -O1 <file>.hip -o /tmp/hipq_asan

HSA_XNACK=1 \
LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  /tmp/hipq_asan
```
Three things are all required and each fails differently if omitted: `xnack+` in
the offload arch (device ASan needs page migration), `HSA_XNACK=1` at run time (a
mismatch aborts at load with a "target ID" error), and the `LD_PRELOAD` of the
ASan runtime when `-shared-libasan` is used. ROCm also ships an ASan-instrumented
copy of its own libraries under `$(hipconfig --rocmpath)/lib/asan` (when the install
has them) — put it first on `LD_LIBRARY_PATH` when the report points inside
rocBLAS/hipBLAS rather than your kernel.

For a kernel `dlopen`ed into a host process (a Python test, a generated library),
build the kernel with the same flags and `LD_PRELOAD` the ASan runtime into the
host — the same technique as `cpp-quality` gate 5.

### 5. Serialized-dispatch run — ordering and attribution
Async dispatch hides both races and error attribution. Run the same binary again
with every kernel and copy serialized:
```bash
AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 AMD_LOG_LEVEL=3 /tmp/hipq_asan
```
`AMD_SERIALIZE_KERNEL=3` waits before AND after each dispatch, so the first failing
kernel is the one named. **A result that changes between this run and the normal
run is a synchronization bug**, not a flake — that difference is the only automated
race signal ROCm gives you, so treat gate 5's disagreement with gate 4 as a
finding, not as noise.

`AMD_LOG_LEVEL=3` prints every HIP API call and its status; grep it for non-zero
statuses when a run "works" but produces wrong numbers.

#### Standing in for the missing initcheck
There is no uninitialized-memory tool, so create the signal by hand: fill every
output buffer with a poison pattern (a signalling NaN, or `0xA5`) before the
kernel, and assert no poison survives. This is what catches "the kernel never
launched" — the failure mode a zero-filled buffer hides, because fresh device
memory reads as zeros and a zero result looks like a plausible answer.

**Report** each gate's status, and name any gate DEFERRED for lack of hardware
(no xnack, no GPU) rather than skipping it silently.

## B. Writing correct HIP

### B.1 Error handling
Identical in substance to `cuda-quality` B.1 with `hip` names — read that section;
the points that matter most:
- Every HIP call is checked, via one `HIP_CHECK(expr)` macro using
  `hipGetErrorString`.
- After every launch: `hipGetLastError()` immediately (configuration), then again
  at the next synchronization point (execution). `hipPeekAtLastError()` when the
  error must not be cleared.
- Errors are sticky. Never swallow one.
- ⛔ **rocPRIM/hipCUB keep CUB's two-call workspace protocol, including the
  null-pointer-means-query rule.** Check the size query, allocate
  `std::max<size_t>(bytes, 1)` (a zero-byte `hipMalloc` yields a null pointer with
  `hipSuccess`), check the allocation, check the work call. A null workspace makes
  the second call silently re-query the size and perform no work, leaving the
  output untouched — which on fresh device memory reads as a clean array of zeros.
- Same for any workspace API: rocBLAS, rocSPARSE, MIOpen.

### B.2 Streams
As `cuda-quality` B.2, with `hipStreamCreateWithFlags(&s, hipStreamNonBlocking)`
opting out of null-stream serialization, and `hipEventRecord` /
`hipStreamWaitEvent` for cross-stream dependencies. Two HIP-specific notes:
- `hipMemcpy` (synchronous form) on some ROCm versions does not synchronize other
  streams the way the CUDA analogue does. Do not lean on it as a barrier; use
  `hipStreamSynchronize` or an event.
- `hipMallocAsync` / stream-ordered allocation is present only on newer ROCm and
  not on every device. Probe for it rather than assuming.

### B.3 Device code — where HIP differs from CUDA most
- ⛔ **`warpSize` is NOT 32.** It is 64 on CDNA (gfx90a, gfx942) and 32 on RDNA
  (gfx10xx, gfx11xx). It is a **runtime** value in HIP, not a compile-time constant
  — `constexpr int kWarp = 32;` is the single most common porting bug, and it
  produces silently wrong reductions rather than a crash. There is no supported
  compile-time replacement: `__AMDGCN_WAVEFRONT_SIZE__` is deprecated ("compile-time
  -constant access to the wavefront size will be removed in a future release") and
  is therefore a hard ERROR under gate 2's `-Werror`, and
  `__builtin_amdgcn_wavefrontsize()` is not a constant expression either. Size LDS
  for the 64 case and read `warpSize` at run time. Verified against ROCm 7.1.
- Lane masks are 64-bit: `__ballot()` returns `unsigned long long`. Code ported
  from CUDA's 32-bit `unsigned` masks truncates silently.
- HIP's `__shfl_*` take a `width` argument and have no `_sync` variants; AMD
  wavefronts do execute in lockstep, so the mask discipline CUDA needs since Volta
  does not apply — but do not write code that depends on that if it must also
  build for NVIDIA.
- `__syncthreads()` must be reached by every thread of the block, as in CUDA.
  With no synccheck to catch it, a barrier under divergent control flow has to be
  found by reading; treat any `__syncthreads()` inside an `if` whose condition is
  not block-uniform as a finding.
- LDS (`__shared__`) races have no automated tool either. Every write-then-read of
  LDS across threads needs a `__syncthreads()` between them; check each one by hand
  and say in the report that you did.
- Do not promote to double by accident (`2.0` vs `2.0f`) —
  `-Wdouble-promotion` in gate 2 is what catches it.
- `__launch_bounds__(maxThreadsPerBlock)` bounds VGPR allocation and prevents
  scratch spills; check occupancy with `rocprofv3`.
- Atomics: prefer `hip::atomic_ref` / `__hip_atomic_*` with an explicit memory
  order and scope. Unsafe fast FP atomics are enabled by
  `-munsafe-fp-atomics` on some targets — never turn that on where bitwise
  reproducibility matters.

### B.4 Build and portability
- List `--offload-arch` explicitly for anything shipped; a code object bundle with
  no image for the present GPU fails at dispatch, not at load.
- Host and device share the same headers and must agree on the C++ standard —
  hipcc otherwise falls back to its own default (currently `gnu++17`), which
  disagrees with a host half built as C++20. CMake's `CMAKE_HIP_STANDARD` is the
  place to set it.
- `roc-obj` / `llvm-objdump --disassemble` on the code object to read generated
  ISA; `rocprofv3` for counters and timeline.
- When porting from CUDA with `hipify-perl` / `hipify-clang`, the output is a
  starting point: re-run every gate on it, and audit warp size, ballot width and
  workspace protocols by hand — hipify does not fix any of the three.

- **Comments: zero fluff, ratio <= 0.2.** At most 1 comment line per 5 lines of code, per
  file and per block. Explain only the non-obvious **why**, never the **what**; never restate
  signatures or types; single line where possible. A long rationale over a short definition
  belongs in the commit message, not the source — leave one line saying which.

After writing or modernizing, run all five gates in section A on the result.
