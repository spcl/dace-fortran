---
name: c23-quality
description: >-
  Rigorous quality-check workflow for a C (C23) file, and modern C23 style rules
  for writing new C. Use whenever the user says "check this C file", "quality-check
  the .c", "review this C file", "lint this C", "run the C checks", "clang-tidy this
  C", "cppcheck this", "sanitize this C", "is this C clean", "run -fanalyzer on this",
  or "write modern C23"/"modernize this C". Runs clang-format, clang-tidy, cppcheck,
  the gcc static analyzer, plus an AddressSanitizer and an UndefinedBehaviorSanitizer
  RUN — six gates, warnings as errors, clean == zero diagnostics from all of them.
---

# c23-quality

Two jobs: (A) QUALITY-CHECK an existing C file through six gates; (B) enforce
modern C23 idioms when WRITING C. `<file>.c` is the placeholder for the target
throughout — swap in the real path. Every command is copy-pasteable. This is C,
not C++: compile with `gcc`/`clang` (not `g++`), `-std=c23`, `--language=c`.
gcc 15+ and clang 19+ accept `-std=c23` with native `constexpr`/`static_assert`.

## Golden rule

**All six gates run. Warnings are errors. A clean pass = zero diagnostics from
every tool + a clean ASan run + a clean UBSan run.** Do not report "looks good"
until all six are green. Fix findings at the source (no suppress-to-pass); the
cppcheck suppressions below are only for third-party/system noise.

Hand-written vs generated code — this changes the clang-tidy/cppcheck check set:
- **Hand-written** code (the default here): the COMPREHENSIVE set below.
- **Machine-GENERATED** code (e.g. codegen output): narrow clang-tidy to
  `clang-analyzer-*` only, `bugprone-*`/style/naming OFF — emitted code trips
  every style rule and most `bugprone-*` are false positives. The path-sensitive
  analyzer is the only useful compile-time gate; the ASan run is the real heap gate.

## A. The six gates (run in this order)

### 1. clang-format (format first, in place)
Use the project's `.clang-format` if one exists at or above the file; else a modern default.
(clang-format's `Standard:` knob is C++-only; for C files there is no `-std` to set.)
```bash
# project style if present, else a modern default (fallback only when none is found):
if find "$(dirname <file>.c)" -maxdepth 4 -name .clang-format | grep -q .; then
  clang-format -i --style=file <file>.c
else
  clang-format -i --style='{BasedOnStyle: LLVM, ColumnLimit: 120}' <file>.c
fi
```

### 2. clang-tidy (COMPREHENSIVE for hand-written C)
For C, drop the C++-only families (`modernize-*`, `cppcoreguidelines-*`) and add `cert-*`.
```bash
clang-tidy \
  --checks='-*,bugprone-*,cert-*,clang-analyzer-*,performance-*,portability-*,readability-*' \
  --header-filter='.*' \
  --warnings-as-errors='*' \
  <file>.c -- -std=c23 -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast
```
`--header-filter=.*` so the file's own headers are checked too. Prefer
`clang-tidy-21` if installed (needed for full C23 parsing). If a CMake compile DB
exists, add `-p <build-dir>` so includes/macros resolve (configure it with
`cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON <build-dir>`).

GENERATED-code variant (narrow set, analyzer only):
```bash
clang-tidy --checks='-*,clang-analyzer-*' --header-filter='$^' <generated>.c -- -std=c23
```

### 3. cppcheck
```bash
cppcheck --enable=warning,performance,portability,style \
  --std=c23 --language=c \
  --inline-suppr --error-exitcode=1 --quiet \
  --suppress=preprocessorErrorDirective \
  --suppress=missingIncludeSystem \
  --suppress='*:*/external/*' \
  <file>.c
```
Suppressions cover third-party/system noise only (vendored-header platform `#error`s,
findings inside `external/`, system-include gaps) — never our own bugs. Add
`--check-level=exhaustive` for a deeper (slower) pass. If a compile DB exists,
prefer `--project=<build-dir>/compile_commands.json` over the bare file.

### 4. gcc static analyzer (syntax-only, no build)
```bash
gcc -std=c23 -fsyntax-only -fanalyzer -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast <file>.c
```
`-fanalyzer` turns on the whole `-Wanalyzer-*` family (double-free, use-after-free,
null-deref, malloc/file leaks, mismatched dealloc, tainted-array-index, write-to-const).
Treat every `-Wanalyzer-*` line as a defect to fix. Add `-Werror` to make it hard-fail.
The analyzer is stronger at higher `-O`, but `-fsyntax-only` keeps it a no-build gate;
use `-O2 -c -o /dev/null` instead if you want the optimizer's extra reach.

### 5. AddressSanitizer — build and RUN once
Static analysis is not enough; the file must actually run under ASan.
```bash
gcc -std=c23 -fsanitize=address -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_asan
ASAN_OPTIONS=detect_leaks=1 /tmp/cq_asan   # exercise the real entry point / test
```
Catches heap/stack/global overflows, use-after-free, use-after-return, leaks.
`detect_leaks=1` is the Linux default. Use `detect_leaks=0` ONLY when the process
is dominated by an external runtime whose leaks you don't own — state the rationale
when you do. For a `dlopen`'d object, build it with the same flags and
`LD_PRELOAD=$(gcc -print-file-name=libasan.so)` into the host process.

### 6. UndefinedBehaviorSanitizer — build and RUN once
```bash
gcc -std=c23 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_ubsan
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/cq_ubsan
```
`halt_on_error=1` so the first UB aborts with a trace — any hit is a bug. Catches
signed-overflow, out-of-range shifts, null-deref, misalignment, bad float↔int
casts, integer div-by-zero, invalid `bool`/enum loads, and `unreachable()` reached.
`-fno-sanitize-recover=all` also aborts on first hit if you prefer it baked into the
binary. ASan and UBSan can share one build (`-fsanitize=address,undefined`); keeping
them separate isolates which sanitizer fired.

**Report** each gate's status. Only "clean" when all six pass with zero output.

## B. Writing modern C23 (lean, use the new keywords)

Prefer plain functions + small concrete structs + tight scope. C23 narrows the gap
to C++ but still has NO templates, NO concepts, NO `constexpr` FUNCTIONS, NO
`consteval` — for generic code use `_Generic`, `typeof`, or macros. Use C23's new
native features in preference to the old C11/C17 workarounds:

- **`constexpr` objects** for true compile-time constants (over `enum`/`static const`):
  `constexpr double PI = 3.141592653589793;`, `constexpr size_t CAP = 256;`. Typed,
  scoped, usable in constant expressions. (`constexpr` applies to objects only; there
  are still no `constexpr` functions — use `static inline`.)
- **`static_assert`, `bool`/`true`/`false`, `nullptr` are now KEYWORDS.** Drop
  `#include <stdbool.h>` and the `<assert.h>` `static_assert` macro. Use
  `static_assert(sizeof(T) == 8, "ABI");` and `nullptr`/`nullptr_t` over `NULL`.
- **`[[nodiscard]]` / `[[maybe_unused]]` / `[[fallthrough]]` / `[[deprecated]]` /
  `[[noreturn]]`** standard attributes over `__attribute__((...))`. Put
  `[[nodiscard]]` on any must-check return (allocators, parse/IO results).
- **`typeof` / `typeof_unqual`** for type-generic locals and macros (drops GNU
  `__typeof__`): `typeof(*p) tmp = *p;`. `typeof_unqual` strips `const`/`volatile`.
- **`_BitInt(N)`** for exact-width integers when `<stdint.h>` widths don't fit
  (e.g. `_BitInt(24)`, `unsigned _BitInt(3)`); otherwise keep `int32_t`/`uint64_t`/
  `size_t`/`ptrdiff_t` from `<stdint.h>` for portable widths.
- **`enum E : underlying_type { ... }`** to fix an enum's underlying type
  (`enum Op : uint8_t { OP_ADD, OP_MUL };`) — stable size, no int promotion surprises.
- **`auto`** type inference for obvious local types (`auto it = find(...);`) — keep
  it for locals whose type is noise, not for public signatures.
- **`unreachable()`** (from `<stddef.h>`) to mark truly impossible branches; pairs
  with UBSan, which traps if one is actually reached.
- **`#embed "data.bin"`** to inline binary/asset data instead of an `xxd`-generated array.
- **Binary literals `0b1010`** for bitmasks/flags where hex is less readable.
- **No silent implicit conversions — cast EXPLICITLY.** C has no `static_cast`, so
  write every lossy / narrowing / sign-changing / int↔float conversion as a deliberate
  `(type)` cast so the intent (and the truncation) is visible at the call site. Watch
  the usual C traps: integer promotions, `unsigned`/`signed` mixing, `size_t` vs `int`,
  `double`→`float`, implicit `int` from a bool context. The
  `-Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast`
  flags above make implicit conversions fail the build — fix them with an explicit cast
  at the source, never by silencing the warning. Keep casts rare and intentional; a
  cast you cannot justify is usually a type or design bug.

Still-valid C guidance (unchanged by C23):
- **`const` and `restrict` correctness** — `const` on non-written pointees; `restrict`
  on non-aliasing pointer params in hot paths (only when aliasing is truly impossible).
- **Designated initializers** with `= {0}` zeroing the rest: never leave fields indeterminate.
- **`static inline` functions over function-like macros** — no double-evaluation, real
  types. Reserve macros for token pasting, `X`-macros, conditional compilation.
- **Check every return code** (`malloc`, `realloc`, `fopen`, `snprintf`, `pthread_*`);
  mark the APIs `[[nodiscard]]`.
- **`sizeof(*ptr)` in allocations**, not the type name: `p = malloc(n * sizeof(*p));`
  (or `calloc(n, sizeof(*p))` for overflow-safe zeroing).
- **No VLAs in headers / public interfaces**, and avoid VLAs generally.
- **Minimal scope for declarations** — declare at first use, initialize on declaration,
  loop counters inside the `for`; `static` (internal linkage) for anything not exported.

After writing or modernizing, run all six gates in section A on the result.

- **Comments: zero fluff, ratio <= 0.2.** At most 1 comment line per 5 lines of code, per
  file and per block. Explain only the non-obvious **why**, never the **what**; never restate
  signatures or types; single line where possible. A long rationale over a short definition
  belongs in the commit message, not the source — leave one line saying which.

## References

Consulted 2026-08-04:
- Clang-Tidy checks & usage — https://clang.llvm.org/extra/clang-tidy/
- Cppcheck manual — https://cppcheck.sourceforge.io/manual.html
- GCC `-fanalyzer` / `-Wanalyzer-*` options — https://gcc.gnu.org/onlinedocs/gcc/Static-Analyzer-Options.html
- GCC sanitizer (ASan/UBSan) instrumentation flags — https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
- Clang UndefinedBehaviorSanitizer — https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
- "A gentle introduction to static analyzers for C" (nrk) — https://nrk.neocities.org/articles/c-static-analyzers
- Chris Wellons / nullprogram, modern C practices — https://nullprogram.com/blog/2023/10/08/
- C23 language changes (canonical feature list) — https://en.cppreference.com/w/c/23
- C23 status / gcc & clang support — https://gcc.gnu.org/c99status.html and https://clang.llvm.org/c_status.html
