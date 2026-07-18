# QE Kernel E2E: Parse, Bindings, Compile, Save SDFG

End-to-end guide for taking a Quantum ESPRESSO Fortran kernel through the ``dace-fortran`` bridge: parse the source, emit Fortran bindings, compile the SDFG to a shared object, persist the SDFG to disk for later reuse.

Captured as a **task document** rather than in the main ``README`` (README already long) -- surface the pointer from there.

## Pre-requisites

* ``flang-new-21`` on ``PATH`` (HLFIR frontend) -- CI installs it from ``apt.llvm.org``, see ``.github/workflows/fortran-ci.yml``.
* ``gfortran`` on ``PATH`` (reference compile).
* Python 3.13 (the bridge's binary build targets ``cpython-313``).
* The bridge built: ``cd dace_fortran/build && cmake .. && make -j``.

## Step 1 — Parse the QE kernel into an SDFG

``dace_fortran.build_sdfg`` parses a Fortran source string, runs the bridge passes, returns a :class:`dace.SDFG`:

```python
import dace_fortran
from pathlib import Path

src = Path("tests/qe/exx_bp/ast_v1_vexx_bp_k_gpu.f90").read_text()

# QE's source has an empty ``MODULE fft_interfaces`` block that
# flang rejects under the strict known-bad-implicit-interface
# check; restore it before parsing.  Helper lives in the same
# test directory.
from qe.exx_bp.test_vexx_bp_k_gpu_parse import _restore_fft_interfaces
src = _restore_fft_interfaces(src)

sdfg = dace_fortran.build_sdfg(
    src,
    out_dir="build/qe",                       # scratch dir for emitted .f90/.hlfir
    entry="_QMexx_bpPvexx_bp_k_gpu",          # mangled symbol of the target sub
    name="vexx_bp_k_gpu",                     # base filename for the scratch files
)
```

``entry`` mangled name follows the ``_Q<scope><name>`` Flang ABI: ``_QM<module>P<procedure>`` for a module procedure, ``_QP<procedure>`` for a free subroutine.

Current status — see ``project_qe_kernel_status.md`` for the live xfail gate and remaining gaps. Today QE hits a ``KeyError`` at SDFG arglist lookup downstream of the module-level struct flatten path; gating follow-up summarised in ``tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py``'s ``test_vexx_bp_k_gpu_parses`` xfail reason.

## Step 2 — Save the SDFG to disk

The returned ``SDFG`` exposes ``save`` (gzipped JSON) and ``to_json`` (plain dict) — use ``save`` for persistence so the file round-trips through ``SDFG.from_file``:

```python
sdfg.save("build/qe/vexx_bp_k_gpu.sdfgz")
```

Reload later without re-running the bridge:

```python
import dace
sdfg = dace.SDFG.from_file("build/qe/vexx_bp_k_gpu.sdfgz")
```

## Step 3 — Emit Fortran bindings

The bridge ships a bindings emitter that wraps an SDFG with a gfortran-callable Fortran shim — same ABI as the original Fortran procedure, so the reference caller can swap between the original kernel and the SDFG by linking against a different ``.so``.

The full ICON velocity case in ``tests/icon/full/test_velocity_full_bindings_e2e.py`` is the canonical example. Two emission modes:

* **Inline** — generate the Fortran wrapper as a string for direct ``gfortran -c`` compile alongside the caller (lowest-friction; one scratch file).
* **Library** — generate a ``.mod`` + ``.o`` pair via ``build_fortran_library``, link against a separate static archive (re-usable across multiple SDFG consumers).

```python
from dace_fortran.bindings import build_fortran_library, FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface

# Auto-derive the OriginalInterface from the kernel's HLFIR (no
# hand-author needed for the velocity / dycore shape).
interface = build_auto_interface(sdfg)

# Generate the wrapper module + companion .o.
build_fortran_library(
    sdfg,
    interface,
    out_dir="build/qe/bindings",
    name="vexx_bp_k_gpu_dace",
)
```

For QE specifically the binding has to ``c_f_pointer``-alias every flattened module-level struct field (``vcut_a``, ``vcut_cutoff``, ...) back to its original ``vcut % a`` form so the caller stays ABI-identical. This part is gated on the same module-level struct flatten path that ``test_vexx_bp_k_gpu_parses`` xfails on.

## Step 4 — Compile the SDFG

``SDFG.compile`` runs DaCe's codegen + the configured C++ compiler, returns a ``CompiledSDFG`` handle:

```python
compiled = sdfg.compile(out_dir="build/qe/compile")
# compiled.filename is the path to the .so
```

The bridge's ``conftest.py`` pins DaCe's CPU flags to ``-O0 -fno-fast-math -ffp-contract=off`` for strict-IEEE numerical parity with the gfortran reference. Drop the pin (or set DaCe's ``compiler.cpu.args``) for production builds.

## Step 5 — End-to-end run (reference vs SDFG)

Deterministic random-input pattern from ``tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py::test_vexx_bp_k_gpu_numerical_correctness``:

```python
import numpy as np
rng = np.random.default_rng(42)

# Reference: compile & call the original kernel.
_, init, run = _compile_reference(tmp_path)   # see test_vexx_bp_k_gpu_parse.py
lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1
init(lda, n, m, npol, max_ibands)
psi_ref, hpsi_ref = _make_random_inputs(lda, npol, max_ibands)
run(psi_ref, hpsi_ref, lda, n, m)

# SDFG side: identical seeded inputs, identical Fortran-side ABI
# via the emitted binding.
psi_sdfg = psi_ref.copy()
hpsi_sdfg = hpsi_ref.copy()
sdfg(psi=psi_sdfg, hpsi=hpsi_sdfg,
     lda=np.int32(lda), n=np.int32(n), m=np.int32(m),
     npol=np.int32(npol), max_ibands=np.int32(max_ibands))

np.testing.assert_allclose(hpsi_sdfg, hpsi_ref, rtol=1e-12, atol=1e-12)
```

## Open gaps (as of 2026-06-10)

* QE ``test_vexx_bp_k_gpu_parses`` xfails on a module-level struct ``KeyError: 'vcut_a'`` (debugging in progress, see commit ``cc514ac`` for the audit fixes that closed the upstream traceToDecl/buildExpr/buildIndexExpr paths and ``fb7c4ee`` for the module-level struct synthesis that works on minimal repros but not yet on the full QE source).
* Bindings layer for module-level struct fields (``vcut_a``, ``vcut_cutoff``, ...) -- caller-side ``c_f_pointer`` aliasing not yet wired for QE.

When both gaps close, the test in Step 5 flips from xfail to PASS and this guide becomes a regression-gate reference.

## See also

* ``tests/icon/full/test_velocity_full_bindings_e2e.py`` — full bindings end-to-end pattern (ICON velocity_tendencies).
* ``tests/icon/full/test_dycore_velocity_external_e2e.py`` — the C-ABI cross-language pattern.
* ``tests/icon/graupel/test_aes_graupel_numerical_correctness.py`` — graupel harness (currently xfail on a structural ``InvalidSDFGEdgeError``).
