"""HLFIR frontend test helpers: compile inline Fortran to ``.hlfir`` via ``flang-new-21``, build SDFGs.  ``have_flang()`` reports availability so callers can skip collection."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HLFIR_DIR = _REPO_ROOT / "dace" / "frontend" / "hlfir"

# when set, build_sdfg(...).build() dumps its SDFG here; "1"/"true"/"yes" means _DEFAULT_DUMP_DIR.
_DUMP_ENV = "__DACE_HLFIR_GEN_TEST_SDFGS"
_DEFAULT_DUMP_DIR = Path("/tmp/hlfir_test_sdfgs")


def _dump_dir() -> Path | None:
    val = os.environ.get(_DUMP_ENV)
    if not val:
        return None
    if val.lower() in ("1", "true", "yes", "on"):
        return _DEFAULT_DUMP_DIR
    return Path(val)


# LLVM/flang 21 only (matches build_bridge.py).  Ubuntu ships flang-new-21/flang-21 as identical symlinks; probe both, $FC overrides.
_FLANG_NAMES = ("flang-new-21", "flang-21", "flang-new", "flang")


def _resolve_flang() -> str | None:
    """Absolute path to an LLVM-flang binary: ``$FC`` if it self-identifies as flang, else the first ``_FLANG_NAMES`` hit on PATH."""
    fc = os.environ.get("FC")
    if fc:
        fc_path = shutil.which(fc) or (fc if os.path.isfile(fc) else None)
        if fc_path:
            try:
                out = subprocess.check_output([fc_path, "--version"], stderr=subprocess.STDOUT,
                                              timeout=5).decode(errors="replace")
                if "flang version" in out:
                    return fc_path
            except (OSError, subprocess.SubprocessError):
                pass
    for name in _FLANG_NAMES:
        p = shutil.which(name)
        if p is not None:
            return p
    return None


_FLANG = _resolve_flang()


def have_flang() -> bool:
    return _FLANG is not None


# strict-FP flags keep an SDFG binding and its gfortran reference byte-identical; -ffree-line-length-none for long generated signatures.
FLANG_PORTABLE_FFLAGS = ["-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none"]


def gfortran_compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    """gfortran-compile ``sources`` into shared object ``out_so``.  Used by e2e binding tests linking via ctypes (not f2py) so LOGICAL/struct ABIs match a real Fortran caller."""
    cmd = ["gfortran", "-shared", "-fPIC", *FLANG_PORTABLE_FFLAGS, f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


def f2py_compile(
    src,
    out_dir: Path,
    mod_name: str,
    extra_f90flags: str | None = None,
    only: tuple[str, ...] | None = None,
):
    """Build Fortran source via gfortran/f2py, return the compiled module.  Skips (pytest.skip) when gfortran/meson missing, so callers can call unconditionally.

    ``only``: subroutine names to expose -- dodges crackfortran's ``KeyError`` on derived-type dummies in unexposed inner subroutines.
    Policy: e2e tests compare against this non-transformed reference, never hand-tuned literal expectations.
    """
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_text = src if not isinstance(src, Path) else None
    if src_text is not None:
        src_file = out_dir / f"{mod_name}.f90"
        src_file.write_text(src_text)
    else:
        src_file = src
    extra_args = [f"--f90flags={extra_f90flags}"] if extra_f90flags else []
    # meson backend (py>=3.12) ignores --f90flags, reads FFLAGS instead; lift line-length cap for f2py's long generated signature.
    env = {**os.environ, "FFLAGS": (os.environ.get("FFLAGS", "") + " -ffree-line-length-none").strip()}
    # retries on transient ENOMEM; rebuilds under a fresh name if `only` routine is missing (crackfortran flake under -n auto)
    from _helpers import f2py_build_and_import
    return f2py_build_and_import(src_file,
                                 out_dir=out_dir,
                                 mod_name=mod_name,
                                 only=only,
                                 extra_args=extra_args,
                                 env=env)


def compile_to_hlfir(source: str,
                     out_dir: Path,
                     name: str = "src",
                     *,
                     preprocess: bool = False,
                     merge: bool = True) -> Path:
    """Write ``source`` to ``<out_dir>/<name>.f90``, compile to HLFIR, return the path.

    ``merge`` (default on): inline USE-d modules into one TU; no-op for self-contained input.
    Integer-valued REAL powers (``x**2.0``) are always expanded to multiplies so bridge/gfortran stay bit-identical.
    ``preprocess``: opt-in ``IF (intvar)`` rewrite for legacy INTEGER-as-IF-condition code flang-new-21 rejects; off by default.
    """
    assert _FLANG is not None, "flang-new-21 not available"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{name}.f90"
    from dace_fortran.preprocess import preprocess_fortran_source
    source = preprocess_fortran_source(source, search_dirs=[out_dir], merge=merge, if_intvar=preprocess)
    src.write_text(source)
    hlfir = out_dir / f"{name}.hlfir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(src), "-o", str(hlfir)])
    return hlfir


def _per_test_suffix() -> str:
    """Test-derived SDFG-name suffix from ``PYTEST_CURRENT_TEST``; empty outside pytest.

    Without it, same-named SDFGs across tests share a .so filename under xdist -- the dynamic loader returns a cached handle and the second test silently runs stale code.
    """
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not raw or "::" not in raw:
        return ""
    nodeid = raw.rsplit(" ", 1)[0]
    file_part, _, test_part = nodeid.partition("::")
    stem = Path(file_part).stem
    if stem.endswith("_test"):
        stem = stem[:-len("_test")]
    if test_part.startswith("test_"):
        test_part = test_part[len("test_"):]
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", f"{stem}_{test_part}").strip("_")
    return f"_{sanitized}" if sanitized else ""


class _TestBuilder:
    """Proxy around ``SDFGBuilder``: renames the built SDFG with a per-test suffix (distinct .so per xdist worker) and optionally dumps it; everything else passes through unchanged."""

    def __init__(self, inner, name: str, suffix: str, dump_dir: Path | None):
        self._inner = inner
        self._name = name
        self._suffix = suffix
        self._dump_dir = dump_dir

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def build(self):
        sdfg = self._inner.build()
        if self._suffix:
            sdfg.name = f"{sdfg.name}{self._suffix}"
        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._dump_dir / f"{self._name}{self._suffix}.sdfgz"
            sdfg.save(str(out_path), compress=True)
        return sdfg


def build_sdfg(source: str,
               out_dir: Path,
               name: str = "src",
               pipeline=None,
               entry: str | None = None,
               defines=(),
               merge_engine: str = "regex"):
    """Test funnel over :func:`dace_fortran.build.make_builder`: adds the per-test xdist-safe SDFG naming / dump-dir wrapper on top of the real builder.  ``entry=None`` auto-resolves from the single procedure in ``source``."""
    from dace_fortran.build import make_builder
    builder = make_builder(source,
                           entry=entry,
                           name=name,
                           pipeline=pipeline,
                           out_dir=out_dir,
                           defines=defines,
                           merge_engine=merge_engine)
    suffix = _per_test_suffix()
    dump = _dump_dir()
    if suffix or dump is not None:
        return _TestBuilder(builder, name, suffix, dump)
    return builder


def build_on_root(comm, build_fn, *, root: int = 0, broadcast: bool = True):
    """Run ``build_fn`` on rank ``root`` only, broadcasting failure as a ``RuntimeError`` on every rank -- otherwise a build exception on root leaves non-root ranks hanging at the next collective until CI times out.

    ``broadcast=False``: return value (e.g. a non-picklable SDFG) stays root-only, for callers that follow with their own collective (:func:`dace.sdfg.utils.distributed_compile`).
    """
    import traceback
    rank = comm.Get_rank()
    error = None
    result = None
    if rank == root:
        try:
            result = build_fn()
        except BaseException as exc:  # noqa: BLE001 -- re-raised on every rank
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    if broadcast:
        error, result = comm.bcast((error, result), root=root)
    else:
        error = comm.bcast(error, root=root)
    if error is not None:
        raise RuntimeError(f"MPI build on rank {root} failed; all ranks abort to avoid a "
                           f"collective deadlock. Rank {root} traceback:\n{error}")
    return result


def run_passes_dump(source: str, out_dir: Path, name: str = "src", pipeline: str = "builtin.module()") -> str:
    """Compile Fortran to HLFIR, run ``pipeline``, return the IR dump -- for tests inspecting post-pass MLIR directly rather than through SDFG extraction."""
    from dace_fortran.build_bridge import hb
    hlfir = compile_to_hlfir(source, out_dir, name)
    mod = hb.HLFIRModule()
    if not mod.parse_file(str(hlfir)):
        raise RuntimeError(f"cannot parse {hlfir}")
    if pipeline:
        mod.run_passes(pipeline)
    return mod.dump()
