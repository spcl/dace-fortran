"""Build a Fortran-callable shared library from a built SDFG.

This is the dace-fortran-only entrypoint that ties the whole binding
contract together.  ``dace.SDFG.compile`` stays vanilla DaCe -- it
just runs ``generate_code`` and builds the kernel ``.so``.  Everything
Fortran-specific lives here and *only* here:

1. compile the SDFG (vanilla DaCe codegen + library build);
2. verify the live SDFG still matches the ``FrozenSignature``
   snapshotted at ``build()`` time -- drift raises
   :class:`SignatureDriftError` *before* any binding is emitted, so a
   wrapper that disagrees with the kernel is never produced;
3. emit the ``<entry>_bindings.f90`` Fortran wrapper;
4. gfortran-compile that wrapper together with the kernel ``.so`` (and
   any caller-supplied Fortran sources) into one linked, Fortran-
   callable shared library.

The drift check used to be a hook inside ``dace/codegen/codegen.py``;
it now lives at this layer so dace-core carries no Fortran coupling.
"""
import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dace_fortran.bindings.emit_bindings import emit_bindings
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface

#: Mandatory flags -- a shared, position-independent, long-line module.
_SHARED_FLAGS = ("-shared", "-fPIC", "-ffree-line-length-none")

#: Debug flags -- optimised (``-O3``) with debug info and strict IEEE
#: (no fast-math, no fp-contraction, rounding-aware) so an
#: SDFG-vs-reference comparison stays bit-reproducible (matches the
#: e2e numerical policy / velocity debug default).
_DEBUG_FLAGS = ("-O3", "-g", "-fno-fast-math", "-ffp-contract=off", "-frounding-math")

#: Release flags -- ``-O3 -ffast-math``; trades IEEE reproducibility
#: for speed (the binding-builder analogue of the velocity ``--release``).
_RELEASE_FLAGS = ("-O3", "-ffast-math")

_MODE_FLAGS = {"debug": _DEBUG_FLAGS, "release": _RELEASE_FLAGS}


@dataclass
class FortranLibrary:
    """A built Fortran-callable shared library.

    :ivar so_path: the linked ``.so`` (binding + kernel + extra sources).
    :ivar sdfg_so: the vanilla-compiled SDFG kernel ``.so`` it links against.
    :ivar bindings_f90: the emitted ``<entry>_bindings.f90`` wrapper.
    """

    so_path: Path
    sdfg_so: Path
    bindings_f90: Path

    def load(self) -> ctypes.CDLL:
        """Open the library with :class:`ctypes.CDLL`.

        If the DaCe kernel needs an OpenMP runtime, supply it via
        ``LD_PRELOAD`` in the environment -- the runtime / its path is
        deliberately not hard-coded here so any implementation
        (``libgomp``, LLVM ``libomp``, ...) works.

        :returns: the opened library.
        """
        return ctypes.CDLL(str(self.so_path))


def build_fortran_library(
    sdfg,
    iface: OriginalInterface,
    plan: FlattenPlan,
    out_dir: str,
    *,
    name: str = None,
    prelude_sources: Sequence = (),
    extra_sources: Sequence = (),
    mode: str = "debug",
    flags: Sequence = None,
    verify: bool = True,
) -> FortranLibrary:
    """Emit + verify + link a Fortran-callable library for ``sdfg``.

    :param sdfg: the SDFG returned by ``SDFGBuilder.build()`` (carries
                 ``_frozen_signature``).
    :param iface: caller-facing Fortran surface of the entry subroutine.
    :param plan: the ``hlfir-flatten-structs`` AoS->SoA plan.
    :param out_dir: scratch directory for the binding + linked ``.so``.
    :param name: library/base name; defaults to ``sdfg.name``.
    :param prelude_sources: ``.f90`` sources the emitted binding
                            depends on (e.g. driver modules whose
                            derived types the binding ``use``s).
                            Compiled *before* the binding.
    :param extra_sources: ``.f90`` sources that depend on the binding
                          (e.g. a caller / shim that ``use``s the
                          binding module).  Compiled *after* it.
    :param mode: ``'debug'`` (default, bit-reproducible: ``-O3 -g`` +
                 no fast-math / no fp-contraction / rounding-aware) or
                 ``'release'`` (``-O3 -ffast-math``).  Ignored when
                 ``flags`` is given.
    :param flags: explicit optimisation/fp flag list, overriding
                  ``mode`` entirely (``-shared``/``-fPIC``/``-fopenmp``
                  are always added).
    :param verify: run the frozen-signature drift check (default on).
    :returns: a :class:`FortranLibrary` handle.
    :raises SignatureDriftError: if the live SDFG drifted from the
            snapshot -- raised before the binding is emitted.
    :raises ValueError: on an unknown ``mode``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or sdfg.name

    if flags is not None:
        opt_flags = tuple(flags)
    elif mode in _MODE_FLAGS:
        opt_flags = _MODE_FLAGS[mode]
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'debug', "
                         f"'release', or an explicit flags= list")

    frozen = getattr(sdfg, "_frozen_signature", None)
    if frozen is None:
        raise ValueError("build_fortran_library requires an SDFG built by "
                         "SDFGBuilder.build() (no _frozen_signature attached). "
                         "Plain SDFGs are unaffected -- dace-core codegen is "
                         "vanilla; the drift contract is dace-fortran-only.")
    # Drift / binding-correctness gate: refuse to emit a wrapper that
    # disagrees with the kernel.  Checked before compile so it fails
    # fast and never produces a stale binding.
    if verify:
        frozen.verify_against(sdfg)

    compiled = sdfg.compile()
    sdfg_so = Path(compiled._lib._library_filename)

    bindings_f90 = out_dir / f"{name}_bindings.f90"
    emit_bindings(frozen, iface, plan, str(bindings_f90))

    so_path = out_dir / f"lib{name}.so"
    # gfortran compiles sources left-to-right with no dependency
    # reordering: modules the binding ``use``s must precede it, and
    # sources that ``use`` the binding must follow it.
    cmd = [
        "gfortran", *_SHARED_FLAGS, *opt_flags, "-fopenmp", f"-J{out_dir}",
        *[str(s) for s in prelude_sources],
        str(bindings_f90), *[str(s) for s in extra_sources], "-o",
        str(so_path), f"-L{sdfg_so.parent}", f"-Wl,-rpath,{sdfg_so.parent}", f"-l:{sdfg_so.name}"
    ]
    subprocess.check_call(cmd, cwd=out_dir)
    return FortranLibrary(so_path=so_path, sdfg_so=sdfg_so, bindings_f90=bindings_f90)
