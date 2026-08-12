"""Run f2dace stage 2 on ``baseline_inputs/velocity_modified.f90``.

The input file is already a stage-1 (preprocessed) self-contained Fortran
AST -- it inlines every ``mo_*`` dependency reachable from
``mo_velocity_advection.velocity_tendencies`` and pre-stubs
``mo_exception``/``mo_real_timer``. So we skip
``create_preprocessed_ast`` and call ``create_singular_sdfg_from_ast``
directly.

The icon-artifacts recipe also runs::

    sed -i 's/LOGICAL(KIND *= *1)/INTEGER(KIND=4)/g'

between stages -- f2dace doesn't lower Fortran ``LOGICAL`` cleanly. The
file we ship has zero ``LOGICAL(KIND=1)`` occurrences (verified at port
time), so the substitution is a no-op for ours; we still apply it
defensively so future updates of the AST snapshot don't regress.

Output: ``<output>`` (default ``baseline/velocity_no_nproma.sdfgz``),
the input expected by ``generate_baselines.py``.
"""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path


_ENTRY_POINT = ("mo_velocity_advection", "velocity_tendencies")


def _normalize_logicals(src: str) -> str:
    return re.sub(r"LOGICAL\(KIND\s*=\s*1\)", "INTEGER(KIND=4)", src)


def _import_f2dace():
    try:
        from dace.frontend.fortran.fortran_parser import (
            ParseConfig,
            SDFGConfig,
            create_internal_ast,
            create_sdfg_from_internal_ast,
        )
    except ImportError as e:
        sys.stderr.write(
            "[sdfg_from_velocity_f90] cannot import f2dace fortran_parser. "
            "Make sure DaCe is checked out on a branch that ships the "
            "Fortran frontend (e.g. f2dace/staging) and that this Python "
            "interpreter sees it.\n"
            f"  ImportError: {e}\n"
        )
        sys.exit(2)
    return ParseConfig, SDFGConfig, create_internal_ast, create_sdfg_from_internal_ast


def main(argv=None):
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument(
        "--input",
        default="baseline_inputs/velocity_modified.f90",
        help="Path to the preprocessed velocity AST in Fortran "
        "(default: baseline_inputs/velocity_modified.f90).",
    )
    argp.add_argument(
        "--output",
        default="baseline/velocity_no_nproma.sdfgz",
        help="Path to write the SDFG to "
        "(default: baseline/velocity_no_nproma.sdfgz).",
    )
    argp.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ``sdfg.simplify()`` (default: simplify after generation).",
    )
    argp.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the post-sed normalized .f90 in the temp dir for debugging.",
    )
    args = argp.parse_args(argv)

    src_path = Path(args.input).resolve()
    if not src_path.is_file():
        sys.stderr.write(f"[sdfg_from_velocity_f90] missing input: {src_path}\n")
        sys.exit(1)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = src_path.read_text()
    norm = _normalize_logicals(src)
    if norm != src:
        print(f"[sdfg_from_velocity_f90] LOGICAL(KIND=1) -> INTEGER(KIND=4) substitutions applied")

    tmpdir = Path(tempfile.mkdtemp(prefix="velocity_f2dace_"))
    norm_path = tmpdir / "velocity_modified_norm.f90"
    norm_path.write_text(norm)

    print(f"[sdfg_from_velocity_f90] input    : {src_path}")
    print(f"[sdfg_from_velocity_f90] entry    : {'.'.join(_ENTRY_POINT)}")
    print(f"[sdfg_from_velocity_f90] output   : {out_path}")

    ParseConfig, SDFGConfig, create_internal_ast, create_sdfg_from_internal_ast = _import_f2dace()

    cfg = ParseConfig(sources=[norm_path], entry_points=[_ENTRY_POINT])
    own_ast, program = create_internal_ast(cfg)

    sdfg_cfg = SDFGConfig(
        {_ENTRY_POINT[-1]: _ENTRY_POINT},
        normalize_offsets=True,
        multiple_sdfgs=False,
    )
    gmap = create_sdfg_from_internal_ast(own_ast, program, sdfg_cfg)
    if _ENTRY_POINT[-1] not in gmap:
        sys.stderr.write(
            f"[sdfg_from_velocity_f90] f2dace returned {list(gmap.keys())}; "
            f"expected key '{_ENTRY_POINT[-1]}'\n"
        )
        sys.exit(3)

    sdfg = gmap[_ENTRY_POINT[-1]]
    sdfg.name = out_path.stem

    sdfg.save(str(out_path), compress=str(out_path).endswith(".sdfgz"))
    if not args.no_simplify:
        sdfg.simplify()
        sdfg.save(str(out_path), compress=str(out_path).endswith(".sdfgz"))

    if not args.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"[sdfg_from_velocity_f90] wrote SDFG to {out_path}")


if __name__ == "__main__":
    main()
