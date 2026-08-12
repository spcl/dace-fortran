"""Load the 4 baseline variant SDFGs and build ``velocity_per``.

Expected input layout (produced by ``generate_baselines.py``)::

    baseline/velocity_no_nproma_if_prop_lvn_only_0_istep_1.sdfgz
    baseline/velocity_no_nproma_if_prop_lvn_only_0_istep_2.sdfgz
    baseline/velocity_no_nproma_if_prop_lvn_only_1_istep_1.sdfgz
    baseline/velocity_no_nproma_if_prop_lvn_only_1_istep_2.sdfgz

Thin wrapper around ``compile_if_propagated_sdfgs``: loads the 4 SDFGs,
delegates codegen + compile + link to the shared helper. ``stage=0`` tells
the helper to skip the stages 2-9 source-patch fix-ups. Codegen output
lands under ``codegen/baseline/<variant>/`` (later stages will use
``codegen/stage<N>/``) instead of the DaCe default ``.dacecache/``.

``include/call_velocity.h`` is a committed artifact that only needs to be
regenerated when the SDFG signature actually changes -- run
``python tools/gen_call_site.py`` by hand in that case (and
``tools/verify_call_site.py`` to confirm).
"""

import argparse
import itertools
from pathlib import Path

import dace

from utils.compile_if_propagated_sdfgs import compile_if_propagated_sdfgs


_VARIANT_TEMPLATE = "velocity_no_nproma_if_prop_lvn_only_{lvn}_istep_{istep}"


def _variant_names():
    return [
        _VARIANT_TEMPLATE.format(lvn=lvn, istep=istep)
        for lvn, istep in itertools.product((0, 1), (1, 2))
    ]


def _load_variants(baseline_dir: Path):
    sdfgs = []
    for name in _variant_names():
        path = baseline_dir / f"{name}.sdfgz"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run `python generate_baselines.py` first."
            )
        sdfgs.append(dace.SDFG.from_file(str(path)))
    return sdfgs


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--baseline-dir", default="baseline")
    argp.add_argument("--codegen-dir", default="codegen/baseline")
    argp.add_argument("--main", default="main.cpp")
    argp.add_argument("--output", default="velocity_baseline")
    argp.add_argument(
        "--debug",
        dest="release",
        action="store_false",
        default=True,
        help="-O0 + DACE_VELOCITY_DEBUG (default: -O3 release build)",
    )
    argp.add_argument("--gpu", action="store_true", help="Build with nvcc/hipcc")
    args = argp.parse_args()

    repo = Path(__file__).resolve().parent
    sdfgs = _load_variants(Path(args.baseline_dir))

    # Namespace each variant's build folder under ``codegen/baseline/<name>/``
    # so the generated headers don't collide, and so baseline codegen output is
    # kept separate from later stages' (``codegen/stageX/<name>/``).
    codegen_root = repo / args.codegen_dir
    for sdfg in sdfgs:
        sdfg.build_folder = str(codegen_root / sdfg.name)

    compile_if_propagated_sdfgs(
        sdfgs,
        gpu=args.gpu,
        release=args.release,
        generate_code=True,
        lib=False,
        main_name=args.main,
        stage=0,
        debuginfo=True,
        output=args.output,
    )


if __name__ == "__main__":
    main()
