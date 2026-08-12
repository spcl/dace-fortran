"""Translate ``layout_candidates.json`` into a list of ``PermuteConfig``.

The JSON is produced by the SC26-Layout-AD access analysis (see
``Experiments/E6_VelocityTendencies/access_analysis/``). It clusters the
loop nests of velocity_advection by access pattern and lists the arrays
each nest touches. We emit:

* ``unpermuted`` — identity, no arrays permuted.
* ``nlev_first`` — every array reachable from a nest with ``shape='v.h'``
  is permuted to put the level (vertical) axis first.
* ``index_only`` — only connectivity / index arrays are permuted.

The 95-cell sweep from icon-artifacts is intentionally not ported; it is
data, and what the translator emits is one representative config per
single-axis decision (the user can hand-extend by listing more configs in
this file). Everything else is orchestration.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .permute_layout import PermuteConfig

# Heuristic: 3-D arrays that fit (nproma, nlev, nblks) are level-firsted as
# [1, 0, 2]. 2-D (nproma, nblks) become [1, 0]. Connectivity index arrays
# (3-D triple of (nproma, nblks, nshape)) get an "n" permutation. We emit
# only one canonical permutation per group; extend by editing the maps
# below.
_THREE_D_LEVEL_FIRST: List[int] = [1, 0, 2]
_TWO_D_LEVEL_FIRST:   List[int] = [1, 0]
_CONN_PERM:           List[int] = [2, 0, 1]   # (nproma, nblks, nshape) -> (nshape, nproma, nblks)


# Forced overrides: arrays that should ALWAYS be permuted, regardless of
# which configuration is selected. The Fortran source declares
# ``levmask(nblks_c, nlev)`` (column-major: ``jb`` fast-varying), but
# f2dace lowers it as C row-major ``levmask[jb, jk]`` -- the OPPOSITE of
# the original memory order. The line-600 ``ANY(levmask(:, jk))`` reduce
# wants ``jb`` contiguous; transposing to ``[1, 0]`` (i.e.
# ``levmask[jk, jb]``) restores the Fortran-canonical layout AND fixes
# the reduce stride. Empirically validated to be uniformly better.
#
# Anything added here is merged into every emitted PermuteConfig
# (including ``unpermuted``), so the "baseline" config is more accurately
# "everything default EXCEPT these forced overrides".
# Both the host-side ``levmask`` and the GPU-mirror ``gpu_levmask`` are
# listed: stage 4's ``OffloadVelocityToGPU`` mirrors host arrays into
# ``gpu_<name>`` siblings, and post-stage-4 SDFGs reference the GPU
# name only. The bare ``levmask`` entry stays for callers that run
# permute_layout pre-stage-4 (e.g. test fixtures).
# TEMPORARILY DISABLED while debugging stage 6 numerics on the
# velocity pipeline -- the rename loop's subscript rewrite for
# pure-scratch transients with multiple body writers is producing
# wrong output. Re-enable by restoring the levmask + gpu_levmask
# entries once that's resolved.
_FORCE_PERMUTED: Dict[str, List[int]] = {
}


def _apply_force_permuted(configs: List['PermuteConfig'],
                          sdfg_arrays: Dict[str, int] | None) -> List['PermuteConfig']:
    """Merge ``_FORCE_PERMUTED`` into every config's ``permute_map``.

    Returns a NEW list of ``PermuteConfig`` instances so the auto-computed
    ``inverse_permute_map`` is rebuilt for each one. Force overrides are
    skipped silently when the array is absent / has a different
    dimensionality in ``sdfg_arrays`` -- the same dim-filter logic used
    for the heuristic configs.
    """
    forced: Dict[str, List[int]] = {}
    for name, perm in _FORCE_PERMUTED.items():
        if sdfg_arrays is not None:
            ndim = sdfg_arrays.get(name)
            if ndim is None or ndim != len(perm):
                continue
        forced[name] = list(perm)
    if not forced:
        return configs
    out: List[PermuteConfig] = []
    for c in configs:
        merged = dict(c.permute_map)
        for name, perm in forced.items():
            # Force overrides any heuristic permutation -- the heuristic
            # has no way to disambiguate (e.g.) a 2D ``levmask`` from the
            # 3D arrays it shares a v.h nest with, and would otherwise
            # silently emit an N-D permutation that doesn't match the
            # array's real rank.
            merged[name] = list(perm)
        out.append(PermuteConfig(name=c.name, permute_map=merged))
    return out


def fortran_to_sdfg_array_name(s: str) -> str:
    """Map a Fortran-flavored array reference to the SDFG-internal name.

    >>> fortran_to_sdfg_array_name('p_diag%vn_ie')
    '__CG_p_diag__m_vn_ie'
    >>> fortran_to_sdfg_array_name('z_kin_hor_e')
    'z_kin_hor_e'
    """
    if '%' not in s:
        return s
    return '__CG_' + s.replace('%', '__m_').strip()


def _array_dim(name: str, sdfg_arrays: Dict[str, int] | None) -> int | None:
    if sdfg_arrays is None:
        return None
    return sdfg_arrays.get(name)


def configs_from_candidates(json_path: str | Path,
                            sdfg_arrays: Dict[str, int] | None = None) -> List[PermuteConfig]:
    """Return a list of ``PermuteConfig`` derived from ``layout_candidates.json``.

    ``sdfg_arrays`` is an optional ``{name: ndim}`` map; when provided we
    skip arrays whose dimensionality doesn't match the permutation we
    want to apply, instead of failing inside ``PermuteDimensions``.
    """
    data = json.loads(Path(json_path).read_text())

    nlev_first_3d: set[str] = set()
    nlev_first_2d: set[str] = set()
    conn_arrays:    set[str] = set()

    for nest in data.get('all_nests', []):
        shape = nest.get('shape', '')
        for fortran_name in nest.get('arrays', []):
            name = fortran_to_sdfg_array_name(fortran_name)
            if shape == 'v.h':
                nlev_first_3d.add(name)
            elif shape == 'h':
                nlev_first_2d.add(name)
            elif shape == 'v':
                # vertical-only arrays: leaving identity is the safe default
                continue
        # Connectivity indices live in the data even without an explicit
        # 'shape' label; the analysis tags them via class_label.
        if 'connectivity' in (nest.get('class_label') or '').lower():
            for fortran_name in nest.get('arrays', []):
                conn_arrays.add(fortran_to_sdfg_array_name(fortran_name))

    def _filter_by_dim(names: set[str], wanted_dim: int) -> set[str]:
        if sdfg_arrays is None:
            return names
        return {n for n in names if sdfg_arrays.get(n) == wanted_dim}

    configs: List[PermuteConfig] = [PermuteConfig(name='unpermuted', permute_map={})]

    nlev_first_map: Dict[str, List[int]] = {}
    for n in _filter_by_dim(nlev_first_3d, 3):
        nlev_first_map[n] = list(_THREE_D_LEVEL_FIRST)
    for n in _filter_by_dim(nlev_first_2d, 2):
        nlev_first_map[n] = list(_TWO_D_LEVEL_FIRST)
    if nlev_first_map:
        configs.append(PermuteConfig(name='nlev_first', permute_map=nlev_first_map))

    index_only_map: Dict[str, List[int]] = {}
    for n in _filter_by_dim(conn_arrays, 3):
        index_only_map[n] = list(_CONN_PERM)
    if index_only_map:
        configs.append(PermuteConfig(name='index_only', permute_map=index_only_map))

    return _apply_force_permuted(configs, sdfg_arrays)


def _nest_config_name(nid: int, shape: str) -> str:
    return f"nest_{nid}_{shape.replace('.', '')}"


def _per_nest_permute_map(nest: dict,
                          sdfg_arrays: Dict[str, int] | None) -> Dict[str, List[int]]:
    shape = nest.get('shape', '')
    pmap: Dict[str, List[int]] = {}
    for fortran_name in nest.get('arrays', []):
        name = fortran_to_sdfg_array_name(fortran_name)
        if sdfg_arrays is not None and name not in sdfg_arrays:
            continue
        ndim = sdfg_arrays.get(name) if sdfg_arrays is not None else None
        if shape == 'v.h':
            if ndim is None or ndim == 3:
                pmap[name] = list(_THREE_D_LEVEL_FIRST)
            elif ndim == 2:
                pmap[name] = list(_TWO_D_LEVEL_FIRST)
        elif shape == 'h':
            if ndim is None or ndim == 2:
                pmap[name] = list(_TWO_D_LEVEL_FIRST)
    if 'connectivity' in (nest.get('class_label') or '').lower():
        for fortran_name in nest.get('arrays', []):
            name = fortran_to_sdfg_array_name(fortran_name)
            if sdfg_arrays is not None and sdfg_arrays.get(name) != 3:
                continue
            pmap[name] = list(_CONN_PERM)
    return pmap


def extended_configs_from_candidates(json_path: str | Path,
                                     sdfg_arrays: Dict[str, int] | None = None) -> List[PermuteConfig]:
    """``configs_from_candidates`` plus per-nest + curated configs.

    Yields, in order:

    1. The named heuristic configs (``unpermuted``, ``nlev_first``,
       ``index_only``) emitted by :func:`configs_from_candidates`.
    2. One ``nest_<nid>_<shape>`` per ``all_nests`` entry whose ``shape``
       is ``h`` or ``v.h``. ``v``-only nests would map to identity and
       are skipped to avoid duplicating ``unpermuted``.
    3. The researcher-tuned curated set from
       :func:`utils.passes.curated_configs.curated_configs` -- 94
       sweep cells (``cv<0|1>_ch<0|1>_f<0|1>_s<0|1>_n<perm>``) plus
       ``curated_nlev_first`` and ``curated_index_only``.

    Names already produced by an earlier stage are skipped, so the
    curated 95-cell sweep never collides with the heuristic outputs.
    """
    from .curated_configs import curated_configs

    configs = list(configs_from_candidates(json_path, sdfg_arrays=sdfg_arrays))
    seen = {c.name for c in configs}

    data = json.loads(Path(json_path).read_text())
    for nest in data.get('all_nests', []):
        shape = nest.get('shape', '')
        if shape not in ('h', 'v.h'):
            continue
        nid = nest.get('nid')
        if nid is None:
            continue
        name = _nest_config_name(int(nid), shape)
        if name in seen:
            continue
        pmap = _per_nest_permute_map(nest, sdfg_arrays)
        if not pmap:
            continue
        configs.append(PermuteConfig(name=name, permute_map=pmap))
        seen.add(name)

    for cfg in curated_configs(sdfg_arrays=sdfg_arrays):
        if cfg.name in seen:
            continue
        configs.append(cfg)
        seen.add(cfg.name)

    # Apply force-permute overrides (e.g. levmask -> [1, 0]) uniformly
    # across every emitted config -- including ``unpermuted`` and the
    # listed nest_<...>/cohort_<...> configs that didn't come through
    # ``configs_from_candidates`` / ``curated_configs``.
    return _apply_force_permuted(configs, sdfg_arrays)
