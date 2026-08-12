"""Hand-tuned ``PermuteConfig`` set for the velocity-tendencies sweep.

The heuristic-derived configs in :mod:`utils.passes.permute_configs`
(``unpermuted``, ``nlev_first``, ``index_only``, ``nest_<nid>_<shape>``)
are mechanically generated from the E6 access-analysis JSON. They cover
the easy cases but miss researcher-validated specifics -- for instance,
the heuristic emits ``[2, 0, 1]`` for connectivity arrays whereas
empirical sweeps showed ``[0, 2, 1]`` is the actually-correct
permutation for the velocity kernels.

This module is the cleaned-up port of the icon-artifacts research
scratch (``velocity/sc26_layout/permute_stage4.py``). Only the
**transformation data** survived the port: the four per-group array →
permutation maps, the 14 connectivity arrays, the 95-cell sweep
generator, and the two named configs ``curated_nlev_first`` /
``curated_index_only``. Everything else from that file
(``permute_sdfg``, ``add_timers``, ``update_reductions``,
``numa_touch_for_config``, kernel-extraction drivers, plotting, sample
drivers, microbenchmarks) was either already cleanly re-implemented
elsewhere in the pipeline (timers + async reductions live in stage 4;
the canonical permutation pass lives in DaCe at
``dace.transformation.layout.permute_dimensions``) or was research
scratch that does not belong in the AD freeze.

API:

    curated_configs(sdfg_arrays=None) -> List[PermuteConfig]
        96 entries: 94 sweep cells (``cv<0|1>_ch<0|1>_f<0|1>_s<0|1>_n<perm>``
        excluding the all-zero/identity-conn cell which equals
        ``unpermuted``) plus ``curated_nlev_first`` and
        ``curated_index_only``.

The names are runnable directly via
``python -m utils.stages.stage6 --config <name>`` once
``extended_configs_from_candidates`` includes the curated set.
"""

from itertools import permutations as _iter_perms
from typing import Dict, List

from .permute_layout import PermuteConfig

Perm = List[int]
PermMap = Dict[str, Perm]


# ---------------------------------------------------------------------------
# Per-group array → permutation maps (researcher-tuned).
# ---------------------------------------------------------------------------

# cv -- COMPUTE_VERT: vertical-accumulation arrays. jk is the natural
# parallelism axis. Level-first [1, 0, 2] makes jk-access stride-1.
_COMPUTE_VERT_PERMUTED: PermMap = {
    "z_w_con_c_full":   [1, 0, 2],
    "z_w_concorr_me":   [1, 0, 2],
    "z_w_v":            [1, 0, 2],
    "z_w_concorr_mc":   [1, 0],
    "z_w_con_c":        [1, 0],
}

# ch -- COMPUTE_HORIZ: horizontal-stencil arrays. je/jc is the
# innermost loop. Level-first breaks stride-1; included in the sweep
# as a hypothesis to test.
_COMPUTE_HORIZ_PERMUTED: PermMap = {
    "z_ekinh":          [1, 0, 2],
    "z_kin_hor_e":      [1, 0, 2],
    "z_vt_ie":          [1, 0, 2],
    "z_v_grad_w":       [1, 0, 2],
    "zeta":             [1, 0, 2],
    "maxvcfl_arr":      [1, 0, 2],
    "cfl_clipping":     [1, 0],
    "levmask":          [1, 0],
}

# f -- FIELDS: prognostic + diagnostic arrays read in both vertical and
# horizontal loops. 4-D arrays have ntl as the last dim, so [1, 0, 2, 3].
_FIELDS_PERMUTED: PermMap = {
    "__CG_p_prog__m_vn":              [1, 0, 2],
    "__CG_p_prog__m_w":               [1, 0, 2],
    "__CG_p_diag__m_vt":              [1, 0, 2],
    "__CG_p_diag__m_vn_ie":           [1, 0, 2],
    "__CG_p_diag__m_vn_ie_ubc":       [1, 0, 2],
    "__CG_p_diag__m_w_concorr_c":     [1, 0, 2],
    "__CG_p_diag__m_ddt_vn_apc_pc":   [1, 0, 2, 3],
    "__CG_p_diag__m_ddt_w_adv_pc":    [1, 0, 2, 3],
    "__CG_p_diag__m_ddt_vn_cor_pc":   [1, 0, 2, 3],
}

# s -- STENCIL: read-only metric + interpolation coefficients.
_STENCIL_PERMUTED: PermMap = {
    "__CG_p_metrics__m_wgtfac_e":          [1, 0, 2],
    "__CG_p_metrics__m_wgtfacq_e":         [1, 0, 2],
    "__CG_p_metrics__m_wgtfac_c":          [1, 0, 2],
    "__CG_p_metrics__m_ddxn_z_full":       [1, 0, 2],
    "__CG_p_metrics__m_ddxt_z_full":       [1, 0, 2],
    "__CG_p_metrics__m_ddqz_z_half":       [1, 0, 2],
    "__CG_p_metrics__m_ddqz_z_full_e":     [1, 0, 2],
    "__CG_p_metrics__m_coeff1_dwdz":       [1, 0, 2],
    "__CG_p_metrics__m_coeff2_dwdz":       [1, 0, 2],
    "__CG_p_metrics__m_coeff_gradekin":    [1, 0, 2],
    "__CG_p_int__m_e_bln_c_s":             [1, 0, 2],
    "__CG_p_int__m_c_lin_e":               [1, 0, 2],
    "__CG_p_int__m_geofac_n2s":            [1, 0, 2],
    "__CG_p_int__m_geofac_grdiv":          [1, 0, 2],
    "__CG_p_int__m_cells_aw_verts":        [1, 0, 2],
    "__CG_p_int__m_geofac_rot":            [1, 0, 2],
}

# n -- CONN: connectivity index arrays in (nproma, nblks, N) layout.
_CONN_ARRAYS: List[str] = [
    "__CG_p_patch__CG_cells__m_edge_idx",
    "__CG_p_patch__CG_cells__m_edge_blk",
    "__CG_p_patch__CG_cells__m_neighbor_idx",
    "__CG_p_patch__CG_cells__m_neighbor_blk",
    "__CG_p_patch__CG_edges__m_cell_idx",
    "__CG_p_patch__CG_edges__m_cell_blk",
    "__CG_p_patch__CG_edges__m_vertex_idx",
    "__CG_p_patch__CG_edges__m_vertex_blk",
    "__CG_p_patch__CG_edges__m_quad_idx",
    "__CG_p_patch__CG_edges__m_quad_blk",
    "__CG_p_patch__CG_verts__m_cell_blk",
    "__CG_p_patch__CG_verts__m_cell_idx",
    "__CG_p_patch__CG_verts__m_edge_idx",
    "__CG_p_patch__CG_verts__m_edge_blk",
]

_ID_CONN: Perm = [0, 1, 2]
_ALL_CONN_PERMS: List[Perm] = [list(p) for p in _iter_perms([0, 1, 2])]


def _label(p: Perm) -> str:
    return "".join(str(i) for i in p)


def _conn_map(perm: Perm) -> PermMap:
    return {a: perm for a in _CONN_ARRAYS}


def _merge(*maps: PermMap) -> PermMap:
    out: PermMap = {}
    for m in maps:
        out.update(m)
    return out


def _filter_by_dim(pm: PermMap, sdfg_arrays: Dict[str, int] | None) -> PermMap:
    """Drop entries whose target SDFG array isn't present at the
    expected dimensionality. Equivalent to the same-named helper in
    :mod:`utils.passes.permute_configs`."""
    if sdfg_arrays is None:
        return pm
    out: PermMap = {}
    for name, perm in pm.items():
        ndim = sdfg_arrays.get(name)
        if ndim is None:
            continue
        if ndim != len(perm):
            continue
        out[name] = perm
    return out


def _cfg_name(cv: int, ch: int, f: int, s: int, np: Perm) -> str:
    return f"cv{cv}_ch{ch}_f{f}_s{s}_n{_label(np)}"


def _build_sweep_map(cv: int, ch: int, f: int, s: int, np: Perm) -> PermMap:
    parts: List[PermMap] = []
    if cv:
        parts.append(_COMPUTE_VERT_PERMUTED)
    if ch:
        parts.append(_COMPUTE_HORIZ_PERMUTED)
    if f:
        parts.append(_FIELDS_PERMUTED)
    if s:
        parts.append(_STENCIL_PERMUTED)
    if np != _ID_CONN:
        parts.append(_conn_map(np))
    return _merge(*parts)


# ---------------------------------------------------------------------------
# Researcher-validated named configs (override the heuristic
# nlev_first / index_only with empirically-correct permutations).
# ---------------------------------------------------------------------------

# Connectivity uses [0, 2, 1] (NOT the heuristic [2, 0, 1]):
# (nproma, nblks, N) -> (nproma, N, nblks) keeps jc stride-1 after the
# gather on N. Past sweeps rejected [2, 1, 0] and [2, 0, 1].
_CURATED_NLEV_FIRST: PermMap = {
    # Compute-horiz / vert work arrays
    "z_ekinh":                                  [1, 0, 2],
    "z_kin_hor_e":                              [1, 0, 2],
    "z_v_grad_w":                               [1, 0, 2],
    "z_w_v":                                    [1, 0, 2],
    "zeta":                                     [1, 0, 2],
    "z_vt_ie":                                  [1, 0, 2],
    "z_w_concorr_me":                           [1, 0, 2],
    "z_w_concorr_mc":                           [1, 0],
    "z_w_con_c_full":                           [1, 0, 2],
    "z_w_con_c":                                [1, 0],
    "maxvcfl_arr":                              [1, 0, 2],
    "cfl_clipping":                             [1, 0],
    "levmask":                                  [1, 0],
    # Prognostic / diagnostic
    "__CG_p_prog__m_vn":                        [1, 0, 2],
    "__CG_p_prog__m_w":                         [1, 0, 2],
    "__CG_p_diag__m_vt":                        [1, 0, 2],
    "__CG_p_diag__m_vn_ie":                     [1, 0, 2],
    "__CG_p_diag__m_vn_ie_ubc":                 [1, 0, 2],
    "__CG_p_diag__m_w_concorr_c":               [1, 0, 2],
    "__CG_p_diag__m_ddt_vn_apc_pc":             [1, 0, 2, 3],
    "__CG_p_diag__m_ddt_w_adv_pc":              [1, 0, 2, 3],
    "__CG_p_diag__m_ddt_vn_cor_pc":             [1, 0, 2, 3],
    # Metrics
    "__CG_p_metrics__m_wgtfac_e":               [1, 0, 2],
    "__CG_p_metrics__m_wgtfacq_e":              [1, 0, 2],
    "__CG_p_metrics__m_wgtfac_c":               [1, 0, 2],
    "__CG_p_metrics__m_ddxn_z_full":            [1, 0, 2],
    "__CG_p_metrics__m_ddxt_z_full":            [1, 0, 2],
    "__CG_p_metrics__m_ddqz_z_half":            [1, 0, 2],
    "__CG_p_metrics__m_ddqz_z_full_e":          [1, 0, 2],
    "__CG_p_metrics__m_coeff1_dwdz":            [1, 0, 2],
    "__CG_p_metrics__m_coeff2_dwdz":            [1, 0, 2],
    "__CG_p_metrics__m_coeff_gradekin":         [1, 0, 2],
    # Interpolation
    "__CG_p_int__m_e_bln_c_s":                  [0, 1, 2],
    "__CG_p_int__m_c_lin_e":                    [0, 1, 2],
    "__CG_p_int__m_geofac_n2s":                 [0, 1, 2],
    "__CG_p_int__m_geofac_grdiv":               [0, 1, 2],
    "__CG_p_int__m_cells_aw_verts":             [0, 1, 2],
    "__CG_p_int__m_geofac_rot":                 [0, 1, 2],
    # Connectivity (validated [0, 2, 1])
    "__CG_p_patch__CG_verts__m_cell_blk":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_cell_idx":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_cell_blk":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_edge_idx":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_edge_blk":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_vertex_idx":     [0, 2, 1],
    "__CG_p_patch__CG_edges__m_vertex_blk":     [0, 2, 1],
    "__CG_p_patch__CG_edges__m_quad_idx":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_quad_blk":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_neighbor_idx":   [0, 2, 1],
    "__CG_p_patch__CG_cells__m_neighbor_blk":   [0, 2, 1],
}

_CURATED_INDEX_ONLY: PermMap = {
    "__CG_p_int__m_e_bln_c_s":                  [0, 1, 2],
    "__CG_p_int__m_c_lin_e":                    [0, 1, 2],
    "__CG_p_int__m_geofac_n2s":                 [0, 1, 2],
    "__CG_p_int__m_geofac_grdiv":               [0, 1, 2],
    "__CG_p_int__m_cells_aw_verts":             [0, 1, 2],
    "__CG_p_int__m_geofac_rot":                 [0, 1, 2],
    "__CG_p_patch__CG_verts__m_cell_blk":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_cell_idx":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_cell_blk":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_edge_idx":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_edge_blk":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_vertex_idx":     [0, 2, 1],
    "__CG_p_patch__CG_edges__m_vertex_blk":     [0, 2, 1],
    "__CG_p_patch__CG_edges__m_quad_idx":       [0, 2, 1],
    "__CG_p_patch__CG_edges__m_quad_blk":       [0, 2, 1],
    "__CG_p_patch__CG_cells__m_neighbor_idx":   [0, 2, 1],
    "__CG_p_patch__CG_cells__m_neighbor_blk":   [0, 2, 1],
}


def curated_configs(sdfg_arrays: Dict[str, int] | None = None) -> List[PermuteConfig]:
    """Return the full curated config list.

    Layout: 94 sweep cells (``cv<0|1>_ch<0|1>_f<0|1>_s<0|1>_n<perm>``,
    minus the all-zero / identity-conn cell which would duplicate
    ``unpermuted``) followed by ``curated_nlev_first`` and
    ``curated_index_only``. ``sdfg_arrays`` is the same ``{name: ndim}``
    map used by ``configs_from_candidates`` -- when supplied, entries
    whose dimensionality doesn't match the permutation are silently
    dropped instead of failing inside ``PermuteDimensions``.
    """
    out: List[PermuteConfig] = []

    for cv in (0, 1):
        for ch in (0, 1):
            for f in (0, 1):
                for s in (0, 1):
                    for np_perm in _ALL_CONN_PERMS:
                        if cv == 0 and ch == 0 and f == 0 and s == 0 and np_perm == _ID_CONN:
                            # Identity sweep cell == unpermuted; skip to
                            # avoid duplicating the heuristic config.
                            continue
                        pmap = _build_sweep_map(cv, ch, f, s, np_perm)
                        pmap = _filter_by_dim(pmap, sdfg_arrays)
                        if not pmap:
                            continue
                        out.append(PermuteConfig(
                            name=_cfg_name(cv, ch, f, s, np_perm),
                            permute_map=pmap,
                        ))

    nlev_filtered = _filter_by_dim(_CURATED_NLEV_FIRST, sdfg_arrays)
    if nlev_filtered:
        out.append(PermuteConfig(name="curated_nlev_first", permute_map=nlev_filtered))

    idx_filtered = _filter_by_dim(_CURATED_INDEX_ONLY, sdfg_arrays)
    if idx_filtered:
        out.append(PermuteConfig(name="curated_index_only", permute_map=idx_filtered))

    return out
