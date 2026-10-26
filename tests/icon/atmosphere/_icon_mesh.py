"""Random ICON patch connectivity with locality, for the e2e harness ``mesh_buffers``.

Each neighbor slot of element ``g`` points at ``g + d``: mostly a near element (``|d| <= near``, the same or the
adjacent block), sometimes a uniformly random far one, the way a space-filling-curve ordered ICON grid looks. The
refinement-control tables open the whole domain, so every block and every row of a block runs.
"""

import numpy as np

#: The element kind each ``<kind>_<target>_{idx,blk}`` connectivity points at.
TARGET_KIND = {"neighbor": "cells", "cell": "cells", "edge": "edges", "vertex": "verts", "quad": "edges"}

#: ``p_patch % <kind> % <target>_{idx,blk}`` read by ``velocity_tendencies``.
VELOCITY_CONNECTIVITY = (
    ("cells", "neighbor"),
    ("cells", "edge"),
    ("edges", "cell"),
    ("edges", "vertex"),
    ("edges", "quad"),
    ("verts", "cell"),
    ("verts", "edge"),
)


def neighbors(rng: np.random.Generator, count: int, slots: int, near: int, far_fraction: float) -> np.ndarray:
    """``(count, slots)`` 0-based neighbor ids: ``g + d`` reflected into range, or uniform for a far slot."""
    own = np.arange(count)[:, None]
    near_ids = own + rng.integers(-near, near + 1, (count, slots))
    near_ids = np.abs(near_ids)
    near_ids = np.where(near_ids >= count, 2 * (count - 1) - near_ids, near_ids)
    far = rng.random((count, slots)) < far_fraction
    return np.where(far, rng.integers(0, count, (count, slots)), near_ids)


def icon_mesh_buffers(
    *,
    nproma: int,
    nblks: int,
    slots: int,
    refin_extent: int,
    seed: int,
    near: int = 2,
    far_fraction: float = 0.1,
    connectivity=VELOCITY_CONNECTIVITY,
) -> dict[str, np.ndarray]:
    """``mesh_buffers`` for an ``nproma x nblks`` patch: connectivity ``(nproma, nblks, slots)`` with 1-based
    ``idx``/``blk``, whole-domain ``start/end_{index,block}`` of ``refin_extent`` levels and ``nblks_{c,e,v}``."""
    rng = np.random.default_rng(seed)
    count = nproma * nblks
    buffers = {}
    for kind, target in connectivity:
        ids = neighbors(rng, count, slots, near, far_fraction)
        # Element g sits at row g % nproma of block g // nproma (Fortran order: row fastest)
        shape = (nproma, nblks, slots)
        buffers[f"p_patch_{kind}_{target}_idx"] = np.asfortranarray((ids % nproma + 1).reshape(shape, order="F"))
        buffers[f"p_patch_{kind}_{target}_blk"] = np.asfortranarray((ids // nproma + 1).reshape(shape, order="F"))
    for kind in ("cells", "edges", "verts"):
        buffers[f"p_patch_{kind}_start_index"] = np.ones(refin_extent, dtype=np.int32)
        buffers[f"p_patch_{kind}_end_index"] = np.full(refin_extent, nproma, dtype=np.int32)
        buffers[f"p_patch_{kind}_start_block"] = np.ones(refin_extent, dtype=np.int32)
        buffers[f"p_patch_{kind}_end_block"] = np.full(refin_extent, nblks, dtype=np.int32)
    for suffix in ("c", "e", "v"):
        buffers[f"p_patch_nblks_{suffix}"] = np.array([nblks], dtype=np.int32)
    return buffers


def velocity_patch(*, refin_extent: int, nproma: int = 8, nlev: int = 7, nblks: int = 8, seed: int = 1001) -> dict:
    """``run_kernel_e2e`` keyword arguments for ``velocity_tendencies`` on one consistent patch: ``nproma x nblks``
    columns of ``nlev`` full and ``nlev + 1`` half levels, random connectivity with locality. ``refin_extent`` is the
    harness's refinement-control table length.

    The buffers are ``n`` wide in every dimension. ``vn`` and ``w`` carry their true level count because the kernel
    takes a loop bound from their shape (``UBOUND(vn, 2)`` in ``rot_vertex_ri``, ``UBOUND(w, 2)`` in
    ``cells2verts_scalar_ri``).
    """
    n = max(nproma, nlev + 1, nblks)
    return {
        "n": n,
        "int_fill": 1,  # every integer the patch does not set
        "module_seeds": {"nproma": nproma},
        "array_overrides": {"p_patch_nlev": nlev, "p_patch_nlevp1": nlev + 1},
        "extents": {"p_prog_vn_d1": nlev, "p_prog_w_d1": nlev + 1},
        "mesh_buffers": icon_mesh_buffers(nproma=nproma, nblks=nblks, slots=n, refin_extent=refin_extent, seed=seed),
    }
