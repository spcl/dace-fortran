"""Named free symbol that's also an array extent (ICON-O ``n_zlev``) must be sourced from
``size(vn, dim=2)``, not the module global (unset=0 in extracted kernels -> OOB writes)."""
from dace_fortran.bindings.block_builders import _sym_from_array_extent
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenSignature


def _sig():
    """Rank-3 array ``vn(nproma, n_zlev, nblks_e)`` whose middle extent ``n_zlev`` is also a module-global free symbol."""
    vn = FrozenArg(
        fortran_name="vn",
        sdfg_name="vn",
        kind="array",
        dtype="float64",
        rank=3,
        shape=("nproma", "n_zlev", "patch_3d_p_patch_2d_nblks_e"),
        intent="in",
    )
    return FrozenSignature(
        entry="nonlinear_coriolis_3d_fast_scalar",
        mangled="_QPnonlinear_coriolis_3d_fast_scalar",
        args=(vn, ),
        free_symbols=("n_zlev", "nproma"),
        module_symbol_origins={"n_zlev": ("mo_ocean_nml", "n_zlev")},
    )


def test_named_extent_resolves_to_size_of_its_array():
    """``n_zlev`` is the 2nd dim of ``vn`` -> ``size(vn, dim=2)``."""
    assert _sym_from_array_extent("n_zlev", _sig()) == ("vn", 2)


def test_first_extent_dim_is_one_indexed():
    """``nproma`` is the 1st dim -> dim=1 (Fortran 1-based)."""
    assert _sym_from_array_extent("nproma", _sig()) == ("vn", 1)


def test_non_extent_symbol_returns_none():
    """Symbol that isn't any array's extent falls through -> returns None."""
    assert _sym_from_array_extent("no_dual_edges", _sig()) is None


def test_scalar_arg_shape_is_not_matched():
    """Only ARRAY args contribute extents; a scalar arg with a matching name is not picked up."""
    scal = FrozenArg(fortran_name="n_zlev", sdfg_name="n_zlev", kind="scalar", dtype="int32", rank=0)
    sig = FrozenSignature(entry="k", mangled="_QPk", args=(scal, ), free_symbols=("n_zlev", ))
    assert _sym_from_array_extent("n_zlev", sig) is None
