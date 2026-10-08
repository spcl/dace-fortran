"""End-to-end emitter tests -- consumes (FrozenSignature, OriginalInterface,
FlattenPlan) and writes <entry>_bindings.f90 mirroring hlfir-flatten-structs.
Asserts on Fortran shapes present/absent (zero-copy); no compile-and-run here.
"""

from pathlib import Path

import pytest

from dace_fortran.bindings import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)
from dace_fortran.bindings.frozen_signature import FrozenArgKind

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _two_real_array_parts():
    """(frozen, iface, plan) for the two-real-array-struct fixture."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="a",
                sdfg_name="fld_a",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
                from_struct_member="fld%a",
            ),
            FrozenArg(
                fortran_name="b",
                sdfg_name="fld_b",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
                from_struct_member="fld%b",
            ),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="fld", fortran_type="type(t_fields)", rank=0, intent="inout", struct_type="t_fields"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
        used_modules={"mo_fields": ("t_fields",)},
    )
    plan = FlattenPlan(
        entries=(
            FlattenEntry(
                outer_expr="fld%a",
                outer_type="real(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("fld_a",),
                    read_exprs=("fld%a($i1, $i2)",),
                    rank=2,
                    shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
                    aliasable=True,
                ),
            ),
            FlattenEntry(
                outer_expr="fld%b",
                outer_type="real(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("fld_b",),
                    read_exprs=("fld%b($i1, $i2)",),
                    rank=2,
                    shape_exprs=("size(fld%b, dim=1)", "size(fld%b, dim=2)"),
                    aliasable=True,
                ),
            ),
        )
    )
    return frozen, iface, plan


def _two_real_array_struct(tmp_path: Path) -> str:
    """type(t_fields) with two plain real(c_double) members -- everything aliases."""
    frozen, iface, plan = _two_real_array_parts()
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def _complex_split_struct(tmp_path: Path) -> str:
    """``st%z`` complex member + ``st%u`` plain real."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="z_re",
                sdfg_name="st_z_re",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
            ),
            FrozenArg(
                fortran_name="z_im",
                sdfg_name="st_z_im",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
            ),
            FrozenArg(
                fortran_name="u",
                sdfg_name="st_u",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
            ),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="st", fortran_type="type(t_state)", rank=0, intent="inout", struct_type="t_state"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
        used_modules={"mo_state": ("t_state",)},
    )
    plan = FlattenPlan(
        entries=(
            FlattenEntry(
                outer_expr="st%z",
                outer_type="complex(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("st_z_re", "st_z_im"),
                    read_exprs=("real(st%z($i1,$i2), kind=c_double)", "aimag(st%z($i1,$i2))"),
                    write_expr="cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)",
                    rank=2,
                    shape_exprs=("size(st%z, dim=1)", "size(st%z, dim=2)"),
                    aliasable=False,
                ),
            ),
            FlattenEntry(
                outer_expr="st%u",
                outer_type="real(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("st_u",),
                    read_exprs=("st%u($i1, $i2)",),
                    rank=2,
                    shape_exprs=("size(st%u, dim=1)", "size(st%u, dim=2)"),
                    aliasable=True,
                ),
            ),
        )
    )
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def _nested_struct(tmp_path: Path) -> str:
    """Two-level nested struct: st%a%v + st%b%v, both aliased with full %-paths in c_loc(...)."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="a_v",
                sdfg_name="st_a_v",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
            ),
            FrozenArg(
                fortran_name="b_v",
                sdfg_name="st_b_v",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=2,
                shape=("n", "m"),
                intent="inout",
            ),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="st", fortran_type="type(t_outer)", rank=0, intent="inout", struct_type="t_outer"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
        used_modules={"mo_types": ("t_outer",)},
    )
    plan = FlattenPlan(
        entries=(
            FlattenEntry(
                outer_expr="st%a%v",
                outer_type="real(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("st_a_v",),
                    read_exprs=("st%a%v($i1, $i2)",),
                    rank=2,
                    shape_exprs=("size(st%a%v, dim=1)", "size(st%a%v, dim=2)"),
                    aliasable=True,
                ),
            ),
            FlattenEntry(
                outer_expr="st%b%v",
                outer_type="real(c_double)",
                writeback_intent="inout",
                recipe=FlattenRecipe(
                    flat_names=("st_b_v",),
                    read_exprs=("st%b%v($i1, $i2)",),
                    rank=2,
                    shape_exprs=("size(st%b%v, dim=1)", "size(st%b%v, dim=2)"),
                    aliasable=True,
                ),
            ),
        )
    )
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


# --------------------------------------------------------------------------
# Two-real-array struct  --  zero-copy guarantee
# --------------------------------------------------------------------------


def test_two_real_array_struct_all_aliased(tmp_path: Path):
    """Matching layout on every member -- wrapper must not allocate scratch or emit copy loops, pointer aliasing only."""
    src = _two_real_array_struct(tmp_path)
    # no deep-copy artefacts
    assert "allocate(" not in src
    assert "deallocate(" not in src
    assert "do i1 =" not in src and "do i2 =" not in src
    assert src.count("call c_f_pointer(c_loc(fld%a)") == 1
    assert src.count("call c_f_pointer(c_loc(fld%b)") == 1
    assert "real(c_double), pointer :: fld_a(:, :)" in src
    assert "real(c_double), pointer :: fld_b(:, :)" in src


def test_acc_residency_none_is_byte_identical(tmp_path: Path):
    """``acc_residency=None`` (the default) must not change one byte of the
    emitted module -- the ACC staging path is strictly opt-in."""
    baseline = _two_real_array_struct(tmp_path)
    out = tmp_path / "kernel_bindings_accnone.f90"
    frozen, iface, plan = _two_real_array_parts()
    emit_bindings(frozen, iface, plan, str(out), acc_residency=None)
    assert out.read_text() == baseline
    assert "!$ACC" not in baseline.upper()


def test_two_real_array_struct_module_boilerplate(tmp_path: Path):
    """Common boilerplate  --  bind(c), handle, finalize  --  still emitted."""
    src = _two_real_array_struct(tmp_path)
    assert "module kernel_dace_bindings" in src
    assert "use mo_fields, only: t_fields" in src
    assert "bind(c, name='__program_kernel')" in src
    assert "subroutine kernel_dace_finalize()" in src


# --------------------------------------------------------------------------
# Complex-split  --  alloc + do-loop + dealloc
# --------------------------------------------------------------------------


def test_complex_split_emits_copy_in_loop(tmp_path: Path):
    src = _complex_split_struct(tmp_path)
    assert "allocate(st_z_re(size(st%z, dim=1), size(st%z, dim=2)))" in src
    assert "allocate(st_z_im(size(st%z, dim=1), size(st%z, dim=2)))" in src
    assert "st_z_re(i1, i2) = real(st%z(i1,i2), kind=c_double)" in src
    assert "st_z_im(i1, i2) = aimag(st%z(i1,i2))" in src


def test_complex_split_emits_copy_out_loop(tmp_path: Path):
    src = _complex_split_struct(tmp_path)
    assert "st%z(i1, i2) = cmplx(st_z_re(i1,i2), st_z_im(i1,i2), kind=c_double)" in src
    assert "deallocate(st_z_re)" in src
    assert "deallocate(st_z_im)" in src


def test_complex_split_still_aliases_plain_member(tmp_path: Path):
    """The st%u member (plain real) must alias, not copy."""
    src = _complex_split_struct(tmp_path)
    assert "call c_f_pointer(c_loc(st%u)" in src
    # st_u is a pointer, not scratch -- no second allocate.
    assert "allocate(st_u" not in src


# --------------------------------------------------------------------------
# Nested struct  --  full %-path in c_loc
# --------------------------------------------------------------------------


def test_nested_struct_uses_full_path_in_c_loc(tmp_path: Path):
    """st%a%v is more than one-level nesting; Fortran compiler handles the % chain directly."""
    src = _nested_struct(tmp_path)
    assert "call c_f_pointer(c_loc(st%a%v)" in src
    assert "call c_f_pointer(c_loc(st%b%v)" in src


def test_nested_struct_no_copy_overhead(tmp_path: Path):
    """Nested + aliasable => still zero-copy."""
    src = _nested_struct(tmp_path)
    assert "allocate(" not in src
    assert "do i1 =" not in src


# --------------------------------------------------------------------------
# LOGICAL: the SDFG works on the caller's storage
# --------------------------------------------------------------------------


def _logical_kernel(tmp_path: Path, fortran_outer_type: str, dtype: str) -> str:
    """Bindings for ``kernel(flag, n)`` with an intent(inout) LOGICAL array ``flag`` of the given Fortran type."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="flag",
                sdfg_name="flag",
                kind=FrozenArgKind.ARRAY,
                dtype=dtype,
                rank=1,
                shape=("n",),
                intent="inout",
                from_struct_member="",
            ),
            FrozenArg(
                fortran_name="n",
                sdfg_name="n",
                kind=FrozenArgKind.SYMBOL,
                dtype="int32",
                rank=0,
                shape=(),
                intent="in",
                from_struct_member="",
            ),
        ),
        free_symbols=("n",),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="flag", fortran_type=fortran_outer_type, rank=1, shape=("n",), intent="inout"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, FlattenPlan(entries=()), str(out))
    return out.read_text()


@pytest.mark.parametrize(
    "fortran_type, dtype",
    [
        ("logical", "uint32"),
        ("logical(1)", "uint8"),
        ("logical(2)", "uint16"),
        ("logical(4)", "uint32"),
        ("logical(8)", "uint64"),
        ("logical(c_bool)", "uint8"),
    ],
)
def test_logical_array_passes_caller_storage(tmp_path: Path, fortran_type: str, dtype: str):
    """Every LOGICAL kind reaches the SDFG as the caller's own buffer: no scratch, no copy in or out."""
    src = _logical_kernel(tmp_path, fortran_type, dtype)
    call_block = src[src.index("call dace_program_kernel") :]
    assert "c_loc(flag)" in call_block
    assert "allocate(" not in src
    assert "flag =" not in src


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
