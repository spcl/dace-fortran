"""Recipe renderers -- the ONLY code that knows how to turn a
``FlattenRecipe`` into Fortran.

Three renderers: ``render_alias_calls`` (zero-copy ``c_f_pointer``
alias), ``render_copy_in_loop`` (allocate + forward do-loop copy),
``render_copy_out_loop`` (reverse do-loop copy + deallocate).  All
return ``List[str]`` of lines pre-indented to wrapper-body level.
"""

from typing import List, Tuple

from dace_fortran.bindings.flatten_plan import (
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)

_DTYPE_TO_F = {
    'float64': 'real(c_double)',
    'float32': 'real(c_float)',
    'int8': 'integer(c_int8_t)',
    'int16': 'integer(c_int16_t)',
    'int32': 'integer(c_int)',
    'int64': 'integer(c_long)',
    'bool': 'logical(c_bool)',
    'complex64': 'complex(c_float)',
    'complex128': 'complex(c_double)',
}


def _fortran_type(dtype: str) -> str:
    """Map a DaCe dtype string to its Fortran iso_c_binding form."""
    return _DTYPE_TO_F.get(dtype, 'real(c_double)')


def _loop_index_names(rank: int) -> Tuple[str, ...]:
    """Loop-index names the wrapper head declares: ``('i1', 'i2', ..., 'iN')``."""
    return tuple(f"i{d + 1}" for d in range(rank))


# ----------------------------------------------------------------------------
# Alias path
# ----------------------------------------------------------------------------


def render_alias_calls(recipe: FlattenRecipe) -> List[str]:
    """Zero-copy alias emission for an ``aliasable=True`` recipe -- one
    ``call c_f_pointer(c_loc(<outer>), <flat>, [<shape>])`` per flat.

    :raises ValueError: recipe is not ``aliasable``.
    """
    if not recipe.aliasable:
        raise ValueError("render_alias_calls called on non-aliasable recipe")
    shape_list = ", ".join(recipe.shape_exprs)
    out: List[str] = []
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs):
        base = strip_index_args(read_expr)
        # Scalar member: c_f_pointer must NOT get a shape arg -- `[]` is
        # an invalid empty array constructor in Fortran.
        if recipe.rank == 0 or not recipe.shape_exprs:
            out.append(f"    call c_f_pointer(c_loc({base}), {flat})")
        else:
            out.append(f"    call c_f_pointer(c_loc({base}), {flat}, [{shape_list}])")
    return out


# ----------------------------------------------------------------------------
# Forward copy (outer -> flats)
# ----------------------------------------------------------------------------


def render_copy_in_loop(recipe: FlattenRecipe) -> List[str]:
    """Generic forward copy: allocate flats, nested do-loops assign each
    flat from its ``read_expr`` with loop-index placeholders substituted.
    Requires ``aliasable=False``, ``rank >= 1``.

    :raises ValueError: recipe is ``aliasable``.
    """
    if recipe.aliasable:
        raise ValueError("render_copy_in_loop called on aliasable recipe  --  use render_alias_calls")
    out: List[str] = []
    for flat in recipe.flat_names:
        out.append(f"    allocate({flat}({', '.join(recipe.shape_exprs)}))")

    # Loop nest: outermost rank first (column-major).
    idx_names = _loop_index_names(recipe.rank)
    # Substituted indices so placeholders don't leak into the generated comment.
    summary = substitute_indices(recipe.read_exprs[0], idx_names)
    out.append(f"    ! Copy-in: {', '.join(recipe.flat_names)} <- {summary}")
    for d in reversed(range(recipe.rank)):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    body_indent = ' ' * (recipe.rank * 2)
    idx_tuple = ", ".join(idx_names)
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs):
        rhs = substitute_indices(read_expr, idx_names)
        out.append(f"    {body_indent}{flat}({idx_tuple}) = {rhs}")

    # Closing markers, innermost-first.
    for d in range(recipe.rank):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")
    return out


# ----------------------------------------------------------------------------
# Reverse copy (flats -> outer) + dealloc
# ----------------------------------------------------------------------------


def render_copy_out_loop(recipe: FlattenRecipe, outer_expr: str) -> List[str]:
    """Inverse of ``render_copy_in_loop``: pack flat buffers back into
    the outer storage at each position, then deallocate.

    ``outer_expr`` is passed separately (same as
    ``FlattenEntry.outer_expr``) so the renderer doesn't reach back up
    to the entry.
    """
    out: List[str] = [f"    ! Copy-out: {outer_expr} <- {', '.join(recipe.flat_names)}"]

    idx_names = _loop_index_names(recipe.rank)
    idx_tuple = ", ".join(idx_names)
    for d in reversed(range(recipe.rank)):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    body_indent = ' ' * (recipe.rank * 2)
    # write_expr set: reconstruction recipe, outer_expr(idx) = write_expr.
    # write_expr empty: plain single-flat member, exact inverse of its
    # copy-in -- scatter into read_exprs[0] (encodes index placement,
    # e.g. pts(i)%x), giving pts(i)%x = pts_x(i).
    if recipe.write_expr:
        lhs = f"{outer_expr}({idx_tuple})"
        rhs = substitute_indices(recipe.write_expr, idx_names)
    else:
        if len(recipe.flat_names) != 1 or not recipe.read_exprs:
            raise ValueError("render_copy_out_loop: empty write_expr needs a single flat + read_expr")
        lhs = substitute_indices(recipe.read_exprs[0], idx_names)
        rhs = f"{recipe.flat_names[0]}({idx_tuple})"
    out.append(f"    {body_indent}{lhs} = {rhs}")

    for d in range(recipe.rank):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")

    for flat in recipe.flat_names:
        out.append(f"    deallocate({flat})")
    return out


# ----------------------------------------------------------------------------
# AoS + allocatable pack/unpack (Phase 5c-B boundary)
# ----------------------------------------------------------------------------
#
# Padding-to-max contract for an AoS dummy whose elements own
# allocatable/pointer array members.  Full design lives on
# ``FlattenRecipe.aos_alloc`` in ``flatten_plan.py``; this module
# implements the two emitters that read those fields.
#
# Helpers below extract `A($i1)%w` from recipe.read_exprs[0] by splitting
# on "($i2)" -- safe because the bridge always emits aos_alloc recipes as
# `<outer>($i1)%<member>($i2)`.  i1 is the iterator build_wrapper_head
# declares for any non-aliasable rank>=1 recipe.


def _aos_alloc_member_at_i(recipe: FlattenRecipe) -> str:
    """Extract ``<outer>(i1)%<member>`` (no inner index) from an
    aos_alloc recipe's ``read_exprs[0]``, for allocated()/size() queries
    in the pack-in/pack-out emitters."""
    template = recipe.read_exprs[0]  # "A($i1)%w($i2)"
    base = template.split('($i2)')[0] if '($i2)' in template \
        else template.rsplit('(', 1)[0]
    return base.replace('$i1', 'i1')


def render_aos_alloc_pack_in(recipe: FlattenRecipe, outer_expr: str) -> List[str]:
    """Compute ``cap``, allocate the 2D buffer, pack each allocated row's
    live region.  ``recipe.aos_alloc`` must be True."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_in called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    cap = recipe.cap_symbol
    n_extent = recipe.shape_exprs[0] if recipe.shape_exprs else "size(" + outer_expr + ")"
    member_at_i = _aos_alloc_member_at_i(recipe)
    return [
        f"    ! ----- AoS+allocatable pack-in: {outer_expr} -> {flat} (cap = {cap}) -----",
        f"    {cap} = 0",
        f"    do i1 = 1, {n_extent}",
        f"      if (allocated({member_at_i})) then",
        f"        if (size({member_at_i}) > {cap}) {cap} = size({member_at_i})",
        f"      end if",
        f"    end do",
        # Empty-batch sentinel: keep cap >= 1 so the buffer is non-degenerate.
        f"    if ({cap} == 0) {cap} = 1",
        f"    allocate({flat}({n_extent}, {cap}))",
        f"    {flat} = 0",
        f"    do i1 = 1, {n_extent}",
        f"      if (allocated({member_at_i})) then",
        f"        {flat}(i1, 1:size({member_at_i})) = {member_at_i}",
        f"      end if",
        f"    end do",
    ]


def render_aos_alloc_pack_out(recipe: FlattenRecipe, outer_expr: str) -> List[str]:
    """Copy each allocated row's live region back from the buffer, free
    the scratch.  No reallocation -- 5c-B kernels don't change
    per-instance sizes (reserved for 5c-C)."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_out called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    n_extent = recipe.shape_exprs[0] if recipe.shape_exprs else "size(" + outer_expr + ")"
    member_at_i = _aos_alloc_member_at_i(recipe)
    return [
        f"    ! ----- AoS+allocatable pack-out: {flat} -> {outer_expr} -----",
        f"    do i1 = 1, {n_extent}",
        f"      if (allocated({member_at_i})) then",
        f"        {member_at_i} = {flat}(i1, 1:size({member_at_i}))",
        f"      end if",
        f"    end do",
        f"    deallocate({flat})",
    ]
