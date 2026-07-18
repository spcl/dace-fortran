"""Named block builders -- one function per Fortran section of the generated wrapper
module.  Each takes the canonical bundle ``(frozen, iface, plan)`` (or a subset) and
returns one rendered string.  Flattening-plan logic lives in ``loop_copy.py``, called
from ``build_wrapper_body`` / ``build_wrapper_tail``."""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: gfortran caps identifiers at 63 chars (no flag to lift it); longer generated names must be renamed.
_FORTRAN_IDENT_LIMIT = 63


def _shorten_long_idents(src: str, limit: int = _FORTRAN_IDENT_LIMIT) -> str:
    """Rename every identifier longer than ``limit`` to a unique, deterministic ``<=limit`` form.
    Safe because these names are binding-internal (C ABI is positional, bind(c) names are separate
    literals); blake2b digest keeps renames stable across runs and disambiguates shared truncated stems."""
    all_idents = set(re.findall(r"[A-Za-z_]\w*", src))
    long_idents = sorted(i for i in all_idents if len(i) > limit)
    if not long_idents:
        return src
    used = set(all_idents)
    rename: Dict[str, str] = {}
    for name in long_idents:
        digest = hashlib.blake2b(name.encode(), digest_size=4).hexdigest()  # 8 hex chars
        short = f"{name[:limit - 9]}_{digest}"
        while short in used:  # digest collision -- vanishingly unlikely; re-hash
            digest = hashlib.blake2b((short + name).encode(), digest_size=4).hexdigest()
            short = f"{name[:limit - 9]}_{digest}"
        rename[name] = short
        used.add(short)
    for name, short in rename.items():
        src = re.sub(rf"\b{re.escape(name)}\b", short, src)
    return src


from dace_fortran.bindings.flatten_plan import (
    FlattenPlan,
    strip_index_args,
)
from dace_fortran.bindings.fortran_interface import DerivedType, OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature
from dace_fortran.bindings.loop_copy import (
    _fortran_type,
    render_alias_calls,
    render_aos_alloc_pack_in,
    render_aos_alloc_pack_out,
    render_copy_in_loop,
    render_copy_out_loop,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# bind(c) interface block
# ---------------------------------------------------------------------------


def build_c_interface(frozen: FrozenSignature, iface: OriginalInterface, dace_arglist: tuple = ()) -> str:
    """Render the ``interface ... end interface`` block declaring the three C
    entry points the compiled SDFG exports (template: ``templates/c_interface.f90.in``)."""
    tpl = _load("c_interface.f90.in")
    header_lines: List[str] = []
    body_lines: List[str] = []
    for a in _dace_call_order(frozen, dace_arglist):
        if isinstance(a, str):
            # Shape-only free symbol: pass-by-value int, except pgrid symbols
            # (dace_user_comm/dace_user_comm_size) which have non-int dtypes.
            header_lines.append(f"      {a}")
            body_lines.append(f"      {_init_symbol_decl(a, frozen)} :: {a}")
            continue
        header_lines.append(f"      {a.sdfg_name}")
        if a.rank > 0:
            # Array, or length-1 wrapper for a scalar OUTPUT -- either way DaCe passes a pointer.
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'symbol':
            # Free symbol, pass-by-value int of its own width -- must match the
            # wrapper-local decl build_wrapper_head emits for it.
            body_lines.append(f"      {_fortran_c_value_type(a.dtype)}, value :: {a.sdfg_name}")
        elif a.kind == 'mpi_comm':
            # MPI_Comm is a pointer-sized handle (OpenMPI ompi_communicator_t*) --
            # binds as type(c_ptr), value; wrapper feeds it the MPI_Comm_f2c result.
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'scalar':
            # Scalar input is a non-transient SDFG Scalar -- DaCe passes by value,
            # so the Fortran interface must bind by value too (not c_ptr).
            body_lines.append(f"      {_fortran_c_value_type(a.dtype)}, value :: {a.sdfg_name}")
        else:
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
    init_syms = _init_sym_names(frozen)
    init_arg_decls = "".join(f"      {_init_symbol_decl(s, frozen)} :: {s}\n" for s in init_syms)
    rendered = tpl.format(entry=iface.entry,
                          c_arg_decls=",  &\n".join(header_lines),
                          c_arg_decls_body="\n".join(body_lines),
                          init_args=", ".join(init_syms),
                          init_arg_decls=init_arg_decls)
    if any(a.kind == 'mpi_comm' for a in frozen.args) or frozen.user_comm_source:
        # Splice MPI_Comm_f2c into the interface block so the wrapper can convert
        # the Fortran integer handle to the C MPI_Comm the SDFG entry expects.
        rendered = rendered.replace("  end interface", _MPI_COMM_F2C_IFACE + _MPI_COMM_SIZE_IFACE + "  end interface")
    return rendered


# Symbols from emit_mpi._install_user_pgrid; special-cased since FrozenSignature.free_symbols
# carries names only. Keep in lockstep with emit_library._USER_*.
_USER_COMM_SYMBOL_NAME = "dace_user_comm"
_USER_COMM_SIZE_SYMBOL_NAME = "dace_user_comm_size"

_INIT_SYMBOL_DECL_OVERRIDES = {
    _USER_COMM_SYMBOL_NAME: "type(c_ptr), value",
    _USER_COMM_SIZE_SYMBOL_NAME: "integer(c_long_long), value",
}


def _init_symbol_decl(sym: str, frozen=None) -> str:
    """Fortran bind(c) decl for one __dace_init free-symbol parameter.
    Resolution order: (1) _INIT_SYMBOL_DECL_OVERRIDES pgrid symbols, (2) matching
    frozen.args dtype when also a kernel arg (must match what DaCe codegen emits),
    (3) default integer(c_int), value."""
    if sym in _INIT_SYMBOL_DECL_OVERRIDES:
        return _INIT_SYMBOL_DECL_OVERRIDES[sym]
    if frozen is not None:
        for a in frozen.args:
            if a.sdfg_name == sym and a.kind in ('symbol', 'scalar'):
                return f"{_fortran_c_value_type(a.dtype)}, value"
    return "integer(c_int), value"


# MPI_Comm_f2c converts the Fortran integer handle to C's pointer-sized MPI_Comm (OpenMPI).
_MPI_COMM_F2C_IFACE = """
    function dace_mpi_comm_f2c(fcomm) bind(c, name='MPI_Comm_f2c')
      import :: c_int, c_ptr
      integer(c_int), value :: fcomm
      type(c_ptr) :: dace_mpi_comm_f2c
    end function
"""

# MPI_Comm_size sizes __user_pgrid before dace_init; bound via a static size_buf(1)
# to avoid MPI_Comm <-> c_ptr confusion.
_MPI_COMM_SIZE_IFACE = """
    function dace_mpi_comm_size(comm, size) bind(c, name='MPI_Comm_size')
      import :: c_ptr, c_int
      type(c_ptr), value :: comm
      integer(c_int) :: size
      integer(c_int) :: dace_mpi_comm_size
    end function
"""


def _mpi_comm_local(sdfg_name: str) -> str:
    """Wrapper-local c_ptr name holding the MPI_Comm_f2c result (distinct from the
    caller's integer dummy of the same name)."""
    return f"{sdfg_name}__commc"


def _free_sym_names(frozen) -> list:
    """Free symbols DaCe folds into __program (shape symbols like n), sorted,
    excluding those already a frozen.args entry or DaCe-internal.
    Use _init_sym_names for __dace_init instead -- it passes every free symbol, including kernel args."""
    argnames = {a.sdfg_name for a in frozen.args}
    return sorted(s for s in frozen.free_symbols if s not in argnames and not s.startswith('__dace'))


def _init_sym_names(frozen) -> list:
    """Symbol list for __dace_init_<entry> -- DaCe's init routine takes every SDFG
    free symbol (alphabetically), even ones that are also a kernel arg."""
    return sorted(s for s in frozen.free_symbols if not s.startswith('__dace'))


def _dace_call_order(frozen, dace_arglist) -> list:
    """The exact __program_<entry> argument order DaCe codegen emitted
    (dace_arglist = CompiledSDFG._sig). Each name resolves to its FrozenArg;
    unmatched names are free symbols, yielded as str. No dace_arglist -> falls
    back to frozen.args order then sorted free symbols."""
    by_name = {a.sdfg_name: a for a in frozen.args}
    if dace_arglist:
        return [by_name.get(n, n) for n in dace_arglist]
    return list(frozen.args) + _free_sym_names(frozen)


def _render_logical_bridge_copy_in(recipe, outer_expr: str) -> List[str]:
    """Copy-in for a source_logical_kind > 1 flat companion: struct member is
    Fortran LOGICAL(KIND=N) (2/4/8 bytes) but SDFG storage is 1-byte bool.
    Allocates scratch at source extents; Fortran's intrinsic LOGICAL-kind
    conversion handles the per-element width change on assignment."""
    flat = recipe.flat_names[0]
    if recipe.rank == 0:
        return [f"    allocate({flat})", f"    {flat} = {outer_expr}"]
    shape_args = ", ".join(recipe.shape_exprs)
    return [f"    allocate({flat}({shape_args}))", f"    {flat} = {outer_expr}"]


def _render_logical_bridge_copy_out(recipe, outer_expr: str) -> List[str]:
    """Inverse of _render_logical_bridge_copy_in: pack the bool flat back into the
    source struct slot for intent(out)/inout entries, then release the scratch."""
    flat = recipe.flat_names[0]
    return [f"    {outer_expr} = {flat}", f"    deallocate({flat})"]


def _fortran_c_value_type(dtype: str) -> str:
    """Map a frozen-arg dtype string to its iso_c_binding form for a pass-by-value dummy."""
    table = {
        'int32': 'integer(c_int)',
        'int64': 'integer(c_long_long)',
        'float32': 'real(c_float)',
        'float64': 'real(c_double)',
        'bool': 'logical(c_bool)',
        'complex64': 'complex(c_float_complex)',
        'complex128': 'complex(c_double_complex)',
    }
    if dtype not in table:
        raise ValueError(f"_fortran_c_value_type: unsupported scalar dtype {dtype!r} -- "
                         "extend the dtype map for new pass-by-value scalar shapes.")
    return table[dtype]


# ---------------------------------------------------------------------------
# Ref-counted handle state
# ---------------------------------------------------------------------------


def build_handle_state(iface: OriginalInterface) -> str:
    """Render the module-level dace_handle + init_count declarations
    (template: templates/handle_state.f90.in) -- save-scoped, shared by
    <entry>_dace and <entry>_dace_finalize."""
    return _load("handle_state.f90.in").format(entry=iface.entry)


# ---------------------------------------------------------------------------
# Wrapper head  --  dummy decls, flat pointer / scratch decls, symbol / iter locals
# ---------------------------------------------------------------------------


def _enum_args(iface: OriginalInterface, enum_maps: dict) -> dict:
    """Filter enum_maps ({arg_name: {literal_lower: int}}) down to the iface's
    actual outer dummies (case-insensitive match); an unmatched key is silently
    dropped (dummy may have been renamed/eliminated by a later bridge pass)."""
    if not enum_maps:
        return {}
    iface_names = {a.name.lower() for a in iface.args}
    return {a: m for a, m in enum_maps.items() if a.lower() in iface_names}


def _enum_local_name(arg_name: str) -> str:
    """Local INTEGER scratch name for an enum-mapped CHARACTER dummy. Uses the
    dace_enum_<arg> prefix -- the dace_ namespace is reserved for bridge-emitted
    identifiers, so this can't collide with user kernel variables."""
    return f"dace_enum_{arg_name}"


def _enum_literal_case_clause(literal: str) -> str:
    """Render the CASE ('lower', 'UPPER') Fortran list for one enum literal --
    matches QE's ``flag == 'c' .OR. flag == 'C'`` shape collapsed to one entry."""
    lower = literal.lower()
    upper = literal.upper()
    if lower == upper:  # digits / symbols
        return f"CASE ('{lower}')"
    return f"CASE ('{lower}', '{upper}')"


def _optional_local_name(sdfg_name: str) -> str:
    """Wrapper-local holding a forwarded OPTIONAL outer dummy's data. Referencing
    an absent optional directly is undefined, so this local is set from the actual
    when present(), else a degenerate zero, keeping the SDFG's storage always valid."""
    return f"{sdfg_name}__opt"


def _optional_outer_dummies(frozen: FrozenSignature, iface: OriginalInterface) -> list:
    """(OriginalArg, FrozenArg) pairs for every OPTIONAL outer dummy the SDFG
    branches on via <name>_present. Scoped to SCALAR optionals only -- ARRAY
    optionals need their extent symbols guarded too (size(absent) is UB), a
    follow-up not yet done; until then they keep the default-absent presence."""
    optional_names = {a.name.lower() for a in iface.args if getattr(a, 'optional', False)}
    if not optional_names:
        return []
    by_name = {a.name.lower(): a for a in iface.args}
    pairs = []
    for fa in frozen.args:
        if fa.kind != 'scalar' or getattr(fa, 'rank', 0) != 0:
            continue
        fn = (fa.fortran_name or '').lower()
        if fn in optional_names:
            pairs.append((by_name[fn], fa))
    return pairs


def build_wrapper_head(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       enum_maps: dict = None) -> str:
    """Render the <entry>_dace subroutine header + declaration section
    (template: templates/wrapper_head.f90.in). Walks plan.entries: aliasable
    recipes -> pointer decl, non-aliasable -> allocatable/target scratch decl.
    Free symbols not already outer dummies become local integer(c_int) scalars."""
    tpl = _load("wrapper_head.f90.in")
    outer_dummy_names = [a.name for a in iface.args]
    outer_dummy_set = set(outer_dummy_names)
    enum_args = _enum_args(iface, enum_maps or {})

    # An arg declared POINTER upstream needs associated() on its presence fold,
    # which is illegal on plain target -- mirror the POINTER attribute here.
    _free_syms = set(frozen.free_symbols)
    ptr_outer_args = {a.name for a in iface.args if f"{a.name}_allocated" in _free_syms}

    # Enum-mapped arg: SDFG side is INTEGER, caller surface stays CHARACTER(LEN=N).
    # Override fortran_type and drop target (character dummy needs no c_loc --
    # the converted INTEGER local is what reaches the SDFG).
    def _outer_decl(a) -> str:
        if a.name.lower() in enum_args:
            literals = enum_args[a.name.lower()]
            max_len = max((len(lit) for lit in literals), default=1)
            return (f"    character(len={max_len}),"
                    f" intent({a.intent or 'in'}) :: {a.name}")
        opt = ", optional" if getattr(a, 'optional', False) else ""
        attr = "pointer" if a.name in ptr_outer_args else "target"
        return (f"    {a.fortran_type},"
                f" intent({a.intent or 'inout'}), {attr}{opt} :: {a.name}"
                f"{_dim_spec(a.shape, {d.lower() for d in outer_dummy_names})}")

    outer_dummy_decls = "\n".join(_outer_decl(a) for a in iface.args)

    # One integer(c_int) scratch per enum-mapped arg holds the SELECT CASE result
    # the SDFG receives: declared here, populated in the body, passed in the tail.
    enum_local_decls = "\n".join(f"    integer(c_int) :: {_enum_local_name(a.name)}" for a in iface.args
                                 if a.name.lower() in enum_args)

    flat_ptr_lines: List[str] = []
    scratch_lines: List[str] = []
    # Dtypes needing a presence_scratch_<dtype>(1) target: the degenerate binding
    # an ABSENT member's flat POINTER remaps onto (copy-in guard's ELSE branch).
    guard_scratch_dtypes: set = set()
    max_loop_rank = 0
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        ftype = _fortran_type(r.scratch_dtype)
        # Rank-0 member takes no array spec -- ``real :: x()`` is invalid Fortran.
        shape_dims = ("(" + ", ".join(":" for _ in range(r.rank)) + ")") if r.rank > 0 else ""
        # LOGICAL(KIND=N>1) member can't alias directly: c_loc+c_f_pointer as
        # logical(c_bool) reinterprets 4 bytes as 1, corrupting adjacent struct heap
        # metadata (real ICON "free(): invalid next size" bug) -- force scratch+copy.
        if r.aliasable and r.source_logical_kind in (0, 1):
            for flat in r.flat_names:
                flat_ptr_lines.append(f"    {ftype}, pointer :: {flat}{shape_dims}")
            if _recipe_presence_guard(iface, r):
                guard_scratch_dtypes.add(r.scratch_dtype)
        else:
            if r.rank > 0:
                max_loop_rank = max(max_loop_rank, r.rank)
            for flat in r.flat_names:
                scratch_lines.append(f"    {ftype}, allocatable, target :: {flat}{shape_dims}")

    for dt in sorted(guard_scratch_dtypes):
        flat_ptr_lines.append(f"    {_fortran_type(dt)}, target :: {_presence_scratch_name(dt)}(1)")

    # A free symbol that's also a frozen arg must use that arg's C type (e.g.
    # int64 extents) -- a hardcoded integer(c_int) local mismatches the bind(c) dummy.
    sym_dtype = {a.sdfg_name: a.dtype for a in frozen.args if a.kind in ('scalar', 'symbol')}
    # A rank-0 struct member is BOTH a flat companion and a free symbol wanted by
    # value -- skip re-declaring it here (duplicate "already has basic type" error).
    # LIVE entries only: a dead entry declares no companion, so a same-named free
    # symbol still needs its own scalar decl here.
    flat_names = {f for entry in live_entries(frozen, plan) for f in entry.recipe.flat_names}
    # A symbol that's also an orphan/AoS module-global arg already has a target
    # local from those paths -- skip it here (duplicate decl error otherwise).
    module_arg_names = ({a.sdfg_name
                         for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
                        | {a.sdfg_name
                           for a in _aos_module_args(frozen)})

    def _local_decl(s: str) -> str:
        """Wrapper-local Fortran decl for one free symbol. Mirrors build_c_interface's
        init_arg_decls overrides so pgrid params are typed consistently."""
        if s == _USER_COMM_SYMBOL_NAME:
            return "type(c_ptr)"
        if s == _USER_COMM_SIZE_SYMBOL_NAME:
            return "integer(c_long_long)"
        return _fortran_c_value_type(sym_dtype.get(s, 'int32'))

    symbol_decls = "\n".join(f"    {_local_decl(s)} :: {s}" for s in frozen.free_symbols
                             if s not in outer_dummy_set and s not in flat_names and s not in module_arg_names)
    if max_loop_rank:
        iter_decl = "    integer(c_int) :: " + ", ".join(f"i{d + 1}" for d in range(max_loop_rank))
        symbol_decls = (symbol_decls + "\n" + iter_decl) if symbol_decls else iter_decl

    # Orphan module-global args: wrapper-local target per arg, filled from the
    # renamed module import in build_wrapper_body.
    for a, _mod, _member in _orphan_module_args(frozen, iface, plan):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")" if a.rank > 0 else ""
        kw = "allocatable, target" if a.rank > 0 else "target"
        scratch_lines.append(f"    {ftype}, {kw} :: {a.sdfg_name}{spec}")

    # AoS-struct component SoA buffers: allocatable/target per arg + loop index +
    # one cap per member dim (filled in build_wrapper_body, drained in build_wrapper_tail).
    for a in _aos_module_args(frozen):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")" if a.rank else ""
        scratch_lines.append(f"    {ftype}, allocatable, target :: {a.sdfg_name}{spec}")
        its, caps, _mrank, _elem = _aos_loop_pieces(a)
        int_names = its + caps  # scalar-struct (outer_rank 0): no loop vars
        if int_names:
            scratch_lines.append(f"    integer(c_int) :: {', '.join(int_names)}")

    # Absent-optional data buffers with no host source: a degenerate local.
    for a in _unsourced_array_args(frozen, iface, plan):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")"
        scratch_lines.append(f"    {ftype}, allocatable, target :: {a.sdfg_name}{spec}")

    # Unsourced scalar optionals + undeclared shape symbols (decl only; values
    # written in build_wrapper_body).
    for name, ftype, _rhs in _extra_local_symbols(frozen, iface, plan):
        scratch_lines.append(f"    {ftype} :: {name}")

    # Guarded scalar locals for FORWARDED optional dummies, so an omitted optional
    # is never referenced. Filled in build_wrapper_body, passed in build_wrapper_tail.
    for _oa, fa in _optional_outer_dummies(frozen, iface):
        local = _optional_local_name(fa.sdfg_name)
        scratch_lines.append(f"    {_fortran_c_value_type(fa.dtype)} :: {local}")

    bridge_decls, _, _, _ = _build_logical_bridges(frozen, iface)
    if bridge_decls:
        scratch_lines = scratch_lines + bridge_decls

    # One type(c_ptr) local per communicator arg, holding the MPI_Comm_f2c result
    # fed to the SDFG call (outer dummy stays the caller's Fortran integer handle).
    for a in frozen.args:
        if a.kind == 'mpi_comm':
            scratch_lines.append(f"    type(c_ptr) :: {_mpi_comm_local(a.sdfg_name)}")
    # Pgrid path (replaces the opaque-MPI_Comm scalar arg above): scratch for the
    # MPI_Comm_size return + the call's error code. __user_comm(_size) are declared
    # above in symbol_decls.
    if frozen.user_comm_source:
        scratch_lines.append("    integer(c_int) :: dace_user_comm_size_buf")
        scratch_lines.append("    integer(c_int) :: dace_user_comm_size_err")
    if enum_local_decls:
        scratch_lines.append(enum_local_decls)

    return tpl.format(
        entry=iface.entry,
        outer_dummy_list=", ".join(outer_dummy_names),
        outer_dummy_decls=outer_dummy_decls or "    ! (no dummies)",
        flat_ptr_decls="\n".join(flat_ptr_lines) or "    ! (no flat pointers)",
        scratch_decls="\n".join(scratch_lines) or "    ! (no scratch)",
        symbol_decls=symbol_decls or "    ! (no free symbols)",
    )


# ---------------------------------------------------------------------------
# Wrapper body  --  per-entry alias calls / copy-in loops, symbol population
# ---------------------------------------------------------------------------


def partition_symbol_blocks(sym_lines: List[str], buffer_names) -> tuple:
    """Split symbol-population lines into ``(early, late)`` by buffer dependency.

    A shape sym read from a module global/dummy/constant is assigned BEFORE the
    buffer allocates (early); a sym reading the size/lbound of a buffer this wrapper
    allocates can only be assigned AFTER (late). Some assigns are a 5-line
    ``if (g) then / sym=<shape> / else / sym=0 / end if`` guard block whose true-branch
    alone is buffer-derived -- partition by BLOCK, never by line: a per-line split
    shears the block (true-branch late, bare else/end if early -> orphaned ELSE,
    uncompilable Fortran). A block goes late iff any of its lines needs a buffer.
    """

    def _buffer_derived(s: str) -> bool:
        return any(re.search(r'\b' + re.escape(b) + r'\b', s) for b in buffer_names)

    blocks, i = [], 0
    while i < len(sym_lines):
        s = sym_lines[i].strip()
        if s.startswith("if (") and s.endswith("then"):
            j = i + 1
            while j < len(sym_lines) and sym_lines[j].strip() != "end if":
                j += 1
            blocks.append(sym_lines[i:j + 1])
            i = j + 1
        else:
            blocks.append([sym_lines[i]])
            i += 1
    early = [ln for blk in blocks if not any(_buffer_derived(ln) for ln in blk) for ln in blk]
    late = [ln for blk in blocks if any(_buffer_derived(ln) for ln in blk) for ln in blk]
    return early, late


def build_wrapper_body(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       enum_maps: dict = None) -> str:
    """Render the between-declaration-and-SDFG-call block: for each FlattenEntry
    either alias it (zero-copy) or allocate + copy in, then populate SDFG free
    symbols from size(...) on the outer storage."""
    outer_dummy_set = {a.name for a in iface.args}
    enum_args = _enum_args(iface, enum_maps or {})
    body: List[str] = []
    # Enum CHARACTER -> SELECT CASE -> INTEGER scratch; runs FIRST so the value is
    # ready before the symbol-population block (which may reference it).
    if enum_args:
        body.append("    ! ----- String-enum CHARACTER -> integer conversion -----")
        for a in iface.args:
            tbl = enum_args.get(a.name.lower())
            if not tbl:
                continue
            local = _enum_local_name(a.name)
            body.append(f"    select case ({a.name})")
            for literal, value in sorted(tbl.items(), key=lambda kv: kv[1]):
                body.append(f"    {_enum_literal_case_clause(literal)}")
                body.append(f"      {local} = {value}")
            body.append("    case default")
            # -1 is the "unknown enum value" sentinel: the SDFG's IF chain has no
            # branch for it, so the kernel falls through like the source's own
            # default (an explicit error-stop would change observable behaviour).
            body.append(f"      {local} = -1")
            body.append("    end select")
        body.append("")
    # Copy-in is BUILT here but DEFERRED -- appended after the module-global
    # reconstruction blocks below. A double-buffer alias subscripts a record array
    # with a module-global time-level symbol (nold/nnew) seeded only by that block;
    # emitting the alias first subscripts an unallocated nold -> SIGSEGV.
    copyin: List[str] = []
    copyin.append("    ! ----- Copy-in / alias per flatten entry -----")
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        # Four mutually exclusive emitter shapes -- see FlattenRecipe for the flag
        # matrix. source_logical_kind > 1 overrides aliasable with a width-bridging
        # scratch (rationale in build_wrapper_head).
        #
        # A deferred-storage member may be absent at runtime: unguarded c_loc/size
        # then reads garbage descriptor bounds (gfortran's internal_pack smashes the
        # stack). Guard the marshal; ABSENT branch gives the flat a degenerate binding.
        guard = _recipe_presence_guard(iface, r)
        if r.aos_alloc:
            copyin.extend(render_aos_alloc_pack_in(r, entry.outer_expr))
        elif r.aliasable and r.source_logical_kind in (0, 1):
            lines = render_alias_calls(r)
            if guard:
                copyin.append(f"    if ({guard}) then")
                copyin.extend("  " + ln for ln in lines)
                copyin.append("    else")
                scratch = _presence_scratch_name(r.scratch_dtype)
                for flat in r.flat_names:
                    if r.rank > 0:
                        bounds = ", ".join("1:1" for _ in range(r.rank))
                        copyin.append(f"      {flat}({bounds}) => {scratch}")
                    else:
                        copyin.append(f"      {flat} => {scratch}(1)")
                copyin.append("    end if")
            else:
                copyin.extend(lines)
        else:
            if r.aliasable and r.source_logical_kind > 1:
                lines = _render_logical_bridge_copy_in(r, entry.outer_expr)
            else:
                lines = render_copy_in_loop(r)
            if guard:
                copyin.append(f"    if ({guard}) then")
                copyin.extend("  " + ln for ln in lines)
                copyin.append("    else")
                for flat in r.flat_names:
                    degen = "(" + ", ".join("1" for _ in range(r.rank)) + ")" if r.rank > 0 else ""
                    copyin.append(f"      allocate({flat}{degen})")
                    copyin.append(f"      {flat} = {_zero_literal(r.scratch_dtype)}")
                copyin.append("    end if")
            else:
                copyin.extend(lines)

    _, copy_in_lines, _, _ = _build_logical_bridges(frozen, iface)
    if copy_in_lines:
        copyin.append("")
        copyin.append("    ! ----- LOGICAL -> logical(c_bool) bridge (copy-in) -----")
        copyin.extend(copy_in_lines)

    # Symbol population is SPLIT by data dependency: a shape sym from a module
    # global/dummy/constant must be assigned BEFORE the allocates below, while a
    # buffer-EXTENT sym (size of a buffer we allocate) can only be assigned AFTER.
    sym_lines = _build_symbol_assigns(frozen, plan, outer_dummy_set, iface)
    extra_syms = _extra_local_symbols(frozen, iface, plan)
    _buffer_names = {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan) if a.rank > 0}
    _buffer_names |= {a.sdfg_name for a in _aos_module_args(frozen)}
    _buffer_names |= {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    # Copy-in/alias companions are also buffer-derived: a symbol reading their
    # lbound/size can only run AFTER the copy-in associates the companion, else it
    # reads an unassociated pointer (garbage lbound -> out-of-bounds SDFG index).
    if plan is not None:
        _buffer_names |= {f for e in plan.entries for f in e.recipe.flat_names}

    def _buffer_derived(rhs: str) -> bool:
        # RHS reads size/lbound of a buffer this wrapper allocates below.
        return any(re.search(r'\b' + re.escape(b) + r'\b', rhs) for b in _buffer_names)

    early_syms, late_syms = partition_symbol_blocks(sym_lines, _buffer_names)
    early_extra = [(n, ft, r) for (n, ft, r) in extra_syms if not _buffer_derived(r)]
    late_extra = [(n, ft, r) for (n, ft, r) in extra_syms if _buffer_derived(r)]
    if early_syms or early_extra:
        body.append("")
        body.append("    ! ----- Symbol population (input-derived; before allocates) -----")
        body.extend(early_syms)
        for name, _ftype, rhs in early_extra:
            body.append(f"    {name} = {rhs}")

    orphans = _orphan_module_args(frozen, iface, plan)
    if orphans:
        body.append("")
        body.append("    ! ----- Module-global args sourced from use-imports -----")
        for a, _mod, _member in orphans:
            alias = _module_symbol_alias(a.sdfg_name)
            alloc_inside = getattr(a, 'global_alloc_inside', False)
            is_alloc = getattr(a, 'module_origin_allocatable', False)
            is_ptr = getattr(a, 'module_origin_pointer', False)
            # A DEFERRED-storage host global may be unallocated on entry (kernel
            # only reads it on a path the caller need not take); its SDFG extents
            # are its own size symbols, so an explicit allocate would be circular.
            # Guard with allocated/associated; fall back to a degenerate buffer.
            deferred = (is_alloc or is_ptr) and not alloc_inside and a.rank > 0
            if alloc_inside:
                # Kernel ALLOCATEs this global itself: host alias holds no data on
                # entry (reading it is UB); write-back assigns it on exit.
                if a.rank > 0:
                    dims = ", ".join(a.shape) if a.shape else "1"
                    body.append(f"    allocate({a.sdfg_name}({dims}))")
                body.append(f"    ! {a.sdfg_name}: kernel-allocated, no copy-in "
                            f"(host alias unallocated on entry)")
            elif deferred:
                degen = ", ".join("1" for _ in range(a.rank))
                body.append(f"    if ({_present(alias, is_ptr)}) then")
                body.append(f"      {a.sdfg_name} = {alias}")
                body.append(f"    else")
                body.append(f"      allocate({a.sdfg_name}({degen}))  "
                            f"! host unallocated (absent on no-op path)")
                body.append(f"    end if")
            else:
                # Static array or scalar module global: allocate to the arg's own
                # concrete extent, NOT size(alias) -- alias may be a scalar, and
                # size(scalar) is a hard Fortran error.
                if a.rank > 0:
                    dims = ", ".join(a.shape) if a.shape else "1"
                    body.append(f"    allocate({a.sdfg_name}({dims}))")
                body.append(f"    {a.sdfg_name} = {alias}")

    aos_args = _aos_module_args(frozen)
    if aos_args:
        body.append("")
        body.append("    ! ----- Module-global AoS-struct components (AoS->SoA) -----")
        for a in aos_args:
            body.extend(_render_aos_copy_in(a))

    unsourced = _unsourced_array_args(frozen, iface, plan)
    if unsourced:
        body.append("")
        body.append("    ! ----- Absent-optional buffers (degenerate, no host source) -----")
        for a in unsourced:
            # Size the placeholder at the arg's REAL extent when recoverable: the
            # SDFG may WRITE an absent-optional buffer that's actually an inlined
            # LOCAL, and a degenerate (1,1,..) local overruns on a mesh-sized write.
            # Fall back to (1,..) only when the shape isn't symbol-recoverable.
            shape = tuple(str(s) for s in (a.shape or ()))
            if len(shape) == a.rank and all(shape):
                dims = ", ".join(shape)
            else:
                dims = ", ".join("1" for _ in range(a.rank))
            body.append(f"    allocate({a.sdfg_name}({dims}))")
            body.append(f"    {a.sdfg_name} = {_zero_literal(a.dtype)}")

    optional_dummies = _optional_outer_dummies(frozen, iface)
    if optional_dummies:
        body.append("")
        body.append("    ! ----- Forwarded optional dummies: guarded data local -----")
        for oa, fa in optional_dummies:
            local = _optional_local_name(fa.sdfg_name)
            body.append(f"    if (present({oa.name})) then")
            body.append(f"      {local} = {oa.name}")
            body.append(f"    else")
            body.append(f"      {local} = {_zero_literal(fa.dtype)}")
            body.append(f"    end if")

    comm_args = [a for a in frozen.args if a.kind == 'mpi_comm']
    if comm_args:
        body.append("")
        body.append("    ! ----- Fortran integer comm -> C MPI_Comm -----")
        for a in comm_args:
            body.append(f"    {_mpi_comm_local(a.sdfg_name)} = dace_mpi_comm_f2c({a.fortran_name})")

    if frozen.user_comm_source:
        body.append("")
        body.append("    ! ----- Fortran integer comm -> __user_comm / __user_comm_size -----")
        body.append("    ! Convert the caller's MPI_Fint to a C MPI_Comm, query its")
        body.append("    ! size, and hand both to ``dace_init_<entry>`` so the")
        body.append("    ! FortranProcessGrid's MPI_Cart_create runs against the")
        body.append("    ! user's communicator (not MPI_COMM_WORLD).")
        body.append(f"    {_USER_COMM_SYMBOL_NAME} = dace_mpi_comm_f2c({frozen.user_comm_source})")
        body.append(
            f"    dace_user_comm_size_err = dace_mpi_comm_size({_USER_COMM_SYMBOL_NAME}, dace_user_comm_size_buf)")
        body.append(f"    {_USER_COMM_SIZE_SYMBOL_NAME} = int(dace_user_comm_size_buf, c_long_long)")

    # Copy-in/alias (deferred from the top): now safe -- the module-global
    # time-level indices its double-buffer aliases subscript are seeded above.
    body.append("")
    body.extend(copyin)

    # Buffer-derived symbols (extent of a wrapper-allocated buffer) are valid only
    # now that the allocates above have run; input-derived ones were emitted earlier.
    if late_syms or late_extra:
        body.append("")
        body.append("    ! ----- Symbol population (buffer-derived; after allocates) -----")
        body.extend(late_syms)
        for name, _ftype, rhs in late_extra:
            body.append(f"    {name} = {rhs}")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Wrapper tail  --  init-count bump, SDFG call, copy-back, deallocate, end sub
# ---------------------------------------------------------------------------


def build_wrapper_tail(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       dace_arglist: tuple = (),
                       enum_maps: dict = None) -> str:
    """Render the wrapper tail: init-count bump + call dace_program_<entry> +
    copy-back for every non-aliased writeable entry, then deallocate + close.
    Template templates/wrapper_call.f90.in supplies the skeleton; we splice the
    copy-back block in before its end marker."""
    tpl = _load("wrapper_call.f90.in")
    _, _, bridge_copy_out, name_override = _build_logical_bridges(frozen, iface)

    # Enum dummies pass their SELECT CASE INTEGER scratch, not the outer CHARACTER.
    # Extend name_override so _call_actual picks up the swap.
    enum_args = _enum_args(iface, enum_maps or {})
    name_override = dict(name_override)
    if enum_args:
        # name_override is keyed by FrozenArg.sdfg_name; map each iface arg's outer
        # name to its frozen sdfg_name (defensive against a future flatten-rename).
        frozen_by_fortran = {fa.fortran_name.lower(): fa for fa in frozen.args if fa.fortran_name}
        for a in iface.args:
            if a.name.lower() not in enum_args:
                continue
            fa = frozen_by_fortran.get(a.name.lower())
            sdfg_name = fa.sdfg_name if fa is not None else a.name
            name_override[sdfg_name] = _enum_local_name(a.name)

    # Forwarded optional dummies pass their guarded local, never the outer dummy
    # directly -- undefined to reference when the caller omitted it.
    for _oa, fa in _optional_outer_dummies(frozen, iface):
        name_override[fa.sdfg_name] = _optional_local_name(fa.sdfg_name)

    # C interface declares every array arg as type(c_ptr), value: wrap the actual
    # with c_loc(...) explicitly -- implicit pointer->c_ptr conversion only applies
    # to intent-typed dummies, not value dummies (gfortran rejects otherwise).
    def _call_actual(a) -> str:
        if isinstance(a, str):
            # Free symbol: wrapper-local DaCe sees by value (declared+populated earlier).
            return a
        actual = name_override.get(a.sdfg_name, a.sdfg_name)
        if a.kind == 'mpi_comm':
            # The C ``MPI_Comm`` (f2c result), not the integer dummy.
            return _mpi_comm_local(a.sdfg_name)
        if a.kind == 'array' or a.rank > 0:
            return f"c_loc({actual})"
        return actual

    call_args = ",  &\n".join(f"      {_call_actual(a)}" for a in _dace_call_order(frozen, dace_arglist))
    call_block = tpl.format(entry=iface.entry,
                            call_arg_list=call_args,
                            init_call_args=", ".join(_init_sym_names(frozen)))

    copy_out_lines: List[str] = []
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        if r.aos_alloc:
            if entry.writeback_intent in ('out', 'inout'):
                copy_out_lines.extend(render_aos_alloc_pack_out(r, entry.outer_expr))
            else:
                # intent(in): no copy-back, but pack-in's scratch still needs releasing.
                copy_out_lines.append(f"    deallocate({r.flat_names[0]})")
            continue
        # source_logical_kind > 1: scratch was allocated unconditionally in
        # copy-in and needs releasing; out/inout adds <outer>=<flat> first.
        # Presence guard mirrors copy-in: an ABSENT member's writeback must not
        # touch the member, but its degenerate copy-in scratch still needs releasing.
        guard = _recipe_presence_guard(iface, r)
        if r.aliasable and r.source_logical_kind > 1:
            if entry.writeback_intent in ('out', 'inout'):
                lines = _render_logical_bridge_copy_out(r, entry.outer_expr)
            else:
                lines = [f"    deallocate({r.flat_names[0]})"]
            copy_out_lines.extend(_guarded_copy_out(lines, guard, r))
            continue
        if r.aliasable:
            continue
        if entry.writeback_intent not in ('out', 'inout'):
            continue
        # Writeable non-aliasable member -> copy flats back. A reconstruction
        # recipe carries write_expr (e.g. complex re/im -> cmplx); a plain
        # single-flat member has none and inverts its copy-in via read_exprs[0].
        if not r.write_expr and not (len(r.flat_names) == 1 and r.read_exprs):
            continue
        copy_out_lines.extend(_guarded_copy_out(render_copy_out_loop(r, entry.outer_expr), guard, r))

    # Module globals the kernel WRITES are host-shared inout state: copy the SDFG
    # arg's final value back to the host module var (symmetric to copy-in). A
    # scalar source was lifted to a length-1 array, so write back element (1).
    module_writeback_lines: List[str] = []
    for a, _mod, _member in _orphan_module_args(frozen, iface, plan):
        if not a.is_written:
            continue
        alias = _module_symbol_alias(a.sdfg_name)
        actual = name_override.get(a.sdfg_name, a.sdfg_name)
        rhs = f"{actual}(1)" if tuple(a.shape) == ('1', ) else actual
        module_writeback_lines.append(f"    {alias} = {rhs}")

    # AoS-struct components: pack the SoA buffer back into the host struct only if
    # WRITTEN, always deallocate the buffer allocated in the body's copy-in.
    aos_out_lines: List[str] = []
    for a in _aos_module_args(frozen):
        if a.is_written:
            aos_out_lines.extend(_render_aos_copy_out(a))
        else:
            aos_out_lines.append(f"    deallocate({a.sdfg_name})")
    for a in _unsourced_array_args(frozen, iface, plan):
        aos_out_lines.append(f"    deallocate({a.sdfg_name})")

    bridge_block = ""
    if bridge_copy_out:
        bridge_block = "\n    ! ----- logical(c_bool) -> LOGICAL bridge (copy-out + dealloc) -----\n" + "\n".join(
            bridge_copy_out)

    writeback_block = ""
    if module_writeback_lines:
        writeback_block = "\n    ! ----- Write-back for kernel-written module globals -----\n" + "\n".join(
            module_writeback_lines)

    if (not copy_out_lines and not bridge_copy_out and not module_writeback_lines and not aos_out_lines):
        return call_block

    copy_out_block = ""
    if copy_out_lines:
        copy_out_block = "\n    ! ----- Copy-out for writeable deep-copy entries -----\n" + "\n".join(copy_out_lines)
    aos_out_block = ""
    if aos_out_lines:
        aos_out_block = "\n    ! ----- AoS-struct component copy-out / dealloc -----\n" + "\n".join(aos_out_lines)
    marker = f"  end subroutine {iface.entry}_dace"
    pre, post = call_block.split(marker, 1)
    return pre + copy_out_block + bridge_block + writeback_block + aos_out_block + "\n" + marker + post


# ---------------------------------------------------------------------------
# Finalize subroutine
# ---------------------------------------------------------------------------


def build_finalize(iface: OriginalInterface) -> str:
    """Placeholder -- the finalize subroutine is baked into wrapper_call.f90.in
    and emitted with the main wrapper tail. Kept as a named function for API symmetry."""
    del iface  # unused
    return ""


# ---------------------------------------------------------------------------
# Module assembler
# ---------------------------------------------------------------------------


def assemble_module(iface: OriginalInterface, frozen: FrozenSignature, blocks: dict) -> str:
    """Stitch the rendered blocks into the complete Fortran module
    (template: templates/module.f90.in)."""
    use_lines = [f"  use {mod}, only: {', '.join(syms)}" for mod, syms in sorted(iface.used_modules.items())]
    # Module-sourced free symbols: import each member under a <sym>__mod alias so
    # it doesn't clash with the wrapper's own local <sym>. Group by module.
    by_mod: dict = {}
    for sym, (mod, member) in sorted(effective_module_sources(frozen, iface).items()):
        by_mod.setdefault(mod, []).append(f"{_module_symbol_alias(sym)} => {member}")
    for mod, renames in sorted(by_mod.items()):
        use_lines.append(f"  use {mod}, only: {', '.join(sorted(set(renames)))}")
    # AoS-struct components: import the HOST STRUCT plainly (no __mod alias, the
    # copy loop references it directly). A DUMMY-rooted component (empty
    # aos_origin_mod) needs no import -- its root is already a wrapper argument.
    aos_by_mod: dict = {}
    for a in _aos_module_args(frozen):
        if a.aos_origin_mod:
            aos_by_mod.setdefault(a.aos_origin_mod, set()).add(a.aos_origin_struct)
    for mod, structs in sorted(aos_by_mod.items()):
        use_lines.append(f"  use {mod}, only: {', '.join(sorted(structs))}")
    use_statements = "\n".join(use_lines)
    wrapper_body = (blocks['wrapper_head'] + "\n" + blocks['wrapper_body'] + "\n" + blocks['wrapper_tail'])
    module_src = _load("module.f90.in").format(
        entry=iface.entry,
        schema_version=frozen.schema_version,
        use_statements=use_statements,
        c_interface=blocks['c_interface'],
        handle_state=blocks['handle_state'],
        wrapper_body=wrapper_body,
        finalize_body=blocks['finalize'],
    )
    # gfortran rejects identifiers over 63 chars; rename any such name uniquely.
    return _shorten_long_idents(module_src)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shape_references_non_dummy(shape, dummy_set_lower: set) -> bool:
    """True iff an explicit extent in shape names an identifier that is not one of
    this wrapper's dummies. A Fortran explicit-shape dummy bound may reference only
    other dummies/literals/intrinsics -- a localized module global there is illegal
    (gfortran rejects it), so such a dummy renders assumed-size instead (see _dim_spec)."""
    for s in shape or ():
        if not s or s == '?':
            continue  # assumed-shape placeholder -- a separate, legal form
        # Identifiers that are variable references (not ``name(`` calls).
        for m in re.finditer(r"[A-Za-z_]\w*", s):
            ident, end = m.group(0), m.end()
            if end < len(s) and s[end] == '(':
                continue  # intrinsic / function call, not a bound variable
            if ident.lower() not in dummy_set_lower:
                return True
    return False


def _dim_spec(shape, dummy_set_lower: set = None) -> str:
    """Render the dimension spec suffix as postfix shape (name(d1,d2)), not a
    dimension(...) attribute -- the latter after :: is read as a SECOND variable
    declaration (silent mis-declare). An illegal dummy bound collapses the array to
    assumed-SHAPE (:,:,:), not assumed-size (*): assumed-size would drop to rank 1
    and break a full-rank AoS gather loop."""
    if not shape:
        return ""
    if dummy_set_lower is not None and _shape_references_non_dummy(shape, dummy_set_lower):
        return "(" + ",".join(":" for _ in shape) + ")"
    return f"({','.join(s if s != '?' else ':' for s in shape)})"


def _is_default_logical(fortran_type: str) -> bool:
    """True for a caller-visible LOGICAL declaration whose storage layout differs
    from logical(c_bool) (default LOGICAL is 4 bytes) -- such kinds need a
    cast-via-copy at the wrapper boundary so the SDFG sees 1-byte bool layout."""
    s = fortran_type.strip().lower()
    if s == 'logical':
        return True
    if s.startswith('logical(') and 'c_bool' not in s:
        return True
    return False


def _build_logical_bridges(frozen: FrozenSignature, iface: OriginalInterface):
    """Emit scratch buffers + entry/exit copies for a LOGICAL outer dummy the SDFG
    sees as bool: the wrapper's 4-byte logical would corrupt a bool* read, so a
    logical(c_bool) scratch bridges via Fortran's intrinsic kind-conversion.
    Returns (decl_lines, copy_in_lines, copy_out_lines, name_override) -- the
    latter maps sdfg_name to the scratch name the call site should pass instead.
    Already-logical(c_bool) dummies need no bridge; bool intent(in) scalars are
    bridged at the call site instead (see build_wrapper_tail)."""
    decl_lines: List[str] = []
    copy_in_lines: List[str] = []
    copy_out_lines: List[str] = []
    name_override: dict = {}

    iface_by_name = {a.name: a for a in iface.args}
    for fa in frozen.args:
        if fa.dtype != 'bool':
            continue
        oa = iface_by_name.get(fa.fortran_name)
        if oa is None:
            continue
        if not _is_default_logical(oa.fortran_type):
            continue
        # Array dummy  --  explicit scratch buffer + element-wise cast.
        if fa.rank > 0:
            scratch = f"{fa.fortran_name}_cbool"
            shape_dim = "(" + ",".join(":" for _ in range(fa.rank)) + ")"
            decl_lines.append(f"    logical(c_bool), allocatable, target :: {scratch}{shape_dim}")
            # A scalar intent(out)/inout LOGICAL is scalar on the caller side but
            # the bridge lifts it to a length-1 Array on the SDFG (see
            # descriptors.py) -- size(scalar) errors, so allocate to the arg's own
            # extent and bridge through element (1).
            if oa.rank == 0:
                dims = ", ".join(fa.shape) if fa.shape else "1"
                copy_in_lines.append(f"    allocate({scratch}({dims}))")
                copy_in_lines.append(f"    {scratch}(1) = {oa.name}")
                if oa.intent in ('out', 'inout', ''):
                    copy_out_lines.append(f"    {oa.name} = {scratch}(1)")
            else:
                shape_args = ", ".join(f"size({oa.name}, dim={d + 1})" for d in range(fa.rank))
                copy_in_lines.append(f"    allocate({scratch}({shape_args}))")
                copy_in_lines.append(f"    {scratch} = {oa.name}")
                if oa.intent in ('out', 'inout', ''):
                    copy_out_lines.append(f"    {oa.name} = {scratch}")
            copy_out_lines.append(f"    deallocate({scratch})")
            name_override[fa.sdfg_name] = scratch
        # Scalar bool dummy: C interface wants logical(c_bool), value, but the
        # outer dummy is default logical (4 bytes) -- gfortran rejects the direct
        # call ("passed LOGICAL(4) to LOGICAL(1)"), no implicit cast for pass-by-
        # value bind(c). Fix mirrors the array path: cast into a local temp, pass that.
        else:
            scratch = f"{fa.fortran_name}_cbool"
            decl_lines.append(f"    logical(c_bool) :: {scratch}")
            copy_in_lines.append(f"    {scratch} = {oa.name}")
            if oa.intent in ('out', 'inout', ''):
                # Symmetric copy-back; no deallocate -- stack temporary, not allocatable.
                copy_out_lines.append(f"    {oa.name} = {scratch}")
            name_override[fa.sdfg_name] = scratch
            continue

    return decl_lines, copy_in_lines, copy_out_lines, name_override


_OFFSET_SYM_RE = re.compile(r"^offset_(.+)_d(\d+)$")
_EXTENT_SYM_RE = re.compile(r"^(.+)_d(\d+)$")


def _sym_from_intrinsic(sym: str, frozen: FrozenSignature) -> Optional[Tuple[str, str, int]]:
    """Map a free SDFG symbol to the Fortran intrinsic that populates it:
    offset_<arr>_d<i> -> ("lbound", expr, i+1); <arr>_d<i> -> ("size", expr, i+1).
    None when sym isn't an offset/extent of a known array arg."""
    by_sdfg = {a.sdfg_name: a for a in frozen.args}

    def _expr(arr: str) -> Optional[str]:
        a = by_sdfg.get(arr)
        if a is None or a.kind != "array":
            return None
        return a.from_struct_member or a.fortran_name

    m = _OFFSET_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("lbound", e, int(m.group(2)) + 1) if e else None
    m = _EXTENT_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("size", e, int(m.group(2)) + 1) if e else None
    return None


def _sym_from_array_extent(sym: str, frozen: FrozenSignature, exclude=None) -> Optional[Tuple[str, int]]:
    """A free symbol that's a NAMED extent of an array arg (e.g. n_zlev is dim 2 of
    vn(nproma, n_zlev, nblks_e)). Must take precedence over a same-named module
    global: ICON's n_zlev is unset (0) in an extracted kernel, and using it would
    size SDFG transients to zero. ``exclude`` skips degenerate absent-optional args
    so the symbol falls to a PRESENT array's real extent."""
    exclude = exclude or set()
    for a in frozen.args:
        if a.kind != "array":
            continue
        if a.sdfg_name in exclude:
            continue
        shape = tuple(str(s) for s in (a.shape or ()))
        if sym in shape:
            expr = a.from_struct_member or a.fortran_name
            if expr:
                return (expr, shape.index(sym) + 1)
    return None


def _module_symbol_alias(sym: str) -> str:
    """Local rename for a module-sourced free symbol's import -- avoids a name
    clash with the wrapper's own local <sym> (passed by value to the SDFG)."""
    return f"{sym}__mod"


def effective_module_sources(frozen: FrozenSignature, iface: OriginalInterface) -> Dict[str, Tuple[str, str]]:
    """Merge bridge-auto-detected module-global provenance (the primary source,
    FrozenSignature.module_symbol_origins) with hand-authored
    iface.module_symbol_sources, which wins on conflict (override/fallback)."""
    merged: Dict[str, Tuple[str, str]] = dict(getattr(frozen, 'module_symbol_origins', {}) or {})
    merged.update(iface.module_symbol_sources)  # explicit override wins
    return merged


def _orphan_module_args(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """SDFG args that are neither an outer dummy, flat companion, nor extent/offset
    symbol -- Fortran module globals the kernel reads directly (ICON's nrdmax,
    i_am_accel_node, timer handles). Returns (FrozenArg, module, member) tuples."""
    sources = effective_module_sources(frozen, iface)
    dummy = {a.name for a in iface.args}
    flat = {f for e in plan.entries for f in e.recipe.flat_names}
    out = []
    for a in frozen.args:
        n = a.sdfg_name
        if n in dummy or n in flat:
            continue
        if _OFFSET_SYM_RE.match(n) or (a.kind == 'symbol'):
            continue
        if _EXTENT_SYM_RE.match(n) and n not in sources:
            continue
        src = sources.get(n)
        if src is not None:
            out.append((a, src[0], src[1]))
    return out


def _aos_module_args(frozen: FrozenSignature):
    """SDFG args that are the SoA image of an array-of-structs component
    (identified by bridge-stamped aos_origin_struct). Two origins share this path:
    a MODULE-LEVEL global (aos_origin_mod set, needs a use-import) or a
    DUMMY-rooted nested member (aos_origin_mod empty, root already a wrapper arg).
    Restricted to ARRAY args -- a rank-0 member reaches the SDFG as a by-value free
    symbol and must not be re-declared here (duplicate-decl error)."""
    return [a for a in frozen.args if getattr(a, 'aos_origin_struct', '') and getattr(a, 'rank', 0) > 0]


def _unsourced_array_args(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """Array SDFG args with NO host source: not a dummy, flatten companion, module
    global, or AoS component -- data buffers of ABSENT inlined-callee optionals
    (QE's becphi_r). Need a shape-degenerate zero-filled local or c_loc() fails to compile."""
    declared = {f for entry in plan.entries for f in entry.recipe.flat_names}
    declared |= {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
    declared |= {a.sdfg_name for a in _aos_module_args(frozen)}
    declared |= {a.name for a in iface.args}
    return [
        a for a in frozen.args if a.kind == 'array' and getattr(a, 'rank', 0) > 0 and a.sdfg_name not in declared
        and not getattr(a, 'aos_origin_struct', '')
    ]


def _extra_local_symbols(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """Symbols the SDFG call / binding-allocate shape-exprs reference that no
    existing decl path covers: (a) an unsourced SCALAR/symbol arg (absent
    inlined-callee optional's value slot), (b) a bare-identifier SHAPE symbol with
    no free-symbol/dummy source. Returns (name, fortran_type, rhs); non-module-
    origin symbols degrade to a degenerate default, valid since none is READ on
    the no-op path."""
    sources = effective_module_sources(frozen, iface)
    outer = {a.name for a in iface.args}
    flat = {f for e in plan.entries for f in e.recipe.flat_names}
    declared = set(frozen.free_symbols) | outer | flat
    # Names already covered by a wrapper-local/dummy decl elsewhere: array args,
    # orphan/aos SCALAR locals, the comm pgrid params.
    declared |= {a.sdfg_name for a in frozen.args if a.kind == 'array'}
    declared |= {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
    declared |= {a.sdfg_name for a in _aos_module_args(frozen)}
    declared |= {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    declared |= {_USER_COMM_SYMBOL_NAME, _USER_COMM_SIZE_SYMBOL_NAME}
    ident = re.compile(r'^[A-Za-z_]\w*$')

    def _rhs(name: str, dtype: str, is_dim: bool) -> str:
        if name in sources:
            alias = _module_symbol_alias(name)
            return f"int({alias}, c_int)" if dtype != 'bool' else alias
        return "1" if is_dim else _zero_literal(dtype)

    out: dict = {}
    # (a) unsourced scalar / symbol args
    for a in frozen.args:
        if a.kind in ('scalar', 'symbol') and a.sdfg_name not in declared:
            out[a.sdfg_name] = (_fortran_c_value_type(a.dtype), _rhs(a.sdfg_name, a.dtype, False))
    # (b) bare-identifier shape symbols of any arg
    for a in frozen.args:
        for s in (getattr(a, 'shape', ()) or ()):
            s = str(s)
            if ident.match(s) and s not in declared and s not in out:
                out[s] = ("integer(c_int)", _rhs(s, 'int32', True))
    return [(n, ft, rhs) for n, (ft, rhs) in out.items()]


def _zero_literal(dtype: str) -> str:
    """Neutral fill literal for dtype -- .false. for LOGICAL (bare 0 warns as an
    INTEGER->LOGICAL extension), 0 otherwise (Fortran promotes it to real/complex)."""
    return ".false." if dtype == 'bool' else "0"


def _present(expr: str, is_pointer: bool) -> str:
    """Definedness test for a POINTER (``associated``) vs ALLOCATABLE
    (``allocated``) -- using the wrong intrinsic is a hard Fortran type error."""
    return f"associated({expr})" if is_pointer else f"allocated({expr})"


def live_entries(frozen: FrozenSignature, plan: FlattenPlan) -> List:
    """Flatten entries the kernel actually consumes. hlfir-flatten-structs emits
    one entry per describable struct member, including ones no SDFG arg/symbol
    reads (real ICON t_nh_diag: 668 entries vs 508 kernel args). Marshalling a
    dead member is not just waste: an UNDEFINED-association POINTER member (ICON
    Held-Suarez's t2m_bias) makes ASSOCIATED/c_loc UB and trips gfortran's
    internal_pack stack canary. Live iff a flat companion is a kernel arg or symbol."""
    live_names = {a.sdfg_name for a in frozen.args}
    return [e for e in plan.entries if any(f in live_names for f in e.recipe.flat_names)]


def _guarded_copy_out(lines: List[str], guard: str, recipe) -> List[str]:
    """Wrap a non-aliased entry's copy-out in its presence guard; the ABSENT branch
    releases the degenerate scratch the guarded copy-in allocated."""
    if not guard:
        return lines
    out = [f"    if ({guard}) then"]
    out.extend("  " + ln for ln in lines)
    out.append("    else")
    out.extend(f"      deallocate({flat})" for flat in recipe.flat_names)
    out.append("    end if")
    return out


def _presence_scratch_name(dtype: str) -> str:
    """Wrapper-local length-1 target array an ABSENT member's flat POINTER remaps
    onto, so c_loc(<flat>) stays a defined reference."""
    return f"presence_scratch_{dtype}"


def _entry_presence_guard(iface: OriginalInterface, base: str) -> str:
    """associated(<base>)/allocated(<base>) when dotted path base (subscripted
    intermediates allowed) ends in a deferred-storage struct member; '' for a
    plain path. An unallocated member's descriptor is garbage -- c_loc/size on it
    smashes the stack via gfortran's internal_pack (ICON Held-Suarez's
    ddt_ua_*/ddt_va_* tendency pointers). Only the LEAF's storage class is tested;
    Fortran .and. has no guaranteed short-circuit, so a deferred ARRAY-of-records
    intermediate stays unguarded."""
    parts = [p.strip() for p in base.split('%')]
    if len(parts) < 2:
        return ''
    root = parts[0].split('(')[0].strip().lower()
    arg = next((a for a in iface.args if a.name.lower() == root and a.struct_type), None)
    if arg is None:
        return ''
    st = iface.struct_types.get(arg.struct_type)
    for i, seg in enumerate(parts[1:]):
        if st is None:
            return ''
        mname = seg.split('(')[0].strip().lower()
        m = next((mm for mm in st.members if mm.name.lower() == mname), None)
        if m is None:
            return ''
        if i == len(parts) - 2:
            return _present(base, m.alloc == 'pointer') if m.alloc else ''
        st = iface.struct_types.get(m.struct_name) if m.struct_name else None
    return ''


def _recipe_presence_guard(iface: OriginalInterface, recipe) -> str:
    """Presence guard for a flatten recipe's source member (from its first read
    expr, index placeholders stripped). aos_alloc recipes guard per-row internally."""
    if recipe.aos_alloc or not recipe.read_exprs:
        return ''
    return _entry_presence_guard(iface, strip_index_args(recipe.read_exprs[0]))


def _aos_loop_pieces(a):
    """Per-outer-dim loop vars / per-member-dim cap-var names + host element
    accessor for an AoS-component arg. aos_outer_rank == N: N element-index loop
    vars are the SoA buffer's LEADING dims, matching the bridge's
    [element-dims..., member-dims...] layout. aos_outer_rank == 0 (scalar struct
    global): no loop, accessor is the bare struct%member."""
    base = a.sdfg_name
    member_rank = a.rank - a.aos_outer_rank
    if a.aos_outer_rank == 0:
        return [], [], member_rank, f"{a.aos_origin_struct}%{a.aos_member_path}"
    # Fortran identifiers must start with a letter (no leading ``_``).
    its = [f"aos_{base}_i{k}" for k in range(a.aos_outer_rank)]
    caps = [f"aos_{base}_c{j}" for j in range(member_rank)]
    # member_path is %-joined (k / x / a%b): becxx(i0)%k / p_diag%p_vn_dual(i0,i1,i2)%x
    elem = f"{a.aos_origin_struct}({', '.join(its)})%{a.aos_member_path}"
    return its, caps, member_rank, elem


def _aos_member_is_static(a) -> bool:
    """True when the AoS component is a fixed-shape VALUE member (every dim a
    compile-time literal) rather than ALLOCATABLE/POINTER. Unconditionally present,
    so the copy skips the per-element cap-max scan and presence guard."""
    member_dims = a.shape[a.aos_outer_rank:]
    return bool(member_dims) and all(str(d).isdigit() for d in member_dims)


def _render_aos_copy_in(a) -> List[str]:
    """Allocate the SoA buffer (per-member-dim cap = max over elements) + pack the
    host AoS component in. Skips the data copy when global_alloc_inside (kernel
    allocates the component itself; host has no data yet)."""
    its, caps, mrank, elem = _aos_loop_pieces(a)
    if a.aos_outer_rank == 0:
        # SCALAR struct global member, POINTER/ALLOCATABLE, may be undefined on
        # entry. When defined, size the SoA companion from the LIVE member -- a
        # shape-degenerate buffer would under-allocate and the kernel's mesh-bounded
        # writes would smash the heap. When undefined, fall back to a degenerate
        # zero-filled buffer (valid c_loc storage of the right rank).
        member = f"{a.aos_origin_struct}%{a.aos_member_path}"
        zero = _zero_literal(a.dtype)
        out = [f"    ! ----- scalar-struct member: {a.sdfg_name} <- {member} -----"]
        if a.rank and _aos_member_is_static(a):
            # Fixed-shape VALUE member: unconditionally present, no guard needed
            # (allocated/associated are illegal on non-ALLOCATABLE/non-POINTER).
            static_dims = ", ".join(str(d) for d in a.shape[a.aos_outer_rank:])
            out.append(f"    allocate({a.sdfg_name}({static_dims})); {a.sdfg_name} = {member}")
        elif a.rank:
            mp = _present(member, a.aos_member_pointer)
            live_dims = ", ".join(f"size({member}, {k + 1})" for k in range(a.rank))
            degen_dims = ", ".join("1" for _ in range(a.rank))
            out.append(f"    if ({mp}) then")
            out.append(f"      allocate({a.sdfg_name}({live_dims})); {a.sdfg_name} = {member}")
            out.append(f"    else")
            out.append(f"      allocate({a.sdfg_name}({degen_dims})); {a.sdfg_name} = {zero}")
            out.append(f"    end if")
        else:
            out.append(f"    {a.sdfg_name} = {zero}")
        return out
    struct = a.aos_origin_struct
    sp = _present(struct, a.aos_struct_pointer)  # struct allocated/associated?
    zero = _zero_literal(a.dtype)
    # N nested element loops + matching closers; outer-dim allocate specs.
    do_open = [f"      do {itv} = 1, size({struct}, {k + 1})" for k, itv in enumerate(its)]
    do_close = ["      end do" for _ in its]
    outer_dims = ", ".join(f"size({struct}, {k + 1})" for k in range(len(its)))
    one_dims = ", ".join("1" for _ in its)
    idx = ", ".join(its)
    if mrank == 0:
        # SCALAR member of a record array (plain value per element, not
        # allocatable): gather one value per element, no member guard needed.
        # Outer struct may itself be unallocated -> degenerate size-1 buffer.
        out = [
            f"    ! ----- AoS->SoA gather (scalar member): {a.sdfg_name} <- "
            f"{struct}(:)%{a.aos_member_path} -----"
        ]
        out.append(f"    if ({sp}) then")
        out.append(f"      allocate({a.sdfg_name}({outer_dims}))")
        out += do_open
        out.append(f"        {a.sdfg_name}({idx}) = {elem}")
        out += do_close
        out.append(f"    else")
        out.append(f"      allocate({a.sdfg_name}({one_dims})); {a.sdfg_name} = {zero}")
        out.append(f"    end if")
        return out
    if _aos_member_is_static(a):
        # Fixed-shape VALUE member: always present, literal extents -- skip the
        # cap-max scan. Buffer is [outer-dims..., literal-member-dims...].
        member_dims = list(a.shape[a.aos_outer_rank:])
        cap_dims = ", ".join(member_dims)
        slc = ", ".join(f"1:{d}" for d in member_dims)
        out = [
            f"    ! ----- AoS->SoA copy-in (static value member): {a.sdfg_name} <- "
            f"{struct}(:)%{a.aos_member_path} -----"
        ]
        out.append(f"    if ({sp}) then")
        out.append(f"      allocate({a.sdfg_name}({outer_dims}, {cap_dims})); {a.sdfg_name} = {zero}")
        out += do_open
        out.append(f"        {a.sdfg_name}({idx}, {slc}) = {elem}")
        out += do_close
        out.append(f"    else")
        out.append(f"      allocate({a.sdfg_name}({one_dims}, {cap_dims})); {a.sdfg_name} = {zero}")
        out.append(f"    end if")
        return out
    # member_rank > 0: 2D+ ALLOCATABLE/POINTER component per element. Cap = max
    # member extent over elements; struct and component are both guarded.
    mp = _present(elem, a.aos_member_pointer)  # element's component defined?
    out = [f"    ! ----- AoS->SoA copy-in: {a.sdfg_name} <- {struct}(:)%{a.aos_member_path} -----"]
    for c in caps:
        out.append(f"    {c} = 0")
    out.append(f"    if ({sp}) then")
    out += do_open
    if not a.global_alloc_inside:
        out.append(f"        if ({mp}) then")
        for j, c in enumerate(caps):
            out.append(f"          {c} = max({c}, size({elem}, {j + 1}))")
        out.append(f"        end if")
    out += do_close
    out.append(f"    end if")
    for c in caps:
        out.append(f"    if ({c} == 0) {c} = 1")
    cap_dims = ", ".join(caps)
    out.append(f"    if ({sp}) then")
    out.append(f"      allocate({a.sdfg_name}({outer_dims}, {cap_dims}))")
    out.append(f"    else")
    out.append(f"      allocate({a.sdfg_name}({one_dims}, {cap_dims}))")
    out.append(f"    end if")
    out.append(f"    {a.sdfg_name} = {zero}")
    if not a.global_alloc_inside:
        slc = ", ".join(f"1:size({elem}, {j + 1})" for j in range(mrank))
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        if ({mp}) &")
        out.append(f"          {a.sdfg_name}({idx}, {slc}) = {elem}")
        out += do_close
        out.append(f"    end if")
    return out


def _render_aos_copy_out(a) -> List[str]:
    """Pack the SoA buffer back into the host AoS component (only when the arg
    is WRITTEN), allocating each component first if the kernel created it."""
    its, caps, mrank, elem = _aos_loop_pieces(a)
    if a.aos_outer_rank == 0:
        # Scalar-struct member: pack writes back when the host member is defined
        # (companion was sized from it in copy-in); undefined -> nothing to write.
        out = []
        member = f"{a.aos_origin_struct}%{a.aos_member_path}"
        if a.rank and _aos_member_is_static(a):
            # Fixed-shape VALUE member: unconditionally present, no guard needed.
            out.append(f"    {member} = {a.sdfg_name}")
        elif a.rank:
            mp = _present(member, a.aos_member_pointer)
            out.append(f"    if ({mp}) {member} = {a.sdfg_name}")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    struct = a.aos_origin_struct
    sp = _present(struct, a.aos_struct_pointer)
    do_open = [f"      do {itv} = 1, size({struct}, {k + 1})" for k, itv in enumerate(its)]
    do_close = ["      end do" for _ in its]
    idx = ", ".join(its)
    if mrank == 0:
        # Scalar member of a record array: scatter one value per element back.
        out = [
            f"    ! ----- SoA->AoS scatter (scalar member): {struct}(:)%"
            f"{a.aos_member_path} <- {a.sdfg_name} -----"
        ]
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        {elem} = {a.sdfg_name}({idx})")
        out += do_close
        out.append(f"    end if")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    if _aos_member_is_static(a):
        # Fixed-shape VALUE member: always present -- scatter every element back.
        member_dims = list(a.shape[a.aos_outer_rank:])
        slc = ", ".join(f"1:{d}" for d in member_dims)
        out = [
            f"    ! ----- SoA->AoS copy-out (static value member): {struct}(:)%"
            f"{a.aos_member_path} <- {a.sdfg_name} -----"
        ]
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        {elem} = {a.sdfg_name}({idx}, {slc})")
        out += do_close
        out.append(f"    end if")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    mp = _present(elem, a.aos_member_pointer)
    out = [f"    ! ----- SoA->AoS copy-out: {struct}(:)%{a.aos_member_path} <- {a.sdfg_name} -----"]
    out.append(f"    if ({sp}) then")
    out += do_open
    if a.global_alloc_inside:
        alloc_dims = ", ".join(f"size({a.sdfg_name}, {a.aos_outer_rank + j + 1})" for j in range(mrank))
        out.append(f"        if (.not. {mp}) allocate({elem}({alloc_dims}))")
    slc = ", ".join(f"1:size({elem}, {j + 1})" for j in range(mrank))
    out.append(f"        if ({mp}) &")
    out.append(f"          {elem} = {a.sdfg_name}({idx}, {slc})")
    out += do_close
    out.append(f"    end if")
    out.append(f"    deallocate({a.sdfg_name})")
    return out


def _struct_member_symbol_sources(iface: OriginalInterface) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map a struct dummy's member free-symbol to the Fortran expr that reads it.
    A symbolic-only member (loop bound / array extent) gets no FlattenEntry, so this
    rebuilds its name from the static struct_types layout. Bridge naming (``%``->``_``,
    FlattenStructs.cpp designateChainPath): scalar d%m -> d_m; array d%a extent dim i
    -> d_a_d<i> = size(d%a, dim=i+1); lbound -> offset_d_a_d<i>. Nested types recurse
    (d_inner_m <- d%inner%m). Returns (sources, member_paths); the latter backs
    _build_symbol_assigns' _allocated fold for nested struct-member pointers."""
    sources: Dict[str, str] = {}
    member_paths: Dict[str, str] = {}

    def walk(st: DerivedType, sym_prefix: str, access: str) -> None:
        for m in st.members:
            msym = f"{sym_prefix}_{m.name}"
            macc = f"{access}%{m.name}"
            if m.struct_name:
                nested = iface.struct_types.get(m.struct_name)
                if nested is not None:
                    # Array-of-records member must be INDEXED to reach a single
                    # record before descending -- else it's a rank mismatch. Index
                    # EVERY dim (kernel reads the single-domain first element, same
                    # constant index the access generator prepends).
                    idx1 = ", ".join(["1"] * m.rank)
                    walk(nested, msym, f"{macc}({idx1})" if m.rank > 0 else macc)
                continue
            member_paths[msym] = macc
            if m.rank == 0:
                sources[msym] = macc
                continue
            for i in range(m.rank):
                sources[f"{msym}_d{i}"] = f"size({macc}, dim={i + 1})"
                sources[f"offset_{msym}_d{i}"] = f"lbound({macc}, dim={i + 1})"

    for a in iface.args:
        if a.struct_type:
            st = iface.struct_types.get(a.struct_type)
            if st is not None:
                walk(st, a.name, a.name)
    return sources, member_paths


def _build_symbol_assigns(frozen: FrozenSignature, plan: FlattenPlan, outer_dummy_set: set,
                          iface: OriginalInterface) -> List[str]:
    """Emit one assignment per free SDFG symbol from the caller's actual storage:
    offset_<arr>_d<i> -> lbound, <arr>_d<i> -> extent/size. Flatten-plan expr
    preferred when available; falls back to lbound/size on the arg's own Fortran
    expr for plain assumed-shape / non-default lower-bound dummies."""
    _module_sources = effective_module_sources(frozen, iface)
    # Struct-member free symbols with no FlattenEntry (used only as a loop bound /
    # extent, e.g. QE dfftt%ngm) -- sourced from the static struct_types layout.
    _struct_member_sources, _struct_member_paths = _struct_member_symbol_sources(iface)
    # Absent-optional args are degenerate (1,1,...) placeholders: a named blocking
    # symbol must NOT source from one, or every size(<arr>,dim) transient collapses.
    _unsourced_names = {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    arg_by_sdfg = {a.sdfg_name: a for a in frozen.args}
    # SCALAR outer dummies declared OPTIONAL: their <name>_present companion is
    # forwarded from the caller's present(<name>), not defaulted.
    outer_optional_set = {oa.name for oa, _fa in _optional_outer_dummies(frozen, iface)}
    # Cap symbols of aos_alloc recipes are populated by pack-in code before the
    # call -- skip here to avoid a stray TODO or duplicate assignment.
    aos_cap_syms = {
        entry.recipe.cap_symbol
        for entry in plan.entries if entry.recipe.aos_alloc and entry.recipe.cap_symbol
    }
    # <flat>_d<i> is the i-th extent of the flat companion <flat> -- must take
    # that entry's shape_exprs[i] (substring-scanning mis-binds on a multi-member struct).
    flat_shapes: dict = {}
    # A rank-0 entry whose flat companion IS the symbol name is a scalar struct
    # member lifted to a free symbol -- its value is the member itself, read via
    # read_exprs[0] with $i placeholders stripped.
    scalar_member: dict = {}
    # Whether the binding declares each flat's local as POINTER vs allocatable,
    # target (mirrors the struct-arg decl rule) -- selects associated vs allocated
    # for an _allocated fold, since the wrong intrinsic is a hard gfortran type
    # error. Args with no recipe (double-buffer lanes) default to POINTER.
    arg_is_pointer_local: dict = {}
    # Presence guard of each flat's SOURCE member ('' for plain members): extents
    # read the member's descriptor, undefined while absent -- guard, fall back to 0.
    flat_guard: dict = {}
    for entry in plan.entries:
        r = entry.recipe
        is_ptr_local = bool(r.aliasable and r.source_logical_kind in (0, 1))
        guard = _recipe_presence_guard(iface, r)
        for flat in r.flat_names:
            flat_shapes[flat] = r.shape_exprs
            arg_is_pointer_local[flat] = is_ptr_local
            flat_guard[flat] = guard
        if r.rank == 0 and len(r.flat_names) == 1 and r.read_exprs:
            scalar_member[r.flat_names[0]] = strip_index_args(r.read_exprs[0])

    def _guarded_assign(sym: str, rhs: str, guard: str, absent: str = "0") -> List[str]:
        if not guard:
            return [f"    {sym} = int({rhs}, c_int)"]
        return [
            f"    if ({guard}) then",
            f"      {sym} = int({rhs}, c_int)",
            f"    else",
            f"      {sym} = {absent}",
            f"    end if",
        ]

    def _member_sym_guard(sym: str) -> str:
        # Maps offset_<flat>_d<i> / <flat>_d<i> / <flat> to its leaf member path.
        base = sym[len("offset_"):] if sym.startswith("offset_") else sym
        m = _EXTENT_SYM_RE.match(base)
        if m and not _OFFSET_SYM_RE.match(base):
            base = m.group(1)
        path = _struct_member_paths.get(base)
        return _entry_presence_guard(iface, path) if path else ''

    out: List[str] = []
    for sym in frozen.free_symbols:
        if sym in outer_dummy_set:
            continue
        if sym in aos_cap_syms:
            continue
        m = _EXTENT_SYM_RE.match(sym)
        if m and not _OFFSET_SYM_RE.match(sym):
            flat, dim = m.group(1), int(m.group(2))
            shapes = flat_shapes.get(flat)
            if shapes is not None and dim < len(shapes):
                out.extend(_guarded_assign(sym, shapes[dim], flat_guard.get(flat, '')))
                continue
        if sym in scalar_member:
            out.extend(_guarded_assign(sym, scalar_member[sym], flat_guard.get(sym, '')))
            continue
        # A struct-member's lower-bound offset must come from the MEMBER itself,
        # not the c_f_pointer flat companion (whose lbound is always 1): ICON's
        # refinement-control arrays (verts/cells/edges%{start,end}_{block,index})
        # allocate with non-default lower bounds, and offset 1 would read the
        # SDFG's arr[(idx)-offset] out of bounds at negative rl.
        if _OFFSET_SYM_RE.match(sym) and sym in _struct_member_sources:
            # Absent member -> neutral lower bound 1 (never evaluated on the absent branch).
            out.extend(_guarded_assign(sym, _struct_member_sources[sym], _member_sym_guard(sym), absent="1"))
            continue
        # No flatten-plan size expr: derive directly via lbound/size (covers plain
        # assumed-shape / non-default-lower-bound dummies).
        intr = _sym_from_intrinsic(sym, frozen)
        if intr is not None:
            fn, expr, dim = intr
            out.append(f"    {sym} = int({fn}({expr}, dim={dim}), c_int)")
            continue
        # A NAMED array extent must beat the module-global fallback below: the SDFG
        # sized its transients by this symbol, so it must equal the actual
        # allocation, not a possibly-unset namelist global of the same name.
        ext = _sym_from_array_extent(sym, frozen, exclude=_unsourced_names)
        if ext is not None:
            expr, dim = ext
            out.append(f"    {sym} = int(size({expr}, dim={dim}), c_int)")
            continue
        # Last resort: a module global the kernel reads directly, use-imported
        # under the __mod alias -- assign from that import.
        if sym in _module_sources:
            out.append(f"    {sym} = int({_module_symbol_alias(sym)}, c_int)")
            continue
        # Presence of a deferred-storage MODULE global: ALLOCATED/ASSOCIATED(g) ->
        # g_allocated, PRESENT(g) -> g_present. Source from the REAL host so a
        # left-unallocated global takes the ABSENT branch -- the defensive copy-in's
        # degenerate buffer would otherwise look "present".
        pres_base = next((sym[:-len(suf)] for suf in ("_allocated", "_present") if sym.endswith(suf)), None)
        if pres_base is not None:
            fa = arg_by_sdfg.get(pres_base)
            if fa is not None and (getattr(fa, 'module_origin_allocatable', False)
                                   or getattr(fa, 'module_origin_pointer', False)):
                present = _present(_module_symbol_alias(pres_base), getattr(fa, 'module_origin_pointer', False))
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # An AoS-materialised companion's own local is unconditionally
            # allocated (a test on it is meaningless) and is an ALLOCATABLE, so
            # associated(<local>) is a hard gfortran type error -- source presence
            # from the HOST member's real storage class instead.
            if fa is not None and fa.aos_outer_rank > 0 and sym.endswith("_allocated") \
                    and pres_base in _struct_member_paths:
                present = _present(_struct_member_paths[pres_base], fa.aos_member_pointer)
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # A STRUCT-MEMBER ASSOCIATED/ALLOCATED fold the kernel branches on
            # (ICON gates p_nh%prog(nnew)%w's output store on it). Every directly
            # aliased array is c_f_pointer'd to a POINTER local, so associated(base)
            # is true exactly when the caller provided the member -- leaving the
            # flag unset drops the output nondeterministically.
            if fa is not None and fa.kind == 'array' and sym.endswith("_allocated"):
                # A presence-GUARDED entry's flat is always bound (alias or
                # degenerate scratch), so associated(<flat>) is always true --
                # test the HOST member instead via its guard expression.
                host_guard = flat_guard.get(pres_base, '')
                present = host_guard or _present(pres_base, arg_is_pointer_local.get(pres_base, True))
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # A NESTED struct-member POINTER/ALLOCATABLE the kernel branches on --
            # no flat array arg backs it (fa is None), but the static struct layout
            # knows the caller-side dotted path. Pointer default mirrors the
            # array-arg fold above.
            if sym.endswith("_allocated") and pres_base in _struct_member_paths:
                present = _present(_struct_member_paths[pres_base], True)
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
        # An OPTIONAL-presence flag, registered by the bridge for every present(x) fold.
        if sym.endswith("_present"):
            base = sym[:-len("_present")]
            # The wrapper's OWN optional dummy: forward present(base) so a PROVIDED
            # optional reads as present, not the hardwired absent default.
            if base in outer_optional_set:
                out.append(f"    {sym} = int(merge(1, 0, present({base})), c_int)")
                continue
            # No provider, not a wrapper dummy: an inlined-callee optional (QE
            # becphi/becpsi, run_on_gpu). Fortran present() of an omitted optional
            # is .false. -> 0, matching the no-op call path.
            out.append(f"    {sym} = 0  ! optional absent (not forwarded by wrapper)")
            continue
        # A struct dummy's member used only symbolically: no plan entry, but the
        # static struct layout names it -- read straight from the caller's struct.
        if sym in _struct_member_sources:
            out.extend(_guarded_assign(sym, _struct_member_sources[sym], _member_sym_guard(sym)))
            continue
        out.append(f"    ! TODO: no plan entry gives size for free symbol {sym!r}")
    return out
