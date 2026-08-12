"""Generate ``include/call_velocity.h`` from a DaCe-generated variant header.

The 4 compiled variants share an identical 231-parameter signature (verified
at runtime by ``unify_variant_signatures``). This tool parses that signature
once from one variant's generated header, cross-references
``include/velocity_tendencies_no_nproma.h`` to know the loaded struct layout,
and emits a single macro ``VELOCITY_CALL_ARGS(...)`` that expands to the 231
call-site expressions.

``main_per.cu`` uses the macro 4 times (once per variant) for both
``__dace_init`` and ``__program`` calls.

Naming rules (verified against the velocity_tendencies kernel):

* ``__CG_<c>__m_<leaf>``              -> ``<c>.<leaf>`` (auto-address if the
                                         signature expects a pointer and the
                                         struct field is a value)
* ``__CG_<c>__m_SA_<leaf>_d_<N>``     -> ``<c>.__f2dace_SA_<leaf>_d_<N>_s_<fortran_line>``
* ``__CG_<c>__m_SOA_<leaf>_d_<N>``    -> same with ``SOA``
* ``__CG_p_patch__CG_<sub>__m_<leaf>``-> ``p_patch.<sub>-><leaf>`` (and
                                         equivalents for ``SA_`` / ``SOA_``)
* ``__CG_p_patch__CG_cells__CG_decomp_info__m_owner_mask``
                                     -> ``p_patch.cells->decomp_info->owner_mask``
* ``SA_<leaf>_d_<N>_<sub>_p_patch_<k>``   -> ``p_patch.<sub>->__f2dace_SA_<leaf>_d_<N>_s_<line>``
* ``SOA_<leaf>_d_<N>_<sub>_p_patch_<k>``  -> same with ``SOA``
* ``A_<arr>_d_<N>``                   -> ``serde::ARRAY_META_DICT_AT(<arr>).size.at(<N>)``
* ``OA_<arr>_d_<N>``                  -> ``serde::ARRAY_META_DICT_AT(<arr>).lbound.at(<N>)``
* Bare scalar parameter (``dtime`` / ``istep`` / ``z_kin_hor_e`` / ...) ->
                                         same name.

The ``_s_<line>`` suffix in the struct's ``__f2dace_SA_*`` fields is the
Fortran source line where the dimension was declared -- unambiguous for a
given (leaf, dim) within one struct, so we don't need to disambiguate.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_STRUCT_HEAD_RE = re.compile(r"struct\s+(\w+)\s*\{")
_FIELD_RE = re.compile(r"^\s*(.+?)(\s*\*+\s*|\s+)(\w+)\s*(?:=\s*\{\}\s*)?;", re.M)
_SA_SOA_FIELD_RE = re.compile(r"^(__f2dace_(?:SA|SOA)_.+?_d_\d+)_s_\d+$")


def _struct_bodies(text: str):
    """Yield (name, body) pairs. Uses brace counting so ``= {}`` initializers
    don't prematurely close the struct."""
    for head in _STRUCT_HEAD_RE.finditer(text):
        name = head.group(1)
        start = head.end()  # just after the opening '{'
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            # Only yield "true" struct definitions (end with "};"). Forward
            # declarations skip the body and won't produce a match here.
            if i < len(text) and text[i] == ";":
                yield name, text[start : i - 1]


class Struct:
    def __init__(self, name: str):
        self.name = name
        # fname -> (base_type, is_pointer)
        self.fields: Dict[str, Tuple[str, bool]] = {}
        # (prefix "__f2dace_SA_<leaf>_d_<N>") -> full field name
        self.f2dace: Dict[str, str] = {}

    def add_field(self, base: str, ptr: bool, name: str):
        self.fields[name] = (base, ptr)
        m = _SA_SOA_FIELD_RE.match(name)
        if m:
            self.f2dace[m.group(1)] = name


def parse_structs(header_path: Path) -> Dict[str, Struct]:
    text = header_path.read_text()
    structs: Dict[str, Struct] = {}
    for name, body in _struct_bodies(text):
        s = Struct(name)
        for fm in _FIELD_RE.finditer(body):
            base = fm.group(1).strip()
            ptr = "*" in fm.group(2)
            fname = fm.group(3)
            if base.startswith("struct "):
                base = base[len("struct ") :]
            s.add_field(base, ptr, fname)
        structs[name] = s
    return structs


# Container token (as it appears in __CG_<container>__...) -> struct type
# declared in include/velocity_tendencies_no_nproma.h.
_CONTAINER_TYPES = {
    "global_data": "global_data_type",
    "p_diag": "t_nh_diag",
    "p_int": "t_int_state",
    "p_metrics": "t_nh_metrics",
    "p_patch": "t_patch",
    "p_prog": "t_nh_prog",
}

# Subcontainer name under p_patch -> struct field name + pointed-to struct.
# From include/velocity_tendencies_no_nproma.h the names are cells/edges/verts.
_P_PATCH_SUBS = {"cells": "t_grid_cells", "edges": "t_grid_edges", "verts": "t_grid_vertices"}


def _parse_decl_args(header_text: str, signature_re: str) -> List[Tuple[str, bool, str]]:
    m = re.search(signature_re, header_text, re.S)
    if not m:
        raise SystemExit(f"couldn't find declaration matching {signature_re}")
    raw = m.group(1)
    parts = [p.strip() for p in raw.split(",")]
    out: List[Tuple[str, bool, str]] = []
    for p in parts:
        toks = p.split()
        name = toks[-1]
        rest = " ".join(toks[:-1]).strip()
        ptr = "*" in rest
        base = rest.replace("*", " ").replace("__restrict__", " ").strip()
        base = re.sub(r"\s+", " ", base)
        out.append((base, ptr, name))
    return out


def parse_signature(header_path: Path, fn_name_re: str) -> List[Tuple[str, bool, str]]:
    """``__program_*`` signature, skipping the leading state arg."""
    text = header_path.read_text()
    out = _parse_decl_args(text, rf"void {fn_name_re}\((.*?)\);")
    # drop state arg
    return out[1:]


def parse_init_signature(header_path: Path, variant: str) -> List[Tuple[str, bool, str]]:
    """``__dace_init_*`` signature. No state arg; may be a strict subset of
    ``__program_*`` on branches where DaCe only passes scalar init args."""
    text = header_path.read_text()
    # Match "<anything_not_semicolon>__dace_init_<variant>(<args>);"
    return _parse_decl_args(
        text, rf"\*?\s*__dace_init_{variant}\((.*?)\);"
    )


def _struct_field(struct: Struct, leaf: str) -> Tuple[str, bool]:
    if leaf not in struct.fields:
        raise KeyError(f"{struct.name} has no field {leaf!r}")
    return struct.fields[leaf]


def _resolve_sa_soa(struct: Struct, prefix: str) -> str:
    """``prefix`` = ``__f2dace_SA_<leaf>_d_<N>`` or ``__f2dace_SOA_<leaf>_d_<N>``."""
    full = struct.f2dace.get(prefix)
    if full is None:
        raise KeyError(f"{struct.name} has no SA/SOA field {prefix!r}")
    return full


def map_param(
    structs: Dict[str, Struct], cpp_type_base: str, is_pointer: bool, name: str
) -> str:
    """Return the call-site C++ expression for ``name``."""
    # Bare scalars / arrays passed by their local name
    if name in (
        "z_kin_hor_e",
        "z_vt_ie",
        "z_w_concorr_me",
        "dt_linintp_ubc",
        "dtime",
        "istep",
        "ldeepatmo",
        "lvn_only",
        "ntnd",
    ):
        return name

    # A_/OA_ array meta (sizes / lbounds of the top-level z_* arrays)
    m = re.match(r"^(A|OA)_(z_\w+?)_d_(\d+)$", name)
    if m:
        kind, arr, d = m.group(1), m.group(2), m.group(3)
        prop = "size" if kind == "A" else "lbound"
        return f"serde::ARRAY_META_DICT_AT({arr}).{prop}.at({d})"

    # Bare SA_/SOA_ with trailing _<sub>_p_patch_<k>
    m = re.match(r"^(SA|SOA)_(.+?)_d_(\d+)_(cells|edges|verts)_p_patch_\d+$", name)
    if m:
        kind, leaf, d, sub = m.group(1), m.group(2), m.group(3), m.group(4)
        sub_struct = structs[_P_PATCH_SUBS[sub]]
        full = _resolve_sa_soa(sub_struct, f"__f2dace_{kind}_{leaf}_d_{d}")
        return f"p_patch.{sub}->{full}"

    # __CG_p_patch__CG_<sub>__CG_decomp_info__m_<leaf>  -- owner_mask lives here
    m = re.match(
        r"^__CG_p_patch__CG_(cells|edges|verts)__CG_decomp_info__m_(\w+)$", name
    )
    if m:
        sub, leaf = m.group(1), m.group(2)
        return f"p_patch.{sub}->decomp_info->{leaf}"

    # __CG_p_patch__CG_<sub>__m_<leaf> (either a pointer field or SA_/SOA_ size)
    m = re.match(r"^__CG_p_patch__CG_(cells|edges|verts)__m_(.+)$", name)
    if m:
        sub, tail = m.group(1), m.group(2)
        sub_struct = structs[_P_PATCH_SUBS[sub]]
        tm = re.match(r"^(SA|SOA)_(.+?)_d_(\d+)$", tail)
        if tm:
            full = _resolve_sa_soa(
                sub_struct, f"__f2dace_{tm.group(1)}_{tm.group(2)}_d_{tm.group(3)}"
            )
            return f"p_patch.{sub}->{full}"
        base, ptr = _struct_field(sub_struct, tail)
        expr = f"p_patch.{sub}->{tail}"
        if is_pointer and not ptr:
            expr = f"&{expr}"
        return expr

    # __CG_<container>__m_<leaf>
    m = re.match(r"^__CG_(\w+?)__m_(.+)$", name)
    if m:
        container, tail = m.group(1), m.group(2)
        struct_name = _CONTAINER_TYPES.get(container)
        if struct_name is None or struct_name not in structs:
            raise KeyError(f"unknown container {container!r}")
        s = structs[struct_name]
        # try SA/SOA size match first
        tm = re.match(r"^(SA|SOA)_(.+?)_d_(\d+)$", tail)
        if tm:
            full = _resolve_sa_soa(
                s, f"__f2dace_{tm.group(1)}_{tm.group(2)}_d_{tm.group(3)}"
            )
            var = _container_local(container)
            return f"{var}.{full}"
        base, ptr = _struct_field(s, tail)
        var = _container_local(container)
        expr = f"{var}.{tail}"
        if is_pointer and not ptr:
            expr = f"&{expr}"
        return expr

    raise ValueError(f"unmapped param {name!r}")


def _container_local(container: str) -> str:
    return container  # main_per.cu already names them this way


_HEADER = """\
// Auto-generated by tools/gen_call_site.py. Do not edit by hand.
//
// Two macros for the 4 velocity tendencies variants:
//   - VELOCITY_CALL_ARGS  : full argument list for ``__program_*``
//   - VELOCITY_INIT_ARGS  : argument list for ``__dace_init_*`` (on some DaCe
//                           branches this is a strict subset -- scalars only)
// The generated SDFGs share identical signatures across variants
// (unify_variant_signatures enforces this), so one macro per entry fits all
// four variants.

#ifndef __VELOCITY_TENDENCIES_CALL_VELOCITY_H__
#define __VELOCITY_TENDENCIES_CALL_VELOCITY_H__

#include "serde_velocity_no_nproma.h"

"""

_FOOTER = """
#endif // __VELOCITY_TENDENCIES_CALL_VELOCITY_H__
"""


_MACRO_SHARED_PARAMS = (
    "global_data, p_diag, p_int, p_metrics, p_patch, p_prog, "
    "z_kin_hor_e, z_vt_ie, z_w_concorr_me, "
    "dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only, ntnd"
)


def _emit_one_macro(macro: str, exprs: List[str]) -> str:
    indent = " " * 4
    out = [f"#define {macro}({_MACRO_SHARED_PARAMS}) \\"]
    for i, e in enumerate(exprs):
        sep = "," if i + 1 < len(exprs) else ""
        out.append(f"{indent}{e}{sep} \\")
    if len(out) > 1:
        out[-1] = out[-1].rstrip(" \\")
    return "\n".join(out) + "\n"


def emit_macro(
    out_path: Path,
    call_exprs: List[str],
    init_exprs: Optional[List[str]] = None,
):
    text = _HEADER + _emit_one_macro("VELOCITY_CALL_ARGS", call_exprs) + "\n"
    if init_exprs is not None:
        text += _emit_one_macro("VELOCITY_INIT_ARGS", init_exprs) + "\n"
    text += _FOOTER
    out_path.write_text(text)


def generate(
    struct_header: Path,
    variant_header: Path,
    fn_name: str,
    out_path: Path,
) -> Tuple[int, int]:
    """Emit the call-site macros (CALL + INIT). Returns (n_call, n_init)."""
    structs = parse_structs(struct_header)
    call_params = parse_signature(variant_header, fn_name)
    call_exprs = [map_param(structs, base, ptr, name) for base, ptr, name in call_params]

    # ``fn_name`` is the program entry (``__program_<variant>``). Derive the
    # variant name to locate the matching ``__dace_init_<variant>``.
    variant = fn_name[len("__program_"):] if fn_name.startswith("__program_") else None
    init_exprs: Optional[List[str]] = None
    if variant:
        init_params = parse_init_signature(variant_header, variant)
        init_exprs = [map_param(structs, base, ptr, name) for base, ptr, name in init_params]

    emit_macro(out_path, call_exprs, init_exprs)
    return len(call_exprs), len(init_exprs) if init_exprs is not None else 0


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--header",
        default="velocity_tendencies_no_nproma.h",
        help="struct-definitions header under include/",
    )
    argp.add_argument(
        "--variant-header",
        default=".dacecache/velocity_no_nproma_if_prop_lvn_only_0_istep_1/include/velocity_no_nproma_if_prop_lvn_only_0_istep_1.h",
    )
    argp.add_argument(
        "--fn",
        default="__program_velocity_no_nproma_if_prop_lvn_only_0_istep_1",
    )
    argp.add_argument("--out", default="include/call_velocity.h")
    args = argp.parse_args()

    n_call, n_init = generate(
        Path("include") / args.header,
        Path(args.variant_header),
        args.fn,
        Path(args.out),
    )
    print(f"wrote {args.out} with VELOCITY_CALL_ARGS={n_call}, VELOCITY_INIT_ARGS={n_init}")


if __name__ == "__main__":
    main()
