import ast
import dace
import re
from typing import Dict, Set, Callable

from dace.codegen.control_flow import ConditionalBlock
from dace.sdfg.state import LoopRegion
from dace.sdfg.sdfg import CodeBlock
import copy
import dace.sdfg.utils as sdutil


def replace_on_interstate_edges(sdfg: dace.SDFG, repl: Dict[str, str]) -> int:
    """Token-bounded, explicit rewrite of every interstate-edge
    expression (assignment keys + RHS, edge condition) as well as every
    ``ConditionalBlock`` branch condition and every ``LoopRegion``
    ``init`` / ``condition`` / ``update`` expression, across all nested
    SDFGs. Use this as a belt-and-suspenders step after
    ``sdfg.replace_dict``: that method goes through ``ast.parse`` +
    ``ASTFindReplace`` and can silently miss references buried in
    nested shape expressions or quoted forms. Returns the total number
    of successful substitutions.

    Uses ``(?<!\\w)<name>(?!\\w)`` so partial identifier matches (e.g.
    ``foo`` inside ``foo_bar``) are left alone.
    """
    if not repl:
        return 0

    compiled = {
        old: re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
        for old in repl
    }
    count = 0

    def _sub(text):
        nonlocal count
        if not text:
            return text
        out = text
        for old, pat in compiled.items():
            new_text, n = pat.subn(str(repl[old]), out)
            if n:
                out = new_text
                count += n
        return out

    for nested in sdfg.all_sdfgs_recursive():
        # Interstate edges on every control-flow region.
        for cfg in nested.all_control_flow_regions():
            for edge in cfg.edges():
                ise = edge.data
                assigns = ise.assignments or {}
                if assigns:
                    new_assigns = {}
                    for k, v in assigns.items():
                        new_assigns[_sub(k)] = _sub(v) if isinstance(v, str) else v
                    ise.assignments = new_assigns
                if ise.condition is not None:
                    old_cond = ise.condition.as_string
                    new_cond = _sub(old_cond)
                    if new_cond != old_cond:
                        ise.condition = CodeBlock(new_cond)

        # ConditionalBlock branch conditions.
        for block in nested.all_control_flow_blocks():
            if isinstance(block, ConditionalBlock):
                for i, (cond, body) in enumerate(block.branches):
                    if cond is None:
                        continue
                    old = cond.as_string
                    new = _sub(old)
                    if new != old:
                        block.branches[i] = (CodeBlock(new), body)

        # LoopRegion init / condition / update expressions.
        for region in nested.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            for attr in ("loop_condition", "init_statement", "update_statement"):
                code = getattr(region, attr, None)
                if code is None:
                    continue
                old = code.as_string
                new = _sub(old)
                if new != old:
                    setattr(region, attr, CodeBlock(new))

    return count


def _strip_suffix_if_matches(s: str) -> str:
    match = re.search(r"_s_\d+$", s)
    if match:
        return s[: match.start() + 2]  # Do not remove _s
    return s


def _array_same(
    a_name1: str, a_name2: str, array1: dace.data.Array, array2: dace.data.Array
) -> bool:
    if a_name1 != a_name2:
        return False
    if array1.dtype != array2.dtype:
        return False
    if array1.shape != array2.shape:
        return False
    if array1.transient != array2.transient:
        return False
    if array1.storage != array2.storage:
        return False
    return True


def _scalar_same(
    a_name1: str, a_name2: str, scalar1: dace.data.Scalar, scalar2: dace.data.Scalar
) -> bool:
    if a_name1 != a_name2:
        return False
    if scalar1.dtype != scalar2.dtype:
        return False
    if scalar1.transient != scalar2.transient:
        return False
    if scalar1.storage != scalar2.storage:
        return False
    return True


def _container_array_same(
    a_name1: str,
    a_name2: str,
    ca1: dace.data.ContainerArray,
    ca2: dace.data.ContainerArray,
) -> bool:
    if a_name1 != a_name2:
        return False
    if ca1.shape != ca2.shape:
        return False
    if ca1.dtype != ca2.dtype:
        return False
    assert isinstance(ca1, dace.data.Structure)
    assert isinstance(ca2, dace.data.Structure)
    s1: dace.data.Structure = ca1.stype
    s2: dace.data.Structure = ca2.stype
    same = _struct_same(s1.name, s2.name, s1, s2)
    return same


def _struct_same(
    s_name1: str,
    s_name2: str,
    struct1: dace.data.Structure,
    struct2: dace.data.Structure,
) -> bool:
    if s_name1 != s_name2:
        # print(f"Struct names do not match: {s_name1} != {s_name2}")
        return False
    if struct1.dtype != struct2.dtype:
        # print(f"Struct dtypes do not match: {s_name1}: {struct1.dtype} != {struct2.dtype}")
        return False
    if struct1.shape != struct2.shape:
        # print(f"Struct shapes do not match: {s_name1}: {struct1.shape} != {struct2.shape}")
        return False
    for member_name, member in struct1.members.items():
        if member_name not in struct2.members:
            # print(f"Member {member_name} of {s_name1} not found in struct {s_name2}")
            return False
        member2 = struct2.members[member_name]
        if type(member) is not type(member2):
            # print(f"Member types do not match: {member_name} in {s_name1} is {type(member)} but in {s_name2} is {type(member2)}")
            return False
        if isinstance(member, dace.data.Array):
            same = _array_same(member_name, member_name, member, member2)
            if not same:
                # print(f"Array {member_name} in {s_name1} does not match with {s_name2}")
                return False
        elif isinstance(member, dace.data.Scalar):
            same = _scalar_same(member_name, member_name, member, member2)
            if not same:
                # print(f"Scalar {member_name} in {s_name1} does not match with {s_name2}")
                return False
        elif isinstance(member, dace.data.Structure):
            same = _struct_same(member_name, member_name, member, member2)
            if not same:
                # print(f"Structure {member_name} in {s_name1} does not match with {s_name2}")
                return False
        elif isinstance(member, dace.data.ContainerArray):
            same = _container_array_same(member_name, member_name, member, member2)
            if not same:
                # print(f"ContainerArray {member_name} in {s_name1} does not match with {s_name2}")
                return False
    return True


def _symbols_same(s1: Set[str], s2: Set[str]) -> bool:
    if len(s1) != len(s2):
        return False
    for sym in s1:
        if sym not in s2:
            return False
    return True


def _prune_name_of_structs(sdfg: dace.SDFG) -> None:
    pass


def _print_names(name: str, struct: dace.data.Structure, depth: int) -> None:
    "  " * depth
    # print(f"{indent}Name: {name}, Type: {struct.name}")
    for member_name, member in struct.members.items():
        if isinstance(member, dace.data.Structure):
            _print_names(member_name, member, depth + 1)
        elif isinstance(member, dace.data.ContainerArray):
            # print(f"{indent}  Member Container Array: {member_name}")
            _print_names(member_name, member.stype, depth + 1)
        else:
            # print(f"{indent}  Member: {member_name}, Type: {member.dtype}, Transient: {member.transient}, Storage: {member.storage}")
            pass


def _repl_struct_names(
    sdfg: dace.SDFG, name: str, struct: dace.data.Structure, depth: int
) -> None:
    "  " * depth
    _new_members = dict()
    for member_name, member in struct.members.items():
        # print(f"{indent}Processing member: {member_name}: {member_name} => {_strip_suffix_if_matches(member_name)}")
        new_member_name = _strip_suffix_if_matches(member_name)
        # print(f"{indent}Name (From) {member_name} => (TO) {new_member_name}")
        _new_members[new_member_name] = member
    struct.members = _new_members

    for member_name, member in struct.members.items():
        # print(member_name, type(member))
        if isinstance(member, dace.data.Structure):
            _repl_struct_names(sdfg, member_name, member, depth + 1)
        elif isinstance(member, dace.data.ContainerArray):
            _repl_struct_names(sdfg, member_name, member.stype, depth + 1)
        else:
            if not isinstance(member, dace.data.Array) and not isinstance(
                member, dace.data.Scalar
            ):
                raise TypeError(
                    f"Unsupported member type {type(member)} in structure {struct.name}"
                )

    # if depth == 0:
    #    #print(struct)

    if depth == 0:
        ##print(struct.members)
        assert name in sdfg.arrays.keys(), (
            f"Structure (name) {name} not in arrays of SDFG {sdfg.name}"
        )
        assert struct in sdfg.arrays.values(), (
            f"Structure (desc) {struct.name} not in arrays of SDFG {sdfg.name}"
        )
        sdfg.arrays[name] = struct


def _rename_on_if_conds(sdfg: dace.SDFG, src: str, dst: str, recursive=False):
    gpu_host_name_map = {src: dst}

    for _, node in (
        enumerate(sdfg.nodes()) if not recursive else sdfg.all_nodes_recursive()
    ):
        if not isinstance(node, ConditionalBlock):
            continue

        for b in node.branches:
            if b[0] is None:
                continue
            if isinstance(b[0].code, list):
                for i, el in enumerate(b[0].code):
                    if isinstance(el, str):
                        for src, dst in gpu_host_name_map.items():
                            b[0].code[i] = b[0].code[i].replace(src, dst)
                    else:

                        def replace_x_with_y(expr: ast.Expr, repl_dict) -> ast.Expr:
                            expr_str = ast.unparse(expr).strip()
                            for src, dst in repl_dict.items():
                                modified_str = expr_str.replace(src, dst)
                            return ast.parse(modified_str, mode="eval").body

                        b[0].code[i] = replace_x_with_y(b[0].code[i], gpu_host_name_map)
            else:
                assert isinstance(b[0].code, str)
                for src, dst in gpu_host_name_map.items():
                    b[0].code = b[0].code.replace(src, dst)


def _rename_symbols(sdfg: dace.SDFG, check_func: Callable) -> None:
    # Build the symbol mapping
    new_symbols = {}
    for sym_name in sdfg.symbols:
        new_sym = check_func(sym_name)
        if new_sym != sym_name:
            new_symbols[sym_name] = new_sym

    # Ensure replaced on ifs
    sdfg.replace_dict(repldict=new_symbols)
    for old_s, new_s in new_symbols.items():
        _rename_on_if_conds(sdfg, old_s, new_s, False)

    # Go to Nested SDFGs, also fix the symbol mapping
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                node.symbol_mapping = {
                    check_func(k): check_func(v) for k, v in node.symbol_mapping.items()
                }

            if isinstance(node, dace.nodes.NestedSDFG):
                _rename_symbols(node.sdfg, check_func)


def _rename_init_and_global_code(sdfg: dace.SDFG, check_func: Callable) -> None:

    new_global_code = dict()
    new_init_code = dict()

    pattern = r"_s_\d+ "
    replacement = "_s "

    for k, v in sdfg.global_code.items():
        ##print(k, v)
        new_key = re.sub(pattern, replacement, k)
        if isinstance(v, CodeBlock):
            new_value = CodeBlock(
                re.sub(pattern, replacement, v.as_string),
                language=dace.dtypes.Language.CPP,
            )
        else:
            new_value = CodeBlock(
                re.sub(pattern, replacement, v), language=dace.dtypes.Language.CPP
            )
        new_global_code[new_key] = new_value
        if new_key != k:
            del new_global_code[k]

    for k, v in sdfg.init_code.items():
        ##print(k, v)
        new_key = re.sub(pattern, replacement, k)
        if isinstance(v, CodeBlock):
            new_value = CodeBlock(
                re.sub(pattern, replacement, v.as_string),
                language=dace.dtypes.Language.CPP,
            )
        else:
            new_value = CodeBlock(
                re.sub(pattern, replacement, v), language=dace.dtypes.Language.CPP
            )
        new_init_code[new_key] = new_value
        if new_key != k:
            del new_init_code[k]

    sdfg.init_code = new_init_code
    sdfg.global_code = new_global_code


def prune_names(sdfg: dace.SDFG) -> dace.SDFG:
    new_sdfg = copy.deepcopy(sdfg)
    sdutil.set_nested_sdfg_parent_references(new_sdfg)
    cname = "p_int"
    for arr_name, arr in new_sdfg.arrays.items():
        if arr_name == cname:
            _print_names(arr_name, arr, 0)

        if isinstance(arr, dace.data.Structure):
            # print("Repl names for structure:", arr_name)
            _repl_struct_names(new_sdfg, arr_name, arr, 0)

        if arr_name == cname:
            _print_names(arr_name, arr, 0)

    _rename_symbols(new_sdfg, _strip_suffix_if_matches)
    _rename_init_and_global_code(new_sdfg, _strip_suffix_if_matches)

    # AH fuck it, just replace on the json
    new_sdfg.save("/tmp/after_prune.sdfg.json", compress=False)
    with open("/tmp/after_prune.sdfg.json", "r", encoding="utf-8") as f:
        json_str = f.read()

    # Step 2: Remove all _s_<number> patterns
    cleaned_str = re.sub(r"_s_\d+", "_s", json_str)

    # Step 4: Save the cleaned JSON to a new file
    with open("/tmp/after_prune.sdfg.json", "w", encoding="utf-8") as f:
        f.write(cleaned_str)

    new_sdfg = dace.SDFG.from_file("/tmp/after_prune.sdfg.json")

    # raise Exception("Pruned names, but not yet tested. Please test this function before removing this exception.")
    new_sdfg.validate()
    # return new_sdfg
    # new_sdfg.save("after_prune.sdfg.json", compress=False)
    # raise Exception("Pruned names, but not yet tested. Please test this function before removing this exception.")
    return new_sdfg


def compare_structs_from_paths(sdfgs: list[str]):
    sdfg0 = dace.SDFG.from_file(sdfgs[0])
    for i in range(1, len(sdfgs)):
        sdfg1 = dace.SDFG.from_file(sdfgs[i])
        for name, arr in sdfg0.arrays.items():
            if isinstance(arr, dace.data.Structure) and not isinstance(
                arr, dace.data.View
            ):
                if name not in sdfg1.arrays:
                    raise KeyError(f"Array {name} not found in SDFG {sdfgs[i]}")
                arr1 = sdfg1.arrays[name]
                if not _struct_same(name, name, arr, arr1):
                    raise ValueError(
                        f"Arrays {name} do not match between SDFGs {sdfgs[0]} and {sdfgs[i]}"
                    )


def compare_structs(sdfgs: list[dace.SDFG]):
    sdfg0 = sdfgs[0]
    for i in range(1, len(sdfgs)):
        sdfg1 = sdfgs[i]
        for name, arr in sdfg0.arrays.items():
            if isinstance(arr, dace.data.Structure) and not isinstance(
                arr, dace.data.View
            ):
                if name not in sdfg1.arrays:
                    raise KeyError(f"Array {name} not found in SDFG {sdfgs[i]}")
                arr1 = sdfg1.arrays[name]
                if not _struct_same(name, name, arr, arr1):
                    raise ValueError(
                        f"Arrays {name} do not match between SDFGs {sdfgs[0]} and {sdfgs[i]}"
                    )
