import copy
import dace
from dace.codegen.control_flow import ConditionalBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion
from dace.codegen.targets.unroller import product
from dace.properties import CodeBlock
from dace.transformation.passes import DeadStateElimination, SymbolPropagation
from dace.transformation.passes.constant_propagation import ConstantPropagation
import ast
import operator as op

def extract_variables_from_ast(expr_str):
    """
    Extracts all variable names from the provided expression string using AST.
    :param expr_str: The expression string to analyze.
    :return: A set of variable names (strings).
    """
    # Parse the expression string into an AST
    tree = ast.parse(expr_str, mode='eval')

    # Set to store variable names
    variables = set()

    # Define a helper function to recursively visit AST nodes
    def visit_node(node):
        if isinstance(node, ast.Name):  # Variables are represented by ast.Name
            variables.add(node.id)
        # Recursively visit children nodes
        for child in ast.iter_child_nodes(node):
            visit_node(child)

    # Visit the root node of the AST
    visit_node(tree.body)

    return variables

# Example usage
expr_str = "((1 - lvn_only) and 0)"

# Define environment with some variables
def eval_expr_any_value(expr_str, env=None):
    """
    Evaluates a string expression where undefined variables can have any value.
    :param expr_str: The expression to evaluate.
    :param env: The environment (dictionary) of variables with their values.
    :return: Evaluated result or symbolic result for undefined variables.
    """
    if env is None:
        env = {}

    # Parse the expression string into an AST
    tree = ast.parse(expr_str, mode='eval')

    # Define a helper function to safely evaluate the AST
    def eval_node(node, env):
        if isinstance(node, ast.Expression):
            return eval_node(node.body, env)

        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand, env)
            return op.neg(operand) if isinstance(node.op, ast.UAdd) else op.pos(operand)

        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left, env)
            right = eval_node(node.right, env)

            # Check if any operand is an undefined variable (symbolic placeholder)
            if left == "undefined_var" or right == "undefined_var":
                return "undefined_var"  # Return a symbolic placeholder

            if isinstance(node.op, ast.Add):
                return op.add(left, right)
            elif isinstance(node.op, ast.Sub):
                return op.sub(left, right)
            elif isinstance(node.op, ast.Mult):
                return op.mul(left, right)
            elif isinstance(node.op, ast.Div):
                return op.truediv(left, right)

        elif isinstance(node, ast.BoolOp):  # Handling boolean operations (and, or)
            values = [eval_node(value, env) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)

        elif isinstance(node, ast.Name):  # Handling variables
            # Check if variable is in the environment
            if node.id in env:
                return env[node.id]
            else:
                # If the variable is not defined, assume it can take any value
                return "undefined_var"  # Symbolic placeholder for an undefined variable

        elif isinstance(node, ast.Constant):  # Handling constants (numbers, strings)
            return node.value

        else:
            raise TypeError(f"Unsupported AST node type: {type(node)}")

    return eval_node(tree.body, env)

def evaluate_with_possible_values(expr_str, possible_values):
    # Extract variables from the expression
    variables = extract_variables_from_ast(expr_str)

    # Create all combinations of 0 and 1 for the variables
    value_lists = [[int(possible_values[var])] for var in variables]
    all_combinations = list(product(*value_lists))

    # Store the results for each combination
    results = {}

    # Iterate over all combinations of 0 and 1
    for combination in all_combinations:
        # Create an environment with the current combination
        env = dict(zip(variables, combination))

        # Evaluate the expression for the current environment
        result = eval_expr_any_value(expr_str, env)

        # Store the result for the current combination of variables
        results[tuple(combination)] = result

    results_eval = set(results.values())
    return results_eval

def evaluate_interstate_assignments_and_ifs(graph : dace.SDFG | ControlFlowRegion, prop_dict, verbose):
    for e in graph.edges():
        if isinstance(e.data, dace.InterstateEdge):
            new_assignments = dict()
            for assignment, expr_str in e.data.assignments.items():
                try:
                    variables = extract_variables_from_ast(expr_str)
                    if len(variables) == 0:
                        raise Exception("No variables")
                    simplified = evaluate_with_possible_values(expr_str, prop_dict)
                    if len(simplified) == 1:
                        simplified = simplified.pop()
                        if isinstance(simplified, int):
                            simplified = bool(simplified)
                    else:
                        raise ValueError("Eval Error")
                except Exception as ex:
                    simplified = expr_str
                try:
                    simplified=  str(eval(simplified))
                except Exception as ex:
                    simplified = simplified
                if "(1 - 0)" == simplified:
                    simplified = "1"
                elif simplified is False:
                    simplified = "0"
                elif simplified is True:
                    simplified = "1"
                elif simplified == "False":
                    simplified = "0"
                elif simplified == "True":
                    simplified = "1"
                elif "(1 == 1) == 1" == simplified:
                    simplified = "1"
                elif "((1 == 1) == 1)" == simplified:
                    simplified = "1"
                elif "((1 - 0) and 0)" == simplified:
                    simplified = "0"
                elif "(1 - 0) and 0" == simplified:
                    simplified = "0"
                elif "(1 - 1) and 0" == simplified:
                    simplified = "0"
                elif "((1 - 1) and 0)" == simplified:
                    simplified = "0"
                elif "not (((1 < 1))) or (1 > 2))" == simplified:
                    simplified = "0"
                elif "(not (((1 < 1) or (1 > 2))))" == simplified:
                    simplified = "0"
                elif "not (((1 < 1) or (1 > 2)) == 1)" == simplified:
                    simplified = "1"
                elif "(not (((1 < 1) or (1 > 2)) == 1))" == simplified:
                    simplified = "1"
                elif simplified == "1 == 1":
                    simplified = "1"
                if simplified != expr_str:
                    if verbose:
                        print(f"{assignment}: {expr_str} ({type(expr_str)}) -> {simplified}")


                if simplified == False or simplified == 0 or simplified == "0":
                    new_assignments[assignment] = "0"
                else:
                    new_assignments[assignment] = str(simplified)

            e.data.assignments = new_assignments

def rename_on_if_conds(node: ConditionalBlock, src: str, dst: str):
    gpu_host_name_map = {src: dst}

    for b in node.branches:
        if b[0] is None:
            continue
        if isinstance(b[0].code, list):
            for i, el in enumerate(b[0].code):
                if isinstance(el, str):
                    for src,dst in gpu_host_name_map.items():
                        b[0].code[i] = b[0].code[i].replace(src,dst)
                else:
                    def replace_x_with_y(expr: ast.Expr, repl_dict) -> ast.Expr:
                        expr_str = ast.unparse(expr).strip()
                        for src, dst in repl_dict.items():
                            modified_str = expr_str.replace(src, dst)
                        return ast.parse(modified_str, mode="eval").body
                    b[0].code[i] = replace_x_with_y(b[0].code[i], gpu_host_name_map)
        else:
            assert isinstance(b[0].code, str)
            for src,dst in gpu_host_name_map.items():
                b[0].code = b[0].code.replace(src, dst)

# I think I will just detect the "AccessNode -> Tasklet -> AccessNode"
# with memlet.data having the name we want and the destination node
# having the name "if*" in this case we can be pretty sure it is if
# access on what we want then I can assign the value on the interstate
# edge and rely on simplify, I hope this will work
def propagate_if_cond(root: dace.SDFG, sdfg: dace.SDFG, replace_dict: None | dict, verbose):
    sdfg.validate()

    if replace_dict is not None:
        sdfg.replace_dict(replace_dict)
        prop_dict = replace_dict
        # It makes global_data.lextra_diffu
        # to global_data.1 on an edge fix that
        for s in sdfg.states():
            for e in s.edges():
                for src, dst in replace_dict.items():
                    if e.data.data.endswith(f".{dst}"):
                        e.data.data = e.data.data.replace(f".{dst}", f".{src}")

        def repl_assign_if(sdfg: dace.SDFG):
            for s in sdfg.states():
                for n in s.nodes():
                    if n not in s.nodes():
                        continue
                    if isinstance(n, dace.nodes.AccessNode):
                        if len(s.out_edges(n)) == 1:
                            dst1 = s.out_edges(n)[0].dst
                            if (len(s.out_edges(dst1)) == 1 and
                                isinstance(dst1, dace.nodes.Tasklet)):
                                dst2 = s.out_edges(dst1)[0].dst
                                if (len(s.out_edges(dst2)) == 0 and
                                    isinstance(dst2, dace.nodes.AccessNode) and
                                    dst2.label.startswith("_if")):
                                    # We have a match
                                    # Access node -> Tasklet -> Access node
                                    for name, dstexpr in replace_dict.items():
                                        if name in n.data:
                                            if verbose:
                                                print(f"Matched: {n.data} -> {dst1.label} -> {dst2.data} on name {name}")
                                            s.remove_node(dst1)
                                            s.remove_node(dst2)
                                            ies = s.in_edges(n)
                                            s.remove_node(n)
                                            oes = s.parent_graph.out_edges(s)
                                            oe = oes[0]
                                            assert dst2.data not in oe.data.assignments
                                            for ie in ies:
                                                if s.in_degree(ie.src) == 0 and s.out_degree(ie.src) == 0:
                                                    s.remove_node(ie.src)
                                            d = sdfg.arrays[dst2.data]
                                            #sdfg.remove_data(dst2.data, validate=False)
                                            oname = dst2.data
                                            sdfg.remove_data(dst2.data, validate=False)
                                            sdfg.add_symbol(name=oname + "_sym", stype=d.dtype)
                                            oe.data.assignments[oname + "_sym"] = dstexpr

                                            for oe in oes:
                                                dst_node = oe.dst
                                                if isinstance(dst_node, ConditionalBlock):
                                                    rename_on_if_conds(dst_node, oname, oname + "_sym")
        if root == sdfg:
            repl_assign_if(sdfg)
        #for state in sdfg.states():
        #    for node in state.nodes():
        #        if isinstance(node, dace.nodes.NestedSDFG):
        #            repl_assign_if(node.sdfg)

        sdfg.validate()

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    propagate_if_cond(sdfg, node.sdfg, replace_dict, prop_dict, verbose)
        sdfg.validate()

        # Evaluate interstate assignments everywhere
        evaluate_interstate_assignments_and_ifs(sdfg, prop_dict, verbose)
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowRegion):
                evaluate_interstate_assignments_and_ifs(node, prop_dict, verbose)
            if isinstance(node, dace.nodes.NestedSDFG):
                evaluate_interstate_assignments_and_ifs(node.sdfg, prop_dict, verbose)
        sdfg.validate()

        # return # here validates

        ConstantPropagation().apply_pass(sdfg, {})
        sdfg.validate()


    #return # here validates

    for cfg in sdfg.all_control_flow_blocks():
        if cfg not in sdfg.all_control_flow_blocks():
            continue
        if isinstance(cfg, ConditionalBlock):
            ss = []
            #if cfg.branches is not None:
            assert len(cfg.branches) == 1 or len(cfg.branches) == 2

            if len(cfg.branches) == 1:
                s = ""
                if cfg.branches[0][0] is not None:
                    for i in range(len(cfg.branches[0][0].code)):
                        s += " and " + ast.unparse(cfg.branches[0][0].code[i])
                ss.append(s[len(" and "):])
                assert len(ss) == 1
                # If length of ss is 1 and we have 1 == 1 or (1 == 1) == 1 then we can cut out body
                if len(ss) == 1:
                    always_true = False
                    always_false = False
                    try:
                        cond = eval(ss[0])
                        if cond is True:
                            always_true = True
                        if cond is False:
                            always_false = True
                    except Exception as ex:
                        pass
                    if always_false:
                        branch, body =  cfg.branches[0]
                        for n in body.nodes():
                            body.remove_node(n)
                        cfg.remove_branch(body)
                        s = cfg.parent_graph.add_state("cond_repl")
                        for ie in cfg.parent_graph.in_edges(cfg):
                            cfg.parent_graph.add_edge(ie.src, s, copy.deepcopy(ie.data))
                        for oe in cfg.parent_graph.out_edges(cfg):
                            cfg.parent_graph.add_edge(s, oe.dst, copy.deepcopy(oe.data))
                        cfg.parent_graph.remove_node(cfg)
                    # Makes access problematic key not found
                    #if always_true:
                    #    cfg.branches[0][0].code = ["1 == 1"]
                    #if always_true:
                    #    cfg.branches[0] = (CodeBlock("1 == 1"), cfg.branches[0][1])
            if len(cfg.branches) == 2:
                branch0, body0 = cfg.branches[0]
                branch1, body1 = cfg.branches[1]
                s0 = ""
                if branch0 is not None:
                    for i in range(len(branch0.code)):
                        s0 += " and " + ast.unparse(branch0.code[i])
                if s0 != "":
                    s0 = s0[len(" and "):]
                s1 = ""
                if branch1 is not None:
                    for i in range(len(branch1.code)):
                        s1 += " and " + ast.unparse(branch1.code[i])
                if s1 != "":
                    s1 = s1[len(" and "):]
                assert not(s0 == "" and s1 == "")
                assert not(s0 != "" and s1 != "")

                else_branch, else_body = None, None
                c_branch, c_body = None, None
                cond = None
                if s0 == "":
                    else_branch = branch0
                    else_body = body0
                    c_branch = branch1
                    c_body = body1
                    cond = s1
                if s1 == "":
                    else_branch = branch1
                    else_body = body1
                    c_branch = branch0
                    c_body = body0
                    cond = s0
                always_true = False
                always_false = False
                try:
                    if eval(cond) is True:
                        always_true = True
                    if eval(cond) is False:
                        always_false = True
                except Exception as ex:
                    pass
                assert always_false or always_true or (not always_false and not always_true)
                # if always_false add the else branch
                if always_false or always_true:
                    ncfg = ConditionalBlock(cfg.label, cfg.sdfg, cfg.parent_graph)

                if always_false:
                    copy_else_body = copy.deepcopy(else_body)
                    for s in copy_else_body.all_states():
                        for n in s.nodes():
                            if isinstance(n, dace.nodes.NestedSDFG):
                                n.sdfg.parent_graph = copy_else_body
                                n.sdfg.parent_sdfg = copy_else_body.sdfg
                    ncfg.add_branch(CodeBlock("1 == 1"), copy_else_body)
                elif always_true:
                    copy_c_body = copy.deepcopy(c_body)
                    for s in copy_c_body.all_states():
                        for n in s.nodes():
                            if isinstance(n, dace.nodes.NestedSDFG):
                                n.sdfg.parent_graph = copy_c_body
                                n.sdfg.parent_sdfg = copy_c_body.sdfg
                    ncfg.add_branch(CodeBlock("1 == 1"), copy_c_body)

                if always_false or always_true:
                    ies = cfg.parent_graph.in_edges(cfg)
                    oes = cfg.parent_graph.out_edges(cfg)
                    for ie in ies:
                        cfg.parent_graph.add_edge(ie.src, ncfg, copy.deepcopy(ie.data))
                    for oe in oes:
                        cfg.parent_graph.add_edge(ncfg, oe.dst, copy.deepcopy(oe.data))
                    cfg.parent_graph.remove_node(cfg)

    sdfg.reset_cfg_list()
    sdfg.validate()

    # Remove all cfg nodes that are 0 == 1
    # DeadStateElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # return # here validates
    # return

    # All non trurthy ifs have been removed, now copy-out the body of trurthy ifs
    # e.g. cond is 1 == 1
    for cfg in sdfg.all_control_flow_blocks():
        if isinstance(cfg, ConditionalBlock):
            ss = []
            #if cfg.branches is not None:
            for j in range(len(cfg.branches)):
                s = ""
                if cfg.branches[j][0] is not None:
                    for i in range(len(cfg.branches[j][0].code)):
                        s += " " + ast.unparse(cfg.branches[j][0].code[i])
                ss.append(s[1:])

            # If length of ss is 1 and we have 1 == 1 or (1 == 1) == 1 then we can cut out body
            if len(ss) == 1:
                if verbose:
                    print(cfg.label, ", cond:", ss)
                try:
                    evaluated = eval(ss[0]) is True
                except Exception as ex:
                    evaluated = False
                if ss[0] == "1 == 1" or ss[0] == "(1 == 1) == 1" or evaluated:
                    node_map = dict()
                    assert len(cfg.branches) == 1
                    body = cfg.branches[0][1]
                    parent_graph = cfg.parent_graph
                    src_edges = set()
                    dst_edges = set()
                    for ie in parent_graph.in_edges(cfg):
                        src_edges.add(ie)
                    for oe in parent_graph.out_edges(cfg):
                        dst_edges.add(oe)

                    #sdfg.remove_node(cfg)
                    parent_graph.remove_node(cfg)

                    for n in body.nodes():
                        node_map[n] = copy.deepcopy(n)
                        for _s in node_map[n].all_states() if (not isinstance(node_map[n], dace.SDFGState) and not isinstance(node_map[n], ContinueBlock)) else [node_map[n]]:
                            for _n in _s.nodes():
                                if isinstance(_n, dace.nodes.NestedSDFG):
                                    _n.sdfg.parent_graph = parent_graph
                                    _n.sdfg.parent_sdfg = sdfg

                        parent_graph.add_node(node_map[n])

                    for e in body.edges():
                        assert e.src in node_map
                        assert e.dst in node_map
                        parent_graph.add_edge(node_map[e.src], node_map[e.dst], copy.deepcopy(e.data))

                    new_src_nodes = set([n for n in node_map.values() if parent_graph.in_degree(n) == 0])
                    new_dst_nodes = set([n for n in node_map.values() if parent_graph.out_degree(n) == 0])

                    assert len(new_src_nodes) == 1
                    assert len(new_dst_nodes) == 1
                    #assert len(src_edges) <= 1, f"{[(e.src, e.dst) for e in src_edges]}"
                    assert len(dst_edges) <= 1, f"{[(e.src, e.dst) for e in dst_edges]}"

                    new_src_node = new_src_nodes.pop()
                    if len(src_edges) > 0:
                        for src_edge in src_edges:
                            parent_graph.add_edge(src_edge.src, new_src_node, copy.deepcopy(src_edge.data))

                    if len(dst_edges) == 1:
                        dst_edge = dst_edges.pop()
                        new_dst_node = new_dst_nodes.pop()
                        parent_graph.add_edge(new_dst_node, dst_edge.dst, copy.deepcopy(dst_edge.data))

    sdfg.validate()

    # If continue block remove all following nodes
    # Clean all nodes reachable from a continue block
    fixpoint = True
    while fixpoint:
        fixpoint = False
        for n in sdfg.all_control_flow_blocks():
            if isinstance(n, ContinueBlock):
                parent = n.parent_graph
                if parent.out_degree(n) > 0:
                    fixpoint = True
                    nodes_to_rm = set()

                    stack = set([n])
                    while len(stack) > 0:
                        _n = stack.pop()
                        nodes_to_rm.add(_n)
                        for e in parent.out_edges(_n):
                            stack.add(e.dst)
                    for _n in nodes_to_rm:
                        parent.remove_node(_n)
                    break
    #for cfg in sdfg.all_control_flow_blocks():
    #    if isinstance(cfg, ConditionalBlock):
    #        print(cfg.label, len(cfg.branches))
    #        s = ""
    #        for i in range(len(cfg.branches[0][0].code)):
    #        s += " " + ast.unparse(cfg.branches[0][0].code[i]) + "\n"
    sdfg.validate()
