"""Find unordered same-container access pairs inside a single SDFG state.

Two AccessNodes for one container in the same state are a hazard when neither is reachable
from the other: nothing in the graph orders them, so the emitted statement order is a
scheduler tie-break. Source line numbers from debuginfo say which way the Fortran meant it:

    RW   one end writes, the other reads -- a WAR or a RAW, source order decides which
    WAW  two writers
    RAR  harmless, not reported

Run standalone against a built SDFG, or import ``scan``/``report`` from a driver.
"""
import sys
from collections import defaultdict

from dace import SDFG
from dace.sdfg import nodes


def reachable_from(state, source):
    """Every node reachable from ``source`` along dataflow edges."""
    seen = set()
    stack = [source]
    while stack:
        node = stack.pop()
        for edge in state.out_edges(node):
            if edge.dst not in seen:
                seen.add(edge.dst)
                stack.append(edge.dst)
    return seen


def access_line(node, state):
    """Source line of the tasklet/nested SDFG this access node talks to, or None."""
    lines = []
    for edge in state.in_edges(node) + state.out_edges(node):
        other = edge.src if edge.dst is node else edge.dst
        debuginfo = other.debuginfo if isinstance(other, (nodes.Tasklet, nodes.NestedSDFG)) else None
        if debuginfo is not None:
            lines.append(debuginfo.start_line)
    return min(lines) if lines else None


def classify(first, second):
    """Hazard kind for an unordered pair.

    WAR and RAW are not separable after the fact: the frontend's statement order is exactly what
    the missing edge failed to record, and debuginfo here is per-construct boilerplate, not
    per-statement. Report the ambiguous pair as RW; only the emitter knows which way it meant.
    """
    if first['writes'] and second['writes']:
        return 'WAW'
    if not first['writes'] and not second['writes']:
        return None
    return 'RW'


def scan(sdfg):
    """All unordered same-container pairs in ``sdfg``, recursing into nested SDFGs."""
    hazards = []
    for state in sdfg.states():
        by_container = defaultdict(list)
        for node in state.data_nodes():
            by_container[node.root_data].append(node)

        for container, access_nodes in by_container.items():
            if len(access_nodes) < 2:
                continue
            descendants = {id(n): reachable_from(state, n) for n in access_nodes}
            described = [{
                'node': n,
                'writes': state.in_degree(n) > 0,
                'reads': state.out_degree(n) > 0,
                'line': access_line(n, state),
            } for n in access_nodes]

            for i, first in enumerate(described):
                for second in described[i + 1:]:
                    if second['node'] in descendants[id(first['node'])]:
                        continue
                    if first['node'] in descendants[id(second['node'])]:
                        continue
                    kind = classify(first, second)
                    if kind is None:
                        continue
                    hazards.append({
                        'sdfg': sdfg.label,
                        'state': state.label,
                        'container': container,
                        'kind': kind,
                        'lines': (first['line'], second['line']),
                        'roles': (('w' if first['writes'] else '') + ('r' if first['reads'] else ''),
                                  ('w' if second['writes'] else '') + ('r' if second['reads'] else '')),
                    })

    for nested, _ in sdfg.all_nodes_recursive():
        if isinstance(nested, nodes.NestedSDFG):
            hazards.extend(scan(nested.sdfg))
    return hazards


def report(hazards, title=''):
    counts = defaultdict(int)
    for hazard in hazards:
        counts[hazard['kind']] += 1
    print(f'=== {title} ===')
    print('total:', len(hazards), dict(counts))
    for hazard in sorted(hazards, key=lambda h: (h['kind'], h['container'], h['state'])):
        print('  {kind:8s} {container:32s} {state:28s} lines={lines} roles={roles}'.format(**hazard))
    return counts


if __name__ == '__main__':
    for path in sys.argv[1:]:
        report(scan(SDFG.from_file(path)), path)
