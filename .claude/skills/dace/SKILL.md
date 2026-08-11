---
name: dace
description: >-
  Contributor guide for DaCe (SDFG-based compiler IR, spcl/dace). Use whenever
  working in a dace repository: writing or fixing transformations, passes,
  symbolic/symbol handling, subsets/memlets, SDFG serialization, control-flow
  regions, or preparing PRs against spcl/dace main. Covers repo layout and
  branch discipline, sdutil + transformation helpers, symbolic.py semantics,
  graph library (networkx on main, dace.graphlib on extended), and the known
  gotchas that silently miscompile: empty memlets, WCR edges, reset_cfg_list
  cost, symbol dtype identity, determinism rules.
---

# dace

How to contribute to DaCe without breaking things. Read the GOTCHAS section
before touching core files — most silent miscompiles come from the rules there.

## Repos and branch discipline

Two working copies exist on this machine:

- `~/Work/Maintainer/dace` — tracks `spcl/dace` **main**. PRs go here.
- `~/Work/dace` — the **extended** experimental branch. Diverged: has
  `dace/graphlib/`, `vector_inference.py`, `performance_evaluation/`,
  `gpu_specialization/` and more that main lacks. Grep before assuming any
  file:line cited here matches extended.

**Minimal-diff rule (hard):** changes landing in a PR against main must be as
small as possible. Canonicalization/vectorization passes may be changed
heavily when needed; **core files** — `dace/symbolic.py`, `dace/subsets.py`,
`dace/memlet.py`, `dace/sdfg/sdfg.py`, `dace/sdfg/state.py`,
`dace/sdfg/graph.py`, core transformations — only surgically, with a verified
reason per hunk. Do not change main's behavior from the extended branch.

## House workflow rules

- Never push without explicit instruction; ask every time. Never `git stash`,
  never blind `--theirs`/`--ours` on conflicts. No co-author trailers —
  commits carry the user's name only.
- Run `pre-commit run --files <touched files>` before every commit (CI blocks
  merge on formatting). Never `--no-verify`. Never weaken or delete tests.
- pytest with `--maxfail=20`, always with `PYTHONHASHSEED=0` and the MPI env
  prefix or runs hang:
  ```bash
  PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp \
  PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl \
  MPI4PY_RC_INITIALIZE=0 PYTHONPATH=$PWD pytest -q --maxfail=20 tests/...
  ```
- **Always run dace with `PYTHONHASHSEED=0`.** Hash-randomized runs make set
  iteration order — and thus transformation order, codegen, and whether an
  ordering-edge bug even shows up — change run to run. A bug that reproduces
  only under one seed is undebuggable; pin the seed on every dace/python
  invocation (tests, repros, sweeps, pipelines), not just pytest.
- Use the system python3; no pyenv installs. Ask the user for missing
  packages. GPU/mpi4py legs can't run on a CPU-only box — mark them DEFERRED,
  don't skip silently.
- Tests are `tests/**/*_test.py` (GPU: `*_cudatest.py`). CI takes ~40 min;
  after pushing, watch `gh pr checks <N> --repo spcl/dace`.

## Architecture in one breath

- An `SDFG` **is** a `ControlFlowRegion` (`dace/sdfg/sdfg.py:434`). Regions
  are directed graphs of `ControlFlowBlock`s connected by `InterstateEdge`s
  (condition + assignments).
- Blocks are either an `SDFGState` — a dataflow multigraph
  (`OrderedMultiDiConnectorGraph`, nodes + memlet edges with connectors) — or
  a nested region: `ControlFlowRegion`, `LoopRegion` (init/cond/update),
  `ConditionalBlock` (list of (condition, region) branches).
  Break/continue/return are their own blocks. **The hierarchy is recursive**:
  regions inside regions, and `NestedSDFG` nodes inside states hold whole
  child SDFGs. `sdfg.nodes()` sees only ONE region's blocks — use
  `all_control_flow_regions(recursive=True)` to reach if-branches/loop bodies.
- Classes live in `dace/sdfg/state.py` (`SDFGState:1361`,
  `AbstractControlFlowRegion:2643`, `ControlFlowRegion:3206`,
  `LoopRegion:3219`, `ConditionalBlock:3822`) and `dace/sdfg/nodes.py`
  (`NestedSDFG:587`).

## Graph library: main vs extended

- **Main:** networkx, wrapped in DaCe-serializable classes in
  `dace/sdfg/graph.py` (`Edge`/`MultiConnectorEdge`, `DiGraph`,
  `OrderedMultiDiConnectorGraph`, ...). Edges carry typed `.data`;
  `state._nx` is the escape hatch into raw nx algorithms.
- **Extended:** `dace/graphlib/` is a drop-in nx replacement with selectable
  backend (`networkx` default, `rustworkx` opt-in via config `graph.backend`).
  **Write `from dace import graphlib as nx`, not `import networkx as nx`.**
  Transitive closure and max-flow/min-cut always run on real networkx; `.nx`
  handles must deepcopy/serialize.

## Key utilities (learn these before writing a pass)

- `from dace.sdfg import utils as sdutil` — `dace/sdfg/utils.py`, ~70 helpers:
  - traversal: `dfs_topological_sort`, `scope_aware_topological_sort`,
    `find_upstream_nodes`/`find_downstream_nodes`, `weakly_connected_component`,
    `traverse_sdfg_with_defined_symbols`, `postdominators`
  - edge/connector: `change_edge_src`/`change_edge_dest` (rewire, don't
    recreate), `remove_edge_and_dangling_path`, `consolidate_edges`,
    `in_edge_with_name`/`out_desc_with_name`
  - memlets/scopes: `canonicalize_memlet_trees`, `dynamic_map_inputs`,
    `get_global_memlet_path_src/dst`, `is_parallel`, `local_transients`,
    `trace_nested_access`
  - data/symbols: `get_used_data`, `get_used_symbols`, `prune_symbols`
  - structure: `fuse_states`, `inline_sdfgs`, `inline_control_flow_regions`,
    `set_nested_sdfg_parent_references`
- `dace/transformation/helpers.py` — `nest_state_subgraph`, `state_fission`,
  `tile`, `permute_map`, `offset_map`, `redirect_edge`, `unsqueeze_memlet`,
  `split_interstate_edges`, `modified_symbols_between`, `scope_tree_recursive`.
- Transformation framework: `dace/transformation/transformation.py`
  (`PatternTransformation`: `expressions()` / `can_be_applied()` / `apply()`),
  `pass_pipeline.py` (`Pass`, `Pipeline`), `passes/pattern_matching.py`
  (`PatternMatchAndApplyRepeated`), implementations under `dataflow/`,
  `interstate/`, `subgraph/`, `passes/`.
- `dace/symbolic.py` — `symbol(name, dtype)` (default `DEFAULT_SYMBOL_TYPE` =
  int32), `pystr_to_symbolic` (string → sympy, use it, never `sympy.sympify`),
  `symstr`, `issymbolic`, `evaluate`, `simplify` (cached — use this one),
  `SymExpr` (main+overapprox pair), `equal` (tri-state: True/False/None),
  `equalize_symbols`/`inequal_symbols`, `serialize_symbolic` /
  `deserialize_symbolic`, `serialization_symbol_dtypes(authority)` ctxmgr.

## GOTCHAS — the part that bites

1. **Empty memlets are ordering edges, not degenerate data edges.**
   `Memlet.is_empty()` (`dace/memlet.py:231`) — data/subset/other_subset all
   None; used to enforce happens-before (e.g. tasklet→scope, control-flow
   ordering). Every field test on one lands in the wrong branch or raises;
   deleting one makes results `PYTHONHASHSEED`-dependent. ~50 `is_empty()`
   guards exist in transformations — add yours when iterating memlets. Don't
   conflate with `SDFGState.is_empty()` (node count).
2. **WCR memlets are atomics: read-modify-write to the destination**, not pure
   writes (`dace/runtime/include/dace/reduction.h:80` does
   `*ptr = wcr(*ptr, value)`). Dead-store/copy-forward passes that treat them
   as writes silently drop the accumulator read.
3. **Symbol identity is name-based; dtype is NOT part of identity on main.**
   `_eval_subs` compares by name (`symbolic.py:181`). Two same-named symbols
   with different dtypes (the frontend mints map params as int64 via
   `result_type_of`, consumers re-mint int32 via bare `symbol('i')`) alias
   silently — and putting dtype in the hash instead makes 49 tests fail
   (measured; reverted). Rules: never re-mint a symbol from a bare name —
   resolve the instance from the expression's `free_symbols`; compare via
   `equalize_symbols`/`inequal_symbols`, not raw `==` across dtype boundaries;
   wrap serialization in `serialization_symbol_dtypes({name: dtype})` when a
   scope-authoritative dtype exists. SymPy caches (`Function.__new__`,
   `eval()`) return equal-named wrong-dtype exprs — symbol-bearing DaCe
   function expressions are constructed uncached for exactly this reason.
4. **`//` in symbolic strings.** `pystr_to_symbolic` maps `//` to `int_floor`;
   plain `sympy.sympify` gives a rational. Emit `int_floor`/`int_ceil` in
   memlet/shape/interstate strings.
5. **`reset_cfg_list()`/`reset_sdfg_list()` recompute the whole CFG tree and
   propagate to every region** (`state.py:2740`) — expensive. When you
   deepcopy graph parts, fix parents manually instead: set `_parent`,
   `_parent_sdfg`, `parent_nsdfg_node` from the memo (see
   `SDFG.__deepcopy__` `sdfg.py:560`, `NestedSDFG.__deepcopy__`
   `nodes.py:654`). Detached copies have `state.sdfg is None` — helpers
   reading `.sdfg.arrays` break; thread the SDFG explicitly.
6. **Transformations = minimal graph operations.** It's a graph IR: rewire
   edges (`change_edge_src/dest`) instead of remove+recreate; constructors
   regenerate IDs while `__deepcopy__` preserves guid — remove+recreate
   silently changes identity. And a pass/transformation that does NOT apply
   must leave the SDFG bit-identical — decide on a deepcopy if unsure
   (LoopFission once returned "nothing applied" after destroying ordering).
7. **Range/subset bounds must stay symbolic.** `subsets.py` coerces in
   `__init__` and `__setitem__` (`tuple_to_symexpr`, `symbolic_range_tuple`).
   A raw Python int stored in a bound explodes much later in an unrelated
   pass as `'int' object has no attribute 'match'`. Coerce at write time.
8. **Never a plain `set` in a transformation.** Iteration order leaks into
   codegen — same SDFG compiles differently per `PYTHONHASHSEED`. Use
   `OrderedSet` (a setup.py dep). Canonical order = topological sort, ties by
   insertion index; never assert on SDFG hashes.
9. **Never set `sdfg.start_block` manually** — the getter prefers cached /
   unique-source and silently ignores the override → orphaned entry, dominator
   KeyErrors. Use `add_state(..., is_start_block=True)`.
10. **Never reuse a Memlet or Subset object across two edges** — validation
    rejects it (`Duplicate subset detected`); build fresh objects per edge.
11. **Memlet `volume` ≠ subset size** (`memlet.py:30`): volume is a separate
    symbolic property (max if `dynamic`, 0 = unbounded). Check which one a
    pass actually needs.
12. **`scope_dict()` restarts per NestedSDFG** — a node at the top of an
    NSDFG state has no map scope even inside a GPU kernel; thread an
    `in_kernel` flag down instead of trusting scope lookups. Related codegen
    trap: `allocate_array`'s `dfg` arg is the first state the data appears in,
    not the allocation scope.
13. **Interstate-edge scoping:** condition executes BEFORE assignments
    (`sdfg.py:272` `used_symbols`); `i = i + 1` keeps `i` free. Wrong scoping
    loses loop-carried symbols.
14. **Perf foot-guns:** `sdfg.nodes()` is insertion order (don't rely on
    topo); `graph.node_id()` is O(V) — never in loops. `save()` writes a hash
    that `from_json` ignores — no round-trip verification.
15. **`dace.map` is data-parallel by definition** — a cross-iteration
    dependence in a map is a race; it belongs in a sequential loop. The SDFG
    call set is program args ∪ free symbols: symbols aren't in the
    `dace.program` signature but must be passed at call time.
16. **`@lru_cache` always `typed=True`**, never on sympy/mutable objects —
    untyped collapses `1`/`1.0`/`True`/dtype objects onto one key.
17. **Squeeze semantics: a length-1 SLICE keeps its dim, an integer INDEX
    drops it** (numpy): `A[:, 0:1]` → `(N,1)`, `A[:, 0]` → `(N,)`;
    `[0..N, 0..1]` → `(N,1)`, `[0..N, 0]` → `(N,)`. But
    `Range.squeeze()` drops EVERY size-1 dim and cannot tell a slice
    singleton from an index singleton — conflating them is a silent shape
    miscompile (`x.T - x` collapsed `(N,N)`→`(N,)` → zeros). The provenance
    exists only at parse time (`memlet_parser._fill_missing_slices` →
    `MemletExpr.slice_dims`); squeeze sites must pass it as `ignore_indices`
    (fixed at `newast._add_read_slice`; `make_slice` still squeezes blind —
    known open). Related: #7.

## Python frontend: writing a case, and looking at the graph

Every spelling below was executed against this tree before being written down. A skill page that
documents a form the parser rejects is worse than one that says nothing, so re-probe rather than
trust this list after a frontend change.

### The smallest reproducer, and its SDFG

```python
import dace, numpy as np
N = dace.symbol('N', dtype=dace.int64)

@dace.program
def k(a: dace.float64[N], out: dace.float64[N]):
    out[:] = a * 2.0

sdfg = k.to_sdfg(simplify=False)   # PARSE ONLY -- no C++ compiler runs, cheap enough for a sweep
sdfg.validate()                    # the check that a hand-built graph usually fails
csdfg = k.compile()                # separate step; this is what invokes the toolchain
```

`simplify=False` is the right default when reproducing a bug: simplify can fuse the very state or
memlet the report is about. Turn it on only to show the bug survives it.

Getting the picture out:

```python
sdfg.save('k.sdfg')                # JSON; open in the VS Code extension or sdfv
sdfg.save('k.sdfg', readable=True) # diffable -- use in a bug report instead of a screenshot
sdfg.view()                        # opens it directly
k.to_sdfg(save=True)               # writes into the build folder as it parses
print(sdfg.to_json()[:2000])       # when a diff is what you actually want to inspect
```

For a *sweep* over many programs, run each parse in its own SUBPROCESS. DaCe's parse state is
process-global, so one program that wedges or corrupts it makes every later verdict in that process
untrustworthy -- measured: 24 kernels came back as crashes that all parse fine alone.

### Frontend features worth knowing before you hand-build an SDFG

Reach for these first: they say the same thing as a hand-built graph and survive a frontend change.

```python
# Schedule hint on a map -- the @ operator, the one BinOp allowed in a for-iterator
for i in dace.map[0:N] @ dace.ScheduleType.Sequential: ...
for i in dace.map[0:N] @ dace.dtypes.ScheduleType.CPU_Multicore: ...
# (parsed in newast._parse_for_iterator; anything but ast.MatMult there is a DaceSyntaxError)

# Storage hint on a local -- an ANNOTATED assignment whose annotation is a data descriptor.
# visit_AnnAssign reads `.storage` off it and applies it to the resulting array.
buf: dace.data.Array(dace.float64, (N,), storage=dace.StorageType.CPU_ThreadLocal) = np.zeros((N,))

# Transients without an annotation
buf = dace.define_local([N], dace.float64)
acc = dace.define_local_scalar(dace.float64)

# Explicit dataflow when you need the exact tasklet/memlet shape
for i in dace.map[0:N]:
    with dace.tasklet:
        inp << a[i]
        res >> out[i]
        res = inp * 3.0
```

`StorageType` names the memory space (`CPU_Heap`, `CPU_ThreadLocal`, `GPU_Global`, `GPU_Shared`,
`Register`); `ScheduleType` names how a map is executed (`Sequential`, `CPU_Multicore`,
`GPU_Device`, `GPU_ThreadBlock`). Both live in `dace.dtypes` and are re-exported from `dace`.

### Multi-rank runs need per-rank build directories

Ranks compile DIFFERENT SDFGs into the same `.dacecache` and the same precompiled-header cache, and
neither is written atomically. The failure is not only noisy (`FileExistsError`, library-load
errors, a job timing out while one rank waits on a build another is rewriting) -- a rank can load
the `.so` another rank is halfway through writing, and the run VALIDATES WRONG.

```python
rank = os.environ.get('OMPI_COMM_WORLD_RANK') or os.environ.get('SLURM_PROCID')  # also PMI_RANK,
dace.Config.set('default_build_folder', value=f'.dacecache/rank{rank}')          # MV2_COMM_WORLD_RANK
os.environ['DACE_BUILD_CACHE_DIR'] = f'/dev/shm/dace_build_cache_{user}/rank{rank}'
```

`DACE_BUILD_CACHE_DIR` is the root `codegen/build_cache.cache_root` reads; it already defaults to
RAM (`/dev/shm`, falling back to `~/.cache/dace/build_cache`), so rank-suffixing only partitions
what is in memory anyway. Probe every launcher variable -- missing one silently returns the whole
job to a single shared folder.

## Contribution guidelines (from CONTRIBUTING.md)

- Google Python Style Guide; power features allowed; new functions need type
  hints. Supported Python versions: match what `setup.py` declares (check
  before using new syntax).
- **No direct class/function imports**, except `SDFG, SDFGState, Memlet,
  InterstateEdge`. No `import *`. Imports at top; inline import needs an
  adjacent reason comment (e.g. `# Avoid import loop`).
- Sphinx docstrings: blank line before `:param:` blocks or docs break.
- Formatting gate: yapf pep8 column_limit 120 + ruff-check via pre-commit;
  CI enforces it.

## Quick commands

```bash
cd ~/Work/Maintainer/dace
git fetch origin && git status            # main-tracking repo
pre-commit run --files dace/symbolic.py   # format+lint gate
OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp \
PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl \
MPI4PY_RC_INITIALIZE=0 PYTHONPATH=$PWD \
  pytest -q --maxfail=10 tests/python_frontend/  # example shard
gh pr checks <PR> --repo spcl/dace        # CI after push (~40 min)
```
