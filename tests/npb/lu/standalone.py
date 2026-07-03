from pathlib import Path

import dace
import dace_fortran
from dace_fortran.bindings import emit_bindings, FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface

HERE = Path(__file__).resolve().parent
OUT = HERE / "lu_out"  # absolute: _emit_hlfir runs flang with cwd=out_dir
LU = HERE / "lu.F90"

# 1. Load lu.F90 and generate the SDFG.  Entry = module::procedure.
sdfg = dace_fortran.build_sdfg_from_files(
    [LU],
    entry="lu::dolu",
    name="lu",
    out_dir=OUT / "hlfir",
)

# 2. Save the SDFG.
sdfg.save(str(OUT / "lu.sdfg"))

# 3. Generate + save the Fortran binding (emit only -- no link).
compiled = sdfg.compile()
dace_arglist = tuple(getattr(compiled, "_sig", None) or ())
plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw)
iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
binding = OUT / f"{sdfg.name}_bindings.f90"
emit_bindings(sdfg._frozen_signature, iface, plan, str(binding), dace_arglist)

print(f"wrote {OUT}/lu.sdfg and {binding}")
