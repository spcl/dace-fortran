#!/bin/bash
# Re-apply the VERIFIED manual fixes to this sample's generated artifacts and
# rebuild the .so chain.  Pattern-based (not line-anchored like the .patch),
# so it survives regenerated identifier numbering; idempotent, so running it
# on already-fixed artifacts is a no-op.
#
# Fixes applied (full narrative: NUMERICAL_CORRECTNESS.md; upstream items:
# ../../dace-fortran-fixes-needed-6c99810.md):
#   bindings f90 : *_present symbols 0 -> 1; abort-on-_r guard clause ->
#                  comment (flattened r/i input checks demand present _r
#                  dummies; verify drivers pass zeros)
#   kernel cpp   : copy-ins for declared-but-never-assigned scal_* locals;
#                  eigts1/2/3 negative-lbound offsets (-1 -> +(d0-1)/2)
# then: recompile kernel .so, patchelf soname, stage-6 relink of the binding.
#
#   usage: ./apply_manual_fixes.sh          (from the sample directory)
#   env:   KERNEL (default: this directory's name)
set -e
cd "$(dirname "$0")"
KERNEL=${KERNEL:-$(basename "$PWD")}
LIB=$PWD/outputs/lib
BINDINGS=$LIB/${KERNEL}_bindings.f90
CPP=$PWD/.dacecache/$KERNEL/src/cpu/$KERNEL.cpp
[ -f "$BINDINGS" ] || { echo "ERROR: $BINDINGS not found" >&2; exit 1; }
[ -f "$CPP" ]      || { echo "ERROR: $CPP not found" >&2; exit 1; }

python3 - "$BINDINGS" "$CPP" <<'EOF'
import re, sys
bindings, cpp = sys.argv[1], sys.argv[2]

# ---------------- bindings: presence symbols + _r guard ----------------
s = open(bindings).read()
n_pres = 0
def fix_present(m):
    global n_pres
    n_pres += 1
    name = m.group(1)
    comment = ("FIXED: char-folding flattens the r/i input checks too"
               if name.endswith('_r') else
               "FIXED: guard enforces presence; data IS forwarded (c_loc below)")
    return f"    {name}_present = 1  ! {comment}"
s = re.sub(r"^    (\w+)_present = 0  ! optional absent \(not forwarded by wrapper\)$",
           fix_present, s, flags=re.M)
guard_pat = re.compile(
    r"    if \(present\(bec\w+_r\)[^\n]*\) then\n"
    r"(?:.*\n)*?"
    r"    end if\n",
    re.M)
GUARD_REPL = ("    ! (guard on real bec arrays removed: the flattened input checks demand\n"
              "    !  present _r dummies; the gamma compute arms are not in this SDFG)\n")
s, n_guard = guard_pat.subn(GUARD_REPL, s, count=1)
open(bindings, 'w').write(s)
print(f"bindings: {n_pres} presence symbol(s) fixed, {n_guard} guard clause(s) removed"
      + ("" if (n_pres or n_guard) else "  (already fixed)"))

# ---------------- kernel cpp: scal_* copy-ins + eigts lbound offsets ----
s = open(cpp).read()
n_copy = 0
for typ, var in re.findall(r"^\s*(bool|int|double) (scal_\w+);", s, flags=re.M):
    src = var[len("scal_"):]
    if re.search(rf"{var} = |&{var}\b", s):
        continue  # already initialized (CopyND or a previous run of this script)
    s = re.sub(rf"^(\s*{typ} {var};)$",
               rf"\1 {var} = {src}[0];  /* FIXED: missing copy-in */",
               s, count=1, flags=re.M)
    n_copy += 1
n_eig = 0
for i in (1, 2, 3):
    pat = re.compile(rf"(eigts{i}\[\(\(\(eigts{i}_d0 \* \(_loop_it_\d+ - 1\)\) \+ mill_at\d+\)) - 1\)\]")
    s, n = pat.subn(rf"\1 + ((eigts{i}_d0 - 1) / 2))]  /* FIXED: lbound -nr offset */", s)
    n_eig += n
open(cpp, 'w').write(s)
print(f"kernel cpp: {n_copy} copy-in(s) inserted, {n_eig} eigts offset(s) fixed"
      + ("" if (n_copy or n_eig) else "  (already fixed)"))
EOF

echo "== recompile kernel .so =="
# flag set from .dacecache/<K>/build/compile_commands.json (the CMake cache
# there is stale, so the command is replayed directly)
D=$PWD/.dacecache/$KERNEL
c++ -DDACE_BINARY_DIR=\""$D/build"\" -D${KERNEL}_EXPORTS -I/opt/dace/dace/runtime/include \
    -Wno-unused-parameter -Wno-unused-label -march=native -fno-math-errno -fno-trapping-math \
    -fno-signed-zeros -freciprocal-math -O3 -DNDEBUG -std=c++20 -fPIC -fopenmp -shared \
    "$CPP" -o "$D/build/lib$KERNEL.so"
cp "$D/build/lib$KERNEL.so" "$LIB/libdace_kernel_$KERNEL.so"
patchelf --set-soname "libdace_kernel_$KERNEL.so" "$LIB/libdace_kernel_$KERNEL.so"

echo "== relink binding .so (stage-6 recipe) =="
GFC="-shared -fPIC -ffree-line-length-none -O3 -g -fno-fast-math -ffp-contract=off -frounding-math -fopenmp -fallow-argument-mismatch -std=legacy"
( cd "$LIB" && gfortran $GFC -J. mpi_stub_gnu.f90 ../$KERNEL.F90 ${KERNEL}_bindings.f90 \
    -o "lib$KERNEL.so" -L. -Wl,-rpath,'$ORIGIN' -l:libdace_kernel_$KERNEL.so 2>/dev/null )

echo "done: $LIB/lib$KERNEL.so + libdace_kernel_$KERNEL.so rebuilt with fixes"
echo "now verify:  ./build_verify.sh && ./verify_*  data/<MAT>   (must PASS before use)"
