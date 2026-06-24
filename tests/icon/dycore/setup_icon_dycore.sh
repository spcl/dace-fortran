#!/usr/bin/env bash
# Reproducible setup for the ICON dynamical-core (mo_solve_nonhydro) parse
# test.  We do NOT need a working ICON binary -- only enough of a build that
# flang can cpp-preprocess + emit HLFIR for the dycore TU and its USE-closure.
# So `make` is allowed to fail late: `bear` captures compile_commands.json
# (the per-TU -cpp / -I / -D flags + build order) as make issues each
# command, which is all the bridge's tier-3 emitter consumes.
#
# Usage:  ICON_DIR=/path/to/checkout BUILD_DIR=/path/to/build ./setup_icon_dycore.sh
set -euo pipefail

ICON_TAG=${ICON_TAG:-icon-2026.04-public}
ICON_URL=${ICON_URL:-https://gitlab.dkrz.de/icon/icon-model.git}
HERE=$(cd "$(dirname "$0")"; pwd)
# Default ``ICON_DIR`` to the in-tree submodule checkout at
# ``tests/icon/full/icon-model``; the build dir lands next to this
# script so the parse test owns its own state (no workspace-level
# ``$HOME/_icon_build`` -- gone with the layout change of 2026-06).
# ``REPO`` walks up: <script>/../../.. -> repo root.
REPO=$(cd "$HERE/../../.."; pwd)
ICON_DIR=${ICON_DIR:-$REPO/tests/icon/full/icon-model}
BUILD_DIR=${BUILD_DIR:-$HERE/.icon_build}

# 1. System deps (Debian/Ubuntu).  YAXT + CDI are bundled ICON submodules
#    (built by make), so they are NOT apt packages.
if [ "${SKIP_APT:-0}" != 1 ]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    git ca-certificates bear m4 autoconf rsync python3 \
    gfortran libopenmpi-dev openmpi-bin \
    libnetcdf-dev libnetcdff-dev libhdf5-dev \
    libxml2-dev libfyaml-dev liblapack-dev libblas-dev zlib1g-dev
  # flang-new-21 (the HLFIR emitter) -- LLVM apt repo; adjust per distro.
  command -v flang-new-21 >/dev/null || echo "WARN: install flang-new-21 (LLVM 21) for the bridge"
fi

# 2. Submodule init for the in-tree ICON checkout; a custom out-of-tree
#    ``$ICON_DIR`` is shallow-cloned at the pinned tag.  Either way, the
#    nested ICON submodules (yaxt, cdi, mtime, tixi, ...) get pulled in.
if [ "$ICON_DIR" = "$REPO/tests/icon/full/icon-model" ]; then
  git -C "$REPO" submodule update --init --recursive tests/icon/full/icon-model
else
  if [ ! -d "$ICON_DIR/.git" ]; then
    git clone --depth 1 --branch "$ICON_TAG" "$ICON_URL" "$ICON_DIR"
  fi
  git -C "$ICON_DIR" submodule update --init --recursive
fi

# 3. The flang-new config wrapper (MPI Fortran wrapper -> flang via OMPI_FC).
install -m755 "$HERE/config/generic_flang" "$ICON_DIR/config/generic/flang"

# 4. Configure for compile_commands capture only -- disable components the
#    dycore does not need so make reaches mo_solve_nonhydro sooner.
#
#    LIBS override: ICON's config wrappers hardcode ``-leccodes`` into ``LIBS``
#    (for GRIB2), and autoconf's "whether the Fortran compiler works" probe
#    LINKS its trivial program against the full ``LIBS`` -- so with GRIB2
#    disabled (eccodes intentionally NOT in this lean dep set) that probe fails
#    with "Fortran compiler cannot create executables" before any ICON source
#    is touched.  The wrappers take ``LIBS=${LIBS-...}``, so exporting it here
#    (eccodes dropped; every remaining lib is apt-installed) wins and lets the
#    probe link.  GRIB2 is off anyway, so the dycore TU never needs eccodes.
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
LIBS="${LIBS:--lxml2 -lfyaml -llapack -lblas -lnetcdff -lnetcdf -lstdc++}" \
"$ICON_DIR/config/generic/${ICON_FC:-gcc}" \
  --disable-ecrad --disable-art --disable-jsbach --disable-coupling \
  --disable-grib2 --disable-rttov --without-external-yac

# 5. Capture compile_commands.json.  `|| true`: make may fail after the
#    dycore -- we only need the commands bear recorded up to that point.
bear -- make -j"$(nproc)" || true
echo "compile_commands.json: $BUILD_DIR/compile_commands.json"
grep -c mo_solve_nonhydro "$BUILD_DIR/compile_commands.json" 2>/dev/null \
  && echo "dycore TU captured" || echo "WARN: dycore not reached -- inspect the make failure"
