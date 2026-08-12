#!/bin/bash
# Clone the ICON source we build at its PINNED sha, submodules included, into
# $ICON_SRC.  LOGIN-NODE ONLY -- compute nodes have no internet; the sbatch
# jobs merely assert the clone exists at the pin (same contract as
# fetch_icon_grids.sh).
#
# Idempotent: a clone already at the pin with submodules initialised is left
# alone.  A clone at a DIFFERENT sha is reported loudly and the script fails
# rather than silently reusing or resetting someone's tree.
#
# ICON source is never vendored into this repo: the reproducible tree is
# upstream@pin + scripts/icon_patches/*.patch, applied by the lane configure
# scripts (scripts/configure_icon_{gcc,nvhpc}_daint.sh).  The clone lands in
# the gitignored samples/_work root, so nothing of ICON is ever tracked.
#
#   bash scripts/fetch_icon_source.sh
#   ICON_SRC=/path/to/icon-model bash scripts/fetch_icon_source.sh

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK_ROOT=${WORK_ROOT:-$REPO/samples/_work}
ICON_SRC=${ICON_SRC:-$WORK_ROOT/icon-model}
ICON_URL=${ICON_URL:-https://gitlab.dkrz.de/icon/icon-model.git}
ICON_SHA=${ICON_SHA:-8597da45ef4b86323f3fb844caedc4ae5e1ffc01}
ICON_TAG=${ICON_TAG:-icon-2026.04-public}

echo "[fetch_icon_source] pin ${ICON_TAG} (${ICON_SHA})"

if [ -d "${ICON_SRC}/.git" ]; then
  have=$(git -C "${ICON_SRC}" rev-parse HEAD)
  if [ "${have}" != "${ICON_SHA}" ]; then
    echo "ERROR: ${ICON_SRC} is at ${have}, pin is ${ICON_SHA}" >&2
    echo "       refusing to touch it -- move it aside or set ICON_SHA deliberately." >&2
    exit 1
  fi
  echo "[fetch_icon_source] clone already at the pin"
else
  echo "[fetch_icon_source] cloning ${ICON_URL} -> ${ICON_SRC}"
  git clone "${ICON_URL}" "${ICON_SRC}"
  git -C "${ICON_SRC}" checkout --detach "${ICON_SHA}"
fi

# Submodules carry the externals ICON's build configures (cdi, rte-rrtmgp, ...).
git -C "${ICON_SRC}" submodule update --init --recursive
missing=$(git -C "${ICON_SRC}" submodule status --recursive | grep -c '^-' || true)
[ "${missing}" = 0 ] || { echo "ERROR: ${missing} submodules still uninitialised" >&2; exit 1; }

echo "[fetch_icon_source] ICON_SHA=$(git -C "${ICON_SRC}" rev-parse HEAD)"
echo "[fetch_icon_source] submodules initialised"
echo "[fetch_icon_source] next: the lane configure script applies scripts/icon_patches/*.patch"
