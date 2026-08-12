#!/bin/bash
# Stage the ICON global (_G) Held-Suarez sweep grids from the MPIM grid
# server into $GRID_DIR/<gridid>/.  LOGIN-NODE ONLY -- compute nodes have no
# internet; the sbatch jobs merely assert the files exist.
#
# Attempts R02B04..R02B09; a resolution with no public _G.nc on the server
# logs GRID_MISSING=<res> and the script carries on (exit stays 0).
# Server truth (probed 2026-08-11): the mpim pool serves the classic ICON-A
# family; the newer rotated 0049..0055 family in mpim_grids.xml is NOT
# hosted there.
#
#   GRID_DIR=... GRIDS="R02B04 R02B05" bash scripts/fetch_icon_grids.sh

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK_ROOT=${WORK_ROOT:-$REPO/samples/_work}
GRID_DIR=${GRID_DIR:-$WORK_ROOT/icon-grids}
BASE=http://icon-downloads.mpimet.mpg.de/grids/public/mpim
GRIDS=${GRIDS:-R02B04 R02B05 R02B06 R02B07 R02B08 R02B09}

grid_id() {
  case "$1" in
    R02B04) echo 0013 ;;
    R02B05) echo 0019 ;;
    R02B06) echo 0021 ;;
    R02B07) echo 0023 ;;
    R02B08) echo 0025 ;;
    R02B09) echo 0015 ;;
    *) return 1 ;;
  esac
}

mkdir -p "${GRID_DIR}"
MAP="${GRID_DIR}/GRIDMAP.tsv"
[ -f "${MAP}" ] || printf '# res\tgrid_id\tfile\n' > "${MAP}"

rc=0
for res in ${GRIDS}; do
  if ! id=$(grid_id "${res}"); then
    echo "GRID_MISSING=${res}  # no grid id mapped"
    continue
  fi
  name="icon_grid_${id}_${res}_G.nc"
  url="${BASE}/${id}/${name}"
  dst="${GRID_DIR}/${id}/${name}"

  status=$(curl -s -o /dev/null -w '%{http_code}' -I "${url}")
  if [ "${status}" != 200 ]; then
    echo "GRID_MISSING=${res}  # HTTP ${status} for ${url}"
    continue
  fi
  remote=$(curl -sI "${url}" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  local_sz=$(stat -c%s "${dst}" 2>/dev/null || echo 0)
  mkdir -p "${GRID_DIR}/${id}"
  if [ -n "${remote}" ] && [ "${local_sz}" = "${remote}" ]; then
    echo "[fetch_icon_grids] ${res} (${id}): already staged, $((local_sz/1048576)) MB"
  else
    echo "[fetch_icon_grids] ${res} (${id}): fetching $((${remote:-0}/1048576)) MB ..."
    # wget -c resumes a partial file from an interrupted earlier run.
    wget -cq -O "${dst}" "${url}" || { echo "FAIL: ${url}" >&2; rc=1; continue; }
  fi
  grep -q "	${id}	" "${MAP}" || printf '%s\t%s\t%s\n' "${res}" "${id}" "${dst}" >> "${MAP}"
done

echo "[fetch_icon_grids] done rc=${rc}; map: ${MAP}"
cat "${MAP}"
exit "${rc}"
