#!/usr/bin/env bash
# download_data.sh -- fetch the R02B05 / nproma=20480 ICON dataset.
#
# Three modes, in order of preference:
#
#   1. ``LOCAL_DATA_DIR`` set -> symlink ``OUTPUT_DIR`` to that path
#      and exit. Useful for dev / CI on hosts that already have the
#      data unpacked elsewhere; avoids the 9.4 GB download entirely.
#   2. ``OUTPUT_DIR`` already populated -> skip (idempotent re-run).
#   3. Otherwise: ``curl`` the tarball from ``URL``, optionally
#      verify SHA256, extract via ``tar -xJf``.
#
# Output goes to ``${EXP_DIR}/data_r02b05`` -- the directory
# ``generate_baselines.py`` expects (--data-dir default).
#
# Env overrides:
#   LOCAL_DATA_DIR      use a local data dir via symlink (no download)
#   URL                 PolyBox / Zenodo / mirror URL
#                       (default: ETH PolyBox)
#   EXPECTED_SHA256     when set, the downloaded tarball is verified
#                       against this checksum before extraction
#   OUTPUT_DIR          override for the extraction destination
#                       (default: ${EXP_DIR}/data_r02b05)

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

URL="${URL:-https://polybox.ethz.ch/index.php/s/WoxiditKJjgdEfR/download/nproma20480_data_files.tar.xz}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/data_r02b05}"
TAR_FILE="${EXP_DIR}/nproma20480_data_files.tar.xz"

# Mode 1: symlink to a local data dir.
if [[ -n "${LOCAL_DATA_DIR:-}" ]]; then
  if [[ ! -d "${LOCAL_DATA_DIR}" ]]; then
    echo "[download_data] LOCAL_DATA_DIR=${LOCAL_DATA_DIR} not a directory" >&2
    exit 1
  fi
  # If OUTPUT_DIR is an empty real dir (e.g. mkdir'd by an earlier
  # run), drop it so the symlink can take its place. Anything else is
  # left alone -- bail rather than clobber populated data.
  if [[ -L "${OUTPUT_DIR}" ]]; then
    rm -f "${OUTPUT_DIR}"
  elif [[ -d "${OUTPUT_DIR}" && -z "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
    rmdir "${OUTPUT_DIR}"
  elif [[ -d "${OUTPUT_DIR}" ]]; then
    echo "[download_data] ${OUTPUT_DIR} already populated; refusing to symlink over it" >&2
    exit 1
  fi
  ln -sfn "${LOCAL_DATA_DIR}" "${OUTPUT_DIR}"
  echo "[download_data] symlinked ${OUTPUT_DIR} -> ${LOCAL_DATA_DIR}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# Mode 2: idempotent skip on already-populated dir.
if [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "[download_data] ${OUTPUT_DIR} already populated; skipping download"
  exit 0
fi

# Mode 3: download.
echo "[download_data] fetching ${URL}"
curl -fL -o "${TAR_FILE}" "${URL}"

if [[ -n "${EXPECTED_SHA256:-}" ]]; then
  echo "[download_data] verifying sha256"
  echo "${EXPECTED_SHA256}  ${TAR_FILE}" | sha256sum -c -
fi

echo "[download_data] extracting to ${OUTPUT_DIR}"
tar -xJf "${TAR_FILE}" -C "${OUTPUT_DIR}"
rm -f "${TAR_FILE}"

echo "[download_data] done. $(find "${OUTPUT_DIR}" -maxdepth 1 -type f | wc -l) file(s) under ${OUTPUT_DIR}"
