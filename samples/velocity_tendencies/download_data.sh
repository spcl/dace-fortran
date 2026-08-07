#!/usr/bin/env bash
# Fetch the R02B05 velocity_tendencies serde dataset (nproma=81920; cells 81920 /
# edges 122880 / verts 40962).  3-mode pattern from
# VelocityTendenciesPipeline/tools/download_data.sh: LOCAL_DATA_DIR symlink /
# idempotent skip / curl + optional EXPECTED_SHA256 + tar.  Patch applies in the
# download mode only; symlinked or pre-populated data is assumed already patched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL="${URL:-https://polybox.ethz.ch/index.php/s/WoxiditKJjgdEfR/download/nproma20480_data_files.tar.xz}"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/data_r02b05}"
TAR_FILE="${OUTPUT_DIR%/}.tar.xz"

# ICON_VELOCITY_DATA_DIR: site-local checkout probe (e.g. an icon-artifacts clone), no baked default.
if [[ -z "${LOCAL_DATA_DIR:-}" && -d "${ICON_VELOCITY_DATA_DIR:-}" ]]; then
  LOCAL_DATA_DIR="${ICON_VELOCITY_DATA_DIR}"
fi

if [[ -n "${LOCAL_DATA_DIR:-}" ]]; then
  if [[ ! -d "${LOCAL_DATA_DIR}" ]]; then
    echo "[download_data] LOCAL_DATA_DIR=${LOCAL_DATA_DIR} not a directory" >&2
    exit 1
  fi
  # Drop an empty leftover dir; never clobber populated data.
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

if [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "[download_data] ${OUTPUT_DIR} already populated; skipping download"
  exit 0
fi

echo "[download_data] fetching ${URL}"
curl -fL -o "${TAR_FILE}" "${URL}"

if [[ -n "${EXPECTED_SHA256:-}" ]]; then
  echo "${EXPECTED_SHA256}  ${TAR_FILE}" | sha256sum -c -
fi

echo "[download_data] extracting to ${OUTPUT_DIR}"
tar -xJf "${TAR_FILE}" -C "${OUTPUT_DIR}"
rm -f "${TAR_FILE}"

# Mandatory struct-alignment patch, inlined from icon-artifacts/velocity/
# patch_out_is_associated.sh (spcl/icon-artifacts @ 78c6b52d): strips records the
# serde structs do not carry, so the reader walks the files member-for-member.
echo "[download_data] patching serde records"
(
  cd "${OUTPUT_DIR}"
  sed -i \
    -e '/# ddt_vn_adv_is_associated/{N;d;}' \
    -e '/# ddt_vn_cor_is_associated/{N;d;}' \
    -e '/# ddt_vn_cor_pc/{N;N;d;}' \
    -e '/# vn_ie_ubc/,/# vt/{/# vt/!d;}' \
    p_diag.*.data
  sed -i \
    -e '/# id/{N;d;}' \
    -e '/# nlev/{N;d;}' \
    -e '/# nlevp1/{N;d;}' \
    -e '/# nshift/{N;d;}' \
    p_patch.*.data
  sed -i \
    -e '/# lvert_nest/{N;d;}' \
    global_data.*.data
)

echo "[download_data] done. $(find "${OUTPUT_DIR}" -maxdepth 1 -type f | wc -l) file(s) under ${OUTPUT_DIR}"
