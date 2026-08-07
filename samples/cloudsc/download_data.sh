#!/usr/bin/env bash
# Fetch the ECMWF dwarf-p-cloudsc input deck (input.h5 + reference.h5) into OUTPUT_DIR
# (default <here>/data).  Source: github.com/ecmwf-ifs/dwarf-p-cloudsc @ f078f8f,
# config-files/*.h5, Apache-2.0 -- attribution + vendoring decision in samples/README.md.
# Modes, in order (pattern: VelocityTendenciesPipeline/tools/download_data.sh): skip-if-
# verified -> local probe (DWARF_CLOUDSC_DATA_DIR, then a sibling dwarf checkout) -> curl
# from the pinned commit, never a branch, so upstream movement cannot change bytes.
# md5 pin enforced in EVERY mode.  Env: OUTPUT_DIR, DWARF_CLOUDSC_DATA_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=f078f8f
BASE_URL="https://raw.githubusercontent.com/ecmwf-ifs/dwarf-p-cloudsc/${COMMIT}/config-files"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/data}"

FILES=(input.h5 reference.h5)
declare -A MD5=(
    [input.h5]=b3282794694f035e8371a8aec1b93a1f
    [reference.h5]=92d401388e74fbe5340baad19cff106a
)

verified() {
    local dir="$1" f
    for f in "${FILES[@]}"; do
        [[ -f "${dir}/${f}" ]] || return 1
        echo "${MD5[$f]}  ${dir}/${f}" | md5sum -c --quiet - >/dev/null 2>&1 || return 1
    done
}

present=0
for f in "${FILES[@]}"; do
    [[ -f "${OUTPUT_DIR}/${f}" ]] && present=1
done
if [[ "$present" == 1 ]]; then
    if verified "${OUTPUT_DIR}"; then
        echo "[download_data] ${OUTPUT_DIR} already holds md5-verified copies; skipping"
        exit 0
    fi
    echo "[download_data] FATAL: files in ${OUTPUT_DIR} fail the pinned md5s;" >&2
    echo "[download_data] delete them and re-run (refusing to clobber silently)" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CANDIDATES=(
    "${DWARF_CLOUDSC_DATA_DIR:-}"
    "${HERE}/../../../dwarf-p-cloudsc/config-files"
)
for cand in "${CANDIDATES[@]}"; do
    [[ -n "$cand" && -d "$cand" ]] || continue
    if verified "$cand"; then
        cp "${cand}/input.h5" "${cand}/reference.h5" "${OUTPUT_DIR}/"
        echo "[download_data] copied md5-verified deck from ${cand}"
        exit 0
    fi
done

for f in "${FILES[@]}"; do
    echo "[download_data] fetching ${BASE_URL}/${f}"
    curl -fL --retry 3 -o "${OUTPUT_DIR}/${f}.part" "${BASE_URL}/${f}"
    if ! echo "${MD5[$f]}  ${OUTPUT_DIR}/${f}.part" | md5sum -c --quiet -; then
        rm -f "${OUTPUT_DIR}/${f}.part"
        echo "[download_data] FATAL: ${f} from ${COMMIT} fails the pinned md5" >&2
        exit 1
    fi
    mv "${OUTPUT_DIR}/${f}.part" "${OUTPUT_DIR}/${f}"
done
echo "[download_data] done: ${OUTPUT_DIR}"

# Fallback when raw.githubusercontent.com is blocked -- shallow clone at the pin, then copy:
#   tmp="$(mktemp -d)"; git clone --depth 1 https://github.com/ecmwf-ifs/dwarf-p-cloudsc "${tmp}/dwarf"
#   git -C "${tmp}/dwarf" fetch --depth 1 origin "${COMMIT}"; git -C "${tmp}/dwarf" checkout "${COMMIT}"
#   cp "${tmp}/dwarf/config-files/"{input,reference}.h5 "${OUTPUT_DIR}/"; rm -rf "${tmp}"  # md5-verify as above
