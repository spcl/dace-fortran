#!/usr/bin/env bash
# Fetch the vexx_bp_k BaTiO3_nat005_hse dump deck (4 boundary-dump slots +
# static tables, one self-consistent QE trajectory) into OUTPUT_DIR (default
# <here>/data, flat layout: vexx_{0..3,static,it1,it2}_<variable>.{bin,txt}).
# This single copy is shared by every lane: baseline/cpu, baseline/gpu, and
# the SDFG binding harness.
#
# Pattern: samples/cloudsc/download_data.sh -- modes in order: skip-if-
# verified -> local probe (VEXX_BP_K_DATA_DIR, then the originating
# experiments tree, renaming its legacy vexx_dump_* filenames) -> curl from a
# RELEASE-PINNED url, never a branch, so upstream movement cannot change
# bytes.  Checksums enforced in EVERY mode: the tarball md5 pin below for
# fetches, and the per-file MANIFEST.md5 shipped with the deck for local /
# extracted trees.  Env: OUTPUT_DIR, VEXX_BP_K_DATA_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/data}"

# ---- pins: fill BASE_URL when the artifact is uploaded (release tag/DOI,
# never a branch); TARBALL_MD5 must match the packed artifact.
TARBALL=vexx_bp_k_batio3_nat005_hse_v1.tar.gz
BASE_URL="https://github.com/bcmchoong/vexx-bp-k-data/releases/download/BaTiO3_nat005_hse06"
TARBALL_MD5=4d31a79e684eba63d9de60689e638f53

verified_tree() {
    # every file listed in the deck's manifest matches
    local dir="$1"
    [[ -f "${dir}/MANIFEST.md5" ]] || return 1
    (cd "${dir}" && md5sum -c --quiet MANIFEST.md5 >/dev/null 2>&1)
}

if compgen -G "${OUTPUT_DIR}/vexx_*" > /dev/null 2>&1; then
    if verified_tree "${OUTPUT_DIR}"; then
        echo "[download_data] ${OUTPUT_DIR} already holds a manifest-verified deck; skipping"
        exit 0
    fi
    echo "[download_data] FATAL: ${OUTPUT_DIR} has vexx_* files but fails MANIFEST.md5;" >&2
    echo "[download_data] delete them and re-run (refusing to clobber silently)" >&2
    exit 1
fi

# ---- local probe: an already-verified copy somewhere on this machine.
# Legacy trees (the QE-side instrumentation writes vexx_dump_* names) are
# accepted and renamed on copy; their manifest is rewritten to match.
CANDIDATES=(
    "${VEXX_BP_K_DATA_DIR:-}"
)
for cand in "${CANDIDATES[@]}"; do
    [[ -n "$cand" && -d "$cand" ]] || continue
    if verified_tree "$cand"; then
        mkdir -p "${OUTPUT_DIR}"
        for f in "${cand}"/vexx_*; do
            b="$(basename "$f")"
            cp "$f" "${OUTPUT_DIR}/${b/_dump/}"
        done
        [[ -f "${cand}/PROVENANCE.txt" ]] && cp "${cand}/PROVENANCE.txt" "${OUTPUT_DIR}/"
        sed 's/ vexx_dump_/ vexx_/' "${cand}/MANIFEST.md5" > "${OUTPUT_DIR}/MANIFEST.md5"
        verified_tree "${OUTPUT_DIR}" || {
            echo "[download_data] FATAL: copy from ${cand} fails re-verification" >&2
            exit 1
        }
        echo "[download_data] copied manifest-verified deck from ${cand}"
        exit 0
    fi
    echo "[download_data] note: ${cand} present but not manifest-verified; ignoring"
done

# ---- network fetch: pinned release asset, .part + md5 + verified extract
if [[ "${TARBALL_MD5}" == FIXME* || "${BASE_URL}" == *FIXME* ]]; then
    echo "[download_data] FATAL: no verified local copy found and the release" >&2
    echo "[download_data] pins (BASE_URL / TARBALL_MD5) are not filled in yet." >&2
    exit 1
fi
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
echo "[download_data] fetching ${BASE_URL}/${TARBALL}"
curl -fL --retry 3 -o "${tmp}/${TARBALL}.part" "${BASE_URL}/${TARBALL}"
if ! echo "${TARBALL_MD5}  ${tmp}/${TARBALL}.part" | md5sum -c --quiet -; then
    echo "[download_data] FATAL: ${TARBALL} fails the pinned md5" >&2
    exit 1
fi
mkdir -p "${tmp}/x"
tar -xzf "${tmp}/${TARBALL}.part" -C "${tmp}/x"
verified_tree "${tmp}/x" || {
    echo "[download_data] FATAL: extracted deck fails its own MANIFEST.md5" >&2
    exit 1
}
mkdir -p "${OUTPUT_DIR}"
mv "${tmp}/x"/* "${OUTPUT_DIR}/"
echo "[download_data] done: ${OUTPUT_DIR}"
