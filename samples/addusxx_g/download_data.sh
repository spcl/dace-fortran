#!/usr/bin/env bash
# Fetch addusxx_g dump decks into per-material directories under <here>/data:
#
#   data/BaTiO3_nat005/   BaTiO3 5-atom HSE06 (PAW Ba,O + US Ti), 2 sets
#   data/BaO_nat002/      BaO rocksalt HSE06 (PAW Ba,O), 2 sets
#
# Every deck is flat (adx_{0,1}_<var>.bin/.txt per set + adxndx_static_*
# shared tables + MANIFEST.md5 + PROVENANCE.txt); set 0 = first call
# (q=Gamma), set 1 = first general-q call.  Feeds the SDFG binding in
# outputs/lib and verify_addusxx alike.
#
# Pattern: samples/vexx_bp_k/download_data.sh -- per deck, modes in order:
# skip-if-verified -> local probe (env dir, then the originating COMBINED
# addusxx+newdxx experiments tree dump_addnewdxx; this sample's adx_*/static
# subset is copied out and the manifest filtered to match) -> curl from the
# RELEASE-PINNED url, never a branch.  Checksums enforced in EVERY mode: the
# per-deck tarball md5 pins below for fetches, the shipped MANIFEST.md5 for
# local / extracted trees.
#
#   usage:  ./download_data.sh                 all decks
#           ./download_data.sh BaO_nat002     one deck (repeatable)
#   env:    DATA_ROOT=<here>/data, ADDUSXX_G_DATA_DIR (extra probe dir,
#           combined or already-split tree; applies to the deck whose
#           manifest verifies there)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${HERE}/data}"
RELEASES="https://github.com/bcmchoong/vexx-bp-k-data/releases/download"
RELEASE_TAG="addusxx_g_combined"
PFX="adx"

# ---- deck pin table: name | tarball | tarball md5 | local probe dir
DECKS="BaTiO3_nat005 BaO_nat002"
deck_row() {
    case "$1" in
      BaTiO3_nat005) echo "addusxx_g_batio3_nat005_hse_v1.tar.gz 4233cc7138c58fb06e2a91f3b3357c96 /workspace/experiments/BaTiO3_nat005_hse/dump_addnewdxx" ;;
      BaO_nat002)    echo "addusxx_g_bao_nat002_hse_v1.tar.gz cb0d7da0aefd7852e75dd067726bd6a6 /workspace/experiments/BaO_nat002/dump_addnewdxx" ;;
      *) return 1 ;;
    esac
}

verified_tree() {
    local dir="$1"
    [[ -f "${dir}/MANIFEST.md5" ]] || return 1
    (cd "${dir}" && md5sum -c --quiet MANIFEST.md5 >/dev/null 2>&1)
}

# keep only this sample's subset of a (possibly combined addusxx+newdxx)
# manifest: adx_* sets, shared adxndx_static_* tables, PROVENANCE.txt
filter_manifest() {
    awk -v p="^(${PFX}_|adxndx_static_|PROVENANCE\\.txt$)" '$2 ~ p' "$1"
}

fetch_deck() {
    local name="$1" row tarball md5 probe out
    row=$(deck_row "$name") || { echo "[download_data] unknown deck '$name' (know: $DECKS)" >&2; return 1; }
    read -r tarball md5 probe <<< "$row"
    out="${DATA_ROOT}/${name}"

    if compgen -G "${out}/${PFX}_*" > /dev/null 2>&1; then
        if verified_tree "${out}"; then
            echo "[download_data] ${name}: ${out} already manifest-verified; skipping"
            return 0
        fi
        echo "[download_data] FATAL: ${out} has ${PFX}_* files but fails MANIFEST.md5;" >&2
        echo "[download_data] delete it and re-run (refusing to clobber silently)" >&2
        return 1
    fi

    # local probe: verify the candidate's own manifest, then copy out this
    # sample's subset with a filtered manifest and re-verify the copy
    local cand
    for cand in "${ADDUSXX_G_DATA_DIR:-}" "$probe"; do
        [[ -n "$cand" && -d "$cand" ]] || continue
        verified_tree "$cand" || continue
        local sub
        sub="$(filter_manifest "${cand}/MANIFEST.md5")"
        [[ -n "$sub" ]] || continue
        mkdir -p "${out}"
        local f
        while read -r _ f; do cp "${cand}/${f}" "${out}/"; done <<< "$sub"
        printf '%s\n' "$sub" > "${out}/MANIFEST.md5"
        verified_tree "${out}" || { echo "[download_data] FATAL: copy from ${cand} fails re-verification" >&2; return 1; }
        echo "[download_data] ${name}: copied manifest-verified deck from ${cand}"
        return 0
    done

    # pinned release fetch
    local tmp
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '${tmp}'" RETURN
    echo "[download_data] ${name}: fetching ${RELEASES}/${RELEASE_TAG}/${tarball}"
    curl -fL --retry 3 -o "${tmp}/${tarball}.part" "${RELEASES}/${RELEASE_TAG}/${tarball}"
    if ! echo "${md5}  ${tmp}/${tarball}.part" | md5sum -c --quiet -; then
        echo "[download_data] FATAL: ${tarball} fails the pinned md5" >&2
        return 1
    fi
    mkdir -p "${tmp}/x"
    tar -xzf "${tmp}/${tarball}.part" -C "${tmp}/x"
    verified_tree "${tmp}/x" || { echo "[download_data] FATAL: extracted ${name} deck fails its own MANIFEST.md5" >&2; return 1; }
    mkdir -p "${out}"
    mv "${tmp}/x"/* "${out}/"
    echo "[download_data] ${name}: done -> ${out}"
}

rc=0
targets="${*:-$DECKS}"
for name in $targets; do
    fetch_deck "$name" || rc=1
done
exit $rc
