#!/usr/bin/env bash
# Fetch vexx_bp_k dump decks into per-material directories under <here>/data:
#
#   data/BaTiO3_nat005/   BaTiO3 5-atom HSE06 (PAW Ba,O + US Ti), 4 slots
#   data/BaO_nat002/      BaO rocksalt HSE06 (PAW Ba,O), 2 slots
#
# Every deck is flat (vexx_{0..3,static,itN}_<var>.{bin,txt} + MANIFEST.md5 +
# PROVENANCE.txt) and shared by every lane (baseline/cpu, gpu, sdfg).
#
# Pattern: samples/cloudsc/download_data.sh -- per deck, modes in order:
# skip-if-verified -> local probe (env dir, then the originating experiments
# tree; legacy vexx_dump_* names renamed on copy) -> curl from the
# RELEASE-PINNED url, never a branch.  Checksums enforced in EVERY mode: the
# per-deck tarball md5 pins below for fetches, the shipped MANIFEST.md5 for
# local / extracted trees.
#
#   usage:  ./download_data.sh                 all decks
#           ./download_data.sh BaO_nat002     one deck (repeatable)
#   env:    DATA_ROOT=<here>/data, VEXX_BP_K_DATA_DIR (extra probe dir,
#           applies to the deck whose manifest it carries)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${HERE}/data}"
RELEASES="https://github.com/bcmchoong/vexx-bp-k-data/releases/download"

# ---- deck pin table: name | release tag | tarball | tarball md5 | local probe
DECKS="BaTiO3_nat005 BaO_nat002"
deck_row() {
    case "$1" in
      BaTiO3_nat005) echo "BaTiO3_nat005_hse06 vexx_bp_k_batio3_nat005_hse_v1.tar.gz 4d31a79e684eba63d9de60689e638f53 /workspace/experiments/BaTiO3_nat005_hse/dump_vexx_omp32" ;;
      BaO_nat002)    echo "BaO_nat002_hse06 vexx_bp_k_bao_nat002_hse_v1.tar.gz 1870a8702b005cf14c40dcaa700696f9 /workspace/experiments/BaO_nat002/dump_vexx_omp32" ;;
      *) return 1 ;;
    esac
}

verified_tree() {
    local dir="$1"
    [[ -f "${dir}/MANIFEST.md5" ]] || return 1
    (cd "${dir}" && md5sum -c --quiet MANIFEST.md5 >/dev/null 2>&1)
}

fetch_deck() {
    local name="$1" row tag tarball md5 probe out
    row=$(deck_row "$name") || { echo "[download_data] unknown deck '$name' (know: $DECKS)" >&2; return 1; }
    read -r tag tarball md5 probe <<< "$row"
    out="${DATA_ROOT}/${name}"

    if compgen -G "${out}/vexx_*" > /dev/null 2>&1; then
        if verified_tree "${out}"; then
            echo "[download_data] ${name}: ${out} already manifest-verified; skipping"
            return 0
        fi
        echo "[download_data] FATAL: ${out} has vexx_* files but fails MANIFEST.md5;" >&2
        echo "[download_data] delete it and re-run (refusing to clobber silently)" >&2
        return 1
    fi

    # local probe (legacy vexx_dump_* trees renamed on copy)
    local cand
    for cand in "${VEXX_BP_K_DATA_DIR:-}" "$probe"; do
        [[ -n "$cand" && -d "$cand" ]] || continue
        if verified_tree "$cand"; then
            mkdir -p "${out}"
            local f b
            for f in "${cand}"/vexx_*; do
                b="$(basename "$f")"
                cp "$f" "${out}/${b/_dump/}"
            done
            [[ -f "${cand}/PROVENANCE.txt" ]] && cp "${cand}/PROVENANCE.txt" "${out}/"
            sed 's/ vexx_dump_/ vexx_/' "${cand}/MANIFEST.md5" > "${out}/MANIFEST.md5"
            verified_tree "${out}" || { echo "[download_data] FATAL: copy from ${cand} fails re-verification" >&2; return 1; }
            echo "[download_data] ${name}: copied manifest-verified deck from ${cand}"
            return 0
        fi
    done

    # pinned release fetch
    local tmp
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '${tmp}'" RETURN
    echo "[download_data] ${name}: fetching ${RELEASES}/${tag}/${tarball}"
    curl -fL --retry 3 -o "${tmp}/${tarball}.part" "${RELEASES}/${tag}/${tarball}"
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
