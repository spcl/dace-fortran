#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL="${URL:-https://polybox.ethz.ch/public.php/dav/files/QEEpykQizxHDtB8/?accept=zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/data_r02b06}"
TAR_FILE="${TAR_FILE:-${OUTPUT_DIR%/}.tar.xz}"
NPZ_DIR="${NPZ_DIR:-$(dirname "${OUTPUT_DIR}")}"
TIMESTEP="${TIMESTEP:-1}"
NPROMAS="${NPROMAS:-32 491520}"
KEEP_TAR="${KEEP_TAR:-1}"
CURL_TRIES="${CURL_TRIES:-3}"
PY="${PYTHON:-python3}"

rc=0
say() { echo "[prepare_r02b06] $*"; }
fail() {
    echo "[prepare_r02b06] ERROR: $*" >&2
    rc=1
}
finish() {
    echo "PREPARE_R02B06_EXIT=$rc"
    exit "$rc"
}

download() {
    if [ -s "$TAR_FILE" ] && xz -t "$TAR_FILE" 2> /dev/null; then
        say "tarball already complete, skipping download: $TAR_FILE"
        return 0
    fi
    local try=1
    while [ "$try" -le "$CURL_TRIES" ]; do
        if [ "$try" -eq 1 ] && [ ! -s "$TAR_FILE" ]; then
            say "downloading $URL -> $TAR_FILE (attempt $try/$CURL_TRIES)"
            curl -fL --retry 5 --retry-delay 10 -o "$TAR_FILE" "$URL"
        else
            say "resuming $TAR_FILE from $(stat -c %s "$TAR_FILE" 2> /dev/null || echo 0) bytes (attempt $try/$CURL_TRIES)"
            curl -fL --retry 5 --retry-delay 10 -C - -o "$TAR_FILE" "$URL"
        fi
        if xz -t "$TAR_FILE" 2> /dev/null; then
            say "xz -t OK: $TAR_FILE ($(stat -c %s "$TAR_FILE") bytes)"
            return 0
        fi
        say "xz -t rejected the archive after attempt $try"
        try=$((try + 1))
    done
    return 1
}

have_data() { [ -n "$(ls -A "$OUTPUT_DIR" 2> /dev/null)" ]; }

extract() {
    mkdir -p "$OUTPUT_DIR" || return 1
    say "extracting to $OUTPUT_DIR"
    tar -xJf "$TAR_FILE" -C "$OUTPUT_DIR" || return 1
    say "extracted $(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l) file(s)"
    [ "$KEEP_TAR" = 1 ] || rm -f "$TAR_FILE"
    return 0
}

patch_records() {
    local p="$OUTPUT_DIR/p_patch.${TIMESTEP}.data"
    [ -f "$p" ] || {
        echo "no $p" >&2
        return 1
    }
    if ! head -c 256 "$p" | grep -q '^# id$'; then
        say "serde records already aligned, no patch needed"
        return 0
    fi
    say "patching serde records (same edits as download_data.sh)"
    (
        cd "$OUTPUT_DIR" || exit 1
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
}

convert_all() {
    local n out
    mkdir -p "$NPZ_DIR" || return 1
    for n in $NPROMAS; do
        out="$NPZ_DIR/velocity_r02b06_nproma${n}.npz"
        if [ -s "$out" ]; then
            say "npz already there, skipping: $out"
            continue
        fi
        say "converting nproma=$n -> $out"
        (
            cd "$NPZ_DIR" \
                && "$PY" "$HERE/convert_data.py" --data-dir "$OUTPUT_DIR" --timestep "$TIMESTEP" \
                    --nproma "$n" --out "$out.partial"
        ) && mv -f "$out.partial" "$out" || fail "convert nproma=$n failed"
    done
}

say "url=$URL"
say "data=$OUTPUT_DIR npz=$NPZ_DIR timestep=$TIMESTEP npromas='$NPROMAS' python=$PY"
if have_data; then
    say "$OUTPUT_DIR already populated, skipping download and extraction"
else
    download || {
        fail "could not fetch a valid $TAR_FILE from $URL"
        finish
    }
    extract || {
        fail "extraction of $TAR_FILE into $OUTPUT_DIR failed"
        finish
    }
fi
patch_records || fail "serde record patch failed"
[ "$rc" -eq 0 ] || finish
convert_all
finish
