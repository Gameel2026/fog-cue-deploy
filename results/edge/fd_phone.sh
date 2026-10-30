#!/data/data/com.termux/files/usr/bin/bash
set -e
DEV="${1:?usage: bash fd_phone.sh DEVICE_LABEL [build|bench|dataset|stream LAPTOP_IP|all]}"
STEP="${2:-all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out_$DEV"
mkdir -p "$OUT"
cd "$HERE"

build() {
  clang -O2 -ffp-contract=off -std=c11 -o fd_edge fd_edge.c fd_edge_cli.c -lm
  sha256sum fd_edge.c fd_edge_cli.c fd_edge.h model*.bin replay.bin streams_*.bin > "$OUT/${DEV}_hashes.txt"
  uname -a > "$OUT/${DEV}_system.txt"
  getprop ro.build.version.release >> "$OUT/${DEV}_system.txt" 2>/dev/null || true
  echo "build ok"
}

bench() {
  termux-wake-lock 2>/dev/null || true
  ./fd_edge bench model.bin replay.bin "$OUT/$DEV" 5
}

dataset() {
  while read -r name model; do
    [ -n "$name" ] || continue
    ./fd_edge dataset "$model" "streams_${name}.bin" "$OUT/${DEV}_dataset_${name}.csv"
  done < streams_models.txt
}

stream() {
  termux-wake-lock 2>/dev/null || true
  ./fd_edge stream model.bin "${1:?laptop IP needed}" "$OUT/$DEV"
}

case "$STEP" in
  build) build ;;
  bench) bench ;;
  dataset) dataset ;;
  stream) stream "$3" ;;
  all) build; bench; dataset ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
