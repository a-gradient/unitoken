#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <checkout> <output-directory> <suite-config>" >&2
  exit 2
fi

checkout=$1
output_dir=$2
suite_config=$3
scan_max_bytes=$((5 * 1024 * 1024))
scan_repeats=9

mkdir -p "$output_dir"
cd "$checkout"

# Keep base/head ordering from turning fixture page-cache state into a PR delta.
find fixtures -maxdepth 1 -type f -exec sha256sum {} + >/dev/null

cargo bench --bench regression --no-run

# Each revision consumes its own config. The report renderer treats cases that
# exist on only one side as missing, which lets benchmark coverage evolve
# without requiring an older revision to understand a newer schema.
cargo bench --bench regression -- suite \
  --config "$suite_config" \
  --output-dir "$output_dir"

# The scan command was introduced after the initial regression protocol. Older
# base revisions legitimately lack it, while the candidate report is required
# by the validator once the command exists in that checkout.
if [[ -f benches/regression/scan.rs ]]; then
  cargo bench --bench regression -- pretokenizer-scan \
    --max-bytes "$scan_max_bytes" \
    --repeats "$scan_repeats" \
    --output "$output_dir/pretokenizer-scan.json"

  # Keep this separate from the default report: default-build regressions and
  # optional SIMD regressions answer different questions. The report records
  # whether this build can actually use AVX2, NEON, or only the scalar fallback.
  cargo bench --features simd --bench regression -- pretokenizer-scan \
    --max-bytes "$scan_max_bytes" \
    --repeats "$scan_repeats" \
    --output "$output_dir/pretokenizer-scan-simd.json"
fi
