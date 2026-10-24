#!/usr/bin/env bash
set -euo pipefail

for input in data/raw/real/*.mp4; do
    [ -f "$input" ] || continue

    name="$(basename "$input" .mp4)"
    output="data/processed/test/$name"

    echo "Processing: $input"
    python src/compress.py \
        --input "$input" \
        --output-dir "$output"
done

echo "All test-video compression jobs completed."
