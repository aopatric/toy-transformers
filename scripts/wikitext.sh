#!/usr/bin/env bash
# Download WikiText-103 (raw) from the Hugging Face mirror: ~100M words of Wikipedia articles.
# Parquet shards land in data/wikitext-103/; src/wikitext.py trains a BPE tokenizer on them and caches ids.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
OUT="data/wikitext-103"
FILES=(
    train-00000-of-00002.parquet
    train-00001-of-00002.parquet
    validation-00000-of-00001.parquet
    test-00000-of-00001.parquet
)

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
    if [[ -s "$OUT/$f" ]]; then
        echo "$OUT/$f already exists, skipping"
        continue
    fi
    # Download to a temp name so an interrupted run doesn't leave a truncated file that looks complete.
    curl -fL --progress-bar "$BASE/$f" -o "$OUT/$f.part"
    mv "$OUT/$f.part" "$OUT/$f"
done
echo "wrote $OUT ($(du -sh "$OUT" | cut -f1))"
