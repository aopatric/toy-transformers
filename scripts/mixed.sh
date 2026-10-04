#!/usr/bin/env bash
# Download the code and math parts of the mixed corpus (WikiText-103 comes from scripts/wikitext.sh):
#   - Python from GitHub: codeparrot/codeparrot-clean, first shard
#   - math, mostly LaTeX (arXiv sources, math StackExchange, formal proofs): hoskinson-center/proof-pile,
#     first train shard, plus its dev shard for validation
# Shards are decompressed on the fly and cut at a byte budget, so only what we use is kept on disk.
set -euo pipefail
cd "$(dirname "$0")/.."

HF="https://huggingface.co/datasets"
OUT="data/mixed"
mkdir -p "$OUT"

# fetch URL BYTES DEST: stream a .gz JSON-lines file, keep its first BYTES of decompressed text, and drop the
# last line, which the cut almost always leaves half-written
fetch() {
    local url="$1" bytes="$2" dest="$3"
    if [[ -s "$dest" ]]; then
        echo "$dest already exists, skipping"
        return
    fi
    # head closes the pipe early on purpose, so curl and gunzip exiting on SIGPIPE is expected here
    (curl -fsL "$url" | gunzip | head -c "$bytes" || true) | sed '$d' > "$dest.part"
    [[ -s "$dest.part" ]] || { echo "$dest is empty after download" >&2; exit 1; }
    mv "$dest.part" "$dest"
    echo "wrote $dest ($(du -h "$dest" | cut -f1), $(wc -l < "$dest") docs)"
}

fetch "$HF/codeparrot/codeparrot-clean/resolve/main/file-000000000001.json.gz" 700000000 "$OUT/code.jsonl"
fetch "$HF/hoskinson-center/proof-pile/resolve/main/train/proofpile_train_0.jsonl.gz" 700000000 "$OUT/math_train.jsonl"
fetch "$HF/hoskinson-center/proof-pile/resolve/main/dev/proofpile_dev.jsonl.gz" 20000000 "$OUT/math_val.jsonl"
