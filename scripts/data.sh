#!/usr/bin/env bash
# Download a small word-level corpus: Alice's Adventures in Wonderland (Project Gutenberg #11).
# ~26k words, ~3k word types lowercased -> a dense bigram table is ~9M entries, easy to hold in memory.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://www.gutenberg.org/cache/epub/11/pg11.txt"
OUT="data/alice.txt"

if [[ -s "$OUT" ]]; then
    echo "$OUT already exists, skipping"
    exit 0
fi

mkdir -p data
raw="$(mktemp)"
trap 'rm -f "$raw"' EXIT

curl -fsSL "$URL" -o "$raw"

# Strip BOM and CRLF first so the line-anchored marker matches below work.
sed -i '1s/^\xEF\xBB\xBF//; s/\r$//' "$raw"

grep -q '^\*\*\* START OF THE PROJECT GUTENBERG' "$raw" || { echo "START marker not found" >&2; exit 1; }
grep -q '^\*\*\* END OF THE PROJECT GUTENBERG' "$raw"   || { echo "END marker not found" >&2; exit 1; }

# Keep only the text between the Gutenberg license header and footer.
sed -n '/^\*\*\* START OF THE PROJECT GUTENBERG/,/^\*\*\* END OF THE PROJECT GUTENBERG/{//!p}' "$raw" > "$OUT"

[[ -s "$OUT" ]] || { echo "$OUT is empty after stripping" >&2; exit 1; }
echo "wrote $OUT ($(wc -w < "$OUT") words)"
