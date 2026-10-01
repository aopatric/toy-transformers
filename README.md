# toy-transformers

Replicating [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) with small models.

## Setup

```bash
uv sync                 # creates .venv with torch (CUDA), numpy, ipykernel
./scripts/data.sh       # downloads Alice in Wonderland to data/alice.txt (0-layer notebook)
./scripts/wikitext.sh   # downloads WikiText-103 to data/wikitext-103/
./scripts/mixed.sh      # downloads Python code + proof-pile math to data/mixed/ (1-layer notebook, with WikiText)
uv run python -m src.data       # sanity check: token count + vocab size
uv run python -m src.wikitext   # first run trains the BPE tokenizer and caches token ids (~1 min)
uv run python -m src.mixed      # first run trains the 16k mixed BPE and caches token ids (~2.5 min)
```

Open notebooks with the `.venv` kernel. `from src.data import load_corpus` works from anywhere, because `src` is installed as an editable package.

## Layout

- `scripts/data.sh` downloads the corpus and strips the Gutenberg header/footer
- `src/data.py` has the word-level tokenizer and the `Corpus` object (`tokens`, `ids`, `itos`, `stoi`), plus `TokenWindows`
- `scripts/wikitext.sh` downloads WikiText-103 (raw) parquet shards from Hugging Face
- `src/wikitext.py` trains an 8k byte-level BPE on WikiText-103 and loads it as `WikiText` (`train_ids`, `val_ids`, `test_ids`, `token_str`)
- `scripts/mixed.sh` streams the first ~700 MB of a codeparrot-clean shard and a proof-pile shard (plus proof-pile's dev shard)
- `src/mixed.py` trains a 16k LaTeX-aware byte-level BPE on wiki + code + math and loads per-source streams as `MixedCorpus`
- `notebooks/` holds one notebook per experiment
