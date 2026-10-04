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

Run the tests with `uv run pytest` (the `real_ckpt` ones need a GPU and a checkpoint in `checkpoints/`; `REAL_CKPT=<file>` picks which one).
Set `CPU_ONLY=1` before starting a notebook kernel to run without a GPU, and `SMOKE=1` for a one-minute tiny training run that writes its own checkpoint.

Open notebooks with the `.venv` kernel. `from src.data import load_corpus` works from anywhere, because `src` is installed as an editable package.

## Layout

- `scripts/data.sh` downloads the corpus and strips the Gutenberg header/footer
- `src/data.py` has the word-level tokenizer and the `Corpus` object (`tokens`, `ids`, `itos`, `stoi`), plus `TokenWindows`
- `scripts/wikitext.sh` downloads WikiText-103 (raw) parquet shards from Hugging Face
- `src/wikitext.py` trains an 8k byte-level BPE on WikiText-103 and loads it as `WikiText` (`train_ids`, `val_ids`, `test_ids`, `token_str`)
- `scripts/mixed.sh` streams the first ~700 MB of a codeparrot-clean shard and a proof-pile shard (plus proof-pile's dev shard)
- `src/mixed.py` trains a 16k LaTeX-aware byte-level BPE on wiki + code + math and loads per-source streams as `MixedCorpus`
- `src/model.py` is the one-layer attention-only transformer (fused forward for training, the step-by-step walkthrough, `paper_weights()` for the paper's layout), the path-expansion function, and checkpoint save/load
- `src/train.py` batches, lr schedule, validation, and `train()` (resumes from a partial checkpoint, never retrains over a finished one)
- `src/counts.py` bigram counts, the direct path vs the bigram statistics, which tokens to leave out when reading circuits
- `src/circuits.py` QK/OV circuit rows, readouts, eigenvalues and the copying score
- `src/skiptrigrams.py` skip-trigrams: the circuits propose candidates, one pass over the corpus scores them, and the in-context-learning pattern labels
- `src/display.py` the printing and plotting helpers the notebooks use
- `tests/` pytest suite: tiny random models on CPU and GPU, brute-force counters, planted skip-trigrams, and `real_ckpt` tests on a trained checkpoint
- `notebooks/` holds one notebook per experiment
