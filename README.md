# toy-transformers

Replicating [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) with small models.

## Setup

```bash
uv sync                 # creates .venv with torch (CUDA), numpy, ipykernel
./scripts/data.sh       # downloads Alice in Wonderland to data/alice.txt
uv run python -m src.data   # sanity check: token count + vocab size
```

Open notebooks with the `.venv` kernel. `from src.data import load_corpus` works from anywhere, because `src` is installed as an editable package.

## Layout

- `scripts/data.sh` downloads the corpus and strips the Gutenberg header/footer
- `src/data.py` has the word-level tokenizer and the `Corpus` object (`tokens`, `ids`, `itos`, `stoi`)
- `notebooks/` holds one notebook per experiment
