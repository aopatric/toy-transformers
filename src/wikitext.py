"""WikiText-103 with a byte-level BPE tokenizer trained on it, for notebooks that need more data than Alice.

Kept separate from src/data.py, whose word-level Alice corpus the 0-layer notebook depends on.

Usage:
    ./scripts/wikitext.sh               # download the parquet shards once
    from src.wikitext import load_wikitext
    wt = load_wikitext()                # first call trains the tokenizer and encodes every split, then caches
    wt.train_ids[:10], wt.vocab_size, wt.decode(wt.val_ids[:50]), wt.token_str(123)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "wikitext-103"
SPLITS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def clean(line: str) -> str:
    """Undo WikiText's word-level spacing so BPE sees ordinary prose.

    The raw dump separates every token with spaces and marks in-number joins as " @-@ ", " @,@ ", " @.@ ".
    Straight quotes are left alone: whether a spaced '"' opens or closes a quote is ambiguous.
    """
    line = re.sub(r" @(.)@ ", r"\1", line)
    line = re.sub(r" ([,.;:!?%)\]])", r"\1", line)
    line = re.sub(r"([(\[]) ", r"\1", line)
    line = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", line)
    return line.strip(" ")


def read_lines(split: str) -> list[str]:
    paths = [DATA_DIR / f for f in SPLITS[split]]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run scripts/wikitext.sh first")
    lines = []
    for p in paths:
        lines += pq.read_table(p).column("text").to_pylist()
    # Rows are paragraphs ending in "\n", plus blank rows between them; drop the blanks.
    return [clean(line) for line in lines if line.strip()]


def train_tokenizer(lines: list[str], vocab_size: int) -> Tokenizer:
    # Byte-level BPE (as in GPT-2): any string is encodable, and a leading space is part of the token (" the").
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(lines, trainer=trainer, length=len(lines))
    return tok


@dataclass
class WikiText:
    tokenizer: Tokenizer
    train_ids: np.ndarray  # int64
    val_ids: np.ndarray
    test_ids: np.ndarray

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode([int(i) for i in ids])

    def token_str(self, i: int) -> str:
        """One token as text, e.g. " the". Lone bytes of a multi-byte character decode to "�"."""
        return self.tokenizer.decode([int(i)])


def load_wikitext(vocab_size: int = 8192, max_train_tokens: int | None = None) -> WikiText:
    """Load WikiText-103 as BPE ids, training and caching the tokenizer and ids on first use.

    max_train_tokens truncates the (~120M-token) train split, since TokenWindows puts the whole
    stream on the device as int64.
    """
    tok_path = DATA_DIR / f"bpe-{vocab_size}.json"
    if tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        tokenizer = train_tokenizer(read_lines("train"), vocab_size)
        tokenizer.save(str(tok_path))

    ids = {}
    for split in SPLITS:
        # uint16 on disk halves the cache size; only valid while the vocab fits
        ids_path = DATA_DIR / f"{split}-bpe-{vocab_size}.npy"
        if not ids_path.exists():
            assert vocab_size <= 2**16
            encoded = tokenizer.encode_batch(read_lines(split))
            np.save(ids_path, np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in encoded]))
        ids[split] = np.load(ids_path, mmap_mode="r")

    train = ids["train"][:max_train_tokens]
    return WikiText(
        tokenizer=tokenizer,
        train_ids=np.asarray(train, dtype=np.int64),
        val_ids=np.asarray(ids["val"], dtype=np.int64),
        test_ids=np.asarray(ids["test"], dtype=np.int64),
    )


if __name__ == "__main__":
    wt = load_wikitext()
    print(f"vocab {wt.vocab_size:,}; train {len(wt.train_ids):,}, val {len(wt.val_ids):,}, test {len(wt.test_ids):,} tokens")
    print("sample:", wt.decode(wt.val_ids[:60]))
    print("tokens:", [wt.token_str(i) for i in wt.val_ids[:20]])
