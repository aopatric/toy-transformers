"""Mixed corpus of Wikipedia prose, Python code and LaTeX-heavy math, with one byte-level BPE trained on all three.

Kept separate from src/data.py (Alice, 0-layer notebook) and src/wikitext.py (WikiText-only BPE).

Sources, each kept as its own token stream so batches can be mixed by weight and val loss read per domain:
    wiki  WikiText-103, via scripts/wikitext.sh (official train/val splits)
    code  Python from codeparrot/codeparrot-clean, via scripts/mixed.sh (last VAL_FRAC of files held out)
    math  hoskinson-center/proof-pile: arXiv LaTeX, math StackExchange, formal proofs (dev shard = val)

Usage:
    ./scripts/wikitext.sh && ./scripts/mixed.sh
    from src.mixed import load_mixed
    mc = load_mixed()          # first call trains the tokenizer and encodes every stream, then caches
    mc.train["code"][:10], mc.val["math"], mc.vocab_size, mc.token_str(123), mc.decode(ids)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from src.wikitext import read_lines

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "mixed"
SOURCES = ("wiki", "code", "math")
EOT = "<|endoftext|>"
VAL_FRAC = 0.02                 # code has no official val split
TOKENIZER_SAMPLE_CHARS = 100_000_000   # per source, so Wikipedia's size doesn't dominate the BPE merges
ENCODE_BATCH = 2000             # docs per encode_batch call, to keep Encoding objects from piling up in RAM


def _jsonl(path: Path, field: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/mixed.sh first")
    with open(path) as f:
        return [json.loads(line)[field] for line in f]


def read_docs(source: str, split: str) -> list[str]:
    """Documents for one source and split. Code and math are kept verbatim: whitespace and brackets are content."""
    if source == "wiki":
        # WikiText rows are paragraphs, already cleaned of its word-level spacing; one row per "doc", no EOT
        # between them, as articles run on across rows
        return read_lines("train" if split == "train" else "val")
    if source == "code":
        docs = _jsonl(DATA_DIR / "code.jsonl", "content")
        n_val = int(len(docs) * VAL_FRAC)
        return docs[: len(docs) - n_val] if split == "train" else docs[len(docs) - n_val :]
    if source == "math":
        return _jsonl(DATA_DIR / f"math_{split}.jsonl", "text")
    raise ValueError(source)


def _sample(docs: list[str], max_chars: int) -> Iterator[str]:
    total = 0
    for d in docs:
        if total >= max_chars:
            return
        total += len(d)
        yield d


# GPT-2's pre-tokenizer split, plus a first alternative that keeps a LaTeX command whole (" \frac", "\left"), and
# punctuation runs that stop before one ("$\left" -> "$", "\left"). plain GPT-2 splits "\left" into "\" + "left",
# which would hide the paper's \left … \right skip-trigrams behind a shared "\" token
PRETOKENIZE = (
    r" ?\\[a-zA-Z]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?(?:[^\s\p{L}\p{N}\\]|\\(?![a-zA-Z]))+|\s+(?!\S)|\s+"
)


def train_tokenizer(vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def balanced():
        for source in SOURCES:
            yield from _sample(read_docs(source, "train"), TOKENIZER_SAMPLE_CHARS)

    tok.train_from_iterator(balanced(), trainer=trainer)
    return tok


def encode_stream(tokenizer: Tokenizer, docs: list[str], eot: int | None) -> np.ndarray:
    """Encode docs into one uint16 stream, with `eot` after each doc when given."""
    parts = []
    for i in range(0, len(docs), ENCODE_BATCH):
        for e in tokenizer.encode_batch(docs[i : i + ENCODE_BATCH]):
            parts.append(np.asarray(e.ids, dtype=np.uint16))
            if eot is not None:
                parts.append(np.array([eot], dtype=np.uint16))
    return np.concatenate(parts)


@dataclass
class MixedCorpus:
    tokenizer: Tokenizer
    train: dict[str, np.ndarray]   # source -> int32 token stream
    val: dict[str, np.ndarray]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self.tokenizer.token_to_id(EOT)

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode([int(i) for i in ids])

    def token_strs(self) -> list[str]:
        """every token of the vocab as text, indexed by id"""
        return [self.token_str(i) for i in range(self.vocab_size)]

    def token_str(self, i: int) -> str:
        """One token as text, e.g. " the". Lone bytes of a multi-byte character decode to "�"."""
        return self.tokenizer.decode([int(i)], skip_special_tokens=False)


def load_mixed(vocab_size: int = 16384) -> MixedCorpus:
    """Load every source's train and val streams, training and caching the tokenizer and ids on first use."""
    assert vocab_size <= 2**16, "streams are cached as uint16"
    tok_path = DATA_DIR / f"bpe-mixed-{vocab_size}.json"
    if tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        tokenizer = train_tokenizer(vocab_size)
        tokenizer.save(str(tok_path))

    streams: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}}
    for split in streams:
        for source in SOURCES:
            path = DATA_DIR / f"{source}-{split}-bpe{vocab_size}.npy"
            if not path.exists():
                eot = None if source == "wiki" else tokenizer.token_to_id(EOT)
                np.save(path, encode_stream(tokenizer, read_docs(source, split), eot))
            streams[split][source] = np.load(path).astype(np.int32)
    return MixedCorpus(tokenizer=tokenizer, train=streams["train"], val=streams["val"])


if __name__ == "__main__":
    mc = load_mixed()
    print(f"vocab {mc.vocab_size:,}, eot id {mc.eot_id}")
    for split, streams in (("train", mc.train), ("val", mc.val)):
        print(split, {s: f"{len(ids) / 1e6:.1f}M" for s, ids in streams.items()})
    for source in SOURCES:
        ids = mc.val[source][:40]
        print(f"{source}: {[mc.token_str(i) for i in ids]}")
