"""Load the word-level corpus downloaded by scripts/data.sh into a Python object.

Usage:
    from src.data import load_corpus
    corpus = load_corpus()
    corpus.tokens[:10], corpus.ids[:10], corpus.vocab_size

    from src.data import make_loader
    for x, y in make_loader(corpus.ids, context_len=32, batch_size=64):
        ...  # x, y: (64, 32) int64, y is x shifted left by one
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.device import DEVICE

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "alice.txt"

# Word-level tokenizer: lowercase, then a token is either
#   - a run of letters/digits, optionally with internal apostrophes or hyphens ("don't", "rabbit-hole"), or
#   - a single punctuation character (".", ",", "!", "“", ...).
# Whitespace is discarded. Curly apostrophes are normalized to ASCII first.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return TOKEN_RE.findall(text)


@dataclass
class Corpus:
    text: str
    tokens: list[str]
    itos: list[str]  # id -> token
    stoi: dict[str, int]  # token -> id
    ids: np.ndarray  # int64 token ids, same length as tokens

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return len(self.ids)

    def tensor(self, device: str | torch.device = DEVICE) -> torch.Tensor:
        return torch.from_numpy(self.ids).to(device)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[t] for t in tokenize(s)]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


def load_corpus(path: str | Path = DEFAULT_PATH) -> Corpus:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/data.sh first")

    text = path.read_text(encoding="utf-8")
    tokens = tokenize(text)

    # Deterministic vocab order (frequency desc, then alphabetical) so ids are stable
    # across processes -- rows of a hand-built bigram table line up with W_U @ W_E.
    counts = Counter(tokens)
    itos = sorted(counts, key=lambda t: (-counts[t], t))
    stoi = {t: i for i, t in enumerate(itos)}
    ids = np.array([stoi[t] for t in tokens], dtype=np.int64)

    return Corpus(text=text, tokens=tokens, itos=itos, stoi=stoi, ids=ids)


def train_val_split(ids: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous split: the last `val_frac` of the token stream is held out.

    For recovering bigram statistics exactly, train on everything (val_frac=0) so the
    model's optimum matches the table you count from corpus.ids.
    """
    n_val = int(len(ids) * val_frac)
    return ids[: len(ids) - n_val], ids[len(ids) - n_val :]


class TokenWindows(Dataset):
    """Fixed-length next-token windows over a token stream.

    Item i is (x, y) with x = ids[s : s+T] and y = ids[s+1 : s+T+1], where s = i * stride.

    With the default stride = context_len the windows tile the stream, so each adjacent
    pair (ids[j], ids[j+1]) appears at most once per epoch; the last < context_len pairs
    that don't fill a window are dropped. With context_len=1 every pair appears exactly
    once, so the loss minimizer of a 0-layer model is exactly the empirical bigram
    distribution. stride < context_len overlaps windows and over-counts pairs in the
    middle of the stream relative to the ends.
    """

    def __init__(
        self,
        ids: np.ndarray,
        context_len: int,
        stride: int | None = None,
        device: str | torch.device = DEVICE,
    ):
        if len(ids) < context_len + 1:
            raise ValueError(f"need at least {context_len + 1} tokens, got {len(ids)}")
        self.ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        self.context_len = context_len
        self.stride = stride or context_len

    def __len__(self) -> int:
        return (len(self.ids) - self.context_len - 1) // self.stride + 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Bounds check matters: `for x, y in dataset` stops only on IndexError, and slicing
        # past the end of a tensor silently returns empty windows instead of raising.
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"window {i} out of range for {n} windows")
        s = i * self.stride
        chunk = self.ids[s : s + self.context_len + 1]
        return chunk[:-1], chunk[1:]


def collate_windows(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of (x, y) windows into (B, T) input and target tensors."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def make_loader(
    ids: np.ndarray,
    context_len: int,
    batch_size: int,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 0,
    device: str | torch.device = DEVICE,
) -> DataLoader:
    """DataLoader of (x, y) minibatches, each (batch_size, context_len) int64 on `device`.

    Seeded so batch order is reproducible across notebook reruns.
    """
    # The token stream lives on `device` and batches are sliced from it there, so there's
    # no per-step host->device copy. Keep num_workers=0 (the default) with CUDA tensors.
    dataset = TokenWindows(ids, context_len, stride, device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(seed),
    )


if __name__ == "__main__":
    c = load_corpus()
    print(f"{len(c):,} tokens, vocab {c.vocab_size:,}")
    print("top 20:", c.itos[:20])
    print("sample:", c.decode(c.ids[1000:1030]))

    x, y = next(iter(make_loader(c.ids, context_len=8, batch_size=4)))
    print("batch:", tuple(x.shape), x.dtype, x.device)
    print("  x:", c.decode(x[0]))
    print("  y:", c.decode(y[0]))
