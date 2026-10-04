"""Bigram statistics of the corpus, and how well the direct path of the model (W_U W_E) matches them."""

import torch


@torch.no_grad()
def count_bigrams(ids, vocab_size, chunk=50_000_000):
    """(V, V) long counts of every [t, t+1] pair in one token stream; chunked so the int64 keys stay small"""
    V = vocab_size
    counts = torch.zeros(V * V, dtype=torch.long, device=ids.device)
    for i in range(0, len(ids) - 1, chunk):
        a = ids[i : i + chunk + 1].long()
        counts += torch.bincount(a[:-1] * V + a[1:], minlength=V * V)
    return counts.view(V, V)


def bigram_counts(streams, vocab_size, chunk=50_000_000):
    """(V, V) float32 counts pooled over several streams (a pair never spans two streams). float32 is half the memory"""
    total = torch.zeros(vocab_size, vocab_size, device=next(iter(streams)).device)
    for ids in streams:
        total += count_bigrams(ids, vocab_size, chunk).float()
    return total


def bigram_probs(counts):
    """row-normalized counts: row t is P(next token | t). rows for tokens never seen stay all zero"""
    return counts / counts.sum(dim=-1, keepdim=True).clamp(min=1)


@torch.no_grad()
def direct_path_logits(model, tokens):
    """logits the direct path alone gives after each token, (n, V): column t of W_U W_E, no attention involved"""
    W_E, W_U, *_ = model.paper_weights()
    return (W_U @ W_E[:, tokens]).T


def topk_overlap(scores, probs, k):
    """per row, how much of the empirical top-k next tokens (those that actually occur) is in the predicted top-k.
    scores and probs are (n, V); returns (n,) in [0, 1], and nan for a row whose token was never seen"""
    predicted = scores.topk(k, dim=-1).indices                       # (n, k)
    top_p, actual = probs.topk(k, dim=-1)
    real = top_p > 0                                                 # a token with < k continuations has fewer targets
    hit = (predicted[:, :, None] == actual[:, None, :]) & real[:, None, :]
    return hit.any(dim=1).sum(-1) / real.sum(-1)


@torch.no_grad()
def direct_path_vs_bigram(model, counts, n_tokens=1000, k=10, exclude=None):
    """top-k overlap between the direct path and the empirical bigrams, over the n most frequent tokens
    (`exclude` is an optional bool mask over the vocab of tokens to skip). returns (token ids, overlap per token)"""
    totals = counts.sum(-1)
    if exclude is not None:
        totals = totals.masked_fill(exclude, -1)
    tokens = totals.topk(n_tokens).indices
    return tokens, topk_overlap(direct_path_logits(model, tokens), bigram_probs(counts)[tokens], k)


@torch.no_grad()
def one_token_logits(model, tokens):
    """logits the whole model gives for each token alone as a one-token context, (n, V): the direct path plus every
    head attending to the token itself. this is the model's own bigram prediction"""
    return model(tokens[:, None])[:, 0]


def excluded_tokens(token_counts, token_strs, eot_id, min_count=1000, keep_top=None):
    """bool mask over the vocab of tokens to leave out when reading circuits. leaves out rare tokens (their embeddings
    barely moved from random init, so their circuit scores are noise), lone bytes of multi-byte characters (they print
    as �) and the document separator. keep_top, if given, keeps only that many of the most frequent tokens still allowed"""
    out = token_counts < min_count
    out |= torch.tensor(["�" in s for s in token_strs], device=token_counts.device)
    out[eot_id] = True
    if keep_top is not None:
        ranked = token_counts.masked_fill(out, -1).argsort(descending=True)   # the tokens still allowed, most frequent first
        out[ranked[keep_top:]] = True
    return out
