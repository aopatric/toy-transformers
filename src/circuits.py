"""QK and OV circuits of each head, read straight off the weights in the paper's notation, and the copying score.

Both circuits are (vocab, vocab), so these helpers only build the rows asked for and never the whole matrix.
"""

import torch


@torch.no_grad()
def qk_scores(model, h, dests):
    """QK circuit E^T W_QK E / sqrt(d_head), rows for the given destination tokens: (n_dests, V).
    entry [d, s] is the score when destination d looks back at source s (the pre-softmax attention score)"""
    W_E, _, W_Q, W_K, *_ = model.paper_weights()
    return (W_Q[h] @ W_E[:, dests]).T @ (W_K[h] @ W_E) / model.d_head**0.5


@torch.no_grad()
def ov_logits(model, h, srcs):
    """OV circuit W_U W_OV E, one row per source token: (n_srcs, V).
    entry [s, o] is how much attending to source s raises the logit of output o"""
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    return (W_U @ (W_O[h] @ (W_V[h] @ W_E[:, srcs]))).T


def _top(scores, k, mask):
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    top = scores.topk(k)
    ok = top.values.isfinite()
    return top.indices[ok], top.values[ok]


def top_sources(model, h, dest, k=10, mask=None):
    """the k sources head h's destination token most wants to attend to: (token ids, scores).
    mask is an optional bool tensor over the vocab of tokens to leave out"""
    return _top(qk_scores(model, h, torch.tensor([dest], device=model.W_E.device))[0], k, mask)


def top_outputs(model, h, src, k=10, mask=None):
    """the k output tokens whose logits head h boosts most when it attends to this source: (token ids, scores)"""
    return _top(ov_logits(model, h, torch.tensor([src], device=model.W_E.device))[0], k, mask)


@torch.no_grad()
def ov_eigenvalues(model, h):
    """the nonzero eigenvalues of the OV circuit W_U W_OV E (a V x V matrix of rank <= d_head), as a (d_head,) complex
    tensor. they equal those of the small matrix W_V E W_U W_O, so we never build the big one"""
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    return torch.linalg.eigvals((W_V[h] @ W_E) @ (W_U @ W_O[h]))


@torch.no_grad()
def ov_eigenvector(model, h, negative=True):
    """the real eigenvalue of head h's OV circuit with the most negative (negative=False: most positive) value, and its
    eigenvector over the vocab: (eigenvalue, (V,) tensor). attending to tokens weighted by the eigenvector comes out
    scaled by the eigenvalue, so tokens at one end of it are boosted and tokens at the other end suppressed.
    The overall sign of the eigenvector is arbitrary. The eigenvector u of the small matrix (W_V E)(W_U W_O) maps to
    W_U W_O u, which is an eigenvector of W_U W_O W_V E with the same eigenvalue"""
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    eig, vec = torch.linalg.eig((W_V[h] @ W_E) @ (W_U @ W_O[h]))
    real = eig.imag.abs() <= 1e-6 * eig.abs()
    if not real.any():
        raise ValueError(f"head {h} has no real eigenvalue")
    scores = eig.real.masked_fill(~real, float("inf") if negative else float("-inf"))
    i = scores.argmin() if negative else scores.argmax()
    u = vec[:, i]
    u = (u / u[u.abs().argmax()]).real                      # a real eigenvalue has a real eigenvector, up to a phase
    return eig[i].real, (W_U @ W_O[h]) @ u


def copying_score(model, h):
    """sum(eigenvalues) / sum(|eigenvalues|) of the OV circuit: +1 if every eigenvalue is positive (the head
    boosts the tokens it attends to), -1 if every one is negative, ~0 for a head that doesn't copy"""
    eig = ov_eigenvalues(model, h)
    return (eig.sum().real / eig.abs().sum()).item()


def copying_scores(model):
    """copying score of every head, (n_heads,)"""
    return torch.tensor([copying_score(model, h) for h in range(model.n_heads)])


def sample_readouts(model, excluded, n=8, k=5, heads=None, seed=0):
    """random (head, destination) pairs among the tokens not in `excluded`, each with the destination's top k sources
    and the top k outputs of the best source. returns a list of (head, dest, source ids, output ids)"""
    g = torch.Generator().manual_seed(seed)
    allowed = (~excluded).nonzero().squeeze(1).cpu()
    heads = list(range(model.n_heads)) if heads is None else list(heads)
    rows = []
    for _ in range(n):
        h = heads[torch.randint(len(heads), (1,), generator=g).item()]
        dest = allowed[torch.randint(len(allowed), (1,), generator=g).item()].item()
        sources, _ = top_sources(model, h, dest, k, mask=excluded)
        outputs, _ = top_outputs(model, h, sources[0].item(), k, mask=excluded)
        rows.append((h, dest, sources, outputs))
    return rows


@torch.no_grad()
def unembedding_cosine(model, a, b):
    """cosine similarity between the unembedding vectors (rows of W_U) of token ids a and b, elementwise: (n,).
    tokens the model predicts in similar situations (' two' and ' three') point the same way; the embedding vectors
    themselves barely do, so this is the similarity to use for "a token like b" """
    W_U = model.paper_weights()[1]
    return torch.nn.functional.cosine_similarity(W_U[a], W_U[b], dim=-1)
