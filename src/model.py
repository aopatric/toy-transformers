"""One-layer attention-only transformer, with the path-expansion view of it from the paper.

Weights are stored the usual torch way (one row per token) so training and the fused forward are plain and fast.
`paper_weights()` hands back the same weights in the paper's layout (one column per token), which is what the
walkthrough, the demo and the circuit code read, so the notebook can use the paper's notation directly.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# checkpoints written by the first notebook name the architecture keys differently
ARCH_KEYS = {"vocab_size": "vocab_size", "hidden_dim": "d_model", "num_heads": "n_heads", "head_dim": "d_head"}


class OneLayerAttentionOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_head: int, device=None):
        super().__init__()
        self.vocab_size, self.d_model, self.n_heads, self.d_head = vocab_size, d_model, n_heads, d_head

        # every weight is scaled by 1/sqrt(its dimension) so activations stay O(1) through each matmul: W_E and the
        # QKV/unembed weights by the model width, W_O by the width of all heads together so their sum doesn't swamp
        # the residual stream. no layernorm or biases, so the circuit formulas stay exact
        def init(*shape, scale=1.0):
            return nn.Parameter(torch.randn(*shape, device=device) * scale)

        self.W_E = init(vocab_size, d_model, scale=d_model**-0.5)
        self.W_U = init(d_model, vocab_size, scale=d_model**-0.5)
        self.W_Q = init(n_heads, d_model, d_head, scale=d_model**-0.5)
        self.W_K = init(n_heads, d_model, d_head, scale=d_model**-0.5)
        self.W_V = init(n_heads, d_model, d_head, scale=d_model**-0.5)
        self.W_O = init(n_heads, d_head, d_model, scale=(d_head * n_heads) ** -0.5)

    def arch(self) -> dict:
        return dict(vocab_size=self.vocab_size, d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head)

    def paper_weights(self):
        """(W_E, W_U, W_Q, W_K, W_V, W_O) in the paper's layout, as transposed views of the parameters:
        W_E (d, V), W_U (V, d), W_Q/W_K/W_V (heads, d_head, d), W_O (heads, d, d_head)"""
        return self.W_E.T, self.W_U.T, self.W_Q.mT, self.W_K.mT, self.W_V.mT, self.W_O.mT

    def forward_per_head(self, x, return_patterns=False):
        """The walkthrough: one head at a time, one column per token, in the paper's notation.
        x is token ids, (T,) or (B, T); returns logits (..., T, V) and optionally the patterns (..., heads, T, T)"""
        W_E, W_U, W_Q, W_K, W_V, W_O = self.paper_weights()
        x0 = W_E[:, x].movedim(0, -2)                            # embed: (..., d, T)
        T = x0.shape[-1]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x0.device), 1)

        # every head reads x0 (not the residual being built) and adds its output into the residual stream
        x1, patterns = x0.clone(), []
        for h in range(self.n_heads):
            q, k, v = W_Q[h] @ x0, W_K[h] @ x0, W_V[h] @ x0      # (..., d_head, T)
            scores = q.mT @ k / self.d_head**0.5
            A = scores.masked_fill(future, float("-inf")).softmax(-1)   # A[i, j]: how much token i attends to token j
            x1 = x1 + W_O[h] @ (v @ A.mT)
            patterns.append(A)

        logits = (W_U @ x1).mT                                   # unembed: (..., T, V)
        return (logits, torch.stack(patterns, dim=-3)) if return_patterns else logits

    def forward(self, x):
        """The same function, all heads at once with the fused attention kernel (what training uses).
        It never forms the patterns, so it can't return them"""
        x0 = self.W_E[x]                                                  # (..., T, d)
        q = torch.einsum("...td,hde->...hte", x0, self.W_Q)               # (..., heads, T, d_head)
        k = torch.einsum("...td,hde->...hte", x0, self.W_K)
        v = torch.einsum("...td,hde->...hte", x0, self.W_V)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x1 = x0 + torch.einsum("...hte,hed->...td", z, self.W_O)
        return x1 @ self.W_U


@torch.no_grad()
def logits_by_path(model, x):
    """Path expansion for one window x (T,): the logits split into the direct path and one attention path per head.
    Returns direct (V, T) and attention (heads, V, T); direct + attention.sum(0) is the model's logits, transposed"""
    W_E, W_U, W_Q, W_K, W_V, W_O = model.paper_weights()
    E = W_E[:, x]                                                # (d, T), one column per token
    future = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool, device=E.device), 1)

    direct = W_U @ E                                             # token -> logits, no attention involved
    attention = []
    for h in range(model.n_heads):
        W_QK, W_OV = W_Q[h].T @ W_K[h], W_O[h] @ W_V[h]
        scores = E.T @ W_QK @ E / model.d_head**0.5              # QK circuit: which sources each destination wants
        A = scores.masked_fill(future, float("-inf")).softmax(-1)
        attention.append(W_U @ (W_OV @ E @ A.T))                 # OV circuit: what each attended source does to the logits
    return direct, torch.stack(attention)


def save_checkpoint(path, model, opt=None, step=0, config=None, history=(), overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict() if opt else None, step=step,
                    config=dict(config or {}), history=list(history)), path)


def load_checkpoint(path, model, opt=None, device=None):
    """Load weights (and optimizer state, if given) in place; returns the saved dict (step, config, history)"""
    ckpt = torch.load(path, map_location=device)
    for old, new in ARCH_KEYS.items():
        if old in ckpt["config"] and ckpt["config"][old] != getattr(model, new):
            raise ValueError(f"checkpoint has {old}={ckpt['config'][old]}, model has {new}={getattr(model, new)}")
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    return ckpt
