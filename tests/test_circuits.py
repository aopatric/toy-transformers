import pytest
import torch

from src.circuits import copying_score, copying_scores, ov_eigenvalues, ov_eigenvector, ov_logits, qk_scores, sample_readouts, top_outputs, top_sources
from src.model import OneLayerAttentionOnlyTransformer as Model
from src.model import logits_by_path


@pytest.fixture
def model(tiny_config, device):
    torch.manual_seed(0)
    return Model(**tiny_config).double().to(device)


@pytest.fixture
def tiny_tokens(tiny_tokens, device):
    return tiny_tokens.to(device)


def test_qk_scores_reproduce_the_real_patterns(model, tiny_tokens):
    x = tiny_tokens[0]
    _, patterns = model.forward_per_head(x, return_patterns=True)
    future = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool, device=x.device), 1)
    for h in range(model.n_heads):
        scores = qk_scores(model, h, x)[:, x]                       # destination rows, source columns
        assert scores.shape == (len(x), len(x))
        assert torch.allclose(scores.masked_fill(future, float("-inf")).softmax(-1), patterns[h])


def test_qk_scores_match_the_full_circuit(model):
    W_E, _, W_Q, W_K, *_ = model.paper_weights()
    dests = torch.tensor([5, 0, 33], device=model.W_E.device)
    for h in range(model.n_heads):
        full = W_E.T @ (W_Q[h].T @ W_K[h]) @ W_E / model.d_head**0.5
        assert torch.allclose(qk_scores(model, h, dests), full[dests])


def test_ov_logits_match_the_full_circuit(model):
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    srcs = torch.tensor([5, 0, 33], device=model.W_E.device)
    for h in range(model.n_heads):
        full = W_U @ (W_O[h] @ W_V[h]) @ W_E                         # (out, source), the paper's order
        assert torch.allclose(ov_logits(model, h, srcs), full.T[srcs])


def test_ov_logits_are_the_attention_path_lookup(model, tiny_tokens):
    x = tiny_tokens[0]
    _, attention = logits_by_path(model, x)
    _, patterns = model.forward_per_head(x, return_patterns=True)
    for h in range(model.n_heads):
        assert torch.allclose(attention[h].T, patterns[h] @ ov_logits(model, h, x))


def test_top_sources_and_outputs_match_argsort(model):
    for h in range(model.n_heads):
        ids, scores = top_sources(model, h, 7, k=5)
        row = qk_scores(model, h, torch.tensor([7], device=model.W_E.device))[0]
        assert ids.tolist() == row.argsort(descending=True)[:5].tolist() and torch.allclose(scores, row[ids])
        ids, scores = top_outputs(model, h, 7, k=5)
        row = ov_logits(model, h, torch.tensor([7], device=model.W_E.device))[0]
        assert ids.tolist() == row.argsort(descending=True)[:5].tolist() and torch.allclose(scores, row[ids])


def test_top_respects_mask(model):
    ids, _ = top_sources(model, 0, 7, k=5)
    mask = torch.zeros(model.vocab_size, dtype=torch.bool, device=model.W_E.device)
    mask[ids[:2]] = True
    masked, _ = top_sources(model, 0, 7, k=5, mask=mask)
    assert not set(masked.tolist()) & set(ids[:2].tolist())
    mask[:] = True
    mask[:3] = False
    assert len(top_outputs(model, 0, 7, k=5, mask=mask)[0]) == 3      # fewer allowed tokens than k


def test_ov_eigenvalues_match_the_full_matrix(model):
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    for h in range(model.n_heads):
        full = torch.linalg.eigvals(W_U @ (W_O[h] @ W_V[h]) @ W_E)                # (V, V) matrix
        small = ov_eigenvalues(model, h)
        assert len(small) == model.d_head
        ri = torch.view_as_real
        assert torch.cdist(ri(small), ri(full)).min(dim=1).values.max() < 1e-6   # every small eigenvalue is in the full set
        assert full.abs().sort().values[: model.vocab_size - model.d_head].max() < 1e-8  # and the rest are zero


def test_ov_eigenvector_is_an_eigenvector_of_the_full_matrix(model):
    W_E, W_U, _, _, W_V, W_O = model.paper_weights()
    for h in range(model.n_heads):
        full = W_U @ (W_O[h] @ W_V[h]) @ W_E
        eig = ov_eigenvalues(model, h)
        for negative in (True, False):
            value, vec = ov_eigenvector(model, h, negative=negative)
            assert torch.allclose(full @ vec, value * vec, atol=1e-8 * vec.abs().max())
            real = eig[eig.imag.abs() <= 1e-6 * eig.abs()].real
            assert value == (real.min() if negative else real.max())


def planted(sign, device="cpu"):
    """one head whose OV circuit is sign * identity, so its eigenvalues are all sign"""
    d = 12
    m = Model(vocab_size=d, d_model=d, n_heads=1, d_head=d, device=device)
    with torch.no_grad():
        m.W_E.copy_(torch.eye(d)); m.W_U.copy_(torch.eye(d)); m.W_V[0].copy_(torch.eye(d)); m.W_O[0].copy_(sign * torch.eye(d))
    return m


def test_copying_score_planted(device):
    assert copying_score(planted(+1, device), 0) == pytest.approx(1.0)
    assert copying_score(planted(-1, device), 0) == pytest.approx(-1.0)


def test_copying_score_range_and_untrained_near_zero(device):
    torch.manual_seed(0)
    m = Model(vocab_size=200, d_model=64, n_heads=6, d_head=16, device=device)
    scores = copying_scores(m)
    assert scores.shape == (6,) and scores.abs().max() <= 1 and scores.abs().max() < 0.4
    assert all(scores[h] == pytest.approx(copying_score(m, h)) for h in range(6))


def test_sample_readouts(model):
    excluded = torch.zeros(model.vocab_size, dtype=torch.bool, device=model.W_E.device)
    excluded[:20] = True
    rows = sample_readouts(model, excluded, n=6, k=4, seed=3)
    assert len(rows) == 6
    for h, dest, sources, outputs in rows:
        assert not excluded[dest]
        assert not excluded[sources].any() and not excluded[outputs].any()
        assert sources.tolist() == top_sources(model, h, dest, 4, mask=excluded)[0].tolist()
        assert outputs.tolist() == top_outputs(model, h, sources[0].item(), 4, mask=excluded)[0].tolist()
    assert [r[:2] for r in rows] == [r[:2] for r in sample_readouts(model, excluded, n=6, k=4, seed=3)]
    assert {r[0] for r in sample_readouts(model, excluded, n=10, heads=[2], seed=0)} == {2}


@pytest.mark.real_ckpt
def test_real_copying_scores_are_mostly_positive_and_untrained_is_not(real_model):
    # some heads legitimately do something else (one dominant negative eigenvalue), so this checks the bulk, not every head
    torch.manual_seed(0)
    fresh = Model(real_model.vocab_size, real_model.d_model, real_model.n_heads, real_model.d_head, device="cuda")
    trained, untrained = copying_scores(real_model), copying_scores(fresh)
    assert (trained > 0.2).float().mean() > 0.7 and trained.mean() > 0.4 and trained.max() > 0.8   # measured: 100%/0.70/0.91 and 84%/0.49/0.95
    assert untrained.abs().max() < 0.1


@pytest.mark.real_ckpt
def test_real_circuits_reproduce_real_patterns(real_model, real_corpus):
    text = "import numpy as np\nimport torch.nn as nn\n\nclass Encoder(nn.Module):\n    def forward(self, x):\n        return np"
    x = torch.tensor(real_corpus.encode(text), device="cuda")
    with torch.no_grad():
        _, patterns = real_model.forward_per_head(x, return_patterns=True)
    future = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool, device="cuda"), 1)
    for h in range(real_model.n_heads):
        from_circuit = qk_scores(real_model, h, x)[:, x].masked_fill(future, float("-inf")).softmax(-1)
        assert (from_circuit - patterns[h]).abs().max() < 1e-3
