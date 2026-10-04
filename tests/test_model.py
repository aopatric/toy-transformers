import copy

import pytest
import torch
import torch.nn.functional as F

from src.model import OneLayerAttentionOnlyTransformer as Model
from src.model import load_checkpoint, logits_by_path, save_checkpoint


@pytest.fixture
def model(tiny_config, device):
    torch.manual_seed(0)
    return Model(**tiny_config).double().to(device)


@pytest.fixture
def tiny_tokens(tiny_tokens, device):
    return tiny_tokens.to(device)


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_shapes_and_param_count(model, tiny_config):
    V, d, h, dh = (tiny_config[k] for k in ("vocab_size", "d_model", "n_heads", "d_head"))
    assert sum(p.numel() for p in model.parameters()) == 2 * V * d + 4 * h * dh * d
    W_E, W_U, W_Q, W_K, W_V, W_O = model.paper_weights()
    assert W_E.shape == (d, V) and W_U.shape == (V, d)
    assert W_Q.shape == W_K.shape == W_V.shape == (h, dh, d)
    assert W_O.shape == (h, d, dh)


def test_paper_weights_are_views(model):
    for view, param in zip(model.paper_weights(), (model.W_E, model.W_U, model.W_Q, model.W_K, model.W_V, model.W_O)):
        assert view.data_ptr() == param.data_ptr()


def test_qk_and_ov_products_match_stored_layout(model):
    _, _, W_Q, W_K, W_V, W_O = model.paper_weights()
    for h in range(model.n_heads):
        assert torch.allclose(W_Q[h].T @ W_K[h], model.W_Q[h] @ model.W_K[h].T)
        assert torch.allclose(W_O[h] @ W_V[h], (model.W_V[h] @ model.W_O[h]).T)


def reference_logits(model, x):
    """one token at a time, plain python loops over the stored-layout weights"""
    out = []
    for t in range(len(x)):
        resid = model.W_E[x[t]].clone()
        for h in range(model.n_heads):
            q = model.W_E[x[t]] @ model.W_Q[h]
            ks = torch.stack([model.W_E[x[s]] @ model.W_K[h] for s in range(t + 1)])
            vs = torch.stack([model.W_E[x[s]] @ model.W_V[h] for s in range(t + 1)])
            a = (ks @ q / model.d_head**0.5).softmax(0)
            resid = resid + (a @ vs) @ model.W_O[h]
        out.append(resid @ model.W_U)
    return torch.stack(out)


def test_walkthrough_matches_loop_reference(model, tiny_tokens):
    x = tiny_tokens[0]
    assert rel_err(model.forward_per_head(x), reference_logits(model, x)) < 1e-10


def test_patterns_are_causal_and_normalized(model, tiny_tokens):
    _, patterns = model.forward_per_head(tiny_tokens, return_patterns=True)
    B, T = tiny_tokens.shape
    assert patterns.shape == (B, model.n_heads, T, T)
    assert torch.allclose(patterns.sum(-1), torch.ones(B, model.n_heads, T, dtype=patterns.dtype, device=patterns.device))
    assert torch.all(patterns.triu(1) == 0)


def test_batch_equals_single_windows(model, tiny_tokens):
    batched = model.forward_per_head(tiny_tokens)
    for x, row in zip(tiny_tokens, batched):
        assert torch.allclose(model.forward_per_head(x), row)


def test_future_tokens_dont_change_earlier_logits(model, tiny_tokens):
    x = tiny_tokens[0]
    y = x.clone()
    y[7:] = (y[7:] + 1) % model.vocab_size
    for fwd in (model.forward_per_head, model):
        assert torch.allclose(fwd(x)[:7], fwd(y)[:7])


def test_fused_matches_walkthrough(model, tiny_tokens):
    assert rel_err(model(tiny_tokens), model.forward_per_head(tiny_tokens)) < 1e-10
    assert rel_err(model(tiny_tokens[0]), model.forward_per_head(tiny_tokens[0])) < 1e-10


def test_fused_matches_walkthrough_fp32(tiny_config, tiny_tokens, device):
    torch.manual_seed(1)
    m = Model(**tiny_config).to(device)
    assert rel_err(m(tiny_tokens), m.forward_per_head(tiny_tokens)) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda (run unsandboxed)")
def test_fused_matches_walkthrough_on_gpu():
    torch.manual_seed(2)
    m = Model(vocab_size=500, d_model=128, n_heads=8, d_head=16, device="cuda")
    x = torch.randint(500, (2, 64), device="cuda")
    with torch.no_grad():
        assert rel_err(m(x), m.forward_per_head(x)) < 1e-4


def test_paths_sum_to_logits(model, tiny_tokens):
    x = tiny_tokens[0]
    direct, attention = logits_by_path(model, x)
    assert direct.shape == (model.vocab_size, len(x))
    assert attention.shape == (model.n_heads, model.vocab_size, len(x))
    assert rel_err((direct + attention.sum(0)).T, model(x)) < 1e-10


def test_zeroed_heads_leave_only_the_direct_path(model, tiny_tokens):
    x = tiny_tokens[0]
    direct, _ = logits_by_path(model, x)
    quiet = copy.deepcopy(model)
    quiet.W_O.data.zero_()
    assert torch.allclose(quiet(x), direct.T)
    assert torch.allclose(logits_by_path(quiet, x)[1], torch.zeros_like(logits_by_path(quiet, x)[1]))


def test_one_head_alone_matches_its_attention_path(model, tiny_tokens):
    x = tiny_tokens[0]
    direct, attention = logits_by_path(model, x)
    for h in range(model.n_heads):
        solo = copy.deepcopy(model)
        keep = torch.zeros(model.n_heads, dtype=torch.bool, device=model.W_O.device)
        keep[h] = True
        solo.W_O.data[~keep] = 0
        assert torch.allclose(solo(x), (direct + attention[h]).T)


def test_source_effect_depends_only_on_its_token(model, tiny_tokens):
    # attention path at t = sum over sources of (attention weight) * (row of a per-token OV table)
    x = tiny_tokens[0]
    _, attention = logits_by_path(model, x)
    _, patterns = model.forward_per_head(x, return_patterns=True)
    for h in range(model.n_heads):
        ov_table = model.W_E @ model.W_V[h] @ model.W_O[h] @ model.W_U      # (V_source, V_out)
        expected = patterns[h] @ ov_table[x]                                 # (T, V)
        assert torch.allclose(attention[h].T, expected)


def test_checkpoint_round_trip(model, tiny_config, tmp_path, device):
    opt = torch.optim.Adam(model.parameters())
    cfg = dict(vocab_size=40, hidden_dim=16, num_heads=4, head_dim=8)
    save_checkpoint(tmp_path / "a.pt", model, opt, step=7, config=cfg, history=[(7, 1.0, {"wiki": 2.0})])
    other = Model(**tiny_config).double().to(device)
    ckpt = load_checkpoint(tmp_path / "a.pt", other, torch.optim.Adam(other.parameters()), device=device)
    assert ckpt["step"] == 7 and ckpt["history"] == [(7, 1.0, {"wiki": 2.0})]
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_refuses_overwrite(model, tmp_path):
    save_checkpoint(tmp_path / "a.pt", model)
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "a.pt", model)
    save_checkpoint(tmp_path / "a.pt", model, overwrite=True)


def test_checkpoint_config_mismatch_raises(model, tiny_config, tmp_path):
    save_checkpoint(tmp_path / "a.pt", model, config=dict(hidden_dim=999))
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "a.pt", Model(**tiny_config))


@pytest.mark.real_ckpt
def test_real_checkpoint_reproduces_training_val_loss(real_model, real_corpus):
    from conftest import REAL_CKPT_NAME
    from src.mixed import REPO_ROOT, SOURCES
    from src.train import fixed_val_starts, val_losses

    history = torch.load(REPO_ROOT / "checkpoints" / REAL_CKPT_NAME, map_location="cpu")["history"]
    val = {s: torch.from_numpy(real_corpus.val[s]).cuda() for s in SOURCES}
    losses = val_losses(real_model, val, fixed_val_starts(val, 1024, 64), 1024)
    for source in SOURCES:
        assert abs(losses[source] - history[-1][2][source]) < 0.05


def test_every_parameter_is_initialized_at_its_own_dimension_scale():
    torch.manual_seed(0)
    V, d, h, dh = 3000, 256, 8, 32
    model = Model(V, d, h, dh)
    expected = dict(W_E=d**-0.5, W_U=d**-0.5, W_Q=d**-0.5, W_K=d**-0.5, W_V=d**-0.5, W_O=(h * dh) ** -0.5)
    for name, std in expected.items():
        assert getattr(model, name).std().item() == pytest.approx(std, rel=0.05), name
    # so the residual stream and the logits start at O(1) or smaller, whatever the width
    x = torch.randint(V, (4, 64))
    with torch.no_grad():
        assert model.W_E[x].norm(dim=-1).mean().item() == pytest.approx(1.0, rel=0.05)
        assert model(x).std().item() < 1.0
