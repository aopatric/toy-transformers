import math

import pytest
import torch

from src.display import plot_history
from src.model import OneLayerAttentionOnlyTransformer as Model
from src.model import load_checkpoint
from src.train import fixed_val_starts, lr_at, sample_batch, train, unigram_val_losses, val_losses, windows

V = 12


def pattern_streams(device, n=2000):
    """a deterministic cycle 0, 1, ..., V-1, 0, ...: next token = this token + 1, which even a bigram model learns"""
    ids = (torch.arange(n) % V).to(device)
    return {"a": ids, "b": ids.clone()}


def tiny_model(device):
    torch.manual_seed(0)
    return Model(V, 16, 2, 8, device=device)


def test_windows_shift_by_one(device):
    ids = torch.arange(100, device=device)
    x, y = windows(ids, torch.tensor([0, 10], device=device), 5)
    assert x.tolist() == [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]
    assert torch.equal(y, x + 1)


def test_sample_batch_shapes_and_single_source_windows(device):
    streams = {s: torch.full((500,), i + 1, device=device) for i, s in enumerate("abc")}
    x, y = sample_batch(streams, {"a": 0.5, "b": 0.3, "c": 0.2}, 64, 16)
    assert x.shape == y.shape == (64, 16)
    assert all(len(row.unique()) == 1 for row in x)                    # every window comes from one source


def test_sample_batch_follows_the_mix(device):
    streams = {s: torch.full((500,), i + 1, device=device) for i, s in enumerate("abc")}
    g = torch.Generator().manual_seed(0)
    x, _ = sample_batch(streams, {"a": 0.5, "b": 0.3, "c": 0.2}, 5000, 8, generator=g)
    share = [(x[:, 0] == i).float().mean().item() for i in (1, 2, 3)]
    assert share == pytest.approx([0.5, 0.3, 0.2], abs=0.03)


def test_sample_batch_is_seedable(device):
    streams = {"a": torch.arange(1000, device=device)}
    a = sample_batch(streams, {"a": 1.0}, 8, 10, generator=torch.Generator().manual_seed(1))[0]
    b = sample_batch(streams, {"a": 1.0}, 8, 10, generator=torch.Generator().manual_seed(1))[0]
    assert torch.equal(a, b)


def test_lr_schedule():
    peak, warmup, total = 1e-3, 10, 100
    assert lr_at(0, peak, warmup, total) == pytest.approx(peak / warmup)
    assert lr_at(warmup - 1, peak, warmup, total) == pytest.approx(peak)
    assert lr_at(warmup, peak, warmup, total) == pytest.approx(peak)
    assert lr_at(total, peak, warmup, total) == pytest.approx(0.1 * peak)
    after = [lr_at(s, peak, warmup, total) for s in range(warmup, total + 1)]
    assert all(a >= b for a, b in zip(after, after[1:]))


def test_fixed_val_starts_are_deterministic_and_in_bounds(device):
    streams = {"a": torch.arange(300, device=device), "b": torch.arange(500, device=device)}
    one, two = (fixed_val_starts(streams, 32, 10) for _ in range(2))
    assert all(torch.equal(one[s], two[s]) for s in streams)
    assert all((one[s] + 33 <= len(streams[s])).all() for s in streams)


def test_zero_logits_give_log_vocab_loss(device):
    model = tiny_model(device)
    model.W_U.data.zero_()
    streams = pattern_streams(device)
    losses = val_losses(model, streams, fixed_val_starts(streams, 16, 8), 16)
    assert all(l == pytest.approx(math.log(V), abs=1e-3) for l in losses.values())


def test_unigram_baseline_on_uniform_tokens(device):
    streams = pattern_streams(device)
    assert all(l == pytest.approx(math.log(V), abs=0.02) for l in unigram_val_losses(streams, streams, V).values())


def run_train(model, streams, ckpt, total_steps, device, **kw):
    return train(model, streams, streams, mix={"a": 0.5, "b": 0.5}, context_len=16, batch_size=8, total_steps=total_steps,
                 peak_lr=3e-3, warmup_steps=5, ckpt_path=ckpt, eval_every=kw.pop("eval_every", 50), val_windows=8,
                 ckpt_every=kw.pop("ckpt_every", 1000), **kw)


def test_training_learns_a_simple_stream(device, tmp_path):
    streams = pattern_streams(device)
    model = tiny_model(device)
    history = run_train(model, streams, tmp_path / "m.pt", 150, device)
    assert history[-1][0] == 150
    assert history[-1][2]["a"] < 0.5 * math.log(V)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pt"]       # final checkpoint only, partial cleaned up


def test_resumes_from_partial_checkpoint(device, tmp_path):
    streams = pattern_streams(device)
    first = run_train(tiny_model(device), streams, tmp_path / "m.pt", 10, device, eval_every=5)
    (tmp_path / "m.pt").rename(tmp_path / "m.partial.pt")              # as if the run had died at step 10 of 20
    model = tiny_model(device)
    history = run_train(model, streams, tmp_path / "m.pt", 20, device, eval_every=5)
    assert [h[0] for h in history] == [5, 10, 15, 20] and history[:2] == first
    assert load_checkpoint(tmp_path / "m.pt", tiny_model(device), device=device)["step"] == 20
    assert not (tmp_path / "m.partial.pt").exists()


def test_final_checkpoint_means_no_training(device, tmp_path):
    streams = pattern_streams(device)
    run_train(tiny_model(device), streams, tmp_path / "m.pt", 10, device)
    saved = (tmp_path / "m.pt").read_bytes()
    other = tiny_model(device)
    with torch.no_grad():
        other.W_E.add_(1.0)
    history = run_train(other, streams, tmp_path / "m.pt", 10_000, device)       # would take forever if it trained
    assert (tmp_path / "m.pt").read_bytes() == saved and history[-1][0] == 10
    assert torch.equal(other.W_U, torch.load(tmp_path / "m.pt", map_location=device)["model"]["W_U"])


def test_non_finite_loss_stops_the_run(device, tmp_path):
    model = tiny_model(device)
    model.W_E.data.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="loss is"):
        run_train(model, pattern_streams(device), tmp_path / "m.pt", 5, device)
    assert not (tmp_path / "m.pt").exists()


def test_plot_history_lines():
    history = [(10, 3.0, {"a": 3.1, "b": 3.2}), (20, 2.0, {"a": 2.1, "b": 2.2})]
    fig = plot_history(history, unigram={"a": 4.0, "b": 4.5})
    ax = fig.axes[0]
    assert len(ax.lines) == 1 + 2 + 2                                  # train, two val curves, two unigram references
    assert len(plot_history(history).axes[0].lines) == 3


@pytest.mark.real_ckpt
def test_smoke_training_at_the_real_architecture(real_corpus, tmp_path):
    """20 steps of the real config on the real corpus, to a temp path; the real checkpoint must stay untouched"""
    from conftest import REAL_CKPT_NAME
    from src.mixed import REPO_ROOT, SOURCES

    real = REPO_ROOT / "checkpoints" / REAL_CKPT_NAME
    before = (real.stat().st_mtime_ns, real.stat().st_size)
    train_s = {s: torch.from_numpy(real_corpus.train[s]).cuda() for s in SOURCES}
    val_s = {s: torch.from_numpy(real_corpus.val[s]).cuda() for s in SOURCES}
    torch.manual_seed(0)
    model = Model(real_corpus.vocab_size, 1024, 32, 128, device="cuda")
    history = train(model, train_s, val_s, mix={"wiki": 0.4, "code": 0.3, "math": 0.3}, context_len=1024, batch_size=32,
                    total_steps=20, peak_lr=1e-3, warmup_steps=5, ckpt_path=tmp_path / "smoke.pt", eval_every=10, val_windows=8)
    assert [h[0] for h in history] == [10, 20] and all(math.isfinite(h[1]) for h in history)
    assert history[-1][1] < history[0][1]                              # loss goes down
    assert (tmp_path / "smoke.pt").exists()
    assert (real.stat().st_mtime_ns, real.stat().st_size) == before


def test_history_plot_colors_follow_the_source():
    from src.display import SOURCE_COLORS

    history = [(10, 3.0, {"wiki": 3.1, "code": 3.2}), (20, 2.0, {"wiki": 2.1, "code": 2.2})]
    colors = {line.get_label(): line.get_color() for line in plot_history(history).axes[0].lines}
    assert colors["val wiki"] == SOURCE_COLORS["wiki"] and colors["val code"] == SOURCE_COLORS["code"]
