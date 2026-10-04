"""Training for the one-layer model: batches from a mix of sources, lr schedule, validation, checkpoint and resume.

Streams are dicts of source name -> 1-D token tensor (on the same device as the model)."""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import load_checkpoint, save_checkpoint


def windows(ids, starts, context_len):
    """the (n, T) inputs and next-token targets of the windows that begin at `starts`"""
    chunk = ids[starts[:, None] + torch.arange(context_len + 1, device=ids.device)].long()
    return chunk[:, :-1], chunk[:, 1:]


def sample_batch(streams, mix, n, context_len, generator=None):
    """n random windows, split across the sources by the shares in `mix`, so no window crosses from one source to another"""
    per_source = torch.multinomial(torch.tensor(list(mix.values())), n, replacement=True, generator=generator)
    xs, ys = [], []
    for source, k in zip(mix, per_source.bincount(minlength=len(mix)).tolist()):
        if k:
            ids = streams[source]
            starts = torch.randint(len(ids) - context_len - 1, (k,), generator=generator).to(ids.device)
            x, y = windows(ids, starts, context_len)
            xs.append(x), ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def lr_at(step, peak, warmup, total):
    """linear warmup to peak, then cosine decay to 10% of it"""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def fixed_val_starts(val_streams, context_len, n_windows, seed=0):
    """the same val windows every time: n_windows start positions per source"""
    first = next(iter(val_streams.values()))
    g = torch.Generator(device=first.device).manual_seed(seed)
    return {s: torch.randint(len(ids) - context_len - 1, (n_windows,), device=ids.device, generator=g)
            for s, ids in val_streams.items()}


@torch.no_grad()
def val_losses(model, val_streams, starts, context_len, batch_size=32):
    """mean cross-entropy per source on the fixed val windows, under the same bf16 autocast as training"""
    out = {}
    for source, ids in val_streams.items():
        x, y = windows(ids, starts[source], context_len)
        batches = list(zip(x.split(batch_size), y.split(batch_size)))
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            losses = [F.cross_entropy(model(xb).float().flatten(0, 1), yb.flatten()).item() for xb, yb in batches]
        out[source] = sum(losses) / len(losses)
    return out


def unigram_val_losses(train_streams, val_streams, vocab_size):
    """val loss per source of predicting every token from its train frequency alone (add-one smoothed): the floor
    the model has to beat to have learned anything about context"""
    counts = sum(torch.bincount(ids.long(), minlength=vocab_size) for ids in train_streams.values()).float() + 1
    logp = (counts / counts.sum()).log()
    return {s: -logp[ids.long()].mean().item() for s, ids in val_streams.items()}


def train(model, train_streams, val_streams, *, mix, context_len, batch_size, total_steps, peak_lr, warmup_steps,
          ckpt_path, accum_steps=1, grad_clip=1.0, eval_every=500, val_windows=64, ckpt_every=2000, config=None):
    """Train with Adam and return the history, a list of (step, train loss, {source: val loss}).

    If ckpt_path exists the model is loaded from it and nothing is trained, so re-running never retrains. If only
    <ckpt_path>.partial exists (a crash mid-run) training resumes from it. Checkpoints are written to those paths only."""
    device = next(model.parameters()).device
    partial_path = ckpt_path.with_suffix(".partial.pt")
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)

    history, start = [], 0
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, device=device)
        print(f"loaded {ckpt_path.name} (trained {ckpt['step']} steps); skipping training")
        model.eval()
        return ckpt["history"]
    if partial_path.exists():
        ckpt = load_checkpoint(partial_path, model, opt, device=device)
        history, start = ckpt["history"], ckpt["step"]
        print(f"resuming from {partial_path.name} at step {start}")

    starts = fixed_val_starts(val_streams, context_len, val_windows)
    unigram = unigram_val_losses(train_streams, val_streams, model.vocab_size)
    print(f"{total_steps:,} steps of {accum_steps * batch_size * context_len:,} tokens; unigram val loss "
          + ", ".join(f"{s} {l:.2f}" for s, l in unigram.items()))

    recent = []
    for step in tqdm(range(start, total_steps), desc="training...", initial=start, total=total_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, peak_lr, warmup_steps, total_steps)
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(accum_steps):
            x, y = sample_batch(train_streams, mix, batch_size, context_len)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / accum_steps   # loss in fp32
            loss.backward()
            step_loss += loss.item()
        if not math.isfinite(step_loss):
            raise RuntimeError(f"loss is {step_loss} at step {step}; last partial checkpoint: {partial_path}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        recent.append(step_loss)
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            vals = val_losses(model, val_streams, starts, context_len, batch_size)
            history.append((step + 1, sum(recent) / len(recent), vals))
            recent = []
            tqdm.write(f"step {step + 1}: train {history[-1][1]:.2f}, val " + ", ".join(f"{s} {l:.2f}" for s, l in vals.items()))
        if (step + 1) % ckpt_every == 0:
            save_checkpoint(partial_path, model, opt, step + 1, config, history, overwrite=True)

    if start < total_steps:
        save_checkpoint(ckpt_path, model, opt, total_steps, config, history)
        partial_path.unlink(missing_ok=True)
        print(f"saved {ckpt_path}")
    model.eval()
    return history
