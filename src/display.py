"""Small plotting and printing helpers for the notebook."""

import matplotlib.pyplot as plt


# colors follow the entity: the same corpus source gets the same color in every figure (the first three slots of a
# palette checked for color-blind separation); grey is for everything that isn't a source
SOURCE_COLORS = {"wiki": "#2a78d6", "code": "#eb6834", "math": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#8a8984"


def _axes(figsize=(6, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def plot_history(history, unigram=None):
    """train loss and per-source val loss against step, with each source's unigram loss as a dashed reference"""
    steps = [h[0] for h in history]
    fig, ax = _axes()
    ax.plot(steps, [h[1] for h in history], color=INK, lw=1.5, label="train")
    for source in history[0][2]:
        color = SOURCE_COLORS.get(source)
        ax.plot(steps, [h[2][source] for h in history], color=color, lw=1.5, label=f"val {source}")
        if unigram:
            ax.axhline(unigram[source], color=color, ls="--", lw=0.8)
    ax.set(xlabel="step", ylabel="loss")
    ax.legend(frameon=False)
    plt.close(fig)                  # closed so a notebook shows the returned figure once, not twice
    return fig


def plot_attention(pattern, labels, title=None):
    """one head's attention pattern over a short snippet: row = destination token, column = source token it looks at"""
    size = min(0.3 * len(labels) + 2, 9)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(pattern.detach().cpu(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="source (attended to)", ylabel="destination", title=title)
    plt.close(fig)
    return fig


def plot_eigenvalues(eigs, title=None):
    """the OV circuit's eigenvalues in the complex plane. mass to the right of the vertical axis means copying"""
    eigs = eigs.detach().cpu()
    # not aspect="equal": the real parts span thousands while the imaginary parts span hundreds, so equal axes squash
    # the cloud into a flat strip. the two axes are read separately (sign of the real part is what matters)
    fig, ax = _axes((7, 4.5))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(eigs.real, eigs.imag, s=22, color=SOURCE_COLORS["wiki"], edgecolor="white", linewidth=0.5)
    big = eigs.abs().argmax()
    ax.annotate(f"largest: {eigs.real[big]:.0f}", (eigs.real[big], eigs.imag[big]), xytext=(6, 8),
                textcoords="offset points", ha="left", fontsize=8, color=INK)
    ax.margins(x=0.1, y=0.25)
    ax.set(xlabel="real part (right of 0 = boosts itself)", ylabel="imaginary part", title=title)
    plt.close(fig)
    return fig


def plot_copying_scores(trained, untrained=None):
    """copying score of every head (sum of eigenvalues over sum of their magnitudes), trained vs untrained"""
    fig, ax = _axes((7, 3))
    heads = range(len(trained))
    if untrained is not None:
        ax.scatter(heads, untrained, s=20, color=MUTED, label="untrained")
    ax.scatter(heads, trained, s=20, color=SOURCE_COLORS["wiki"], label="trained")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="head", ylabel="copying score", ylim=(-1, 1))
    if untrained is not None:
        ax.legend(frameon=False)
    plt.close(fig)
    return fig


def _tok(token_strs, i):
    return repr(token_strs[int(i)])


def format_trigram(st, i, token_strs):
    """one line: [source] … [destination] → [output], the dominant corpus source, then the evidence. train: hits k out of
    the n windows (rate), against the rate expected if the source didn't matter, and the smoothed lift; val the same"""
    share = st.hits_by_source[i] / st.hits_by_source[i].sum().clamp(min=1)
    main = share.argmax().item()
    k, n, e = (st.train[x][i].item() for x in ("k", "n", "expected"))
    vk, vn, ve = (st.val[x][i].item() for x in ("k", "n", "expected"))
    return (f"{_tok(token_strs, st.s[i]):>14} … {_tok(token_strs, st.d[i]):<14} → {_tok(token_strs, st.o[i]):<14} "
            f"[{st.sources[main]} {share[main]:.0%}]  train {k:.0f}/{n:.0f} = {k / max(n, 1):.1%} vs {e / max(n, 1):.2%} "
            f"(lift ×{st.lift[i].item():.1f}) | val {vk:.0f}/{vn:.0f} (lift ×{st.val_lift[i].item():.1f})")


def show_trigrams(st, idx, token_strs, title=None):
    if title:
        print(title)
    for i in idx.tolist():
        print("  " + format_trigram(st, i, token_strs))


def domain_coverage(st, idx):
    """how many of the rows idx are dominated by each corpus source: {source: count}"""
    main = st.hits_by_source[idx].argmax(dim=-1).tolist()
    return {name: main.count(j) for j, name in enumerate(st.sources)}
