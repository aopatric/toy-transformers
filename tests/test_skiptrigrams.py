import pytest
import torch

from src.skiptrigrams import count_skip_trigrams, next_occurrence


def brute_force(ids, cand_keys, V, lookback):
    ids = ids.tolist()
    pair_n = torch.zeros(V * V, dtype=torch.long)
    keys = cand_keys.tolist()
    triple_k = [0] * len(keys)
    for t in range(len(ids) - 1):
        window = set(ids[max(0, t - lookback + 1) : t])          # the previous lookback-1 tokens, distinct
        d, o = ids[t], ids[t + 1]
        for s in window:
            pair_n[s * V + d] += 1
            key = (s * V + d) * V + o
            if key in keys:
                triple_k[keys.index(key)] += 1
    return pair_n, torch.tensor(triple_k, dtype=torch.long)


def random_case(V, n, n_cands, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(V, (n,), generator=g)
    cand_keys = torch.randint(V**3, (n_cands,), generator=g).unique()      # some of these never occur
    return ids, cand_keys


def test_next_occurrence():
    ids = torch.tensor([3, 1, 3, 2, 1, 3])
    assert next_occurrence(ids).tolist() == [2, 4, 5, 6, 6, 6]


def test_next_occurrence_matches_brute_force(device):
    ids = torch.randint(5, (80,), generator=torch.Generator().manual_seed(0))
    expected = [next((v for v in range(u + 1, len(ids)) if ids[v] == ids[u]), len(ids)) for u in range(len(ids))]
    assert next_occurrence(ids.to(device)).tolist() == expected


@pytest.mark.parametrize("lookback", [2, 4, 7, 100])
@pytest.mark.parametrize("seed", [0, 1])
def test_count_skip_trigrams_matches_brute_force(lookback, seed, device):
    V = 6
    ids, cand_keys = random_case(V, 90, 60, seed)
    pair_n, triple_k = count_skip_trigrams(ids.to(device), cand_keys.to(device), V, lookback)
    ref_pair, ref_triple = brute_force(ids, cand_keys, V, lookback)
    assert torch.equal(pair_n.cpu(), ref_pair) and torch.equal(triple_k.cpu(), ref_triple)


def test_hand_checked_example(device):
    # tokens a b a c (a=0 b=1 c=2). t=1 (b, then a): window {a}. t=2 (a, then c): window {a, b}. t=3 has nothing after it
    ids = torch.tensor([0, 1, 0, 2], device=device)
    V = 3
    key = lambda s, d, o: (s * V + d) * V + o
    cands = torch.tensor(sorted([key(0, 1, 0), key(1, 0, 2), key(0, 0, 2)]), device=device)
    pair_n, triple_k = count_skip_trigrams(ids, cands, V, lookback=10)
    assert pair_n.view(V, V)[0, 1] == 1 and pair_n.view(V, V)[0, 0] == 1 and pair_n.view(V, V)[1, 0] == 1
    assert pair_n.sum() == 3                                   # position 3 has no token after it, so it isn't counted
    assert dict(zip(cands.tolist(), triple_k.tolist())) == {key(0, 1, 0): 1, key(1, 0, 2): 1, key(0, 0, 2): 1}


def test_pair_counts_accumulate_into_the_given_table(device):
    V = 6
    ids, cand_keys = random_case(V, 60, 20, 3)
    ids, cand_keys = ids.to(device), cand_keys.to(device)
    once, _ = count_skip_trigrams(ids, cand_keys, V, 5)
    shared = torch.zeros(V * V, dtype=torch.long, device=device)
    count_skip_trigrams(ids, cand_keys, V, 5, pair_n=shared)
    count_skip_trigrams(ids, cand_keys, V, 5, pair_n=shared)
    assert torch.equal(shared, 2 * once)


def test_short_streams(device):
    V = 4
    cands = torch.tensor([0], device=device)
    for ids in ([1], [1, 2]):
        pair_n, triple_k = count_skip_trigrams(torch.tensor(ids, device=device), cands, V, 5)
        assert pair_n.sum() == 0 and triple_k.sum() == 0


# ---- candidates, scoring, lift, pipeline --------------------------------------------------------------------------

from collections import Counter  # noqa: E402

from src.circuits import ov_logits, qk_scores  # noqa: E402
from src.model import OneLayerAttentionOnlyTransformer as Model  # noqa: E402
from src.skiptrigrams import (find_skip_trigrams, lift_table, propose_candidates, score_candidates,  # noqa: E402
                              smoothed_lift)


def test_propose_candidates_match_brute_force(tiny_config, device):
    torch.manual_seed(0)
    model = Model(**tiny_config).double().to(device)
    V = model.vocab_size
    g = torch.Generator().manual_seed(1)
    seen = (torch.rand(V, V, generator=g) < 0.4).to(device)
    follows = (torch.rand(V, V, generator=g) < 0.4).to(device)
    excluded = torch.zeros(V, dtype=torch.bool)
    excluded[[3, 11, 20]] = True
    excluded = excluded.to(device)
    dests = (~excluded).nonzero().squeeze(1)
    rows = propose_candidates(model, dests, seen, follows, excluded, n_sources=3, n_outputs=2, chunk=7)

    expected = set()
    for h in range(model.n_heads):
        for d in dests.tolist():
            qk = qk_scores(model, h, torch.tensor([d], device=device))[0].masked_fill(excluded | ~seen[:, d], float("-inf"))
            for s in qk.argsort(descending=True)[:3].tolist():
                if qk[s] == float("-inf"):
                    continue
                ov = ov_logits(model, h, torch.tensor([s], device=device))[0].masked_fill(excluded | ~follows[d], float("-inf"))
                expected |= {(h, s, d, o) for o in ov.argsort(descending=True)[:2].tolist() if ov[o] != float("-inf")}
    assert {tuple(r) for r in rows.tolist()} == expected and len(rows) == len(expected)


def test_propose_candidates_never_uses_excluded_or_unseen(tiny_config, device):
    torch.manual_seed(2)
    model = Model(**tiny_config).to(device)
    V = model.vocab_size
    seen = torch.zeros(V, V, dtype=torch.bool, device=device)
    seen[5, :] = True                                           # only token 5 was ever seen before anything
    follows = torch.zeros(V, V, dtype=torch.bool, device=device)
    follows[:, 7] = True                                        # only token 7 ever follows
    excluded = torch.zeros(V, dtype=torch.bool, device=device)
    rows = propose_candidates(model, torch.arange(V, device=device), seen, follows, excluded, 3, 3)
    assert len(rows) > 0 and (rows[:, 1] == 5).all() and (rows[:, 3] == 7).all()


def test_score_candidates_match_brute_force(device):
    V, lookback = 6, 5
    g = torch.Generator().manual_seed(0)
    streams = {"a": torch.randint(V, (80,), generator=g), "b": torch.randint(V, (60,), generator=g)}
    cands = torch.stack([torch.randint(2, (40,), generator=g), torch.randint(V, (40,), generator=g),
                         torch.randint(V, (40,), generator=g), torch.randint(V, (40,), generator=g)], dim=-1)
    cands = torch.cat([cands, cands[:5]])                       # the same trigram can come from several heads
    stats = score_candidates({k: v.to(device) for k, v in streams.items()}, cands.to(device), V, lookback)
    for name, ids in streams.items():
        keys = (cands[:, 1] * V + cands[:, 2]) * V + cands[:, 3]
        ref_pair, ref_triple = brute_force(ids, keys.unique(), V, lookback)
        _, inverse = keys.unique(return_inverse=True)
        bigrams = Counter(zip(ids[:-1].tolist(), ids[1:].tolist()))
        d_total = Counter(ids[:-1].tolist())
        assert torch.equal(stats[name]["k"].cpu(), ref_triple[inverse])
        assert torch.equal(stats[name]["n"].cpu(), ref_pair[cands[:, 1] * V + cands[:, 2]])
        assert stats[name]["bi"].tolist() == [bigrams[(d, o)] for d, o in cands[:, 2:].tolist()]
        assert stats[name]["d_total"].tolist() == [d_total[d] for d in cands[:, 2].tolist()]


def test_smoothed_lift_damps_coincidences():
    one_off = smoothed_lift(torch.tensor(1.0), torch.tensor(0.0002))       # raw lift would be 5000
    solid = smoothed_lift(torch.tensor(100.0), torch.tensor(0.5))
    assert one_off == pytest.approx(2.0, rel=1e-3) and solid == pytest.approx(101 / 1.5) and solid > 10 * one_off
    assert smoothed_lift(torch.tensor(5.0), torch.tensor(5.0)) == pytest.approx(6 / 6)       # no effect = lift 1
    ks = torch.arange(10.0)
    assert (smoothed_lift(ks, torch.tensor(3.0)).diff() > 0).all()                          # more hits, higher lift


def fake_stats(k, n, bi, d_total):
    return {name: dict(k=torch.tensor([float(k[i])]), n=torch.tensor([float(n[i])]), bi=torch.tensor([float(bi[i])]),
                       d_total=torch.tensor([float(d_total[i])])) for i, name in enumerate(["A", "B"])}


def test_lift_table_compares_against_each_domains_own_baseline():
    # the trigram only ever fires in domain A, where o follows d half the time anyway; domain B never has o after d
    stats = fake_stats(k=[50, 0], n=[100, 0], bi=[500, 0], d_total=[1000, 1000])
    table = lift_table(stats, stats, ("A", "B"))
    assert table["expected"].item() == pytest.approx(50.0)               # 100 windows * 0.5 in A, nothing from B
    assert smoothed_lift(table["k"], table["expected"]).item() == pytest.approx(1.0)       # so there is no real effect
    pooled_expected = 100 * (500 + 0) / (1000 + 1000)                    # what a baseline mixing both domains would say
    assert smoothed_lift(table["k"], torch.tensor(pooled_expected)).item() > 1.9            # ...and would call it ~2x


def test_lift_table_pools_and_restricts_to_sources():
    stats = fake_stats(k=[10, 4], n=[20, 8], bi=[2, 1], d_total=[10, 10])
    both = lift_table(stats, stats, ("A", "B"))
    assert both["k"].item() == 14 and both["n"].item() == 28 and both["expected"].item() == pytest.approx(20 * 0.2 + 8 * 0.1)
    only_b = lift_table(stats, stats, ("B",))
    assert only_b["k"].item() == 4 and only_b["expected"].item() == pytest.approx(0.8)


def planted_model(device):
    """one head wiring token 2 (destination) to attend to token 5 (source), whose OV boosts token 3; head 1 does nothing"""
    V = 8
    model = Model(vocab_size=V, d_model=V, n_heads=2, d_head=1).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.W_E.copy_(torch.eye(V))
        model.W_Q[0, 2, 0] = 4.0
        model.W_K[0, 5, 0] = 4.0
        model.W_V[0, 5, 0] = 1.0
        model.W_O[0, 0, 3] = 1.0
        model.W_U.copy_(5 * torch.eye(V))
    return model.to(device)


def planted_stream(n_segments, seed, device):
    """segments [5 x x x 2 3] (1 in 10) and [x x x x 2 y] (9 in 10): 3 follows 2 only when 5 came earlier"""
    g = torch.Generator().manual_seed(seed)
    filler = torch.tensor([0, 1, 4, 6, 7])
    out = []
    for _ in range(n_segments):
        x = filler[torch.randint(5, (4,), generator=g)].tolist()
        if torch.rand(1, generator=g).item() < 0.1:
            out += [5, *x[:3], 2, 3]
        else:
            out += [*x, 2, filler[torch.randint(5, (1,), generator=g)].item()]
    return torch.tensor(out, device=device)


def test_planted_skip_trigram_is_found(device):
    model = planted_model(device)
    train = {"a": planted_stream(600, 0, device)}
    val = {"a": planted_stream(300, 1, device)}
    excluded = torch.zeros(8, dtype=torch.bool, device=device)
    st = find_skip_trigrams(model, train, val, excluded, n_sources=2, n_outputs=2, lookback=8)
    best = st.lift.argmax().item()
    assert (st.s[best].item(), st.d[best].item(), st.o[best].item()) == (5, 2, 3)
    assert st.lift[best] > 5 and st.val_lift[best] > 3                   # strong on train, and holds on held-out text
    assert ((st.h == 0) & (st.s == 5) & (st.d == 2) & (st.o == 3)).any()
    assert st.hits_by_source.shape == (len(st), 1) and len(st) == len(st.h)


# ---- patterns and selection ----------------------------------------------------------------------------------------

from src.circuits import unembedding_cosine  # noqa: E402
from src.display import domain_coverage, format_trigram  # noqa: E402
from src.skiptrigrams import (COPY, OTHER, SIMILAR, SPLIT, SPLIT_SIMILAR, SkipTrigrams, classify_pattern,  # noqa: E402
                              examples_by_pattern, select_examples)


@pytest.mark.parametrize("s, d, o, expected", [
    # the figure's own examples
    (" two", " One", " two", COPY), (" perfect", " are", " perfect", COPY), ("nbsp", " &", "nbsp", COPY),
    ("lambda", " $\\", "lambda", COPY),
    ("Ralph", " R", "alph", SPLIT), ("Pike", " P", "ike", SPLIT), ("Pixmap", " P", "ixmap", SPLIT),
    (" Lloyd", " L", "loyd", SPLIT),
    ("Ralph", " R", "ALPH", SPLIT_SIMILAR), ("Pike", " P", "ikes", SPLIT_SIMILAR),
    # spelling variants of the source count as similar without any embedding help
    (" perfect", " looks", " Perfect", SIMILAR), (" run", " to", " running", SIMILAR),
    # near misses
    (" the", " of", " cat", OTHER), ("abc", "ab", "xyz", OTHER), (" a", " b", "", OTHER), ("\n", " x", "y", OTHER),
    ("ab", "a", "ab", COPY),                                      # a copy even though d is a piece of s
])
def test_classify_pattern(s, d, o, expected):
    assert classify_pattern(s, d, o) == expected


def test_classify_similar_needs_embedding_closeness_for_different_words():
    assert classify_pattern(" two", " has", " three", cosine=0.8) == SIMILAR
    assert classify_pattern(" two", " has", " three", cosine=0.1) == OTHER
    assert classify_pattern(" two", " has", " three", cosine=0.3, min_cosine=0.2) == SIMILAR


def test_unembedding_cosine(tiny_config, device):
    import torch.nn.functional as F

    model = Model(**tiny_config).to(device)
    a, b = torch.tensor([1, 2, 3], device=device), torch.tensor([3, 2, 9], device=device)
    assert torch.allclose(unembedding_cosine(model, a, b), F.cosine_similarity(model.W_U[:, a].T, model.W_U[:, b].T, dim=-1))
    assert torch.allclose(unembedding_cosine(model, a, a), torch.ones(3, device=device))


def make_st(device, h, k, expected, hits):
    n = len(h)
    h = torch.tensor(h, device=device)
    z = torch.zeros(n, dtype=torch.long, device=device)
    train = dict(k=torch.tensor(k, device=device, dtype=torch.float), n=torch.full((n,), 100.0, device=device),
                 expected=torch.tensor(expected, device=device, dtype=torch.float))
    val = {key: v.clone() for key, v in train.items()}
    return SkipTrigrams(("wiki", "code"), h, z, z, z, train, val, torch.tensor(hits, device=device, dtype=torch.float))


def test_select_examples(device):
    #        rows:         0     1     2     3     4     5     6
    st = make_st(device, h=[0, 0, 0, 1, 1, 1, 1], k=[10, 30, 4, 20, 50, 8, 6], expected=[1, 1, 1, 1, 5, 1, 1],
                 hits=[[10, 0], [0, 30], [4, 0], [20, 0], [0, 50], [8, 0], [6, 0]])
    # lift = (k+1)/(expected+1): 5.5, 15.5, 2.5 | 10.5, 8.5, 4.5, 3.5
    out = select_examples(st, st.h, top_k=2, min_hits=5)
    assert {g: v.tolist() for g, v in out.items()} == {0: [1, 0], 1: [3, 4]}        # row 2 has < 5 hits; top 2 by lift
    assert select_examples(st, st.h, top_k=10, min_hits=5)[1].tolist() == [3, 4, 5, 6]
    assert {g: v.tolist() for g, v in select_examples(st, st.h, top_k=2, min_hits=40).items()} == {1: [4]}
    skip = torch.tensor([0, 0, 0, -1, -1, -1, -1], device=device)
    assert list(select_examples(st, skip, top_k=5, min_hits=1)) == [0]              # -1 skips rows
    assert select_examples(st, st.h, top_k=2, min_hits=1000) == {}


def test_select_examples_distinct_and_val_filter(device):
    # rows 0 and 2 are the same trigram found by two heads; row 3 is a different one
    st = make_st(device, h=[0, 0, 1, 1], k=[10, 30, 40, 20], expected=[1, 1, 1, 1], hits=[[1, 0]] * 4)
    st.s[:] = torch.tensor([1, 2, 1, 3], device=device)
    st.val["k"] = torch.tensor([0.0, 50.0, 40.0, 0.0], device=device)
    both = select_examples(st, torch.zeros(4, dtype=torch.long, device=device), top_k=10)[0].tolist()
    once = select_examples(st, torch.zeros(4, dtype=torch.long, device=device), top_k=10, distinct=True)[0].tolist()
    assert both == [2, 1, 3, 0] and once == [2, 1, 3]                      # the lower-lift duplicate (row 0) is dropped
    held = select_examples(st, torch.zeros(4, dtype=torch.long, device=device), top_k=10, min_val_lift=3)[0].tolist()
    assert held == [2, 1]                                                  # val lifts: 1, 26, 21, 1


def test_examples_by_pattern_and_display(device):
    strs = ["ab", "a", "b", " x", " y", "ab2"]
    # rows (s, d, o): ab..a->b is a split, b..x->b and x..y->x are copies, x..y->ab2 is neither
    rows = [(0, 1, 2), (2, 3, 2), (3, 4, 3), (3, 4, 5)]
    z = torch.tensor
    n = len(rows)
    st = SkipTrigrams(("wiki", "code"), torch.zeros(n, dtype=torch.long), z([r[0] for r in rows]), z([r[1] for r in rows]),
                      z([r[2] for r in rows]), dict(k=z([9.0, 8.0, 7.0, 6.0]), n=torch.full((n,), 20.0), expected=z([1.0, 1.0, 2.0, 1.0])),
                      dict(k=z([4.0] * n), n=torch.full((n,), 20.0), expected=z([1.0] * n)), torch.tensor([[9.0, 0]] * n))
    out = examples_by_pattern(st, strs, lambda a, b: torch.zeros(len(a)), top_k=5, min_hits=5)
    assert {name: (c, rows.tolist()) for name, (c, rows) in out.items()} == {
        SPLIT: (1, [0]), SPLIT_SIMILAR: (0, []), COPY: (2, [1, 2]), SIMILAR: (0, []), OTHER: (1, [3])}
    line = format_trigram(st, 0, strs)
    assert "'ab'" in line and "[wiki 100%]" in line and "9/20" in line and "lift" in line
    assert domain_coverage(st, torch.arange(n)) == {"wiki": 4, "code": 0}


def test_bug_outputs(tiny_config, device):
    from src.circuits import ov_logits as ov
    from src.skiptrigrams import bug_outputs

    torch.manual_seed(0)
    model = Model(**tiny_config).to(device)
    follows_d = torch.rand(model.vocab_size, device=device) < 0.5
    ids, scores, seen = bug_outputs(model, 1, 7, follows_d, k=6)
    row = ov(model, 1, torch.tensor([7], device=device))[0]
    assert ids.tolist() == row.argsort(descending=True)[:6].tolist() and torch.allclose(scores, row[ids])
    assert torch.equal(seen, follows_d[ids])                              # flagged by whether the token can follow d
    assert ids.tolist() == bug_outputs(model, 1, 7, torch.zeros_like(follows_d), k=6)[0].tolist()   # the boosts ignore d


@pytest.mark.real_ckpt
def test_real_checkpoint_skip_trigrams_hold_up_on_held_out_text(real_model, real_corpus):
    from src.circuits import unembedding_cosine
    from src.counts import excluded_tokens
    from src.mixed import SOURCES

    V = real_corpus.vocab_size
    strs = [real_corpus.token_str(i) for i in range(V)]
    train = {s: torch.from_numpy(real_corpus.train[s][:3_000_000]).cuda().long() for s in SOURCES}
    val = {s: torch.from_numpy(real_corpus.val[s][:1_000_000]).cuda().long() for s in SOURCES}
    unigram = sum(torch.bincount(x, minlength=V) for x in train.values())
    st = find_skip_trigrams(real_model, train, val, excluded_tokens(unigram, strs, real_corpus.eot_id, min_count=200))

    top = select_examples(st, torch.zeros(len(st), dtype=torch.long, device="cuda"), top_k=200, min_hits=5, distinct=True)[0]
    assert st.lift[top[0]] > 20                                          # a strong effect exists on the train sample
    assert (st.val_lift[top[:100]] > 3).float().mean() > 0.3             # and a good share of the best 100 holds on held-out text (measured ~0.5 at this small sample)
    by_pattern = examples_by_pattern(st, strs, lambda a, b: unembedding_cosine(real_model, a, b), top_k=5)
    assert by_pattern[COPY][0] > 0 and by_pattern[SIMILAR][0] > 0       # copying and "similar token" trigrams are both found
