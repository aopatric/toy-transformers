from collections import Counter

import pytest
import torch

from src.counts import excluded_tokens, bigram_counts, bigram_probs, count_bigrams, direct_path_logits, direct_path_vs_bigram, topk_overlap
from src.model import OneLayerAttentionOnlyTransformer as Model


def brute_force(ids, V):
    pairs = Counter(zip(ids[:-1].tolist(), ids[1:].tolist()))
    out = torch.zeros(V, V, dtype=torch.long)
    for (a, b), n in pairs.items():
        out[a, b] = n
    return out


@pytest.mark.parametrize("chunk", [3, 7, 50, 10_000])
def test_count_bigrams_matches_brute_force(chunk, device):
    ids = torch.randint(7, (200,), generator=torch.Generator().manual_seed(0))
    counts = count_bigrams(ids.to(device), 7, chunk=chunk)
    assert torch.equal(counts.cpu(), brute_force(ids, 7))
    assert counts.sum() == len(ids) - 1


def test_count_bigrams_short_streams():
    assert count_bigrams(torch.tensor([3]), 5).sum() == 0
    assert count_bigrams(torch.tensor([3, 1]), 5)[3, 1] == 1


def test_pooled_counts_dont_span_streams(device):
    a, b = torch.tensor([0, 1, 2], device=device), torch.tensor([2, 0, 1], device=device)
    pooled = bigram_counts([a, b], 3)
    assert pooled.sum() == 4 and pooled[2, 2] == 0 and pooled[2, 0] == 1


def test_bigram_probs_rows():
    counts = torch.tensor([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 2.0]])
    probs = bigram_probs(counts)
    assert torch.allclose(probs[0], torch.tensor([0.25, 0.75, 0.0]))
    assert probs[1].sum() == 0
    assert torch.allclose(probs[2].sum(), torch.tensor(1.0))


def test_direct_path_logits_match_stored_layout(tiny_config, device):
    model = Model(**tiny_config).to(device)
    tokens = torch.tensor([3, 3, 17, 0], device=device)
    assert torch.allclose(direct_path_logits(model, tokens), model.W_E[tokens] @ model.W_U, atol=1e-6)


def test_topk_overlap_cases():
    probs = torch.tensor([[0.0, 0.5, 0.3, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    perfect = torch.tensor([[0.0, 3.0, 2.0, 1.0, -1.0]] * 2)
    assert topk_overlap(perfect, probs, 3)[0] == 1.0
    wrong = torch.tensor([[3.0, -1.0, -1.0, -1.0, 2.0]] * 2)
    assert topk_overlap(wrong, probs, 2)[0] == 0.0               # predicted {0, 4}, targets {1, 2}
    half = torch.tensor([[0.0, 3.0, -1.0, -1.0, 2.0]] * 2)       # predicted top-3 = {1, 4, 0}; targets {1, 2, 3}
    assert topk_overlap(half, probs, 3)[0] == pytest.approx(1 / 3)
    assert topk_overlap(perfect, probs, 3)[1].isnan()            # token never seen


def test_topk_overlap_fewer_continuations_than_k():
    probs = torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0, 0.0]])       # only two real continuations
    scores = torch.tensor([[0.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    assert topk_overlap(scores, probs, 4)[0] == 1.0


def markov_corpus(V=30, n=60_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(V, V, generator=g) ** 4 + 1e-3
    P = P / P.sum(-1, keepdim=True)
    ids = [0]
    for _ in range(n):
        ids.append(torch.multinomial(P[ids[-1]], 1, generator=g).item())
    return P, torch.tensor(ids)


def model_with_direct_path(logp, V):
    """a model whose direct path W_U W_E is exactly the given (V, V) table, row per token"""
    model = Model(V, V, 1, 1)
    with torch.no_grad():
        model.W_E.copy_(torch.eye(V))
        model.W_U.copy_(logp)
    return model


def test_direct_path_equal_to_true_bigrams_scores_one():
    V = 30
    P, ids = markov_corpus(V)
    counts = bigram_counts([ids], V)
    model = model_with_direct_path(counts.add(1).log(), V)
    tokens, overlap = direct_path_vs_bigram(model, counts, n_tokens=10, k=5)
    assert len(tokens) == 10 and torch.all(overlap == 1.0)


def test_random_direct_path_scores_near_chance():
    V = 30
    P, ids = markov_corpus(V)
    counts = bigram_counts([ids], V)
    torch.manual_seed(1)
    _, overlap = direct_path_vs_bigram(Model(V, 16, 1, 4), counts, n_tokens=30, k=5)
    assert overlap.mean() < 0.5                                   # chance is 5 / 30


def test_direct_path_vs_bigram_picks_frequent_tokens_and_honors_exclude():
    V = 30
    _, ids = markov_corpus(V)
    counts = bigram_counts([ids], V)
    model = model_with_direct_path(counts.add(1).log(), V)
    most = counts.sum(-1).argmax().item()
    tokens, _ = direct_path_vs_bigram(model, counts, n_tokens=5, k=3)
    assert most in tokens.tolist()
    skip = torch.zeros(V, dtype=torch.bool)
    skip[most] = True
    tokens, _ = direct_path_vs_bigram(model, counts, n_tokens=5, k=3, exclude=skip)
    assert most not in tokens.tolist()


def test_one_token_logits_is_the_whole_model_on_one_token(tiny_config, device):
    from src.counts import one_token_logits
    from src.model import logits_by_path

    torch.manual_seed(0)
    model = Model(**tiny_config).to(device)
    tokens = torch.tensor([3, 3, 17, 0], device=device)
    one = one_token_logits(model, tokens)
    assert one.shape == (4, tiny_config["vocab_size"])
    for t, row in zip(tokens, one):
        direct, attention = logits_by_path(model, t[None])
        assert torch.allclose(row, (direct + attention.sum(0))[:, 0], atol=1e-5)
        assert torch.allclose(row, model(t[None])[0], atol=1e-5)


def test_excluded_tokens():
    from src.counts import excluded_tokens

    counts = torch.tensor([5000, 10, 4000, 3000, 9000, 2000])
    strs = ["a", "rare", "�", "b", "<|endoftext|>", "c"]
    mask = excluded_tokens(counts, strs, eot_id=4)
    assert mask.tolist() == [False, True, True, False, True, False]
    assert excluded_tokens(counts, strs, eot_id=4, keep_top=2).tolist() == [False, True, True, False, True, True]
    assert excluded_tokens(counts, strs, eot_id=4, keep_top=1).tolist() == [False, True, True, True, True, True]


@pytest.mark.real_ckpt
def test_real_direct_path_and_one_token_logits_beat_chance(real_model, real_corpus):
    from src.mixed import SOURCES

    V = real_corpus.vocab_size
    counts = bigram_counts([torch.from_numpy(real_corpus.train[s]).cuda() for s in SOURCES], V)
    strs = [real_corpus.token_str(i) for i in range(V)]
    unigram = counts.sum(-1)
    excluded = excluded_tokens(unigram, strs, real_corpus.eot_id)
    tokens, direct = direct_path_vs_bigram(real_model, counts, n_tokens=500, k=10, exclude=excluded)
    chance = 10 / V
    assert direct.mean() > 50 * chance                      # measured ~170x chance
    from src.counts import one_token_logits

    whole = topk_overlap(one_token_logits(real_model, tokens), bigram_probs(counts)[tokens], 10)
    assert whole.mean() > direct.mean()                     # the self-attending heads add bigram signal
