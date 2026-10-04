"""Skip-trigrams [source] ... [destination] -> [output]: the circuits propose candidates, the corpus counts score them.

The full space is heads x V^3 triples, far too many to count, so a head's QK circuit proposes the sources a destination
attends to and its OV circuit proposes the outputs, and one pass over the corpus scores only those candidates."""

from dataclasses import dataclass

import torch

from src.circuits import ov_logits, qk_scores, top_outputs
from src.counts import count_bigrams


@torch.no_grad()
def next_occurrence(ids):
    """nxt[u] = the next position holding the same token as position u (len(ids) if there is none)"""
    order = torch.argsort(ids, stable=True)               # positions grouped by token, ascending within each token
    nxt = torch.full_like(ids, len(ids))
    same = ids[order[1:]] == ids[order[:-1]]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt


@torch.no_grad()
def count_skip_trigrams(ids, cand_keys, vocab_size, lookback, pair_n=None):
    """One pass over a token stream. For every position t (with a token after it), each distinct source token s
    among the previous lookback-1 tokens counts once. Returns
      pair_n[s * V + d]: positions t with ids[t] = d and s in the window before t (dense, V * V; added into the
                         `pair_n` passed in, if any)
      triple_k[i]: how many of those also have ids[t + 1] = o, for each sorted candidate key (s * V + d) * V + o"""
    V = vocab_size
    nxt = next_occurrence(ids)
    if pair_n is None:
        pair_n = torch.zeros(V * V, dtype=torch.long, device=ids.device)
    triple_k = torch.zeros(len(cand_keys), dtype=torch.long, device=ids.device)
    for j in range(1, min(lookback, len(ids) - 1)):
        u = torch.arange(len(ids) - 1 - j, device=ids.device)   # t = u + j, and t + 1 must exist
        t = u + j
        keep = nxt[u] >= t                                       # u is the latest copy of its token before t
        u, t = u[keep], t[keep]
        pair = ids[u] * V + ids[t]
        pair_n.index_add_(0, pair, torch.ones_like(pair))       # in place: a bincount would build a (V * V) table per offset
        key = pair * V + ids[t + 1]
        idx = torch.searchsorted(cand_keys, key).clamp(max=len(cand_keys) - 1)
        hit = cand_keys[idx] == key
        triple_k += torch.bincount(idx[hit], minlength=len(cand_keys))
    return pair_n, triple_k


@torch.no_grad()
def co_occurrence(streams, vocab_size, lookback):
    """two (V, V) bool tables over the counted streams: seen[s, d] = s occurs within the lookback before d at least
    once, and follows[d, o] = o directly follows d at least once. candidates are drawn only from pairs that occur"""
    V = vocab_size
    pair_n = torch.zeros(V * V, dtype=torch.long, device=next(iter(streams.values())).device)
    follows = torch.zeros(V, V, dtype=torch.bool, device=pair_n.device)
    dummy = torch.zeros(1, dtype=torch.long, device=pair_n.device)
    for ids in streams.values():
        count_skip_trigrams(ids, dummy, V, lookback, pair_n=pair_n)
        follows |= count_bigrams(ids, V) > 0
    return pair_n.view(V, V) > 0, follows


@torch.no_grad()
def propose_candidates(model, dests, seen, follows, excluded, n_sources=5, n_outputs=5, chunk=512):
    """The circuits propose skip-trigrams. For every head and destination d: the n_sources tokens its QK circuit most
    wants to attend to (among those seen before d), then for each such source s the n_outputs its OV circuit most
    boosts (among tokens seen after d). Excluded tokens are never used. returns (N, 4) rows (head, s, d, o)"""
    # the OV row depends only on s, so its top outputs are often tokens that never follow d: those are the paper's
    # "skip-trigram bugs", and restricting to tokens seen after d leaves them out
    neg_inf = float("-inf")
    rows = []
    for h in range(model.n_heads):
        for ds in dests.split(chunk):
            qk = qk_scores(model, h, ds).masked_fill(excluded | ~seen[:, ds].T, neg_inf)
            s_score, srcs = qk.topk(n_sources, dim=-1)                     # (n_d, n_sources)
            ok = s_score.isfinite()                                        # a d may have fewer valid sources
            d_flat, s_flat = ds[:, None].expand_as(srcs)[ok], srcs[ok]
            ov = ov_logits(model, h, s_flat).masked_fill(excluded | ~follows[d_flat], neg_inf)
            o_score, outs = ov.topk(n_outputs, dim=-1)
            ok = o_score.isfinite()
            rows.append(torch.stack([torch.full_like(outs, h), s_flat[:, None].expand_as(outs),
                                     d_flat[:, None].expand_as(outs), outs], dim=-1)[ok])
    return torch.cat(rows)


@torch.no_grad()
def score_candidates(streams, cands, vocab_size, lookback):
    """Count every candidate in the corpus, one stream at a time. Per stream, for each candidate row:
      k        windows where s was before d and o came next
      n        windows where d occurred with s before it
      bi       how often o directly follows d in this stream
      d_total  how often d is followed by anything, i.e. the bigram baseline is bi / d_total"""
    V = vocab_size
    s, d, o = cands[:, 1], cands[:, 2], cands[:, 3]
    cand_keys, inverse = torch.unique((s * V + d) * V + o, return_inverse=True)   # sorted, and shared between heads
    stats = {}
    for name, ids in streams.items():
        pair_n, triple_k = count_skip_trigrams(ids, cand_keys, V, lookback)
        bigrams = count_bigrams(ids, V)
        stats[name] = dict(k=triple_k[inverse], n=pair_n[s * V + d], bi=bigrams[d, o], d_total=bigrams.sum(dim=1)[d])
        del pair_n, bigrams
    return stats


def smoothed_lift(k, expected):
    """observed hits over the hits expected if s didn't matter, each with one added so a one-off coincidence can't win"""
    return (k + 1) / (expected + 1)


def lift_table(train_stats, other_stats, sources):
    """Observed vs expected hits for every candidate, pooled over `sources`. The expected hits use each source's own
    bigram rate P(o | d), taken from train: expected = sum over sources of n * bi / d_total. A trigram that only
    lives in one domain is therefore compared against that domain's baseline, not a mix of all three.
    returns dicts with k, n, expected (floats) for `other_stats` (e.g. val), using the train rates"""
    rate = {s: train_stats[s]["bi"] / train_stats[s]["d_total"].clamp(min=1) for s in sources}
    return dict(k=sum(other_stats[s]["k"] for s in sources), n=sum(other_stats[s]["n"] for s in sources),
                expected=sum(other_stats[s]["n"] * rate[s] for s in sources))


@dataclass
class SkipTrigrams:
    """Candidate skip-trigrams with their corpus statistics; row i is head h[i]: [s[i]] ... [d[i]] -> [o[i]]"""
    sources: tuple                  # names of the corpus sources, in the order of hits_by_source's columns
    h: torch.Tensor
    s: torch.Tensor
    d: torch.Tensor
    o: torch.Tensor
    train: dict                     # k, n, expected per row, on the train sample
    val: dict                       # the same on held-out text, with the train baseline
    hits_by_source: torch.Tensor    # (N, n_sources) train hits from each source

    @property
    def lift(self):
        return smoothed_lift(self.train["k"], self.train["expected"])

    @property
    def val_lift(self):
        return smoothed_lift(self.val["k"], self.val["expected"])

    def __len__(self):
        return len(self.h)


def find_skip_trigrams(model, train_streams, val_streams, excluded, *, n_sources=5, n_outputs=5, lookback=256,
                       domain=None):
    """The whole pipeline: candidates from the circuits, scored on the train streams (and on val for a sanity check).
    excluded is a bool mask over the vocab of tokens never to use. domain, if a source name, pools the lift over that
    source only"""
    V = model.vocab_size
    sources = tuple(train_streams)
    pool = sources if domain is None else (domain,)
    seen, follows = co_occurrence(train_streams, V, lookback)
    dests = (~excluded).nonzero().squeeze(1)
    cands = propose_candidates(model, dests, seen, follows, excluded, n_sources, n_outputs)
    del seen, follows
    train = score_candidates(train_streams, cands, V, lookback)
    val = score_candidates(val_streams, cands, V, lookback)
    h, s, d, o = cands.unbind(-1)
    return SkipTrigrams(sources, h, s, d, o, train=lift_table(train, train, pool), val=lift_table(train, val, pool),
                        hits_by_source=torch.stack([train[x]["k"] for x in sources], dim=-1).float())


# the primitive in-context learning patterns, written [source] ... [destination] -> [output]: b is a token, a another,
# b' something similar to b, and [ab] a token spelled as a followed by b
SPLIT, SPLIT_SIMILAR = "[ab]…[a]→[b]", "[ab]…[a]→[b']"
COPY, SIMILAR, OTHER = "[b]…[a]→[b]", "[b]…[a]→[b']", "other"
PATTERNS = (SPLIT, SPLIT_SIMILAR, COPY, SIMILAR, OTHER)       # in the order they are checked


def _variant(a, b):
    """the same word up to case, or one extends the other (ike / ikes)"""
    a, b = a.lower(), b.lower()
    return a == b or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)))


def classify_pattern(s, d, o, cosine=0.0, min_cosine=0.6):
    """Name the in-context pattern of one skip-trigram from the decoded strings of its source s, destination d and output
    o (a leading space doesn't matter). `cosine` is the unembedding similarity of s and o, which decides what counts as
    "similar" for tokens that aren't spelling variants of each other (' two' -> ' three')"""
    sb, db, ob = s.strip(), d.strip(), o.strip()
    if sb and db and len(db) < len(sb) and sb.startswith(db):          # s is spelled d + rest
        rest = sb[len(db):]
        if o == rest:
            return SPLIT
        if ob and _variant(ob, rest):
            return SPLIT_SIMILAR
    if o == s:
        return COPY
    if ob and sb and (_variant(ob, sb) or cosine >= min_cosine):
        return SIMILAR
    return OTHER


def classify_rows(st, idx, token_strs, cosine, min_cosine=0.6):
    """pattern names for rows idx of a SkipTrigrams; cosine is the unembedding similarity of s and o for those rows"""
    s, d, o = st.s[idx].tolist(), st.d[idx].tolist(), st.o[idx].tolist()
    return [classify_pattern(token_strs[a], token_strs[b], token_strs[c], cos, min_cosine)
            for a, b, c, cos in zip(s, d, o, cosine.tolist())]


def dominant_source(st):
    """index into st.sources of the corpus source that supplied most of each row's hits"""
    return st.hits_by_source.argmax(dim=-1)


def distinct_rows(st, idx):
    """of the rows idx (in the order given), keep the first one of each [s] ... [d] -> [o]: several heads can propose the same trigram"""
    V = int(max(st.s.max(), st.d.max(), st.o.max())) + 1
    _, inverse = ((st.s[idx] * V + st.d[idx]) * V + st.o[idx]).unique(return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), len(idx), dtype=torch.long, device=idx.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(idx), device=idx.device), "amin")
    return idx[first.sort().values]


def select_examples(st, group, top_k=10, min_hits=5, min_val_lift=None, distinct=False):
    """the top_k rows by smoothed lift within each group. group is an int tensor (one id per row, e.g. the head or the
    dominant source; -1 skips a row); rows need at least min_hits train hits, and a val lift of at least min_val_lift
    if given. distinct=True lists a trigram once even if several heads found it. returns {group id: row indices}"""
    keep = (st.train["k"] >= min_hits) & (group >= 0)
    if min_val_lift is not None:
        keep &= st.val_lift >= min_val_lift
    idx = keep.nonzero().squeeze(1)
    idx = idx[st.lift[idx].argsort(descending=True)]
    if distinct:
        idx = distinct_rows(st, idx)
    idx = idx[group[idx].argsort(stable=True)]                         # by group, lift order kept within each
    g = group[idx]
    rank = torch.arange(len(g), device=g.device) - torch.searchsorted(g, g)   # position within its group's run
    idx, g = idx[rank < top_k], g[rank < top_k]
    return {int(x): idx[g == x] for x in g.unique()}


def examples_by_pattern(st, token_strs, cosine_of, top_k=10, min_hits=5, min_cosine=0.6):
    """{pattern: (number of distinct trigrams with that pattern, top_k rows by lift)} over the rows with enough hits.
    cosine_of(a, b) gives the unembedding similarity of token id tensors a and b"""
    idx = distinct_rows(st, (st.train["k"] >= min_hits).nonzero().squeeze(1))
    labels = classify_rows(st, idx, token_strs, cosine_of(st.s[idx], st.o[idx]), min_cosine)
    out = {}
    for name in PATTERNS:
        rows = idx[torch.tensor([lab == name for lab in labels], dtype=torch.bool, device=idx.device)]
        out[name] = (len(rows), rows[st.lift[rows].argsort(descending=True)[:top_k]])
    return out


def bug_outputs(model, h, s, follows_d, k=10, excluded=None):
    """The skip-trigram "bugs". Head h's OV circuit boosts the same outputs for source s whichever destination attends
    to it, so some boosted outputs make no sense after a given destination d. returns the top k outputs for s (ids,
    scores) and a bool per output: does it ever follow d? follows_d is a bool (V,) row, e.g. bigram counts[d] > 0"""
    ids, scores = top_outputs(model, h, s, k, mask=excluded)
    return ids, scores, follows_d[ids]
