import matplotlib.pyplot as plt
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.display import plot_attention, plot_copying_scores, plot_eigenvalues
from src.mixed import EOT, MixedCorpus


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def tiny_corpus():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=[EOT], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["hello world, héllo wörld " * 20], trainer=trainer)
    return MixedCorpus(tokenizer=tok, train={}, val={})


def test_token_strs_lists_every_token_by_id():
    corpus = tiny_corpus()
    strs = corpus.token_strs()
    assert len(strs) == corpus.vocab_size == len(set(range(len(strs))))
    assert all(strs[i] == corpus.token_str(i) for i in (0, 5, 77, corpus.vocab_size - 1))
    assert strs[corpus.eot_id] == EOT
    assert "�" in strs                                    # a lone byte of a multi-byte character can't be shown


def test_plot_attention_shows_the_pattern():
    pattern = torch.tril(torch.ones(4, 4)).softmax(-1)
    ax = plot_attention(pattern, ["a", "b", "c", "d"], title="head 0").axes[0]
    assert torch.allclose(torch.tensor(ax.images[0].get_array().data), pattern)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c", "d"] and ax.get_title() == "head 0"


def test_plot_eigenvalues_has_one_point_per_eigenvalue():
    eigs = torch.randn(12, dtype=torch.complex64)
    ax = plot_eigenvalues(eigs).axes[0]
    assert len(ax.collections[0].get_offsets()) == 12
    assert torch.allclose(torch.tensor(ax.collections[0].get_offsets().data[:, 0]), eigs.real.double(), atol=1e-6)


def test_plot_copying_scores():
    ax = plot_copying_scores(torch.rand(8), torch.randn(8) * 0.01).axes[0]
    assert len(ax.collections) == 2 and ax.get_legend() is not None       # two series get a legend
    assert plot_copying_scores(torch.rand(8)).axes[0].get_legend() is None  # one series doesn't need one
