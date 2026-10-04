def test_imports():
    import matplotlib, pytest, torch  # noqa: F401


def test_fixtures(tiny_config, tiny_tokens):
    assert tiny_tokens.shape == (3, 12)
    assert tiny_tokens.max() < tiny_config["vocab_size"]
