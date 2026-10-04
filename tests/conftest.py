import os

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

# the checkpoint the real_ckpt tests run on; REAL_CKPT=<file in checkpoints/> picks another one
REAL_CKPT_NAME = os.environ.get("REAL_CKPT", "1l_attn_mixed.pt")


@pytest.fixture
def tiny_config():
    return dict(vocab_size=40, d_model=16, n_heads=4, d_head=8)


@pytest.fixture
def tiny_tokens(tiny_config):
    g = torch.Generator().manual_seed(0)
    return torch.randint(tiny_config["vocab_size"], (3, 12), generator=g)


def pytest_collection_modifyitems(config, items):
    from src.mixed import REPO_ROOT

    ok = torch.cuda.is_available() and (REPO_ROOT / "checkpoints" / REAL_CKPT_NAME).exists()
    skip = pytest.mark.skip(reason="needs cuda and the real checkpoint (run unsandboxed)")
    for item in items:
        if "real_ckpt" in item.keywords and not ok:
            item.add_marker(skip)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs cuda (run unsandboxed)")
    return request.param


@pytest.fixture(scope="session")
def real_model():
    from src.mixed import REPO_ROOT
    from src.model import OneLayerAttentionOnlyTransformer, load_checkpoint

    path = REPO_ROOT / "checkpoints" / REAL_CKPT_NAME
    cfg = torch.load(path, map_location="cuda")["config"]
    model = OneLayerAttentionOnlyTransformer(cfg["vocab_size"], cfg["hidden_dim"], cfg["num_heads"], cfg["head_dim"], device="cuda")
    load_checkpoint(path, model, device="cuda")
    return model.eval()


@pytest.fixture(scope="session")
def real_corpus():
    from src.mixed import load_mixed

    return load_mixed(16384)
