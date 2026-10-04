from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn.functional as F

from src.model import OneLayerAttentionOnlyTransformer as Model
from src.model import logits_by_path as src_logits_by_path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "1_layer_attention_only_v2.ipynb"


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_every_code_cell_compiles(nb):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell {i}", "exec")


def test_notebook_starts_with_text_and_has_one_demo_cell(nb):
    assert nb.cells[0].cell_type == "markdown"
    assert sum("demo-path-expansion" in c.metadata.get("tags", []) for c in nb.cells) == 1


def test_the_demo_cell_in_the_notebook_is_the_tested_path_expansion(nb, tiny_config, device):
    cells = [c for c in nb.cells if "demo-path-expansion" in c.metadata.get("tags", [])]
    assert len(cells) == 1
    namespace = {"torch": torch, "F": F}
    exec(cells[0].source, namespace)

    torch.manual_seed(0)
    model = Model(**tiny_config).to(device)
    x = torch.randint(tiny_config["vocab_size"], (12,), device=device)
    direct, attention = namespace["logits_by_path"](model, x)
    ref_direct, ref_attention = src_logits_by_path(model, x)
    assert torch.allclose(direct, ref_direct, atol=1e-5) and torch.allclose(attention, ref_attention, atol=1e-5)
    assert torch.allclose((direct + attention.sum(0)).T, model(x), atol=1e-5)                 # and it is the model
