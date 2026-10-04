import torch

from src.device import get_device


def test_cpu_only_arg_forces_cpu():
    assert get_device(cpu_only=True).type == "cpu"


def test_cpu_only_env_forces_cpu(monkeypatch):
    monkeypatch.setenv("CPU_ONLY", "1")
    assert get_device().type == "cpu"


def test_default_follows_cuda(monkeypatch):
    monkeypatch.delenv("CPU_ONLY", raising=False)
    assert get_device().type == ("cuda" if torch.cuda.is_available() else "cpu")
