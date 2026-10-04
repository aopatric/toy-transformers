"""Single place that picks the device every module and notebook uses."""

import os
import warnings

import torch


def get_device(cpu_only: bool | None = None) -> torch.device:
    """cuda if available, cpu otherwise; cpu_only=True (or CPU_ONLY=1 in the environment) forces cpu"""
    if cpu_only is None:
        cpu_only = os.environ.get("CPU_ONLY") == "1"
    if cpu_only:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # usually means the process can't see the driver (e.g. a sandbox), not that there's no gpu
    warnings.warn("CUDA not available, falling back to CPU", stacklevel=2)
    return torch.device("cpu")


DEVICE = get_device()
