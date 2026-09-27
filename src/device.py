"""Single place that picks the device every module and notebook uses."""

import warnings

import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    # Usually means the process can't see the driver (e.g. a sandbox), not that there's no GPU.
    warnings.warn("CUDA not available, falling back to CPU", stacklevel=2)
    DEVICE = torch.device("cpu")
