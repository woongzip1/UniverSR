import importlib
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths in flow matching."""

    @abstractmethod
    def sample_source(self, shape_ref: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        """Sample from the source distribution. shape_ref is used only for shape/device."""

    @abstractmethod
    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate between source x0 and target x1 at time t."""

    @abstractmethod
    def get_target_vector_field(
        self, xt: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the target vector field u_t(xt | x1)."""

class OriginalCFMPath(ConditionalProbabilityPath):
    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def sample_source(self, shape_ref, generator=None):
        return torch.randn(shape_ref.shape, dtype=shape_ref.dtype, device=shape_ref.device, generator=generator)

    def sample_xt(self, x0, x1, t):
        return t * x1 + (1 - t + self.sigma_min * t) * x0

    def get_target_vector_field(self, xt, x0, x1, t):
        return x1 - (1 - self.sigma_min) * x0
    
def get_path(config):
    class_path = config.get("class_path")
    
    if not class_path:
        raise ValueError("Configuration must contain a 'class_path' key")
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"Invalid class_path '{class_path}'. Must contain at least one")
    
    module = importlib.import_module(module_path)
    Class = getattr(module, class_name)
    init_args = config.get("init_args", {})
    return Class(**init_args)
    
    
