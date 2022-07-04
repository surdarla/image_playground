"""model explain"""
# pylint: disable=C0103, E1101, C0116, C0115
# C0103(snake_name), E1101(no_member), C0115(missing-class-docstring), C0116(missing-docstring)
from typing import Optional, List, Callable, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResNet(nn.Module):
    """ResNet from torchvision"""

    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
