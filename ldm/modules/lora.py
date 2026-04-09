import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base, (nn.Conv1d, nn.Conv2d))
        kernel_size = base.kernel_size if isinstance(base.kernel_size, tuple) else (base.kernel_size,)
        assert all(k == 1 for k in kernel_size), "LoRAConv1x1 only supports 1x1 convolutions"

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if isinstance(base, nn.Conv1d):
            self.lora_down = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
            self.lora_up = nn.Conv1d(rank, base.out_channels, kernel_size=1, bias=False)
        else:
            self.lora_down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
            self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


def _should_wrap_module(module_name: str) -> bool:
    exact_suffixes = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "qkv",
        "proj_out",
        "q",
        "k",
        "v",
    )
    return module_name.endswith(exact_suffixes)


def inject_lora_modules(root_module: nn.Module, rank: int = 8, alpha: float = 16.0) -> List[str]:
    replaced = []
    for module_name, _ in list(root_module.named_modules()):
        if not module_name:
            continue
        if not _should_wrap_module(module_name):
            continue

        parent = root_module
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]
        leaf = getattr(parent, leaf_name)

        if isinstance(leaf, nn.Linear):
            setattr(parent, leaf_name, LoRALinear(leaf, rank=rank, alpha=alpha))
            replaced.append(module_name)
        elif isinstance(leaf, (nn.Conv1d, nn.Conv2d)):
            kernel_size = leaf.kernel_size if isinstance(leaf.kernel_size, tuple) else (leaf.kernel_size,)
            if all(k == 1 for k in kernel_size):
                setattr(parent, leaf_name, LoRAConv1x1(leaf, rank=rank, alpha=alpha))
                replaced.append(module_name)
    return replaced


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if "lora_down" in name or "lora_up" in name:
            yield param
