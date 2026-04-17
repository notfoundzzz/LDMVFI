import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch.nn as nn


class _LoRABranchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class _LoRABranchConv1x1(nn.Module):
    def __init__(self, dim: int, out_dim: int, conv_type):
        super().__init__()
        self.down = conv_type(dim[0], dim[1], kernel_size=1, bias=False)
        self.up = conv_type(dim[1], out_dim, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class DualLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        pixel_rank: int = 0,
        pixel_alpha: float = 16.0,
        semantic_rank: int = 0,
        semantic_alpha: float = 16.0,
    ):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.pixel_rank = pixel_rank
        self.semantic_rank = semantic_rank
        self.pixel_alpha = pixel_alpha
        self.semantic_alpha = semantic_alpha
        self.pixel_scaling = pixel_alpha / pixel_rank if pixel_rank > 0 else 0.0
        self.semantic_scaling = semantic_alpha / semantic_rank if semantic_rank > 0 else 0.0
        self.pixel_scale = 1.0
        self.semantic_scale = 1.0

        if pixel_rank > 0:
            branch = _LoRABranchLinear(base.in_features, base.out_features, pixel_rank)
            self.pixel_lora_down = branch.down
            self.pixel_lora_up = branch.up
        else:
            self.pixel_lora_down = None
            self.pixel_lora_up = None

        if semantic_rank > 0:
            branch = _LoRABranchLinear(base.in_features, base.out_features, semantic_rank)
            self.semantic_lora_down = branch.down
            self.semantic_lora_up = branch.up
        else:
            self.semantic_lora_down = None
            self.semantic_lora_up = None

    def forward(self, x):
        out = self.base(x)
        if self.pixel_lora_down is not None:
            out = out + self.pixel_lora_up(self.pixel_lora_down(x)) * self.pixel_scaling * self.pixel_scale
        if self.semantic_lora_down is not None:
            out = out + self.semantic_lora_up(self.semantic_lora_down(x)) * self.semantic_scaling * self.semantic_scale
        return out


class DualLoRAConv1x1(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        pixel_rank: int = 0,
        pixel_alpha: float = 16.0,
        semantic_rank: int = 0,
        semantic_alpha: float = 16.0,
    ):
        super().__init__()
        assert isinstance(base, (nn.Conv1d, nn.Conv2d))
        kernel_size = base.kernel_size if isinstance(base.kernel_size, tuple) else (base.kernel_size,)
        assert all(k == 1 for k in kernel_size), "DualLoRAConv1x1 only supports 1x1 convolutions"

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        conv_type = nn.Conv1d if isinstance(base, nn.Conv1d) else nn.Conv2d
        self.pixel_scaling = pixel_alpha / pixel_rank if pixel_rank > 0 else 0.0
        self.semantic_scaling = semantic_alpha / semantic_rank if semantic_rank > 0 else 0.0
        self.pixel_scale = 1.0
        self.semantic_scale = 1.0

        if pixel_rank > 0:
            branch = _LoRABranchConv1x1((base.in_channels, pixel_rank), base.out_channels, conv_type)
            self.pixel_lora_down = branch.down
            self.pixel_lora_up = branch.up
        else:
            self.pixel_lora_down = None
            self.pixel_lora_up = None

        if semantic_rank > 0:
            branch = _LoRABranchConv1x1((base.in_channels, semantic_rank), base.out_channels, conv_type)
            self.semantic_lora_down = branch.down
            self.semantic_lora_up = branch.up
        else:
            self.semantic_lora_down = None
            self.semantic_lora_up = None

    def forward(self, x):
        out = self.base(x)
        if self.pixel_lora_down is not None:
            out = out + self.pixel_lora_up(self.pixel_lora_down(x)) * self.pixel_scaling * self.pixel_scale
        if self.semantic_lora_down is not None:
            out = out + self.semantic_lora_up(self.semantic_lora_down(x)) * self.semantic_scaling * self.semantic_scale
        return out


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


def _match_patterns(module_name: str, patterns: Sequence[str]) -> bool:
    return any(pattern and pattern in module_name for pattern in patterns)


def normalize_lora_patterns(patterns: Optional[Union[str, Sequence[str]]], default: Sequence[str]) -> Tuple[str, ...]:
    if patterns is None:
        return tuple(default)
    if isinstance(patterns, str):
        parts = [part.strip() for part in patterns.split(",")]
        return tuple(part for part in parts if part)
    return tuple(part for part in patterns if part)


def inject_dual_lora_modules(
    root_module: nn.Module,
    pixel_rank: int = 8,
    pixel_alpha: float = 16.0,
    semantic_rank: int = 8,
    semantic_alpha: float = 16.0,
    pixel_patterns: Optional[Sequence[str]] = None,
    semantic_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    pixel_patterns = normalize_lora_patterns(pixel_patterns, ("input_blocks", "output_blocks"))
    semantic_patterns = normalize_lora_patterns(semantic_patterns, ("middle_block",))

    replaced = []
    pixel_wrapped = []
    semantic_wrapped = []
    for module_name, _ in list(root_module.named_modules()):
        if not module_name or not _should_wrap_module(module_name):
            continue

        use_pixel = pixel_rank > 0 and _match_patterns(module_name, pixel_patterns)
        use_semantic = semantic_rank > 0 and _match_patterns(module_name, semantic_patterns)
        if not use_pixel and not use_semantic:
            continue

        parent = root_module
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]
        leaf = getattr(parent, leaf_name)

        if isinstance(leaf, nn.Linear):
            wrapped = DualLoRALinear(
                leaf,
                pixel_rank=pixel_rank if use_pixel else 0,
                pixel_alpha=pixel_alpha,
                semantic_rank=semantic_rank if use_semantic else 0,
                semantic_alpha=semantic_alpha,
            )
        elif isinstance(leaf, (nn.Conv1d, nn.Conv2d)):
            kernel_size = leaf.kernel_size if isinstance(leaf.kernel_size, tuple) else (leaf.kernel_size,)
            if not all(k == 1 for k in kernel_size):
                continue
            wrapped = DualLoRAConv1x1(
                leaf,
                pixel_rank=pixel_rank if use_pixel else 0,
                pixel_alpha=pixel_alpha,
                semantic_rank=semantic_rank if use_semantic else 0,
                semantic_alpha=semantic_alpha,
            )
        else:
            continue

        setattr(parent, leaf_name, wrapped)
        replaced.append(module_name)
        if use_pixel:
            pixel_wrapped.append(module_name)
        if use_semantic:
            semantic_wrapped.append(module_name)

    return replaced, pixel_wrapped, semantic_wrapped


def set_dual_lora_scales(module: nn.Module, pixel_scale: float = 1.0, semantic_scale: float = 1.0):
    for child in module.modules():
        if hasattr(child, "pixel_scale"):
            child.pixel_scale = pixel_scale
        if hasattr(child, "semantic_scale"):
            child.semantic_scale = semantic_scale


def pixel_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if "pixel_lora_" in name:
            yield param


def semantic_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if "semantic_lora_" in name:
            yield param
