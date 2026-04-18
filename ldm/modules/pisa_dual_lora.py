import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch.nn as nn


def _normalize_items(items: Optional[Union[str, Sequence[str]]], default: Sequence[str]) -> Tuple[str, ...]:
    if items is None:
        return tuple(default)
    if isinstance(items, str):
        parts = [part.strip() for part in items.split(",")]
        return tuple(part for part in parts if part)
    return tuple(part for part in items if part)


class PiSADualLoRALinear(nn.Module):
    """@brief PiSA-style dual-branch LoRA linear layer.

    @example With `pixel_scale=1.0` and `semantic_scale=0.5`,
    the output becomes `base + pixel_delta + 0.5 * semantic_delta`.
    """

    def __init__(
        self,
        base: nn.Linear,
        pixel_rank: int,
        pixel_alpha: float,
        semantic_rank: int,
        semantic_alpha: float,
    ):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.pixel_scaling = pixel_alpha / pixel_rank if pixel_rank > 0 else 0.0
        self.semantic_scaling = semantic_alpha / semantic_rank if semantic_rank > 0 else 0.0
        self.pixel_scale = 1.0
        self.semantic_scale = 1.0

        if pixel_rank > 0:
            self.pixel_lora_down = nn.Linear(base.in_features, pixel_rank, bias=False)
            self.pixel_lora_up = nn.Linear(pixel_rank, base.out_features, bias=False)
            self._reset_branch(self.pixel_lora_down, self.pixel_lora_up)
        else:
            self.pixel_lora_down = None
            self.pixel_lora_up = None

        if semantic_rank > 0:
            self.semantic_lora_down = nn.Linear(base.in_features, semantic_rank, bias=False)
            self.semantic_lora_up = nn.Linear(semantic_rank, base.out_features, bias=False)
            self._reset_branch(self.semantic_lora_down, self.semantic_lora_up)
        else:
            self.semantic_lora_down = None
            self.semantic_lora_up = None

    @staticmethod
    def _reset_branch(down: nn.Module, up: nn.Module):
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.pixel_lora_down is not None:
            out = out + self.pixel_lora_up(self.pixel_lora_down(x)) * self.pixel_scaling * self.pixel_scale
        if self.semantic_lora_down is not None:
            out = out + self.semantic_lora_up(self.semantic_lora_down(x)) * self.semantic_scaling * self.semantic_scale
        return out


class PiSADualLoRAConv1x1(nn.Module):
    """@brief PiSA-style dual-branch LoRA 1x1 convolution layer.

    @example This wrapper is used for 1x1 layers such as
    `proj_in`, `proj_out`, or `skip_connection`.
    """

    def __init__(
        self,
        base: nn.Module,
        pixel_rank: int,
        pixel_alpha: float,
        semantic_rank: int,
        semantic_alpha: float,
    ):
        super().__init__()
        assert isinstance(base, (nn.Conv1d, nn.Conv2d))
        kernel_size = base.kernel_size if isinstance(base.kernel_size, tuple) else (base.kernel_size,)
        assert all(k == 1 for k in kernel_size), "PiSADualLoRAConv1x1 only supports 1x1 convolutions"

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
            self.pixel_lora_down = conv_type(base.in_channels, pixel_rank, kernel_size=1, bias=False)
            self.pixel_lora_up = conv_type(pixel_rank, base.out_channels, kernel_size=1, bias=False)
            self._reset_branch(self.pixel_lora_down, self.pixel_lora_up)
        else:
            self.pixel_lora_down = None
            self.pixel_lora_up = None

        if semantic_rank > 0:
            self.semantic_lora_down = conv_type(base.in_channels, semantic_rank, kernel_size=1, bias=False)
            self.semantic_lora_up = conv_type(semantic_rank, base.out_channels, kernel_size=1, bias=False)
            self._reset_branch(self.semantic_lora_down, self.semantic_lora_up)
        else:
            self.semantic_lora_down = None
            self.semantic_lora_up = None

    @staticmethod
    def _reset_branch(down: nn.Module, up: nn.Module):
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.pixel_lora_down is not None:
            out = out + self.pixel_lora_up(self.pixel_lora_down(x)) * self.pixel_scaling * self.pixel_scale
        if self.semantic_lora_down is not None:
            out = out + self.semantic_lora_up(self.semantic_lora_down(x)) * self.semantic_scaling * self.semantic_scale
        return out


def _module_group(module_name: str) -> str:
    if "input_blocks" in module_name or "down_blocks" in module_name or module_name.startswith("input_blocks.0"):
        return "encoder"
    if "output_blocks" in module_name or "up_blocks" in module_name:
        return "decoder"
    return "others"


def _is_supported_target(module_name: str, target_suffixes: Sequence[str]) -> bool:
    return any(module_name.endswith(suffix) for suffix in target_suffixes)


def _wrap_module(
    leaf: nn.Module,
    pixel_rank: int,
    pixel_alpha: float,
    semantic_rank: int,
    semantic_alpha: float,
):
    if isinstance(leaf, nn.Linear):
        return PiSADualLoRALinear(leaf, pixel_rank, pixel_alpha, semantic_rank, semantic_alpha)
    if isinstance(leaf, (nn.Conv1d, nn.Conv2d)):
        kernel_size = leaf.kernel_size if isinstance(leaf.kernel_size, tuple) else (leaf.kernel_size,)
        if all(k == 1 for k in kernel_size):
            return PiSADualLoRAConv1x1(leaf, pixel_rank, pixel_alpha, semantic_rank, semantic_alpha)
    return None


def inject_pisa_dual_lora_modules(
    root_module: nn.Module,
    pixel_rank: int = 8,
    pixel_alpha: float = 16.0,
    semantic_rank: int = 8,
    semantic_alpha: float = 16.0,
    pixel_groups: Optional[Union[str, Sequence[str]]] = None,
    semantic_groups: Optional[Union[str, Sequence[str]]] = None,
    target_suffixes: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """@brief Inject PiSA-style dual LoRA modules into the LDMVFI UNet.

    @example By default this covers encoder, decoder, and others groups,
    and targets real layer names such as `to_q`, `proj_in`, and `in_layers.2`.
    """

    pixel_groups = _normalize_items(pixel_groups, ("encoder", "decoder", "others"))
    semantic_groups = _normalize_items(semantic_groups, ("encoder", "decoder", "others"))
    target_suffixes = _normalize_items(
        target_suffixes,
        (
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "qkv",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "in_layers.2",
            "out_layers.3",
            "skip_connection",
        ),
    )

    replaced: List[str] = []
    pixel_targets: List[str] = []
    semantic_targets: List[str] = []
    for module_name, _ in list(root_module.named_modules()):
        if not module_name or not _is_supported_target(module_name, target_suffixes):
            continue

        group = _module_group(module_name)
        use_pixel = pixel_rank > 0 and group in pixel_groups
        use_semantic = semantic_rank > 0 and group in semantic_groups
        if not use_pixel and not use_semantic:
            continue

        parent = root_module
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]
        leaf = getattr(parent, leaf_name)
        wrapped = _wrap_module(
            leaf,
            pixel_rank if use_pixel else 0,
            pixel_alpha,
            semantic_rank if use_semantic else 0,
            semantic_alpha,
        )
        if wrapped is None:
            continue

        setattr(parent, leaf_name, wrapped)
        replaced.append(module_name)
        if use_pixel:
            pixel_targets.append(module_name)
        if use_semantic:
            semantic_targets.append(module_name)

    return replaced, pixel_targets, semantic_targets


def set_pisa_dual_lora_scales(module: nn.Module, pixel_scale: float = 1.0, semantic_scale: float = 1.0):
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
