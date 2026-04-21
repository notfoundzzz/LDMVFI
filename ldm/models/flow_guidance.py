import os

import torch
import torch.nn.functional as F

_RAFT_MODEL_CACHE = {}


def _to_gray_uint8(frame: torch.Tensor):
    """@brief 将单帧张量转为 Farneback 所需的灰度 `uint8` 图像。

    @example 输入 `[-1,1]` 范围的 `3xH xW` 张量，输出 `H x W`
    的 `uint8` 灰度图。
    """
    frame_01 = torch.clamp((frame.detach().cpu().float() + 1.0) / 2.0, 0.0, 1.0)
    if frame_01.shape[0] == 3:
        gray = 0.299 * frame_01[0] + 0.587 * frame_01[1] + 0.114 * frame_01[2]
    else:
        gray = frame_01[0]
    return (gray.numpy() * 255.0).round().astype("uint8")


def _normalize_raft_frames(frame_batch: torch.Tensor):
    """@brief 将输入帧转换为 RAFT 更常见的 `[-1,1]` 浮点输入格式。"""
    if frame_batch.min().item() < -0.5:
        return frame_batch.clamp(-1.0, 1.0)
    return (frame_batch * 2.0 - 1.0).clamp(-1.0, 1.0)


def _pad_to_multiple_of_8(frame_batch: torch.Tensor):
    """@brief 将输入 padding 到 8 的倍数，满足 RAFT 下采样约束。"""
    _, _, h, w = frame_batch.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return frame_batch, (0, 0)
    padded = F.pad(frame_batch, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


def _crop_flow(flow: torch.Tensor, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        flow = flow[:, :, :-pad_h, :]
    if pad_w > 0:
        flow = flow[:, :, :, :-pad_w]
    return flow


def _build_raft_model(variant: str, ckpt_path: str, device: torch.device):
    cache_key = (variant, os.path.abspath(ckpt_path), str(device))
    if cache_key in _RAFT_MODEL_CACHE:
        return _RAFT_MODEL_CACHE[cache_key]

    try:
        from torchvision.models.optical_flow import raft_large, raft_small
    except ImportError as exc:
        raise ImportError("RAFT backend requires torchvision.models.optical_flow") from exc

    if not ckpt_path:
        raise ValueError("RAFT backend requires a local flow_raft_ckpt path")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"RAFT checkpoint not found: {ckpt_path}")

    def _create_raft(constructor):
        """@brief 兼容新旧 torchvision 接口创建 RAFT 模型。

        @example 新版通常支持 `weights=None`，旧版可能只接受
        `pretrained=False` 或完全无参构造，因此这里按顺序回退。
        """
        build_attempts = (
            lambda: constructor(weights=None, progress=False),
            lambda: constructor(pretrained=False, progress=False),
            lambda: constructor(pretrained=False),
            lambda: constructor(),
        )
        last_error = None
        for build in build_attempts:
            try:
                return build()
            except TypeError as exc:
                last_error = exc
        raise last_error

    if variant == "small":
        model = _create_raft(raft_small)
    elif variant == "large":
        model = _create_raft(raft_large)
    else:
        raise ValueError(f"Unsupported RAFT variant: {variant}")

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected RAFT checkpoint keys: {unexpected[:10]}")
    if missing:
        raise ValueError(f"Missing RAFT checkpoint keys: {missing[:10]}")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    _RAFT_MODEL_CACHE[cache_key] = model
    return model


def _estimate_bidirectional_flow_farneback(prev_batch: torch.Tensor, next_batch: torch.Tensor):
    """@brief 使用 OpenCV Farneback 估计双向稠密光流。

    @example 对一对 LR 邻帧估计 `prev->next` 和 `next->prev` 光流，
    后续再上采样到目标尺度以生成中间时刻先验。
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Flow guidance requires cv2 to estimate Farneback optical flow") from exc

    flow_prev_to_next = []
    flow_next_to_prev = []
    for prev, nxt in zip(prev_batch, next_batch):
        prev_gray = _to_gray_uint8(prev)
        next_gray = _to_gray_uint8(nxt)
        forward = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_prev_to_next.append(torch.from_numpy(forward).permute(2, 0, 1).float())
        flow_next_to_prev.append(torch.from_numpy(backward).permute(2, 0, 1).float())
    return torch.stack(flow_prev_to_next, dim=0), torch.stack(flow_next_to_prev, dim=0)


@torch.no_grad()
def _estimate_bidirectional_flow_raft(
    prev_batch: torch.Tensor,
    next_batch: torch.Tensor,
    variant: str,
    ckpt_path: str,
):
    """@brief 使用本地 RAFT 权重估计双向稠密光流。

    @example 指定 `variant=large` 且传入本地 `raft-large.pth` 后，
    可得到比 Farneback 更强的 `prev->next` / `next->prev` 光流先验。
    """
    device = prev_batch.device
    model = _build_raft_model(variant=variant, ckpt_path=ckpt_path, device=device)

    prev_batch = _normalize_raft_frames(prev_batch)
    next_batch = _normalize_raft_frames(next_batch)
    prev_batch, pad_hw = _pad_to_multiple_of_8(prev_batch)
    next_batch, _ = _pad_to_multiple_of_8(next_batch)

    flow_prev_to_next = model(prev_batch, next_batch)[-1]
    flow_next_to_prev = model(next_batch, prev_batch)[-1]
    flow_prev_to_next = _crop_flow(flow_prev_to_next, pad_hw)
    flow_next_to_prev = _crop_flow(flow_next_to_prev, pad_hw)
    return flow_prev_to_next.float(), flow_next_to_prev.float()


def _estimate_bidirectional_flow(
    prev_batch: torch.Tensor,
    next_batch: torch.Tensor,
    backend: str,
    raft_variant: str = "large",
    raft_ckpt: str = None,
):
    """@brief 根据配置选择 Farneback 或 RAFT 作为光流后端。"""
    if backend == "farneback":
        return _estimate_bidirectional_flow_farneback(prev_batch, next_batch)
    if backend == "raft":
        return _estimate_bidirectional_flow_raft(
            prev_batch.to(dtype=torch.float32),
            next_batch.to(dtype=torch.float32),
            variant=raft_variant,
            ckpt_path=raft_ckpt,
        )
    raise ValueError(f"Unsupported flow backend: {backend}")


def _resize_flow(flow: torch.Tensor, target_hw):
    target_h, target_w = target_hw
    src_h, src_w = flow.shape[-2:]
    if (src_h, src_w) == (target_h, target_w):
        return flow
    scale_h = float(target_h) / float(src_h)
    scale_w = float(target_w) / float(src_w)
    flow = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
    flow[:, 0] *= scale_w
    flow[:, 1] *= scale_h
    return flow


def _warp_with_half_flow(frame: torch.Tensor, flow: torch.Tensor):
    """@brief 将帧按半步光流 backward warp 到中间时刻。

    @example 对 `prev->next` 光流取 `0.5` 倍，可近似生成 `t=0.5`
    时刻对应的对齐结果。
    """
    b, _, h, w = frame.shape
    device = frame.device
    dtype = frame.dtype
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    base_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    sample_x = base_x - 0.5 * flow[:, 0]
    sample_y = base_y - 0.5 * flow[:, 1]
    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


def build_flow_guided_middle_prior(
    prev_flow_frame: torch.Tensor,
    next_flow_frame: torch.Tensor,
    prev_target_frame: torch.Tensor,
    next_target_frame: torch.Tensor,
    backend: str = "farneback",
    raft_variant: str = "large",
    raft_ckpt: str = None,
):
    """@brief 构造光流引导的中间时刻先验帧。

    @example 当 `backend=farneback` 时，先在较低分辨率邻帧上估计光流，
    再上采样到目标尺度；当 `backend=raft` 时，直接在目标尺度邻帧上估计
    光流，以避免低分辨率输入导致的数值不稳定，再对目标帧做半步 warp。
    """
    if backend == "raft":
        flow_prev_to_next, flow_next_to_prev = _estimate_bidirectional_flow(
            prev_target_frame,
            next_target_frame,
            backend=backend,
            raft_variant=raft_variant,
            raft_ckpt=raft_ckpt,
        )
    else:
        flow_prev_to_next, flow_next_to_prev = _estimate_bidirectional_flow(
            prev_flow_frame,
            next_flow_frame,
            backend=backend,
            raft_variant=raft_variant,
            raft_ckpt=raft_ckpt,
        )
    flow_prev_to_next = _resize_flow(flow_prev_to_next.to(prev_target_frame.device), prev_target_frame.shape[-2:])
    flow_next_to_prev = _resize_flow(flow_next_to_prev.to(next_target_frame.device), next_target_frame.shape[-2:])
    warped_prev = _warp_with_half_flow(prev_target_frame, flow_prev_to_next)
    warped_next = _warp_with_half_flow(next_target_frame, flow_next_to_prev)
    prior = 0.5 * (warped_prev + warped_next)
    return torch.clamp(prior, min=-1.0, max=1.0)
