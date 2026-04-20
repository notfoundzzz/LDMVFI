import torch
import torch.nn.functional as F


def _to_gray_uint8(frame: torch.Tensor):
    """@brief 将单帧张量转为 Farneback 所需的灰度 `uint8` 图像.

    @example 输入 `[-1,1]` 范围的 `3xH xW` 张量，输出 `H x W`
    的 `uint8` 灰度图。
    """
    frame_01 = torch.clamp((frame.detach().cpu().float() + 1.0) / 2.0, 0.0, 1.0)
    if frame_01.shape[0] == 3:
        gray = 0.299 * frame_01[0] + 0.587 * frame_01[1] + 0.114 * frame_01[2]
    else:
        gray = frame_01[0]
    return (gray.numpy() * 255.0).round().astype("uint8")


def _estimate_bidirectional_flow_farneback(prev_batch: torch.Tensor, next_batch: torch.Tensor):
    """@brief 使用 OpenCV Farneback 估计双向稠密光流.

    @example 对一批 LR 邻帧估计 `prev->next` 和 `next->prev` 光流，
    后续再上采样到 SR 尺度以生成中间时刻先验。
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
    """@brief 将帧按半步光流 backward warp 到中间时刻.

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
):
    """@brief 构造光流引导的中间时刻先验帧.

    @example 先在较低分辨率邻帧上估计光流，再将其上采样到目标帧
    分辨率，对 SR 邻帧做半步 warp，并平均得到 `flow-guided middle prior`。
    """
    flow_prev_to_next, flow_next_to_prev = _estimate_bidirectional_flow_farneback(prev_flow_frame, next_flow_frame)
    flow_prev_to_next = _resize_flow(flow_prev_to_next.to(prev_target_frame.device), prev_target_frame.shape[-2:])
    flow_next_to_prev = _resize_flow(flow_next_to_prev.to(next_target_frame.device), next_target_frame.shape[-2:])
    warped_prev = _warp_with_half_flow(prev_target_frame, flow_prev_to_next)
    warped_next = _warp_with_half_flow(next_target_frame, flow_next_to_prev)
    prior = 0.5 * (warped_prev + warped_next)
    return torch.clamp(prior, min=-1.0, max=1.0)
