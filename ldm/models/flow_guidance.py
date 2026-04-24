import os

import torch
import torch.nn.functional as F

_RAFT_MODEL_CACHE = {}


def _to_gray_uint8(frame: torch.Tensor):
    """@brief 灏嗗崟甯у紶閲忚浆涓?Farneback 鎵€闇€鐨勭伆搴?`uint8` 鍥惧儚銆?
    @example 杈撳叆 `[-1,1]` 鑼冨洿鐨?`3xH xW` 寮犻噺锛岃緭鍑?`H x W`
    鐨?`uint8` 鐏板害鍥俱€?    """
    frame_01 = torch.clamp((frame.detach().cpu().float() + 1.0) / 2.0, 0.0, 1.0)
    if frame_01.shape[0] == 3:
        gray = 0.299 * frame_01[0] + 0.587 * frame_01[1] + 0.114 * frame_01[2]
    else:
        gray = frame_01[0]
    return (gray.numpy() * 255.0).round().astype("uint8")


def _normalize_raft_frames(frame_batch: torch.Tensor):
    """@brief 灏嗚緭鍏ュ抚杞崲涓?RAFT 鏇村父瑙佺殑 `[-1,1]` 娴偣杈撳叆鏍煎紡銆?""
    if frame_batch.min().item() < -0.5:
        return frame_batch.clamp(-1.0, 1.0)
    return (frame_batch * 2.0 - 1.0).clamp(-1.0, 1.0)


def _pad_to_multiple_of_8(frame_batch: torch.Tensor):
    """@brief 灏嗚緭鍏?padding 鍒?8 鐨勫€嶆暟锛屾弧瓒?RAFT 涓嬮噰鏍风害鏉熴€?""
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
        """@brief 鍏煎鏂版棫 torchvision 鎺ュ彛鍒涘缓 RAFT 妯″瀷銆?
        @example 鏂扮増閫氬父鏀寔 `weights=None`锛屾棫鐗堝彲鑳藉彧鎺ュ彈
        `pretrained=False` 鎴栧畬鍏ㄦ棤鍙傛瀯閫狅紝鍥犳杩欓噷鎸夐『搴忓洖閫€銆?        """
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
    """@brief 浣跨敤 OpenCV Farneback 浼拌鍙屽悜绋犲瘑鍏夋祦銆?
    @example 瀵逛竴瀵?LR 閭诲抚浼拌 `prev->next` 鍜?`next->prev` 鍏夋祦锛?    鍚庣画鍐嶄笂閲囨牱鍒扮洰鏍囧昂搴︿互鐢熸垚涓棿鏃跺埢鍏堥獙銆?    """
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
    """@brief 浣跨敤鏈湴 RAFT 鏉冮噸浼拌鍙屽悜绋犲瘑鍏夋祦銆?
    @example 鎸囧畾 `variant=large` 涓斾紶鍏ユ湰鍦?`raft-large.pth` 鍚庯紝
    鍙緱鍒版瘮 Farneback 鏇村己鐨?`prev->next` / `next->prev` 鍏夋祦鍏堥獙銆?    """
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
    """@brief 鏍规嵁閰嶇疆閫夋嫨 Farneback 鎴?RAFT 浣滀负鍏夋祦鍚庣銆?""
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
    """@brief 灏嗗抚鎸夊崐姝ュ厜娴?backward warp 鍒颁腑闂存椂鍒汇€?
    @example 瀵?`prev->next` 鍏夋祦鍙?`0.5` 鍊嶏紝鍙繎浼肩敓鎴?`t=0.5`
    鏃跺埢瀵瑰簲鐨勫榻愮粨鏋溿€?    """
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


def build_flow_aligned_input_pair(
    prev_flow_frame: torch.Tensor,
    next_flow_frame: torch.Tensor,
    prev_target_frame: torch.Tensor,
    next_target_frame: torch.Tensor,
    backend: str = "farneback",
    raft_variant: str = "large",
    raft_ckpt: str = None,
):
    """@brief 鏋勯€犲榻愬埌涓棿鏃跺埢鐨勫墠鍚庢潯浠跺抚銆?    @example
    浣跨敤鍙屽悜鍏夋祦灏?`prev_target_frame / next_target_frame` 鍒嗗埆 warp 鍒?    `t=0.5`锛屽啀鐩存帴浣滀负鐢熸垚妯″瀷鐨勪富鏉′欢杈撳叆銆?    """
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
    return torch.clamp(warped_prev, min=-1.0, max=1.0), torch.clamp(warped_next, min=-1.0, max=1.0)


def build_flow_guided_middle_prior(
    prev_flow_frame: torch.Tensor,
    next_flow_frame: torch.Tensor,
    prev_target_frame: torch.Tensor,
    next_target_frame: torch.Tensor,
    backend: str = "farneback",
    raft_variant: str = "large",
    raft_ckpt: str = None,
):
    """@brief 鏋勯€犲厜娴佸紩瀵肩殑涓棿鏃跺埢鍏堥獙甯с€?
    @example 褰?`backend=farneback` 鏃讹紝鍏堝湪杈冧綆鍒嗚鲸鐜囬偦甯т笂浼拌鍏夋祦锛?    鍐嶄笂閲囨牱鍒扮洰鏍囧昂搴︼紱褰?`backend=raft` 鏃讹紝鐩存帴鍦ㄧ洰鏍囧昂搴﹂偦甯т笂浼拌
    鍏夋祦锛屼互閬垮厤浣庡垎杈ㄧ巼杈撳叆瀵艰嚧鐨勬暟鍊间笉绋冲畾锛屽啀瀵圭洰鏍囧抚鍋氬崐姝?warp銆?    """
    warped_prev, warped_next = build_flow_aligned_input_pair(
        prev_flow_frame,
        next_flow_frame,
        prev_target_frame,
        next_target_frame,
        backend=backend,
        raft_variant=raft_variant,
        raft_ckpt=raft_ckpt,
    )
    prior = 0.5 * (warped_prev + warped_next)
    return torch.clamp(prior, min=-1.0, max=1.0)


def build_bidirectional_flow_tensor(
    prev_flow_frame: torch.Tensor,
    next_flow_frame: torch.Tensor,
    prev_target_frame: torch.Tensor,
    next_target_frame: torch.Tensor,
    backend: str = "farneback",
    raft_variant: str = "large",
    raft_ckpt: str = None,
):
    """@brief 鏋勯€犲綊涓€鍖栧悗鐨勫弻鍚戝厜娴佸紶閲忎笌骞呭€奸€氶亾锛屼緵 latent motion encoder 鐩存帴缂栫爜銆?    @example 杈撳嚭褰㈢姸涓?`B x 6 x H x W`锛屽叾涓墠 2 閫氶亾涓?`prev->next`锛?    涓棿 2 閫氶亾涓?`next->prev`锛屾渶鍚?2 閫氶亾涓哄搴旂殑鍏夋祦骞呭€煎浘銆?    """
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
    target_hw = prev_target_frame.shape[-2:]
    flow_prev_to_next = _resize_flow(flow_prev_to_next.to(prev_target_frame.device), target_hw)
    flow_next_to_prev = _resize_flow(flow_next_to_prev.to(next_target_frame.device), target_hw)

    target_h, target_w = target_hw
    norm_w = float(max(target_w - 1, 1))
    norm_h = float(max(target_h - 1, 1))
    flow_prev_to_next = flow_prev_to_next.clone()
    flow_next_to_prev = flow_next_to_prev.clone()
    flow_prev_to_next[:, 0] /= norm_w
    flow_prev_to_next[:, 1] /= norm_h
    flow_next_to_prev[:, 0] /= norm_w
    flow_next_to_prev[:, 1] /= norm_h
    mag_prev_to_next = torch.sqrt(torch.sum(flow_prev_to_next * flow_prev_to_next, dim=1, keepdim=True))
    mag_next_to_prev = torch.sqrt(torch.sum(flow_next_to_prev * flow_next_to_prev, dim=1, keepdim=True))
    return torch.cat([flow_prev_to_next, flow_next_to_prev, mag_prev_to_next, mag_next_to_prev], dim=1)

