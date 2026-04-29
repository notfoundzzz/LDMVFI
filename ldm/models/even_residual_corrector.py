import torch
from torch import nn


class EvenFrameResidualCorrector(nn.Module):
    def __init__(
        self,
        hidden_channels=32,
        num_blocks=4,
        max_residue=0.25,
        use_flow_inputs=False,
        corrector_mode="residual",
        fusion_init_pred_logit=8.0,
        fusion_gate_init_bias=-4.0,
        zero_init_last=True,
    ):
        super().__init__()
        self.max_residue = float(max_residue)
        self.use_flow_inputs = bool(use_flow_inputs)
        self.corrector_mode = str(corrector_mode)
        self.fusion_init_pred_logit = float(fusion_init_pred_logit)
        self.fusion_gate_init_bias = float(fusion_gate_init_bias)
        if self.corrector_mode not in {"residual", "fusion", "confidence_gated"}:
            raise ValueError(f"Unsupported even corrector mode: {self.corrector_mode}")
        in_channels = 15 if self.use_flow_inputs else 9
        if self.corrector_mode == "confidence_gated":
            in_channels += 5
        out_channels = 3
        if self.corrector_mode == "fusion":
            out_channels = 8
        elif self.corrector_mode == "confidence_gated":
            out_channels = 9
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(max(num_blocks - 2, 0)):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        if zero_init_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            if self.corrector_mode in {"fusion", "confidence_gated"}:
                # Bias the initial softmax toward the LDMVFI prediction so
                # fusion mode starts near the existing baseline.
                with torch.no_grad():
                    self.net[-1].bias[0] = self.fusion_init_pred_logit
                    if self.corrector_mode == "confidence_gated":
                        self.net[-1].bias[8] = self.fusion_gate_init_bias

    @staticmethod
    def _candidate_images(prev_sr, pred_even, next_sr, warped_prev, warped_next):
        neighbor_blend = torch.clamp(0.5 * (prev_sr + next_sr), -1.0, 1.0)
        warp_blend = torch.clamp(0.5 * (warped_prev + warped_next), -1.0, 1.0)
        candidates = torch.stack([pred_even, warped_prev, warped_next, neighbor_blend, warp_blend], dim=1)
        return candidates, neighbor_blend, warp_blend

    @staticmethod
    def _confidence_maps(prev_sr, pred_even, next_sr, warped_prev, warped_next, neighbor_blend, warp_blend):
        del neighbor_blend
        maps = [
            torch.mean(torch.abs(warped_prev - warped_next), dim=1, keepdim=True),
            torch.mean(torch.abs(pred_even - warp_blend), dim=1, keepdim=True),
            torch.mean(torch.abs(pred_even - warped_prev), dim=1, keepdim=True),
            torch.mean(torch.abs(pred_even - warped_next), dim=1, keepdim=True),
            torch.mean(torch.abs(prev_sr - next_sr), dim=1, keepdim=True),
        ]
        return torch.cat(maps, dim=1)

    def forward(self, prev_sr, pred_even, next_sr, warped_prev=None, warped_next=None):
        inputs = [prev_sr, pred_even, next_sr]
        if warped_prev is None or warped_next is None:
            warped_prev, warped_next = prev_sr, next_sr
        candidates, neighbor_blend, warp_blend = self._candidate_images(
            prev_sr, pred_even, next_sr, warped_prev, warped_next
        )
        if self.use_flow_inputs:
            inputs.extend([warped_prev, warped_next])
        if self.corrector_mode == "confidence_gated":
            inputs.append(
                self._confidence_maps(prev_sr, pred_even, next_sr, warped_prev, warped_next, neighbor_blend, warp_blend)
            )
        features = self.net(torch.cat(inputs, dim=1))
        if self.corrector_mode == "residual":
            residual = torch.tanh(features)
            return torch.clamp(pred_even + self.max_residue * residual, -1.0, 1.0)

        weights = torch.softmax(features[:, :5], dim=1).unsqueeze(2)
        base = torch.sum(weights * candidates, dim=1)
        residual = torch.tanh(features[:, 5:8])
        proposal = torch.clamp(base + self.max_residue * residual, -1.0, 1.0)
        if self.corrector_mode == "confidence_gated":
            gate = torch.sigmoid(features[:, 8:9])
            return torch.clamp(pred_even + gate * (proposal - pred_even), -1.0, 1.0)
        return proposal


def load_even_corrector(
    ckpt_path,
    device,
    hidden_channels=32,
    num_blocks=4,
    max_residue=0.25,
    corrector_mode="residual",
    fusion_init_pred_logit=8.0,
    fusion_gate_init_bias=-4.0,
):
    payload = torch.load(ckpt_path, map_location="cpu")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = EvenFrameResidualCorrector(
        hidden_channels=int(args.get("hidden_channels", hidden_channels)),
        num_blocks=int(args.get("num_blocks", num_blocks)),
        max_residue=float(args.get("max_residue", max_residue)),
        use_flow_inputs=bool(args.get("use_flow_inputs", False)),
        corrector_mode=str(args.get("corrector_mode", corrector_mode)),
        fusion_init_pred_logit=float(args.get("fusion_init_pred_logit", fusion_init_pred_logit)),
        fusion_gate_init_bias=float(args.get("fusion_gate_init_bias", fusion_gate_init_bias)),
        zero_init_last=False,
    )
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
