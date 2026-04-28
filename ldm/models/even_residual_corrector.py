import torch
from torch import nn


class EvenFrameResidualCorrector(nn.Module):
    def __init__(
        self,
        hidden_channels=32,
        num_blocks=4,
        max_residue=0.25,
        use_flow_inputs=False,
        zero_init_last=True,
    ):
        super().__init__()
        self.max_residue = float(max_residue)
        self.use_flow_inputs = bool(use_flow_inputs)
        in_channels = 15 if self.use_flow_inputs else 9
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
        layers.append(nn.Conv2d(hidden_channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        if zero_init_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, prev_sr, pred_even, next_sr, warped_prev=None, warped_next=None):
        inputs = [prev_sr, pred_even, next_sr]
        if self.use_flow_inputs:
            if warped_prev is None or warped_next is None:
                warped_prev, warped_next = prev_sr, next_sr
            inputs.extend([warped_prev, warped_next])
        residual = torch.tanh(self.net(torch.cat(inputs, dim=1)))
        return torch.clamp(pred_even + self.max_residue * residual, -1.0, 1.0)


def load_even_corrector(ckpt_path, device, hidden_channels=32, num_blocks=4, max_residue=0.25):
    payload = torch.load(ckpt_path, map_location="cpu")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = EvenFrameResidualCorrector(
        hidden_channels=int(args.get("hidden_channels", hidden_channels)),
        num_blocks=int(args.get("num_blocks", num_blocks)),
        max_residue=float(args.get("max_residue", max_residue)),
        use_flow_inputs=bool(args.get("use_flow_inputs", False)),
        zero_init_last=False,
    )
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
