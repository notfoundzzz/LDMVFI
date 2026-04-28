import torch
from torch import nn


class EvenFrameResidualCorrector(nn.Module):
    def __init__(self, hidden_channels=32, num_blocks=4, max_residue=0.25, zero_init_last=True):
        super().__init__()
        self.max_residue = float(max_residue)
        layers = [
            nn.Conv2d(9, hidden_channels, 3, padding=1),
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

    def forward(self, prev_sr, pred_even, next_sr):
        residual = torch.tanh(self.net(torch.cat([prev_sr, pred_even, next_sr], dim=1)))
        return torch.clamp(pred_even + self.max_residue * residual, -1.0, 1.0)


def load_even_corrector(ckpt_path, device, hidden_channels=32, num_blocks=4, max_residue=0.25):
    payload = torch.load(ckpt_path, map_location="cpu")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = EvenFrameResidualCorrector(
        hidden_channels=int(args.get("hidden_channels", hidden_channels)),
        num_blocks=int(args.get("num_blocks", num_blocks)),
        max_residue=float(args.get("max_residue", max_residue)),
        zero_init_last=False,
    )
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
