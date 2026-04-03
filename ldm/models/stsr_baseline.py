import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import utility


def _batch_to_chw(batch_tensor):
    if batch_tensor.dim() == 4 and batch_tensor.shape[-1] == 3:
        return batch_tensor.permute(0, 3, 1, 2).contiguous().float()
    return batch_tensor.float()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleStsrNet(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(hidden_channels, 3, 3, padding=1)

    def forward(self, prev_lr, next_lr, output_size):
        prev_up = F.interpolate(prev_lr, size=output_size, mode="bicubic", align_corners=False)
        next_up = F.interpolate(next_lr, size=output_size, mode="bicubic", align_corners=False)
        base = 0.5 * (prev_up + next_up)
        x = torch.cat([prev_up, next_up], dim=1)
        x = self.head(x)
        x = self.body(x)
        residual = self.tail(x)
        return torch.clamp(base + residual, min=-1.0, max=1.0)


class DeterministicSTSR(pl.LightningModule):
    def __init__(self, learning_rate=2.0e-4, hidden_channels=64, num_blocks=6):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = SimpleStsrNet(hidden_channels=hidden_channels, num_blocks=num_blocks)
        self.loss_fn = nn.L1Loss()

    def _extract_batch(self, batch):
        gt = _batch_to_chw(batch["image"]).to(self.device)
        prev_lr = _batch_to_chw(batch["prev_frame_lr"]).to(self.device)
        next_lr = _batch_to_chw(batch["next_frame_lr"]).to(self.device)
        return prev_lr, next_lr, gt

    def forward(self, prev_lr, next_lr, output_size):
        return self.model(prev_lr, next_lr, output_size)

    def predict_middle_frame(self, prev_lr, next_lr):
        return self(prev_lr, next_lr, prev_lr.shape[-2:])

    def sample_stsr(self, prev_lr, next_lr, output_size):
        return self(prev_lr, next_lr, output_size)

    def shared_step(self, batch, split):
        prev_lr, next_lr, gt = self._extract_batch(batch)
        pred = self(prev_lr, next_lr, gt.shape[-2:])
        loss = self.loss_fn(pred, gt)
        psnr = utility.calc_psnr(gt, pred).mean()
        self.log(f"{split}/loss", loss, prog_bar=(split == "train"), logger=True, on_step=(split == "train"), on_epoch=True)
        self.log(f"{split}/psnr", psnr, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        prev_lr, next_lr, gt = self._extract_batch(batch)
        pred = self(prev_lr, next_lr, gt.shape[-2:])
        prev_up = F.interpolate(prev_lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        next_up = F.interpolate(next_lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        return {
            "inputs_prev": prev_up,
            "inputs_next": next_up,
            "samples": pred,
            "reconstructions": pred,
            "gts": gt,
        }
