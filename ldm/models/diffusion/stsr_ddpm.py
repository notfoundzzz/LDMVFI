import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusionVFI
from ldm.util import instantiate_from_config


class ConditionBridge(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
        )

    def forward(self, frame_lr, output_size):
        frame_up = nn.functional.interpolate(frame_lr, size=output_size, mode="bicubic", align_corners=False)
        residual = self.net(frame_up)
        return torch.clamp(frame_up + residual, min=-1.0, max=1.0)


class LatentDiffusionVFIStsr(LatentDiffusionVFI):
    def __init__(
        self,
        bridge_hidden_channels=32,
        lr_prev_key="prev_frame_lr",
        lr_next_key="next_frame_lr",
        cond_prev_key="prev_frame",
        cond_next_key="next_frame",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_prev_key = lr_prev_key
        self.lr_next_key = lr_next_key
        self.cond_prev_key = cond_prev_key
        self.cond_next_key = cond_next_key
        self.condition_bridge = ConditionBridge(hidden_channels=bridge_hidden_channels)

    def _adapt_condition_frames(self, batch, bs=None):
        prev_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_prev_key).to(self.device)
        next_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_next_key).to(self.device)
        target = super(LatentDiffusionVFI, self).get_input(batch, self.first_stage_key).to(self.device)
        if bs is not None:
            prev_lr = prev_lr[:bs]
            next_lr = next_lr[:bs]
            target = target[:bs]
        output_size = target.shape[-2:]
        prev_cond = self.condition_bridge(prev_lr, output_size)
        next_cond = self.condition_bridge(next_lr, output_size)
        return prev_cond, next_cond, prev_lr, next_lr

    def get_learned_conditioning(self, c):
        phi_prev_list, phi_next_list = None, None
        if isinstance(c, dict) and self.cond_prev_key in c.keys():
            c_prev, phi_prev_list = self.cond_stage_model.encode(c[self.cond_prev_key], ret_feature=True)
            c_next, phi_next_list = self.cond_stage_model.encode(c[self.cond_next_key], ret_feature=True)
            c = torch.cat([c_prev, c_next], dim=1)
        else:
            c = self.cond_stage_model.encode(c)
        return c, phi_prev_list, phi_next_list

    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        return_phi=False,
        bs=None,
    ):
        x = super(LatentDiffusionVFI, self).get_input(batch, k).to(self.device)
        if bs is not None:
            x = x[:bs]

        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        phi_prev_list, phi_next_list = None, None
        assert self.model.conditioning_key is not None
        if cond_key is None:
            cond_key = self.cond_stage_key
        assert cond_key == "past_future_frames"

        prev_cond, next_cond, prev_lr, next_lr = self._adapt_condition_frames(batch, bs=bs)
        xc = {
            self.cond_prev_key: prev_cond,
            self.cond_next_key: next_cond,
            self.lr_prev_key: prev_lr,
            self.lr_next_key: next_lr,
        }

        if not self.cond_stage_trainable or force_c_encode:
            c, phi_prev_list, phi_next_list = self.get_learned_conditioning(xc)
        else:
            c = xc

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z, xc, phi_prev_list, phi_next_list)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if return_phi:
            out.append(phi_prev_list)
            out.append(phi_next_list)
        return out

    @torch.no_grad()
    def sample_stsr(self, prev_lr, next_lr, gt_size, use_ddim=False, ddim_steps=200, ddim_eta=1.0):
        prev_cond = self.condition_bridge(prev_lr.to(self.device), gt_size)
        next_cond = self.condition_bridge(next_lr.to(self.device), gt_size)
        xc = {
            self.cond_prev_key: prev_cond,
            self.cond_next_key: next_cond,
            self.lr_prev_key: prev_lr.to(self.device),
            self.lr_next_key: next_lr.to(self.device),
        }
        c, phi_prev_list, phi_next_list = self.get_learned_conditioning(xc)
        shape = (self.channels, c.shape[2], c.shape[3])
        if use_ddim:
            ddim_sampler = DDIMSampler(self)
            latent, _ = ddim_sampler.sample(ddim_steps, c.shape[0], shape, c, eta=ddim_eta, verbose=False)
        else:
            latent = self.sample_ddpm(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None, verbose=False)
        if isinstance(latent, tuple):
            latent = latent[0]
        out = self.decode_first_stage(latent, xc, phi_prev_list, phi_next_list)
        return torch.clamp(out, min=-1.0, max=1.0)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(self.condition_bridge.parameters())
        if self.cond_stage_trainable:
            params += list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
