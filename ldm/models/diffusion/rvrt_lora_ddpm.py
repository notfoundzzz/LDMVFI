import torch
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddpm import LatentDiffusionVFI
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.modules.ema import LitEma
from ldm.modules.lora import inject_lora_modules, lora_parameters
from ldm.util import instantiate_from_config


class LatentDiffusionVFIRVRTLoRA(LatentDiffusionVFI):
    def __init__(
        self,
        rvrt_root,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        rvrt_tile=(0, 0, 0),
        rvrt_tile_overlap=(2, 20, 20),
        lora_rank=8,
        lora_alpha=16.0,
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

        self.rvrt_frontend = build_sr_frontend(
            sr_mode="rvrt",
            scale=4,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tuple(rvrt_tile),
            tile_overlap=tuple(rvrt_tile_overlap),
        )
        self.rvrt_frontend.eval()
        for param in self.rvrt_frontend.parameters():
            param.requires_grad = False

        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

        self.lora_targets = inject_lora_modules(self.model, rank=lora_rank, alpha=lora_alpha)
        print(f"Injected LoRA into {len(self.lora_targets)} modules")
        if len(self.lora_targets) == 0:
            raise RuntimeError("No LoRA target modules were found in diffusion model")
        if self.use_ema:
            # Rebuild EMA after LoRA injection/freeze so it tracks only current
            # trainable parameters and does not retain stale parameter names.
            self.model_ema = LitEma(self.model)

    def on_fit_start(self):
        super().on_fit_start()
        self.rvrt_frontend = self.rvrt_frontend.to(self.device)

    @torch.no_grad()
    def _super_resolve_neighbors(self, batch, bs=None):
        prev_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_prev_key).to(self.device)
        next_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_next_key).to(self.device)
        if bs is not None:
            prev_lr = prev_lr[:bs]
            next_lr = next_lr[:bs]

        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        seq = torch.stack([prev_01, next_01], dim=1)
        out = self.rvrt_frontend(seq)
        prev_sr = torch.clamp(out[:, 0] * 2.0 - 1.0, min=-1.0, max=1.0)
        next_sr = torch.clamp(out[:, 1] * 2.0 - 1.0, min=-1.0, max=1.0)
        return prev_sr, next_sr

    @torch.no_grad()
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
        x = super(LatentDiffusionVFI, self).get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        prev_sr, next_sr = self._super_resolve_neighbors(batch, bs=bs)
        xc = {self.cond_prev_key: prev_sr, self.cond_next_key: next_sr}
        c, phi_prev_list, phi_next_list = self.get_learned_conditioning(xc)

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

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(lora_parameters(self.model))
        if self.learn_logvar:
            params.append(self.logvar)
        if not params:
            raise RuntimeError("No trainable LoRA parameters found")
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
