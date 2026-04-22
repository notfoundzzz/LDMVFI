import torch
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddpm import LatentDiffusionVFI
from ldm.models.flow_guidance import build_flow_aligned_input_pair, build_flow_guided_middle_prior
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.util import instantiate_from_config


class LatentDiffusionVFIRVRTFlowGuided(LatentDiffusionVFI):
    """纯光流引导训练类。"""

    def __init__(
        self,
        rvrt_root,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        rvrt_tile=(0, 0, 0),
        rvrt_tile_overlap=(2, 20, 20),
        lr_prev_key="prev_frame_lr",
        lr_next_key="next_frame_lr",
        cond_prev_key="prev_frame",
        cond_next_key="next_frame",
        cond_flow_key="flow_prior",
        use_flow_guidance=True,
        flow_guidance_strength=0.25,
        flow_condition_mode="fused",
        flow_backend="farneback",
        flow_raft_variant="large",
        flow_raft_ckpt=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_prev_key = lr_prev_key
        self.lr_next_key = lr_next_key
        self.cond_prev_key = cond_prev_key
        self.cond_next_key = cond_next_key
        self.cond_flow_key = cond_flow_key
        self.use_flow_guidance = bool(use_flow_guidance)
        self.flow_guidance_strength = float(flow_guidance_strength)
        self.flow_condition_mode = str(flow_condition_mode)
        self.flow_backend = str(flow_backend)
        self.flow_raft_variant = str(flow_raft_variant)
        self.flow_raft_ckpt = flow_raft_ckpt
        self.cond_latent_channels = int(getattr(self.cond_stage_model, "embed_dim", self.channels))
        self.flow_condition_fuser = None
        if self.use_flow_guidance and self.flow_condition_mode == "explicit":
            self.flow_condition_fuser = torch.nn.Conv2d(
                self.cond_latent_channels * 3,
                self.cond_latent_channels * 2,
                kernel_size=1,
            )

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
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Full-tuning diffusion UNet with {trainable} trainable parameters")
        print("RVRT frontend frozen")
        print("First-stage and cond-stage frozen")
        print(
            f"Flow guidance enabled: {self.use_flow_guidance}, strength={self.flow_guidance_strength:.3f}, "
            f"mode={self.flow_condition_mode}, backend={self.flow_backend}, raft_variant={self.flow_raft_variant}"
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["rvrt_flow_guidance_metadata"] = {
            "use_flow_guidance": self.use_flow_guidance,
            "flow_guidance_strength": self.flow_guidance_strength,
            "flow_condition_mode": self.flow_condition_mode,
            "flow_backend": self.flow_backend,
            "flow_raft_variant": self.flow_raft_variant,
            "flow_raft_ckpt": self.flow_raft_ckpt,
        }

    def on_fit_start(self):
        super().on_fit_start()
        self.rvrt_frontend = self.rvrt_frontend.to(self.device).eval()

    @torch.no_grad()
    def _super_resolve_neighbors(self, batch, bs=None):
        """LR 邻帧先做 SR。"""
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

    def _build_flow_guided_prior(self, prev_lr, next_lr, prev_target, next_target):
        return build_flow_guided_middle_prior(
            prev_lr,
            next_lr,
            prev_target,
            next_target,
            backend=self.flow_backend,
            raft_variant=self.flow_raft_variant,
            raft_ckpt=self.flow_raft_ckpt,
        )

    def get_learned_conditioning(self, c):
        """编码条件。"""
        phi_prev_list, phi_next_list = None, None
        if isinstance(c, dict) and self.cond_prev_key in c.keys():
            c_prev, phi_prev_list = self.cond_stage_model.encode(c[self.cond_prev_key], ret_feature=True)
            c_next, phi_next_list = self.cond_stage_model.encode(c[self.cond_next_key], ret_feature=True)
            if self.use_flow_guidance and self.cond_flow_key in c:
                c_flow, phi_flow_list = self.cond_stage_model.encode(c[self.cond_flow_key], ret_feature=True)
                mix = self.flow_guidance_strength
                if self.flow_condition_mode == "explicit":
                    c = self.flow_condition_fuser(torch.cat([c_prev, c_flow, c_next], dim=1))
                    if phi_prev_list is not None and phi_next_list is not None and phi_flow_list is not None:
                        phi_prev_list = [
                            None if prev_feat is None else prev_feat + mix * flow_feat
                            for prev_feat, flow_feat in zip(phi_prev_list, phi_flow_list)
                        ]
                        phi_next_list = [
                            None if next_feat is None else next_feat + mix * flow_feat
                            for next_feat, flow_feat in zip(phi_next_list, phi_flow_list)
                        ]
                else:
                    c_prev = (1.0 - mix) * c_prev + mix * c_flow
                    c_next = (1.0 - mix) * c_next + mix * c_flow
                    if phi_prev_list is not None and phi_next_list is not None and phi_flow_list is not None:
                        phi_prev_list = [
                            None if prev_feat is None else (1.0 - mix) * prev_feat + mix * flow_feat
                            for prev_feat, flow_feat in zip(phi_prev_list, phi_flow_list)
                        ]
                        phi_next_list = [
                            None if next_feat is None else (1.0 - mix) * next_feat + mix * flow_feat
                            for next_feat, flow_feat in zip(phi_next_list, phi_flow_list)
                        ]
                    c = torch.cat([c_prev, c_next], dim=1)
            else:
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
        x = super(LatentDiffusionVFI, self).get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        prev_sr, next_sr = self._super_resolve_neighbors(batch, bs=bs)
        xc = {self.cond_prev_key: prev_sr, self.cond_next_key: next_sr}
        if self.use_flow_guidance:
            prev_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_prev_key).to(self.device)
            next_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_next_key).to(self.device)
            if bs is not None:
                prev_lr = prev_lr[:bs]
                next_lr = next_lr[:bs]
            if self.flow_condition_mode == "aligned_input":
                aligned_prev, aligned_next = build_flow_aligned_input_pair(
                    prev_lr,
                    next_lr,
                    prev_sr,
                    next_sr,
                    backend=self.flow_backend,
                    raft_variant=self.flow_raft_variant,
                    raft_ckpt=self.flow_raft_ckpt,
                )
                xc[self.cond_prev_key] = aligned_prev
                xc[self.cond_next_key] = aligned_next
            else:
                xc[self.cond_flow_key] = self._build_flow_guided_prior(prev_lr, next_lr, prev_sr, next_sr)
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
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.flow_condition_fuser is not None:
            params.extend(p for p in self.flow_condition_fuser.parameters() if p.requires_grad)
        if not params:
            raise RuntimeError("No trainable diffusion parameters found")
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        if self.flow_condition_fuser is not None:
            print(f"Optimizer groups: diffusion+flow_fuser lr={self.learning_rate:.2e}")
        else:
            print(f"Optimizer groups: diffusion lr={self.learning_rate:.2e}")
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
