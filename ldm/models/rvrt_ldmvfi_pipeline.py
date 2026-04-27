from contextlib import nullcontext

import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.flow_guidance import (
    build_bidirectional_flow_tensor,
    build_flow_aligned_input_pair,
    build_flow_guided_middle_prior,
)
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.util import instantiate_from_config


class RVRTLDMVFIPipeline:

    def __init__(
        self,
        ldm_config,
        ldm_ckpt,
        sr_mode="bicubic",
        scale=4,
        rvrt_root=None,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        rvrt_flow_mode="spynet",
        rvrt_raft_variant="large",
        rvrt_raft_ckpt=None,
        rvrt_adapter_ckpt=None,
        rvrt_use_flow_adapter=False,
        rvrt_flow_adapter_hidden_channels=16,
        rvrt_flow_adapter_zero_init_last=True,
        rvrt_flow_adapter_max_residue_magnitude=1.0,
        device=None,
        tile=(0, 0, 0),
        tile_overlap=(2, 20, 20),
        use_ema=True,
        strict_checkpoint=True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ema = bool(use_ema)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.last_flow_prior = None

        self.model = instantiate_from_config(ldm_config.model)
        state = torch.load(ldm_ckpt, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        ckpt_has_lora = any("lora_" in key for key in state_dict.keys())
        model_has_lora = any("lora_" in key for key in self.model.state_dict().keys())

        if ckpt_has_lora and not model_has_lora:
            raise ValueError(
                "Checkpoint contains LoRA weights, but the selected config does not instantiate LoRA modules. "
                "Use configs/ldm/rvrt-lora-stsr-x4.yaml for LoRA checkpoints."
            )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        disallowed_unexpected = [key for key in unexpected if not self._is_allowed_unexpected_key(key)]
        if disallowed_unexpected and self.strict_checkpoint:
            raise ValueError(
                "Unexpected checkpoint keys were found while loading the diffusion model. "
                f"First keys: {disallowed_unexpected[:10]}"
            )
        if disallowed_unexpected and not self.strict_checkpoint:
            print(
                "Ignoring unexpected checkpoint keys because strict checkpoint loading is disabled. "
                f"First keys: {disallowed_unexpected[:10]}"
            )
        if model_has_lora:
            missing_lora = [key for key in missing if "lora_" in key]
            if missing_lora:
                raise ValueError(
                    "Selected config expects LoRA weights, but checkpoint is missing them. "
                    f"First keys: {missing_lora[:10]}"
                )
        disallowed_missing = [key for key in missing if not self._is_allowed_missing_key(key)]
        if self.strict_checkpoint and disallowed_missing:
            raise ValueError(
                "Checkpoint is missing required keys for the selected evaluation config. "
                f"First keys: {disallowed_missing[:10]}"
            )
        if rvrt_adapter_ckpt and hasattr(getattr(self.model, "rvrt_frontend", None), "_load_adapter_checkpoint"):
            self.model.rvrt_frontend._load_adapter_checkpoint(rvrt_adapter_ckpt)
        self.model = self.model.to(self.device).eval()
        self.sr_frontend, self.sr_frontend_source = self._build_eval_frontend(
            sr_mode=sr_mode,
            scale=scale,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            rvrt_flow_mode=rvrt_flow_mode,
            rvrt_raft_variant=rvrt_raft_variant,
            rvrt_raft_ckpt=rvrt_raft_ckpt,
            rvrt_adapter_ckpt=rvrt_adapter_ckpt,
            rvrt_use_flow_adapter=rvrt_use_flow_adapter,
            rvrt_flow_adapter_hidden_channels=rvrt_flow_adapter_hidden_channels,
            rvrt_flow_adapter_zero_init_last=rvrt_flow_adapter_zero_init_last,
            rvrt_flow_adapter_max_residue_magnitude=rvrt_flow_adapter_max_residue_magnitude,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        print(f"Evaluation RVRT source: {self.sr_frontend_source}")

    def _is_allowed_missing_key(self, key):
        if key.startswith("rvrt_frontend."):
            # The RVRT frontend is initialized from its own checkpoint before
            # the LDM checkpoint is loaded, so missing frontend keys here do
            # not indicate missing diffusion weights.
            return True
        if key.startswith("model_ema.") and not self.use_ema:
            return True
        if key == "flow_condition_gate":
            return getattr(self.model, "flow_condition_gate", None) is None
        if key.startswith("condition_frame_adapter."):
            return getattr(self.model, "condition_frame_adapter", None) is None
        if key.startswith("latent_motion_encoder."):
            return getattr(self.model, "latent_motion_encoder", None) is None
        if key.startswith("latent_motion_phi_fusers."):
            return getattr(self.model, "latent_motion_phi_fusers", None) is None
        return False

    def _is_allowed_unexpected_key(self, key):
        return key.startswith("rvrt_frontend.")

    def _build_eval_frontend(
        self,
        sr_mode,
        scale,
        rvrt_root,
        rvrt_task,
        rvrt_ckpt,
        rvrt_flow_mode,
        rvrt_raft_variant,
        rvrt_raft_ckpt,
        rvrt_adapter_ckpt,
        rvrt_use_flow_adapter,
        rvrt_flow_adapter_hidden_channels,
        rvrt_flow_adapter_zero_init_last,
        rvrt_flow_adapter_max_residue_magnitude,
        tile,
        tile_overlap,
    ):
        model_frontend = getattr(self.model, "rvrt_frontend", None)
        if sr_mode == "rvrt" and model_frontend is not None:
            frontend = model_frontend.to(self.device).eval()
            frontend_mode = getattr(self.model, "sr_frontend_mode", "rvrt")
            return frontend, f"checkpoint sr_frontend ({frontend_mode})"

        frontend = build_sr_frontend(
            sr_mode=sr_mode,
            scale=scale,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            rvrt_flow_mode=rvrt_flow_mode,
            rvrt_raft_variant=rvrt_raft_variant,
            rvrt_raft_ckpt=rvrt_raft_ckpt,
            rvrt_adapter_ckpt=rvrt_adapter_ckpt,
            rvrt_use_flow_adapter=rvrt_use_flow_adapter,
            rvrt_flow_adapter_hidden_channels=rvrt_flow_adapter_hidden_channels,
            rvrt_flow_adapter_zero_init_last=rvrt_flow_adapter_zero_init_last,
            rvrt_flow_adapter_max_residue_magnitude=rvrt_flow_adapter_max_residue_magnitude,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        return frontend.to(self.device).eval(), "external sr_frontend"

    def _sampling_scope(self, seed):
        if seed is None:
            return nullcontext()

        devices = [self.device] if self.device.type == "cuda" else []
        return torch.random.fork_rng(devices=devices)

    def _prepare_rvrt_sequence(self, seq):
        if getattr(self.sr_frontend, "flow_mode", None) is not None and seq.shape[1] == 7:
            return torch.cat([seq, seq.flip(1)], dim=1)
        return seq

    @torch.no_grad()
    def super_resolve_neighbors(self, prev_lr, next_lr, lr_sequence=None, prev_index=None, next_index=None):
        if lr_sequence is not None:
            seq = lr_sequence.to(self.device)
            seq = self._prepare_rvrt_sequence(seq)
            if prev_index is None:
                prev_index = int(getattr(self.model, "rvrt_prev_index", 2))
            if next_index is None:
                next_index = int(getattr(self.model, "rvrt_next_index", 4))
        else:
            seq = torch.stack([prev_lr, next_lr], dim=1).to(self.device)
            prev_index, next_index = 0, 1
        out = self.sr_frontend(seq)
        if max(prev_index, next_index) >= out.shape[1]:
            raise ValueError(
                f"SR frontend returned {out.shape[1]} frames, cannot select indices {prev_index}/{next_index}"
            )
        raw_prev_sr = torch.clamp(out[:, prev_index] * 2.0 - 1.0, min=-1.0, max=1.0)
        raw_next_sr = torch.clamp(out[:, next_index] * 2.0 - 1.0, min=-1.0, max=1.0)
        if hasattr(self.model, "_adapt_condition_frames"):
            prev_sr, next_sr = self.model._adapt_condition_frames(raw_prev_sr, raw_next_sr)
        else:
            prev_sr, next_sr = raw_prev_sr, raw_next_sr
        return raw_prev_sr, raw_next_sr, prev_sr, next_sr

    @torch.no_grad()
    def interpolate(
        self,
        prev_lr,
        next_lr,
        lr_sequence=None,
        use_ddim=True,
        ddim_steps=200,
        ddim_eta=0.0,
        seed=None,
    ):
        prev_lr = prev_lr.to(self.device)
        next_lr = next_lr.to(self.device)

        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        lr_sequence_01 = None
        if lr_sequence is not None:
            lr_sequence_01 = ((lr_sequence.to(self.device) + 1.0) / 2.0).clamp(0.0, 1.0)
        raw_prev_sr, raw_next_sr, prev_sr, next_sr = self.super_resolve_neighbors(
            prev_01,
            next_01,
            lr_sequence=lr_sequence_01,
        )

        scope = self.model.ema_scope() if self.use_ema and hasattr(self.model, "ema_scope") else nullcontext()
        with scope:
            xc = {"prev_frame": prev_sr, "next_frame": next_sr}
            if getattr(self.model, "use_flow_guidance", False):
                flow_mode = getattr(self.model, "flow_condition_mode", "fused")
                if flow_mode == "aligned_input":
                    aligned_prev, aligned_next = build_flow_aligned_input_pair(
                        prev_lr,
                        next_lr,
                        raw_prev_sr,
                        raw_next_sr,
                        backend=getattr(self.model, "flow_backend", "farneback"),
                        raft_variant=getattr(self.model, "flow_raft_variant", "large"),
                        raft_ckpt=getattr(self.model, "flow_raft_ckpt", None),
                    )
                    if hasattr(self.model, "_adapt_condition_frames"):
                        aligned_prev, aligned_next = self.model._adapt_condition_frames(aligned_prev, aligned_next)
                    xc["prev_frame"] = aligned_prev
                    xc["next_frame"] = aligned_next
                    self.last_flow_prior = None
                elif flow_mode == "latent_explicit":
                    xc[getattr(self.model, "cond_flow_key", "flow_prior")] = build_bidirectional_flow_tensor(
                        prev_lr,
                        next_lr,
                        raw_prev_sr,
                        raw_next_sr,
                        backend=getattr(self.model, "flow_backend", "farneback"),
                        raft_variant=getattr(self.model, "flow_raft_variant", "large"),
                        raft_ckpt=getattr(self.model, "flow_raft_ckpt", None),
                    )
                    self.last_flow_prior = None
                else:
                    self.last_flow_prior = build_flow_guided_middle_prior(
                        prev_lr,
                        next_lr,
                        raw_prev_sr,
                        raw_next_sr,
                        backend=getattr(self.model, "flow_backend", "farneback"),
                        raft_variant=getattr(self.model, "flow_raft_variant", "large"),
                        raft_ckpt=getattr(self.model, "flow_raft_ckpt", None),
                    )
                    xc[getattr(self.model, "cond_flow_key", "flow_prior")] = self.last_flow_prior
            else:
                self.last_flow_prior = None
            c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
            shape = (self.model.channels, c.shape[2], c.shape[3])
            with self._sampling_scope(seed):
                if seed is not None:
                    torch.manual_seed(seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                if use_ddim:
                    ddim = DDIMSampler(self.model)
                    latent, _ = ddim.sample(ddim_steps, c.shape[0], shape, c, eta=ddim_eta, verbose=False, x_T=None)
                else:
                    latent = self.model.sample_ddpm(
                        conditioning=c,
                        batch_size=c.shape[0],
                        shape=shape,
                        x_T=None,
                        verbose=False,
                    )
            if isinstance(latent, tuple):
                latent = latent[0]
            out = self.model.decode_first_stage(latent, xc, phi_prev_list, phi_next_list)
        return torch.clamp(out, min=-1.0, max=1.0), prev_sr, next_sr
