import argparse
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import get_args_parser, build_CNNMLP_model_and_optimizer
from detr.models_variant_c.detr_vae import build as build_variant_c_model


def build_variant_c_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser("Variant C DETR training", parents=[get_args_parser()])
    args, _ = parser.parse_known_args()
    for key, value in args_override.items():
        setattr(args, key, value)
    print("args: ", args)

    model = build_variant_c_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def build_optimizer_for_model(model, lr, lr_backbone, weight_decay):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    param_dicts = [param_group for param_group in param_dicts if param_group["params"]]
    return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_variant_c_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.lr = args_override["lr"]
        self.lr_backbone = args_override["lr_backbone"]
        self.weight_decay = self.optimizer.defaults.get("weight_decay", 0.0)
        self.kl_weight = args_override["kl_weight"]
        self.lambda_roi = args_override.get("lambda_roi", 1.0)
        self.lambda_sem = args_override.get("lambda_sem", 0.1)
        self.lambda_sig = args_override.get("lambda_sig", 0.0)
        self.lambda_recon_grad = args_override.get("lambda_recon_grad", 0.25)
        self.lambda_future_rgb = args_override.get("lambda_future_rgb", 1.0)
        self.lambda_future_grad = args_override.get("lambda_future_grad", 0.25)
        self.lambda_future_latent = args_override.get("lambda_future_latent", 0.1)
        self.lambda_vq = args_override.get("lambda_vq", 1.0)
        self.lambda_vq_commit = args_override.get("lambda_vq_commit", 0.25)
        self.vq_warmup_epochs = int(args_override.get("vq_warmup_epochs", 0))
        # biased quantization dropout: probs[i] = probability of using i+1 active RQ stages
        self.rq_dropout_probs = list(args_override.get("rq_dropout_probs", [0.50, 0.25, 0.15, 0.10]))
        self.rq_dead_restart_interval = int(args_override.get("rq_dead_code_restart_interval", 500))
        self.rq_dead_restart_threshold = float(args_override.get("rq_dead_code_restart_threshold", 0.07))
        self.rq_dead_restart_max_fraction = float(args_override.get("rq_dead_code_restart_max_fraction", 0.05))
        self._rq_steps = 0
        self._rq_usage_counts = None
        self._rq_usage_totals = None
        self.future_rgb_decay_alpha = args_override.get("future_rgb_decay_alpha", 0.03)
        self.future_latent_decay_alpha = args_override.get("future_latent_decay_alpha", 0.01)
        self.future_teacher_mix_steps = max(1, int(args_override.get("future_teacher_mix_steps", 10000)))
        self.focus_masked_region = args_override.get("focus_masked_region", False)
        self.lambda_masked_region = 1.0
        self.ema_momentum = args_override.get("ema_momentum", 0.99)
        self.comm_detach_warmup = args_override.get("comm_detach_warmup", 0)
        self._ema_updates = 0
        self._future_updates = 0
        self.current_epoch = 0
        self.train_stage = "joint"
        print(
            f"KL Weight {self.kl_weight}, lambda_roi {self.lambda_roi}, "
            f"lambda_sem {self.lambda_sem}, lambda_sig {self.lambda_sig}, "
            f"lambda_recon_grad {self.lambda_recon_grad}, "
            f"focus_masked_region {self.focus_masked_region}, "
            f"lambda_vq {self.lambda_vq}, lambda_vq_commit {self.lambda_vq_commit}, "
            f"vq_warmup_epochs {self.vq_warmup_epochs}"
        )

    @staticmethod
    def _is_roi_parameter(name):
        roi_prefixes = (
            "comm_query_embed",
            "comm_task_token",
            "comm_bandwidth_token",
            "comm_layers",
            "comm_pool_proj",
            "comm_film",
            "comm_decoder",
        )
        return name.startswith(roi_prefixes)

    @staticmethod
    def _is_future_parameter(name):
        future_prefixes = (
            "future_action_proj",
            "future_action_pos_embed",
            "future_frame_query_embed",
            "future_horizon_proj",
            "future_action_layers",
            "future_memory_layers",
            "future_pool_proj",
            "future_film",
            "future_decoder",
        )
        return name.startswith(future_prefixes)

    @staticmethod
    def _is_vq_parameter(name):
        vq_prefixes = (
            "memory_to_code",
            "code_to_memory",
            "memory_quantizer",
            "rq_pos_embed",
        )
        return name.startswith(vq_prefixes)

    @staticmethod
    def _is_ema_parameter(name):
        return name.startswith("ema_")

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        return self

    def set_train_stage(self, stage):
        if stage not in {"joint", "act", "roi", "future"}:
            raise ValueError(f"Unknown train stage {stage!r}; expected joint, act, roi, or future.")
        self.train_stage = stage
        for name, param in self.model.named_parameters():
            if self._is_ema_parameter(name):
                param.requires_grad = False
            elif stage == "joint":
                param.requires_grad = True
            elif stage == "act":
                param.requires_grad = not self._is_roi_parameter(name) and not self._is_future_parameter(name)
            elif stage == "roi":
                param.requires_grad = self._is_roi_parameter(name)
            elif stage == "future":
                param.requires_grad = self._is_future_parameter(name) or self._is_vq_parameter(name)
        self.optimizer = build_optimizer_for_model(self.model, self.lr, self.lr_backbone, self.weight_decay)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Set ACTPolicy train stage to {stage}; trainable parameters: {trainable / 1e6:.2f}M")
        return self

    def _should_detach_comm(self):
        warmup = self.comm_detach_warmup
        if isinstance(warmup, bool):
            return warmup
        return self._ema_updates < int(warmup)

    def _use_quantized_memory(self):
        return self.current_epoch >= self.vq_warmup_epochs

    def _sample_rq_active_stages(self):
        """Sample how many RQ stages to use during training (biased toward more stages)."""
        if not self.training:
            return self.model.memory_quantizer.num_stages
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(self.rq_dropout_probs):
            cumulative += p
            if r < cumulative:
                return i + 1
        return len(self.rq_dropout_probs)

    def _ensure_rq_usage_buffers(self, device):
        device = torch.device(device)
        rq = self.model.memory_quantizer
        needs_init = (
            self._rq_usage_counts is None
            or len(self._rq_usage_counts) != rq.num_stages
            or self._rq_usage_counts[0].numel() != rq.codebook_bins
            or self._rq_usage_counts[0].device != device
        )
        if needs_init:
            self._rq_usage_counts = [
                torch.zeros(rq.codebook_bins, dtype=torch.float32, device=device)
                for _ in range(rq.num_stages)
            ]
            self._rq_usage_totals = [
                torch.zeros((), dtype=torch.float32, device=device)
                for _ in range(rq.num_stages)
            ]

    @torch.no_grad()
    def _accumulate_rq_usage(self, codes, num_active_stages):
        if codes is None or num_active_stages is None:
            return
        rq = self.model.memory_quantizer
        self._ensure_rq_usage_buffers(codes.device)
        active = min(int(num_active_stages), rq.num_stages)
        for stage_idx in range(active):
            stage_codes = codes[..., stage_idx].reshape(-1)
            counts = torch.bincount(stage_codes, minlength=rq.codebook_bins).to(torch.float32)
            self._rq_usage_counts[stage_idx].add_(counts)
            self._rq_usage_totals[stage_idx].add_(float(stage_codes.numel()))

    def _reset_rq_usage_buffers(self):
        if self._rq_usage_counts is None or self._rq_usage_totals is None:
            return
        for counts in self._rq_usage_counts:
            counts.zero_()
        for total in self._rq_usage_totals:
            total.zero_()

    def _sig_reg(self, z_comm_pool):
        if z_comm_pool is None or z_comm_pool.shape[0] <= 1:
            return torch.zeros((), device=next(self.parameters()).device)
        std = torch.sqrt(z_comm_pool.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()

    @staticmethod
    def _weighted_l1(pred, target, weight):
        residual = F.l1_loss(pred, target, reduction="none")
        return (residual * weight).sum() / weight.expand_as(residual).sum().clamp(min=1.0)

    @classmethod
    def _weighted_gradient_l1(cls, pred, target, weight):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        weight_dx = torch.maximum(weight[:, :, :, 1:], weight[:, :, :, :-1])
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        weight_dy = torch.maximum(weight[:, :, 1:, :], weight[:, :, :-1, :])
        return cls._weighted_l1(pred_dx, target_dx, weight_dx) + cls._weighted_l1(pred_dy, target_dy, weight_dy)

    @staticmethod
    def _future_weights(horizons, alpha, valid, channels=False):
        horizon_tensor = torch.tensor(horizons, dtype=torch.float32, device=valid.device)
        weights = torch.exp(-float(alpha) * horizon_tensor).view(1, -1)
        weights = weights * valid.float()
        weights = weights / (weights.sum() / valid.float().sum().clamp(min=1.0)).clamp(min=1e-6)
        weights = weights.masked_fill(~valid, 0.0)
        if channels:
            return weights[:, :, None, None, None]
        return weights[:, :, None]

    @staticmethod
    def _decayed_l1(pred, target, valid, horizons, alpha, weight_mask=None):
        weights = ACTPolicy._future_weights(horizons, alpha, valid, channels=True)
        residual = F.l1_loss(pred, target, reduction="none")
        if weight_mask is not None:
            weights = weights * weight_mask[:, :, None]
        return (residual * weights).sum() / weights.expand_as(residual).sum().clamp(min=1.0)

    @classmethod
    def _decayed_gradient_l1(cls, pred, target, valid, horizons, alpha, weight_mask=None):
        pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        weight_dx = None
        weight_dy = None
        if weight_mask is not None:
            weight_dx = torch.maximum(weight_mask[:, :, :, 1:], weight_mask[:, :, :, :-1])
            weight_dy = torch.maximum(weight_mask[:, :, 1:, :], weight_mask[:, :, :-1, :])
        return cls._decayed_l1(pred_dx, target_dx, valid, horizons, alpha, weight_dx) + cls._decayed_l1(
            pred_dy,
            target_dy,
            valid,
            horizons,
            alpha,
            weight_dy,
        )

    @staticmethod
    def _decayed_latent_mse(pred, target, valid, horizons, alpha):
        weights = ACTPolicy._future_weights(horizons, alpha, valid, channels=False)
        residual = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1), reduction="none")
        return (residual * weights).sum() / weights.expand_as(residual).sum().clamp(min=1.0)

    def _use_predicted_future_actions(self):
        if not self.training:
            return self.train_stage == "joint"
        if self.train_stage == "future":
            pred_ratio = min(0.5, 0.5 * self._future_updates / float(self.future_teacher_mix_steps))
        elif self.train_stage == "joint":
            pred_ratio = 0.8
        else:
            pred_ratio = 0.0
        return torch.rand(()).item() < pred_ratio

    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        future_rgb=None,
        future_valid=None,
        future_roi_weight_mask=None,
        future_focus_weight_mask=None,
        return_comm=False,
    ):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]
            is_act_stage = self.train_stage == "act"
            is_roi_stage = self.train_stage == "roi"
            is_future_stage = self.train_stage == "future"
            target_image = None
            future_image = normalize(future_rgb.flatten(0, 1)).reshape_as(future_rgb) if future_rgb is not None else None
            use_predicted_future_actions = self._use_predicted_future_actions()
            future_actions = None if use_predicted_future_actions else actions
            use_rq = self._use_quantized_memory() and not is_act_stage
            num_active_rq_stages = self._sample_rq_active_stages() if use_rq else None
            a_hat, is_pad_hat, (mu, logvar), _, comm_outputs = self.model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                target_image=target_image,
                future_images=future_image,
                future_actions=future_actions,
                detach_comm=self._should_detach_comm(),
                detach_future_actions=is_future_stage,
                force_zero_latent=is_roi_stage,
                run_comm=False,
                run_future=not is_act_stage and future_rgb is not None,
                encode_target_semantics=not is_act_stage,
                use_quantized_memory=use_rq,
                num_active_rq_stages=num_active_rq_stages,
            )
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            loss_dict["l1"] = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            if mu is None or logvar is None:
                loss_dict["kl"] = torch.zeros((), device=qpos.device)
            else:
                total_kld, _, _ = kl_divergence(mu, logvar)
                loss_dict["kl"] = total_kld[0]

            roi_hat = comm_outputs["roi_hat"]
            z_comm_pool = comm_outputs["z_comm_pool"]
            z_target_pool = comm_outputs["z_target_pool"]

            if roi_hat is None:
                roi_loss = torch.zeros((), device=qpos.device)
                recon_grad_loss = torch.zeros((), device=qpos.device)
                masked_region_loss = torch.zeros((), device=qpos.device)
            loss_dict["roi"] = roi_loss
            loss_dict["recon_grad"] = recon_grad_loss
            loss_dict["roi_focus"] = masked_region_loss

            if z_comm_pool is None or z_target_pool is None:
                sem_loss = torch.zeros((), device=qpos.device)
            else:
                sem_loss = F.mse_loss(F.normalize(z_comm_pool, dim=-1), F.normalize(z_target_pool, dim=-1))
            loss_dict["sem"] = sem_loss

            sig_loss = self._sig_reg(z_comm_pool) if self.lambda_sig > 0 else torch.zeros((), device=qpos.device)
            loss_dict["sig"] = sig_loss

            future_rgb_hat = comm_outputs["future_rgb_hat"]
            future_z_hat = comm_outputs["future_z_hat"]
            future_z_target = comm_outputs["future_z_target"]
            if future_rgb_hat is None or future_rgb is None or future_valid is None:
                future_rgb_loss = torch.zeros((), device=qpos.device)
                future_grad_loss = torch.zeros((), device=qpos.device)
                future_latent_loss = torch.zeros((), device=qpos.device)
            else:
                future_weight = future_roi_weight_mask
                future_rgb_loss = self._decayed_l1(
                    future_rgb_hat,
                    future_rgb,
                    future_valid,
                    self.model.future_horizons,
                    self.future_rgb_decay_alpha,
                    future_weight,
                )
                future_grad_loss = self._decayed_gradient_l1(
                    future_rgb_hat,
                    future_rgb,
                    future_valid,
                    self.model.future_horizons,
                    self.future_rgb_decay_alpha,
                    future_weight,
                )
                if future_z_hat is None or future_z_target is None:
                    future_latent_loss = torch.zeros((), device=qpos.device)
                else:
                    future_latent_loss = self._decayed_latent_mse(
                        future_z_hat,
                        future_z_target,
                        future_valid,
                        self.model.future_horizons,
                        self.future_latent_decay_alpha,
                    )
                for horizon_idx, horizon in enumerate(self.model.future_horizons):
                    valid_i = future_valid[:, horizon_idx]
                    if valid_i.any():
                        pred_i = future_rgb_hat[:, horizon_idx][valid_i]
                        target_i = future_rgb[:, horizon_idx][valid_i]
                        loss_dict[f"future_rgb_h{horizon}"] = F.l1_loss(pred_i, target_i)
                        pred_dx = pred_i[:, :, :, 1:] - pred_i[:, :, :, :-1]
                        target_dx = target_i[:, :, :, 1:] - target_i[:, :, :, :-1]
                        pred_dy = pred_i[:, :, 1:, :] - pred_i[:, :, :-1, :]
                        target_dy = target_i[:, :, 1:, :] - target_i[:, :, :-1, :]
                        loss_dict[f"future_grad_h{horizon}"] = F.l1_loss(pred_dx, target_dx) + F.l1_loss(
                            pred_dy,
                            target_dy,
                        )
                        if future_z_hat is not None and future_z_target is not None:
                            loss_dict[f"future_latent_h{horizon}"] = F.mse_loss(
                                F.normalize(future_z_hat[:, horizon_idx][valid_i], dim=-1),
                                F.normalize(future_z_target[:, horizon_idx][valid_i], dim=-1),
                            )
                        else:
                            loss_dict[f"future_latent_h{horizon}"] = torch.zeros((), device=qpos.device)
                    else:
                        loss_dict[f"future_rgb_h{horizon}"] = torch.zeros((), device=qpos.device)
                        loss_dict[f"future_grad_h{horizon}"] = torch.zeros((), device=qpos.device)
                        loss_dict[f"future_latent_h{horizon}"] = torch.zeros((), device=qpos.device)
            loss_dict["future_rgb"] = future_rgb_loss
            loss_dict["future_grad"] = future_grad_loss
            loss_dict["future_latent"] = future_latent_loss
            loss_dict["future_pred_action_ratio"] = torch.tensor(
                float(use_predicted_future_actions),
                device=qpos.device,
            )
            loss_dict["vq"] = comm_outputs["vq_loss"]
            loss_dict["vq_commit"] = comm_outputs["vq_commitment_loss"]
            loss_dict["vq_perplexity"] = comm_outputs["vq_perplexity"]
            if self.training and use_rq:
                self._accumulate_rq_usage(comm_outputs.get("vq_codes"), num_active_rq_stages)
            loss_dict["rq_active_stages"] = torch.tensor(
                float(num_active_rq_stages if num_active_rq_stages is not None else
                      self.model.memory_quantizer.num_stages),
                device=qpos.device,
            )
            stage_perps = comm_outputs.get("vq_stage_perplexities")
            if stage_perps is not None:
                for i, p in enumerate(stage_perps):
                    loss_dict[f"vq_perplexity_stage_{i + 1}"] = p
            else:
                for i in range(self.model.memory_quantizer.num_stages):
                    loss_dict[f"vq_perplexity_stage_{i + 1}"] = torch.zeros((), device=qpos.device)
            if is_act_stage:
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            elif is_future_stage:
                loss_dict["l1"] = torch.zeros((), device=qpos.device)
                loss_dict["kl"] = torch.zeros((), device=qpos.device)
                loss_dict["loss"] = (
                    loss_dict["future_rgb"] * self.lambda_future_rgb
                    + loss_dict["future_grad"] * self.lambda_future_grad
                    + loss_dict["future_latent"] * self.lambda_future_latent
                )
            elif is_roi_stage:
                loss_dict["sem"] = torch.zeros((), device=qpos.device)
                loss_dict["sig"] = torch.zeros((), device=qpos.device)
                loss_dict["loss"] = (
                    loss_dict["roi"] * self.lambda_roi
                    + loss_dict["recon_grad"] * self.lambda_recon_grad
                    + loss_dict["roi_focus"] * self.lambda_masked_region
                )
            else:
                loss_dict["loss"] = (
                    loss_dict["l1"]
                    + loss_dict["kl"] * self.kl_weight
                    + loss_dict["future_rgb"] * self.lambda_future_rgb
                    + loss_dict["future_grad"] * self.lambda_future_grad
                    + loss_dict["future_latent"] * self.lambda_future_latent
                )
            if self._use_quantized_memory() and not is_act_stage:
                loss_dict["loss"] = (
                    loss_dict["loss"]
                    + loss_dict["vq"] * self.lambda_vq
                    + loss_dict["vq_commit"] * self.lambda_vq_commit
                )
            if future_roi_weight_mask is not None:
                loss_dict["future_roi_fg"] = (
                    future_roi_weight_mask > future_roi_weight_mask.amin(dim=(-2, -1), keepdim=True)
                ).float().mean()
            if future_focus_weight_mask is not None:
                loss_dict["future_focus_fg"] = (future_focus_weight_mask > 0).float().mean()
            return loss_dict

        a_hat, _, (_, _), attn_weights, comm_outputs = self.model(
            qpos,
            image,
            env_state,
            use_quantized_memory=True,
        )
        if return_comm:
            return a_hat, attn_weights, comm_outputs
        return a_hat, attn_weights

    def update_ema(self):
        if self.train_stage not in {"joint", "future"}:
            return
        self.model.update_ema(self.ema_momentum)
        self._ema_updates += 1
        if self.train_stage == "future":
            self._future_updates += 1
        self._rq_steps += 1

    def should_restart_dead_codes(self):
        return (
            self.train_stage in {"future", "joint"}
            and self.rq_dead_restart_interval > 0
            and self._rq_steps > 0
            and self._rq_steps % self.rq_dead_restart_interval == 0
        )

    @torch.no_grad()
    def maybe_restart_dead_codes(self, image_data, qpos_data, device):
        """Reinitialize codes with low usage over the accumulated RQ window."""
        rq = self.model.memory_quantizer
        self._ensure_rq_usage_buffers(device)
        was_training = self.training
        self.eval()

        image_data = image_data.to(device)
        qpos_data = qpos_data.to(device)
        vectors = self.model.collect_code_inputs(image_data, qpos_data)  # (B*32, D)

        # Compute per-stage residuals and restart dead codes
        residual = vectors.clone()
        for i, stage in enumerate(rq.stages):
            flat = residual
            c_sq = stage.embedding.weight.pow(2).sum(dim=1)
            chunk = 4096
            assignments = torch.empty(flat.shape[0], dtype=torch.long, device=device)
            for start in range(0, flat.shape[0], chunk):
                v = flat[start : start + chunk]
                dists = v.pow(2).sum(1, keepdim=True) - 2.0 * (v @ stage.embedding.weight.t()) + c_sq
                assignments[start : start + chunk] = dists.argmin(dim=1)

            window_counts = self._rq_usage_counts[i]
            window_total = float(self._rq_usage_totals[i].item())
            if window_total <= 0:
                active = int((window_counts > 0).sum())
                print(f"  RQ stage {i + 1}: skipped dead-code restart; no accumulated assignments "
                      f"(active_window: {active}/{stage.codebook_bins})")
            else:
                expected = window_total / stage.codebook_bins
                dead_threshold = self.rq_dead_restart_threshold * expected
                dead_indices = torch.nonzero(window_counts < dead_threshold, as_tuple=False).flatten()
                n_dead = int(dead_indices.numel())
                max_fraction = self.rq_dead_restart_max_fraction
                if max_fraction <= 0:
                    max_restart = 0
                elif max_fraction >= 1:
                    max_restart = stage.codebook_bins
                else:
                    max_restart = max(1, int(stage.codebook_bins * max_fraction))
                n_restart = min(n_dead, max_restart)

                if n_restart > 0:
                    if n_dead > n_restart:
                        candidate_counts = window_counts[dead_indices]
                        selected = torch.topk(candidate_counts, k=n_restart, largest=False).indices
                        restart_indices = dead_indices[selected]
                    else:
                        restart_indices = dead_indices
                    rand_idx = torch.randint(0, flat.shape[0], (n_restart,), device=device)
                    stage.embedding.weight.data[restart_indices] = flat[rand_idx]
                active = int((window_counts > 0).sum())
                print(
                    f"  RQ stage {i + 1}: window_assignments={int(window_total)} "
                    f"expected={expected:.2f} threshold={dead_threshold:.2f} "
                    f"dead={n_dead}/{stage.codebook_bins} restarted={n_restart} "
                    f"active_window={active}/{stage.codebook_bins}"
                )

            # Advance residual for next stage
            quantized = stage.embedding(assignments)
            residual = flat - quantized

        self._reset_rq_usage_buffers()
        if was_training:
            self.train()

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}
        return self.model(qpos, image, env_state)

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
