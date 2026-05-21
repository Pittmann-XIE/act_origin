import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import get_args_parser, build_CNNMLP_model_and_optimizer
from detr.models_variant_a.detr_vae import build as build_variant_a_model


def build_variant_a_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser("Variant A DETR training", parents=[get_args_parser()])
    args, _ = parser.parse_known_args()
    for key, value in args_override.items():
        setattr(args, key, value)
    print("args: ", args)

    model = build_variant_a_model(args)
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
        model, optimizer = build_variant_a_model_and_optimizer(args_override)
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
        self.focus_masked_region = args_override.get("focus_masked_region", False)
        self.lambda_masked_region = 1.0
        self.ema_momentum = args_override.get("ema_momentum", 0.99)
        self.comm_detach_warmup = args_override.get("comm_detach_warmup", 0)
        self._ema_updates = 0
        self.train_stage = "joint"
        print(
            f"KL Weight {self.kl_weight}, lambda_roi {self.lambda_roi}, "
            f"lambda_sem {self.lambda_sem}, lambda_sig {self.lambda_sig}, "
            f"lambda_recon_grad {self.lambda_recon_grad}, "
            f"focus_masked_region {self.focus_masked_region}"
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
    def _is_ema_parameter(name):
        return name.startswith("ema_")

    def set_train_stage(self, stage):
        if stage not in {"joint", "act", "roi"}:
            raise ValueError(f"Unknown train stage {stage!r}; expected joint, act, or roi.")
        self.train_stage = stage
        for name, param in self.model.named_parameters():
            if self._is_ema_parameter(name):
                param.requires_grad = False
            elif stage == "joint":
                param.requires_grad = True
            elif stage == "act":
                param.requires_grad = not self._is_roi_parameter(name)
            elif stage == "roi":
                param.requires_grad = self._is_roi_parameter(name)
        self.optimizer = build_optimizer_for_model(self.model, self.lr, self.lr_backbone, self.weight_decay)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Set ACTPolicy train stage to {stage}; trainable parameters: {trainable / 1e6:.2f}M")
        return self

    def _should_detach_comm(self):
        warmup = self.comm_detach_warmup
        if isinstance(warmup, bool):
            return warmup
        return self._ema_updates < int(warmup)

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

    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        target_rgb=None,
        roi_weight_mask=None,
        focus_weight_mask=None,
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
            target_image = None
            if self.train_stage == "joint" and target_rgb is not None:
                target_image = normalize(target_rgb)
            a_hat, is_pad_hat, (mu, logvar), _, comm_outputs = self.model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                target_image=target_image,
                detach_comm=self._should_detach_comm(),
                force_zero_latent=is_roi_stage,
                run_comm=not is_act_stage,
                encode_target_semantics=self.train_stage == "joint",
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

            if roi_hat is None or target_rgb is None or roi_weight_mask is None:
                roi_loss = torch.zeros((), device=qpos.device)
                recon_grad_loss = torch.zeros((), device=qpos.device)
                masked_region_loss = torch.zeros((), device=qpos.device)
            else:
                if roi_hat.shape[-2:] != target_rgb.shape[-2:]:
                    roi_hat = F.interpolate(roi_hat, size=target_rgb.shape[-2:], mode="bilinear", align_corners=False)
                roi_weight = roi_weight_mask.unsqueeze(1)
                roi_loss = self._weighted_l1(roi_hat, target_rgb, roi_weight)
                recon_grad_loss = self._weighted_gradient_l1(roi_hat, target_rgb, roi_weight)
                if (
                    self.focus_masked_region
                    and focus_weight_mask is not None
                    and focus_weight_mask.sum().item() > 0
                ):
                    masked_region_loss = self._weighted_l1(roi_hat, target_rgb, focus_weight_mask.unsqueeze(1))
                else:
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
            if is_act_stage:
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
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
                    + loss_dict["roi"] * self.lambda_roi
                    + loss_dict["recon_grad"] * self.lambda_recon_grad
                    + loss_dict["roi_focus"] * self.lambda_masked_region
                    + loss_dict["sem"] * self.lambda_sem
                    + loss_dict["sig"] * self.lambda_sig
                )
            if roi_weight_mask is not None:
                loss_dict["roi_fg"] = (roi_weight_mask > roi_weight_mask.amin(dim=(-2, -1), keepdim=True)).float().mean()
            if focus_weight_mask is not None:
                loss_dict["focus_fg"] = (focus_weight_mask > 0).float().mean()
            return loss_dict

        a_hat, _, (_, _), attn_weights, comm_outputs = self.model(qpos, image, env_state)
        if return_comm:
            return a_hat, attn_weights, comm_outputs
        return a_hat, attn_weights

    def update_ema(self):
        if self.train_stage != "joint":
            return
        self.model.update_ema(self.ema_momentum)
        self._ema_updates += 1

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
