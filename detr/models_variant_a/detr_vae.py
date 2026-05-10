# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Variant A ACT model with a decoupled communication branch.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from ..models.backbone import build_backbone
from ..models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def make_group_norm(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)


class CommCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, query, memory):
        attn_out, _ = self.cross_attn(query=query, key=memory, value=memory)
        query = self.norm1(query + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = self.norm2(query + self.dropout2(ff))
        return query


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_group_norm(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNetFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, layer_channels=(64, 128, 256, 512)):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(channels, hidden_dim, kernel_size=1) for channels in layer_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * len(layer_channels), hidden_dim, kernel_size=1),
            make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),
            ResidualConvBlock(hidden_dim),
        )

    def forward(self, features):
        target_size = features[-1].shape[-2:]
        fused_features = []
        for feature, projection in zip(features, self.proj):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            fused_features.append(feature)
        return self.fuse(torch.cat(fused_features, dim=1))


class CommDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        decoder_dims = [hidden_dim, 256, 192, 128, 96, 64]
        self.stem = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            make_group_norm(hidden_dim),
            nn.SiLU(inplace=True),
            ResidualConvBlock(hidden_dim),
            ResidualConvBlock(hidden_dim),
        )
        self.up_blocks = nn.ModuleList()
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                    make_group_norm(out_dim),
                    nn.SiLU(inplace=True),
                    ResidualConvBlock(out_dim),
                    ResidualConvBlock(out_dim),
                )
            )
        self.refine = nn.Sequential(
            ResidualConvBlock(decoder_dims[-1]),
            nn.Conv2d(decoder_dims[-1], 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.up_blocks:
            x = block(x)
        return torch.sigmoid(self.refine(x))


class DETRVAE(nn.Module):
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        target_camera,
        comm_num_queries,
        comm_layers,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.target_camera = target_camera
        self.target_camera_idx = camera_names.index(target_camera)
        self.transformer = transformer
        self.encoder = encoder
        self.hidden_dim = transformer.d_model
        self.action_head = nn.Linear(self.hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        if backbones is not None:
            self.visual_fusion = ResNetFeatureFusion(self.hidden_dim)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, self.hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(14, self.hidden_dim)
            self.input_proj_env_state = nn.Linear(7, self.hidden_dim)
            self.pos = torch.nn.Embedding(2, self.hidden_dim)
            self.backbones = None

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, self.hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, self.hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, self.hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, self.hidden_dim)

        self.comm_query_embed = nn.Embedding(comm_num_queries, self.hidden_dim)
        self.comm_task_token = nn.Embedding(1, self.hidden_dim)
        self.comm_bandwidth_token = nn.Embedding(1, self.hidden_dim)
        self.comm_layers = nn.ModuleList(
            [
                CommCrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=transformer.nhead,
                    dim_feedforward=self.hidden_dim * 2,
                )
                for _ in range(comm_layers)
            ]
        )
        self.comm_pool_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.comm_film = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.comm_decoder = CommDecoder(self.hidden_dim)

        self.ema_backbone = copy.deepcopy(self.backbones[self.target_camera_idx])
        self.ema_visual_fusion = copy.deepcopy(self.visual_fusion)
        self.ema_projector = copy.deepcopy(self.comm_pool_proj)
        self._freeze_ema()

    def _freeze_ema(self):
        for module in [self.ema_backbone, self.ema_visual_fusion, self.ema_projector]:
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, momentum):
        online_modules = [
            self.backbones[self.target_camera_idx],
            self.visual_fusion,
            self.comm_pool_proj,
        ]
        ema_modules = [self.ema_backbone, self.ema_visual_fusion, self.ema_projector]
        for online_module, ema_module in zip(online_modules, ema_modules):
            for ema_param, online_param in zip(ema_module.parameters(), online_module.parameters()):
                ema_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def _encode_action_latent(self, qpos, actions, is_pad):
        bs = qpos.shape[0]
        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1).permute(1, 0, 2)
        cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
        encoder_is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=encoder_is_pad)[0]
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)
        return mu, logvar, latent_input

    def _encode_visual_tokens(self, qpos, image, latent_input):
        bs, _, _, _, _ = image.shape
        all_cam_features = []
        all_cam_pos = []
        feat_h = feat_w = None
        for cam_id, _ in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = self.visual_fusion(features)
            pos = pos[-1]
            feat_h, feat_w = features.shape[-2:]
            all_cam_features.append(features)
            all_cam_pos.append(pos)

        proprio_input = self.input_proj_robot_state(qpos)
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        pos_full = torch.cat([additional_pos_embed, pos_flat], axis=0)
        src_full = torch.cat([torch.stack([latent_input, proprio_input], axis=0), src_flat], axis=0)

        memory = self.transformer.encoder(src_full, src_key_padding_mask=None, pos=pos_full)
        hs, attn_weights = self.transformer.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_full,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs, attn_weights, memory, feat_h, feat_w

    def _run_comm_branch(self, memory, feat_h, feat_w, detach_comm=False):
        bs = memory.shape[1]
        memory_bsd = memory.permute(1, 0, 2)
        memory_prop = memory_bsd[:, 1:2, :]
        memory_vis = memory_bsd[:, 2:, :]
        if detach_comm:
            memory_prop = memory_prop.detach()
            memory_vis = memory_vis.detach()

        cond = (
            memory_prop.squeeze(1)
            + self.comm_task_token.weight[0].unsqueeze(0)
            + self.comm_bandwidth_token.weight[0].unsqueeze(0)
        )
        comm_query = self.comm_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        comm_query = comm_query + cond.unsqueeze(0)
        memory_vis_t = memory_vis.permute(1, 0, 2)

        for layer in self.comm_layers:
            comm_query = layer(comm_query, memory_vis_t)

        z_comm = comm_query.transpose(0, 1)
        z_comm_pool = self.comm_pool_proj(z_comm.mean(dim=1))

        tokens_per_camera = feat_h * feat_w
        start_idx = self.target_camera_idx * tokens_per_camera
        end_idx = start_idx + tokens_per_camera
        target_tokens = memory_vis[:, start_idx:end_idx, :]
        target_grid = target_tokens.transpose(1, 2).reshape(bs, self.hidden_dim, feat_h, feat_w)

        gamma, beta = self.comm_film(z_comm_pool).chunk(2, dim=-1)
        conditioned_grid = target_grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        roi_hat = self.comm_decoder(conditioned_grid)
        return roi_hat, z_comm_pool

    @torch.no_grad()
    def _encode_ema_target(self, target_image):
        features, _ = self.ema_backbone(target_image)
        target_tokens = self.ema_visual_fusion(features).flatten(2).transpose(1, 2)
        return self.ema_projector(target_tokens.mean(dim=1))

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        target_image=None,
        detach_comm=False,
    ):
        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            mu, logvar, latent_input = self._encode_action_latent(qpos, actions, is_pad)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            hs, attn_weights, memory, feat_h, feat_w = self._encode_visual_tokens(qpos, image, latent_input)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs, attn_weights = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)
            memory = None
            feat_h = feat_w = None

        hs = hs[-1]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        comm_outputs = {
            "roi_hat": None,
            "z_comm_pool": None,
            "z_target_pool": None,
        }
        if memory is not None and feat_h is not None and feat_w is not None:
            roi_hat, z_comm_pool = self._run_comm_branch(memory, feat_h, feat_w, detach_comm=detach_comm)
            comm_outputs["roi_hat"] = roi_hat
            comm_outputs["z_comm_pool"] = z_comm_pool
            if target_image is not None:
                comm_outputs["z_target_pool"] = self._encode_ema_target(target_image)

        return a_hat, is_pad_hat, [mu, logvar], attn_weights, comm_outputs


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5),
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)
            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, _ in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            all_cam_features.append(self.backbone_down_projs[cam_id](features[0]))
        flattened_features = [cam_feature.reshape([bs, -1]) for cam_feature in all_cam_features]
        features = torch.cat(flattened_features + [qpos], axis=1)
        return self.mlp(features)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)


def build(args):
    state_dim = 14
    backbones = []
    original_masks = getattr(args, "masks", False)
    args.masks = True
    for _ in args.camera_names:
        backbones.append(build_backbone(args))
    args.masks = original_masks

    transformer = build_transformer(args)
    encoder = build_encoder(args)
    target_camera = getattr(args, "target_camera", None)
    if target_camera is None:
        target_camera = "top" if "top" in args.camera_names else args.camera_names[0]

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        target_camera=target_camera,
        comm_num_queries=getattr(args, "comm_num_queries", 8),
        comm_layers=getattr(args, "comm_layers", 2),
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def build_cnnmlp(args):
    state_dim = 14
    backbones = []
    for _ in args.camera_names:
        backbones.append(build_backbone(args))

    model = CNNMLP(backbones, state_dim=state_dim, camera_names=args.camera_names)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model
