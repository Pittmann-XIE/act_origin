# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Variant C ACT model with VQ-compressed encoder memory.
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
    def __init__(self, hidden_dim, num_upsample_blocks=5):
        super().__init__()
        decoder_dims = [hidden_dim, 256, 192, 128, 96, 64][: num_upsample_blocks + 1]
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


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_bins, codebook_dim):
        super().__init__()
        self.codebook_bins = int(codebook_bins)
        self.codebook_dim = int(codebook_dim)
        self.embedding = nn.Embedding(self.codebook_bins, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_bins, 1.0 / self.codebook_bins)

    def forward(self, x):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_dim)
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        codes = torch.argmin(distances, dim=1)
        quantized = self.embedding(codes)

        codebook_loss = F.mse_loss(quantized, flat_x.detach())
        commitment_loss = F.mse_loss(flat_x, quantized.detach())
        quantized = flat_x + (quantized - flat_x).detach()

        encodings = F.one_hot(codes, self.codebook_bins).type(flat_x.dtype)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized.reshape(original_shape),
            "codes": codes.reshape(original_shape[:-1]),
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }

    def decode(self, codes):
        return self.embedding(codes)


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
        future_horizons,
        future_image_height,
        future_image_width,
        future_layers,
        codebook_bins,
        codebook_dim,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.target_camera = target_camera
        self.target_camera_idx = camera_names.index(target_camera)
        self.future_horizons = tuple(int(horizon) for horizon in future_horizons)
        self.num_future_frames = len(self.future_horizons)
        self.future_image_height = int(future_image_height)
        self.future_image_width = int(future_image_width)
        self.transformer = transformer
        self.encoder = encoder
        self.hidden_dim = transformer.d_model
        self.codebook_bins = int(codebook_bins)
        self.codebook_dim = self.hidden_dim if codebook_dim is None else int(codebook_dim)
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
        self.memory_to_code = (
            nn.Identity() if self.codebook_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.codebook_dim)
        )
        self.code_to_memory = (
            nn.Identity() if self.codebook_dim == self.hidden_dim else nn.Linear(self.codebook_dim, self.hidden_dim)
        )
        self.memory_quantizer = VectorQuantizer(self.codebook_bins, self.codebook_dim)
        self._cached_pos_full = None

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

        self.future_action_proj = nn.Linear(state_dim, self.hidden_dim)
        self.future_action_pos_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.future_frame_query_embed = nn.Embedding(self.num_future_frames, self.hidden_dim)
        self.future_horizon_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.future_action_layers = nn.ModuleList(
            [
                CommCrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=transformer.nhead,
                    dim_feedforward=self.hidden_dim * 2,
                )
                for _ in range(future_layers)
            ]
        )
        self.future_memory_layers = nn.ModuleList(
            [
                CommCrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=transformer.nhead,
                    dim_feedforward=self.hidden_dim * 2,
                )
                for _ in range(future_layers)
            ]
        )
        self.future_pool_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.future_film = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.future_decoder = CommDecoder(self.hidden_dim, num_upsample_blocks=4)

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

    def _empty_vq_outputs(self, reference):
        scalar = torch.zeros((), dtype=reference.dtype, device=reference.device)
        return {
            "codes": None,
            "codebook_loss": scalar,
            "commitment_loss": scalar,
            "perplexity": scalar,
        }

    def _quantize_memory(self, memory):
        code_input = self.memory_to_code(memory)
        vq_result = self.memory_quantizer(code_input)
        quantized_memory = self.code_to_memory(vq_result["quantized"])
        return quantized_memory, {
            "codes": vq_result["codes"],
            "codebook_loss": vq_result["codebook_loss"],
            "commitment_loss": vq_result["commitment_loss"],
            "perplexity": vq_result["perplexity"],
        }

    def _decode_memory_codes(self, codes):
        code_vectors = self.memory_quantizer.decode(codes)
        return self.code_to_memory(code_vectors)

    def _decode_from_memory(self, memory, pos_full):
        bs = memory.shape[1]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs, attn_weights = self.transformer.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_full,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), attn_weights

    def _encode_visual_tokens(self, qpos, image, latent_input, use_quantized_memory=True):
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
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        pos_full = torch.cat([additional_pos_embed, pos_flat], axis=0)
        src_full = torch.cat([torch.stack([latent_input, proprio_input], axis=0), src_flat], axis=0)

        memory = self.transformer.encoder(src_full, src_key_padding_mask=None, pos=pos_full)
        quantized_memory, vq_outputs = self._quantize_memory(memory)
        memory_for_decode = quantized_memory if use_quantized_memory else memory
        hs, attn_weights = self._decode_from_memory(memory_for_decode, pos_full)
        return hs, attn_weights, memory, memory_for_decode, vq_outputs, pos_full, feat_h, feat_w

    def encode_memory_codes(self, qpos, image, actions=None, is_pad=None):
        if actions is not None and is_pad is not None:
            _, _, latent_input = self._encode_action_latent(qpos, actions, is_pad)
        else:
            bs = qpos.shape[0]
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        _, _, _, _, vq_outputs, pos_full, feat_h, feat_w = self._encode_visual_tokens(
            qpos,
            image,
            latent_input,
            use_quantized_memory=True,
        )
        self._cached_pos_full = pos_full.detach()
        return vq_outputs["codes"], feat_h, feat_w

    def decode_from_memory_codes(self, codes, feat_h, feat_w, batch_size, future_actions=None):
        memory = self._decode_memory_codes(codes)
        if memory.shape[1] != batch_size:
            raise ValueError(f"codes batch size {memory.shape[1]} does not match batch_size {batch_size}")
        pos_full = self._cached_pos_full
        if pos_full is None or pos_full.shape[:2] != memory.shape[:2]:
            raise ValueError("decode_from_memory_codes requires cached pos_full from encode_memory_codes first.")

        hs, attn_weights = self._decode_from_memory(memory, pos_full.to(device=memory.device, dtype=memory.dtype))
        hs = hs[-1]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        comm_outputs = self._empty_comm_outputs(memory)
        if future_actions is not None:
            future_rgb_hat, future_z_hat = self._run_future_branch(memory, feat_h, feat_w, future_actions)
            comm_outputs["future_rgb_hat"] = future_rgb_hat
            comm_outputs["future_z_hat"] = future_z_hat
        return a_hat, is_pad_hat, attn_weights, comm_outputs

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

    @torch.no_grad()
    def _encode_ema_future_targets(self, future_images):
        bs, num_frames, channels, height, width = future_images.shape
        flat_images = future_images.reshape(bs * num_frames, channels, height, width)
        flat_targets = self._encode_ema_target(flat_images)
        return flat_targets.reshape(bs, num_frames, self.hidden_dim)

    def _run_future_branch(self, memory, feat_h, feat_w, action_sequence, detach_actions=False):
        bs = memory.shape[1]
        if detach_actions:
            action_sequence = action_sequence.detach()

        memory_bsd = memory.permute(1, 0, 2)
        memory_prop = memory_bsd[:, 1:2, :]
        memory_vis = memory_bsd[:, 2:, :]

        num_actions = min(action_sequence.shape[1], self.num_queries)
        action_tokens = self.future_action_proj(action_sequence[:, :num_actions])
        action_pos = self.future_action_pos_embed.weight[:num_actions].unsqueeze(0)
        action_tokens = (action_tokens + action_pos).permute(1, 0, 2)

        horizon_tensor = torch.tensor(
            self.future_horizons,
            dtype=memory.dtype,
            device=memory.device,
        ).view(self.num_future_frames, 1)
        horizon_tensor = horizon_tensor / max(1, self.num_queries - 1)
        horizon_embed = self.future_horizon_proj(horizon_tensor).unsqueeze(1)
        frame_query = self.future_frame_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        frame_query = frame_query + horizon_embed + memory_prop.permute(1, 0, 2)

        for action_layer, memory_layer in zip(self.future_action_layers, self.future_memory_layers):
            frame_query = action_layer(frame_query, action_tokens)
            frame_query = memory_layer(frame_query, memory)

        future_tokens = frame_query.transpose(0, 1)
        future_z = self.future_pool_proj(future_tokens)

        tokens_per_camera = feat_h * feat_w
        start_idx = self.target_camera_idx * tokens_per_camera
        end_idx = start_idx + tokens_per_camera
        target_tokens = memory_vis[:, start_idx:end_idx, :]
        target_grid = target_tokens.transpose(1, 2).reshape(bs, self.hidden_dim, feat_h, feat_w)

        gamma, beta = self.future_film(future_z).chunk(2, dim=-1)
        base_grid = target_grid.unsqueeze(1).expand(-1, self.num_future_frames, -1, -1, -1)
        conditioned_grid = base_grid * (1.0 + gamma[:, :, :, None, None]) + beta[:, :, :, None, None]
        conditioned_grid = conditioned_grid.reshape(bs * self.num_future_frames, self.hidden_dim, feat_h, feat_w)
        future_rgb = self.future_decoder(conditioned_grid)
        future_rgb = F.interpolate(
            future_rgb,
            size=(self.future_image_height, self.future_image_width),
            mode="bilinear",
            align_corners=False,
        )
        future_rgb = future_rgb.reshape(
            bs,
            self.num_future_frames,
            3,
            self.future_image_height,
            self.future_image_width,
        )
        return future_rgb, future_z

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        target_image=None,
        future_images=None,
        future_actions=None,
        detach_comm=False,
        detach_future_actions=False,
        force_zero_latent=False,
        run_comm=True,
        run_future=True,
        encode_target_semantics=True,
        use_quantized_memory=True,
    ):
        is_training = actions is not None and not force_zero_latent
        bs, _ = qpos.shape

        if is_training:
            mu, logvar, latent_input = self._encode_action_latent(qpos, actions, is_pad)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            hs, attn_weights, memory, memory_for_decode, vq_outputs, _, feat_h, feat_w = self._encode_visual_tokens(
                qpos,
                image,
                latent_input,
                use_quantized_memory=use_quantized_memory,
            )
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs, attn_weights = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)
            memory = None
            memory_for_decode = None
            feat_h = feat_w = None
            vq_outputs = self._empty_vq_outputs(qpos)

        hs = hs[-1]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        comm_outputs = self._empty_comm_outputs(qpos)
        comm_outputs.update(
            {
                "vq_codes": vq_outputs["codes"],
                "vq_loss": vq_outputs["codebook_loss"],
                "vq_commitment_loss": vq_outputs["commitment_loss"],
                "vq_perplexity": vq_outputs["perplexity"],
            }
        )
        if run_comm and memory_for_decode is not None and feat_h is not None and feat_w is not None:
            roi_hat, z_comm_pool = self._run_comm_branch(memory_for_decode, feat_h, feat_w, detach_comm=detach_comm)
            comm_outputs["roi_hat"] = roi_hat
            comm_outputs["z_comm_pool"] = z_comm_pool
            if encode_target_semantics and target_image is not None:
                comm_outputs["z_target_pool"] = self._encode_ema_target(target_image)

        if run_future and memory_for_decode is not None and feat_h is not None and feat_w is not None:
            action_condition = a_hat if future_actions is None else future_actions
            future_rgb_hat, future_z_hat = self._run_future_branch(
                memory_for_decode,
                feat_h,
                feat_w,
                action_condition,
                detach_actions=detach_future_actions,
            )
            comm_outputs["future_rgb_hat"] = future_rgb_hat
            comm_outputs["future_z_hat"] = future_z_hat
            if encode_target_semantics and future_images is not None:
                comm_outputs["future_z_target"] = self._encode_ema_future_targets(future_images)

        return a_hat, is_pad_hat, [mu, logvar], attn_weights, comm_outputs

    def _empty_comm_outputs(self, reference):
        scalar = torch.zeros((), dtype=reference.dtype, device=reference.device)
        return {
            "roi_hat": None,
            "z_comm_pool": None,
            "z_target_pool": None,
            "future_rgb_hat": None,
            "future_z_hat": None,
            "future_z_target": None,
            "vq_codes": None,
            "vq_loss": scalar,
            "vq_commitment_loss": scalar,
            "vq_perplexity": scalar,
        }


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
        future_horizons=getattr(args, "future_horizons", (0, 5, 15, 30, 60, 99)),
        future_image_height=getattr(args, "future_image_height", 240),
        future_image_width=getattr(args, "future_image_width", 320),
        future_layers=getattr(args, "future_layers", 2),
        codebook_bins=getattr(args, "codebook_bins", 1024),
        codebook_dim=getattr(args, "codebook_dim", None),
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
