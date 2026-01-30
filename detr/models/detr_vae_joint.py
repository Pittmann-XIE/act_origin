# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed

import os
import cv2
import torchvision.transforms as T


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        super().__init__()
        self.num_queries = num_queries
        print(f'deter_vae_joint/num_queries: {self.num_queries}')
        self.camera_names = camera_names
        print(f'deter_vae_joint/camera_names: {self.camera_names}')
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        
        # Separate heads for different action components
        self.action_head_joint = nn.Linear(hidden_dim, 6)  # j0,j1,j2,j3,j4,j5
        self.action_head_distance = nn.Linear(hidden_dim, 1)  # distance_class (will apply sigmoid)
        
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            #===changed for keypoint===
            self.num_keypoints = 64 # You can adjust this (16, 32, 64)
            # Reduce 512 channels to 32 keypoint heatmaps
            self.keypoint_proj = nn.Conv2d(backbones[0].num_channels, self.num_keypoints, kernel_size=1)
            # Based on 480x640 input, ResNet output is 15x20
            self.spatial_softmax = SpatialSoftmax(h=15, w=20, temperature=0.1)
            # Project the (x,y) keypoints into the transformer hidden_dim
            self.keypoint_to_embed = nn.Linear(2, hidden_dim)
            # == end changed for keypoint ===
            self.input_proj_robot_state = nn.Linear(7, hidden_dim)
            print('backbones not none')
        else:
            self.input_proj_robot_state = nn.Linear(7, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
            print('backbones is none')
    
        # encoder extra parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(7, hidden_dim)
        self.encoder_joint_proj = nn.Linear(7, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    # def forward(self, qpos, image, env_state, actions=None, is_pad=None):
    #     """
    #     qpos: batch, qpos_dim
    #     image: batch, num_cam, channel, height, width
    #     env_state: None
    #     actions: batch, seq, action_dim
    #     """
    #     is_training = actions is not None
    #     bs, _ = qpos.shape
        
    #     ### Obtain latent z from action sequence
    #     if is_training:
    #         action_embed = self.encoder_action_proj(actions)
    #         qpos_embed = self.encoder_joint_proj(qpos)
    #         qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
    #         cls_embed = self.cls_embed.weight
    #         cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
    #         encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
    #         encoder_input = encoder_input.permute(1, 0, 2)
    #         cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
    #         is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
    #         pos_embed = self.pos_table.clone().detach()
    #         pos_embed = pos_embed.permute(1, 0, 2)
    #         encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
    #         encoder_output = encoder_output[0]
    #         latent_info = self.latent_proj(encoder_output)
    #         mu = latent_info[:, :self.latent_dim]
    #         logvar = latent_info[:, self.latent_dim:]
    #         latent_sample = reparametrize(mu, logvar)
    #         latent_input = self.latent_out_proj(latent_sample) 
    #     else:
    #         mu = logvar = None
    #         latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
    #         latent_input = self.latent_out_proj(latent_sample)

    #     if self.backbones is not None:
    #         all_cam_features = []
    #         all_cam_pos = []
            
    #         # 1. Decide which cameras to drop for this batch (Training only)
    #         dropped_indices = []
    #         if self.training:
    #             for i in range(len(self.camera_names)):
    #                 if torch.rand(1) < 0.25: # 25% chance per camera
    #                     dropped_indices.append(i)
                
    #             # Safety check: If we accidentally dropped ALL cameras, 
    #             # keep one random one so the model isn't blind.
    #             if len(dropped_indices) == len(self.camera_names):
    #                 keep_idx = torch.randint(0, len(self.camera_names), (1,)).item()
    #                 dropped_indices.remove(keep_idx)

    #         # # 2. Process backbones
    #         # for cam_id, cam_name in enumerate(self.camera_names):
    #         #     features, pos = self.backbones[cam_id](image[:, cam_id])
    #         #     features = features[0]
    #         #     pos = pos[0]
    #         #     proj_feat = self.input_proj(features)
                
    #         #     # Apply the drop
    #         #     if cam_id in dropped_indices:
    #         #         proj_feat = torch.zeros_like(proj_feat)
                
    #         #     all_cam_features.append(proj_feat)
    #         #     all_cam_pos.append(pos)
                
    #         # proprio_input = self.input_proj_robot_state(qpos)
    #         # src = torch.cat(all_cam_features, axis=3)
    #         # pos = torch.cat(all_cam_pos, axis=3)
    #         # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[-1]
    #         #===changed for keypoint===
    #         for cam_id, cam_name in enumerate(self.camera_names):
    #             features, pos = self.backbones[cam_id](image[:, cam_id])
    #             features = features[0] # [B, 512, 15, 20]
                
    #             # --- SPATIAL SOFTMAX BOTTLENECK ---
    #             # 1. Get heatmaps
    #             heatmaps = self.keypoint_proj(features) # [B, 32, 15, 20]
                
    #             # 2. Apply dropout to vision here (zeroing heatmaps)
    #             if self.training and cam_id in dropped_indices:
    #                 heatmaps = torch.zeros_like(heatmaps)
                
    #             # 3. Get [x, y] coordinates
    #             keypoints = self.spatial_softmax(heatmaps) # [B, 32, 2]
                
    #             if not self.training:
    #                 # If this value is near 0, all 32 dots are stuck in the same place!
    #                 print(f"Keypoint Variance: {keypoints.var(dim=0).mean().item():.6f}")
                
    #             # 4. Convert keypoints to transformer tokens
    #             # [B, 32, hidden_dim]
    #             cam_feat_tokens = self.keypoint_to_embed(keypoints)
    #             # ----------------------------------
                
    #             all_cam_features.append(cam_feat_tokens)
                
    #         # Combine all cameras
    #         # Since tokens are already [B, N, C], we cat on the sequence dimension (1)
    #         src = torch.cat(all_cam_features, axis=1) # [B, num_cam * 32, hidden_dim]
            
    #         # Since keypoints carry their own spatial info, 
    #         # we can pass None or a zeroed pos to the transformer
    #         pos = torch.zeros_like(src) 
            
    #         proprio_input = self.input_proj_robot_state(qpos)
            
    #         # Note: You might need to adjust transformer.py slightly 
    #         # because 'src' is now 3D [B, Seq, Dim] instead of 4D [B, C, H, W]
    #         hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[-1]
    #     else:
    #         qpos = self.input_proj_robot_state(qpos)
    #         env_state = self.input_proj_env_state(env_state)
    #         transformer_input = torch.cat([qpos, env_state], axis=1)
    #         hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[-1]
        
    #     # Separate outputs for different components
    #     a_hat_joint = self.action_head_joint(hs)  # (batch, num_queries, 6)
    #     a_hat_distance = torch.sigmoid(self.action_head_distance(hs))  # (batch, num_queries, 1) with sigmoid
        
    #     # Concatenate all components
    #     a_hat = torch.cat([a_hat_joint, a_hat_distance], dim=-1)  # (batch, num_queries, 7)
        
    #     is_pad_hat = self.is_pad_head(hs)
    #     return a_hat, is_pad_hat, [mu, logvar]

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
            is_training = actions is not None
            bs, _ = qpos.shape
            
            ### Obtain latent z logic (unchanged) ...
            if is_training:
                action_embed = self.encoder_action_proj(actions)
                qpos_embed = self.encoder_joint_proj(qpos)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
                cls_embed = self.cls_embed.weight
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
                encoder_input = encoder_input.permute(1, 0, 2)
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0]
                latent_info = self.latent_proj(encoder_output)
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]
                latent_sample = reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample) 
                if torch.rand(1) < 0.25: 
                    latent_input = torch.zeros_like(latent_input)
            else:
                mu = logvar = None
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)

            if self.backbones is not None:
                all_cam_features = []
                
                # 1. Camera Drop Logic (unchanged)
                dropped_indices = []
                if self.training:
                    for i in range(len(self.camera_names)):
                        if torch.rand(1) < 0.25: dropped_indices.append(i)
                    if len(dropped_indices) == len(self.camera_names):
                        dropped_indices.remove(torch.randint(0, len(self.camera_names), (1,)).item())

                # 2. Process backbones + Spatial Softmax
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, _ = self.backbones[cam_id](image[:, cam_id])
                    features = features[0] # [B, 512, 15, 20]
                    
                    heatmaps = self.keypoint_proj(features) # [B, 32, 15, 20]
                    if self.training and cam_id in dropped_indices:
                        heatmaps = torch.zeros_like(heatmaps)
                    
                    keypoints = self.spatial_softmax(heatmaps) # [B, 32, 2]
                    
                    # --- DIAGNOSTICS ---
                    if not self.training:
                        # 1. Check Variance: How much do the dots move across the batch?
                        # High variance = dots are tracking different things in different images.
                        # Low variance = dots are all stuck in the same pixels.
                        var = keypoints.var(dim=0).mean().item()
                        if var < 1e-5:
                            print(f"WARNING: Cam {cam_name} keypoint variance is VERY low ({var:.8f}). Backbone might be dead.")
                        
                        # 2. Plotting: Save one image per batch during eval
                        # We save the first image of the batch (index 0)
                        save_path = f"eval/debug_keypoints_{cam_name}.png"
                        self._save_debug_image(image[0, cam_id], keypoints[0], save_path)
                    # -------------------
                    
                    cam_feat_tokens = self.keypoint_to_embed(keypoints)
                    all_cam_features.append(cam_feat_tokens)
                    
                src = torch.cat(all_cam_features, axis=1) 
                pos = torch.zeros_like(src) 
                proprio_input = self.input_proj_robot_state(qpos)
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[-1]
            else:
                qpos = self.input_proj_robot_state(qpos)
                env_state = self.input_proj_env_state(env_state)
                transformer_input = torch.cat([qpos, env_state], axis=1)
                hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[-1]
            
            a_hat_joint = self.action_head_joint(hs)
            a_hat_distance = torch.sigmoid(self.action_head_distance(hs))
            a_hat = torch.cat([a_hat_joint, a_hat_distance], dim=-1)
            is_pad_hat = self.is_pad_head(hs)
            return a_hat, is_pad_hat, [mu, logvar]

    def _save_debug_image(self, img_tensor, kpts, path):
        """Helper to denormalize image, draw keypoints, and save."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 1. Denormalize
        inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img = inv_normalize(img_tensor).permute(1, 2, 0).cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w, _ = img.shape
        # 2. Draw 32 keypoints
        for i in range(kpts.shape[0]):
            # Spatial softmax outputs -1 to 1. Convert to 0 to pixels.
            x = int((kpts[i, 0].item() + 1) / 2 * w)
            y = int((kpts[i, 1].item() + 1) / 2 * h)
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(path, img)

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


class SpatialSoftmax(nn.Module):
    def __init__(self, h, w, temperature=0.1):
        super().__init__()
        self.h = h
        self.w = w
        self.temperature = temperature

        # Create a grid of coordinates [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, w),
            torch.linspace(-1, 1, h),
            indexing='ij'
        )
        # Shape: [2, H, W] -> [1, 2, H, W]
        grid = torch.stack([pos_x.T, pos_y.T], dim=0).unsqueeze(0)
        self.register_buffer('grid', grid)

    def forward(self, x):
        # x: [batch, channels, h, w]
        batch, c, h, w = x.shape
        
        # 1. Flatten spatial dims and apply Softmax
        # [batch, channels, h*w]
        softmax_attention = torch.nn.functional.softmax(x.view(batch, c, -1) / self.temperature, dim=-1)
        softmax_attention = softmax_attention.view(batch, c, h, w)

        # 2. Compute expected value of coordinates
        # [batch, channels, 2, h, w] * [1, 1, 2, h, w]
        expected_pos = torch.sum(softmax_attention.unsqueeze(2) * self.grid, dim=[3, 4])
        # Output: [batch, channels, 2]
        return expected_pos



def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    print('detr_vae_joint: building backbone for state_dim=7')
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

