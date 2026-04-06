# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
import numpy as np

import IPython
e = IPython.embed


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
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names,use_a_hat,num_jump_samples, decay_gamma):
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
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        
        self.bisim_projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # --- Ablation A: Jump-Step Forward Dynamics ---
        self.time_embed = nn.Embedding(num_queries, 32)
        
        # Input: latent_dim (128) + masked_actions (num_queries * state_dim) + time_embed (32)
        dynamics_input_dim = 128 + (num_queries * state_dim) + 32
        self.jump_forward_dynamics = nn.Sequential(
            nn.Linear(dynamics_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.use_a_hat = use_a_hat # Store the flag
        self.num_jump_samples = num_jump_samples # <--- Store
        self.decay_gamma = decay_gamma           # <--- Store
 
    def forward(self, qpos, image, env_state, actions=None, is_pad=None, next_qpos=None, next_image=None, valid_next=None, sampled_ks=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            # --- MODIFICATION START ---
            hs, attn_weights,memory = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)
            # hs = hs[0] # Original code might have done this, we unpack tuple now
            # --- MODIFICATION END ---
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            # hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
            hs,attn_weights = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)
        # --- MODIFICATION START ---   
        hs = hs[-1] # Take the output from the last decoder layer
        # --- MODIFICATION END ---
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        
        # ==========================================================
        # Bisimulation Forward Dynamics (Ablation A)
        # ==========================================================
        bisim_loss = None
        if actions is not None and next_qpos is not None and next_image is not None and sampled_ks is not None:
            bs = a_hat.shape[0]
            
            phi_global_current = memory.mean(dim=0) # [Batch, Dim]
            z_theta_current = self.bisim_projection_head(phi_global_current)
            
            num_samples = sampled_ks.shape[1]
            
            total_weighted_loss = torch.zeros(bs, device=qpos.device) # [Batch]
            weight_sum = torch.zeros(bs, device=qpos.device)          # [Batch]
            
            # Loop over the feature dimension, NOT the k value
            for idx in range(num_samples):
                k_batch = sampled_ks[:, idx] # Shape: [Batch]
                
                # --- Vectorized Action Masking ---
                if getattr(self, 'use_a_hat', False):
                    source_actions = a_hat.detach() 
                else:
                    source_actions = actions 
                    
                # Create a mask of shape [Batch, num_queries]. True if step <= k for that trajectory.
                seq_idx = torch.arange(self.num_queries, device=qpos.device).unsqueeze(0)
                mask = seq_idx <= k_batch.unsqueeze(1)
                
                masked_actions = source_actions.clone()
                # Use masked_fill_ which correctly broadcasts the [Batch, Seq, 1] mask 
                # across the [Batch, Seq, Action_Dim] tensor
                masked_actions.masked_fill_(~mask.unsqueeze(-1), 0.0) 
                masked_actions_flat = masked_actions.view(bs, -1)
                
                # Predict Jump State z_{t+k}
                k_embed = self.time_embed(k_batch) # PyTorch handles the [Batch] shape automatically
                
                dynamics_input = torch.cat([z_theta_current, masked_actions_flat, k_embed], dim=-1)
                pred_z_theta_next = self.jump_forward_dynamics(dynamics_input)
                
                # Target Network logic
                with torch.no_grad():
                    # Simply pull the pre-sampled targets!
                    target_image = next_image[:, idx] 
                    target_qpos = next_qpos[:, idx]   
                    
                    all_cam_features_next = []
                    all_cam_pos_next = []
                    for cam_id, cam_name in enumerate(self.camera_names):
                        feat_next, pos_next = self.backbones[cam_id](target_image[:, cam_id])
                        all_cam_features_next.append(self.input_proj(feat_next[0]))
                        all_cam_pos_next.append(pos_next[0])
                    
                    proprio_input_next = self.input_proj_robot_state(target_qpos)
                    src_next = torch.cat(all_cam_features_next, axis=3)
                    pos_next = torch.cat(all_cam_pos_next, axis=3)
                    
                    src_next_flat = src_next.flatten(2).permute(2, 0, 1)
                    pos_next_flat = pos_next.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
                    additional_pos_embed_next = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                    pos_next_full = torch.cat([additional_pos_embed_next, pos_next_flat], axis=0)
                    
                    addition_input_next = torch.stack([latent_input, proprio_input_next], axis=0)
                    src_next_full = torch.cat([addition_input_next, src_next_flat], axis=0)
                    
                    memory_next = self.transformer.encoder(src_next_full, src_key_padding_mask=None, pos=pos_next_full)
                    phi_global_next = memory_next.mean(dim=0)
                    target_z_theta_next = self.bisim_projection_head(phi_global_next).detach() 
                    
                # Compute Step Loss
                raw_bisim_loss = F.mse_loss(pred_z_theta_next, target_z_theta_next, reduction='none').mean(dim=-1) # [Batch]
                valid_mask = valid_next[:, idx] # [Batch]
                
                # Vectorized temporal weight
                temporal_weight = getattr(self, 'decay_gamma') ** k_batch.float()
                
                total_weighted_loss += raw_bisim_loss * valid_mask * temporal_weight
                weight_sum += valid_mask * temporal_weight

            # Normalize and combine as before
            per_trajectory_loss = total_weighted_loss / weight_sum.clamp(min=1e-6)
            trajectory_has_valid_step = (weight_sum > 0).float()
            valid_trajectory_count = trajectory_has_valid_step.sum()
            
            if valid_trajectory_count > 0:
                bisim_loss = (per_trajectory_loss * trajectory_has_valid_step).sum() / valid_trajectory_count
            else:
                bisim_loss = total_weighted_loss.sum() 

        return a_hat, is_pad_hat, [mu, logvar], attn_weights, bisim_loss



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
    state_dim = 14 # TODO hardcode
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)
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
        use_a_hat=getattr(args, 'use_a_hat'),
        num_jump_samples=getattr(args, 'num_jump_samples'),
        decay_gamma=getattr(args, 'decay_gamma')
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

