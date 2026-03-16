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
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, fusion_type=0, num_fusion_layers=1, use_qformer=False, use_implicit_distill=False):
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
        
        self.fusion_type = fusion_type
        if self.fusion_type in [1, 2, 3]:
            self.fusion_block = FusionBlock(hidden_dim, transformer.nhead, self.fusion_type, num_layers=num_fusion_layers)
        # Add the use_qformer flag
        self.use_qformer = use_qformer
        if self.use_qformer:
            # Hardcoded for ResNet18 output on a 480x640 image (H/32=15, W/32=20). 
            self.learned_feat1 = nn.Parameter(torch.randn(1, hidden_dim, 15, 20))
            self.learned_pos1 = nn.Parameter(torch.randn(1, hidden_dim, 15, 20))

        # --- NEW: Implicit Attention Distillation Flag and Block ---
        self.use_implicit_distill = use_implicit_distill
        if self.use_implicit_distill:
            self.implicit_attention_block = FusionBlock(hidden_dim, transformer.nhead, fusion_type=1, num_layers=num_fusion_layers+1)
        print(f'DETRVAE: q-former: {self.use_qformer}, implicit: {self.use_implicit_distill}')
            
    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        
        ### Obtain latent z from action sequence (VAE encoding)
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
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            
            # --- FUSION AND DISTILLATION LOGIC START ---
            feature_distill_loss = torch.tensor(0.0).to(qpos.device)
            
            if self.fusion_type == 0:
                src = torch.cat(all_cam_features, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)
                
            elif self.fusion_type in [1, 2, 3]:
                
                # 1. Compute Teacher features ONLY IF the second camera (crop) is provided
                if len(all_cam_features) >= 2:
                    teacher_src, teacher_pos = self.fusion_block(
                        all_cam_features[0], all_cam_features[1], 
                        all_cam_pos[0], all_cam_pos[1]
                    )
                
                # 2. Branch: Q-Former Student
                if getattr(self, 'use_qformer', False):
                    bs = image.shape[0]
                    q_feat1 = self.learned_feat1.expand(bs, -1, -1, -1)
                    q_pos1 = self.learned_pos1.expand(bs, -1, -1, -1)
                    
                    student_src, student_pos = self.fusion_block(
                        all_cam_features[0], q_feat1, 
                        all_cam_pos[0], q_pos1
                    )
                    
                    if len(all_cam_features) >= 2:
                        # Using Cosine Similarity to escape the MSE average trap
                        cos_sim = F.cosine_similarity(student_src, teacher_src.detach(), dim=1)
                        feature_distill_loss = (1.0 - cos_sim).mean()
                    
                    src = student_src
                    pos = student_pos

                # 3. Branch: Implicit Attention Distillation Student
                elif getattr(self, 'use_implicit_distill', False):
                    # Pass cam[0] into both sides of the implicit block to act as a self-attention bottleneck
                    student_src, student_pos = self.implicit_attention_block(
                        all_cam_features[0], all_cam_features[0], 
                        all_cam_pos[0], all_cam_pos[0]
                    )
                    
                    if len(all_cam_features) >= 2:
                        # Using Cosine Similarity to force structural matching
                        cos_sim = F.cosine_similarity(student_src, teacher_src.detach(), dim=1)
                        feature_distill_loss = (1.0 - cos_sim).mean()
                        
                    src = student_src
                    pos = student_pos

                # 4. Branch: Standard Baseline / Teacher (No Distillation)
                else:
                    assert len(all_cam_features) >= 2, "Need at least 2 cameras for standard fusion without distillation."
                    src = teacher_src
                    pos = teacher_pos
            # --- FUSION AND DISTILLATION LOGIC END ---
                    
            hs, attn_weights = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs, attn_weights = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)
            feature_distill_loss = torch.tensor(0.0).to(qpos.device) # Fallback for state-only
            
        hs = hs[0] # Take the output from the last decoder layer
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        return a_hat, is_pad_hat, [mu, logvar], attn_weights, feature_distill_loss
class FusionLayer(nn.Module):
    """ A single layer of Cross-Attention + FFN """
    def __init__(self, d_model, nhead, fusion_type, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type in [1, 2]:
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.activation = nn.ReLU(inplace=True)
            self.dropout_ffn = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            
        elif fusion_type == 3:
            # Stream 1
            self.multihead_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1_1 = nn.LayerNorm(d_model)
            self.dropout1_1 = nn.Dropout(dropout)
            self.linear1_1 = nn.Linear(d_model, dim_feedforward)
            self.activation_1 = nn.ReLU(inplace=True)
            self.dropout_ffn_1 = nn.Dropout(dropout)
            self.linear2_1 = nn.Linear(dim_feedforward, d_model)
            self.norm2_1 = nn.LayerNorm(d_model)
            self.dropout2_1 = nn.Dropout(dropout)

            # Stream 2
            self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1_2 = nn.LayerNorm(d_model)
            self.dropout1_2 = nn.Dropout(dropout)
            self.linear1_2 = nn.Linear(d_model, dim_feedforward)
            self.activation_2 = nn.ReLU(inplace=True)
            self.dropout_ffn_2 = nn.Dropout(dropout)
            self.linear2_2 = nn.Linear(dim_feedforward, d_model)
            self.norm2_2 = nn.LayerNorm(d_model)
            self.dropout2_2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, f0, f1, p0, p1):
        if self.fusion_type == 1:
            q = self.with_pos_embed(f0, p0)
            k = self.with_pos_embed(f1, p1)
            attn_out = self.multihead_attn(query=q, key=k, value=f1)[0]
            out = self.norm1(f0 + self.dropout1(attn_out))
            ffn_out = self.linear2(self.dropout_ffn(self.activation(self.linear1(out))))
            out = self.norm2(out + self.dropout2(ffn_out))
            return out, f1  # f0 is updated, f1 stays the same

        elif self.fusion_type == 2:
            q = self.with_pos_embed(f1, p1)
            k = self.with_pos_embed(f0, p0)
            attn_out = self.multihead_attn(query=q, key=k, value=f0)[0]
            out = self.norm1(f1 + self.dropout1(attn_out))
            ffn_out = self.linear2(self.dropout_ffn(self.activation(self.linear1(out))))
            out = self.norm2(out + self.dropout2(ffn_out))
            return f0, out  # f1 is updated, f0 stays the same

        elif self.fusion_type == 3:
            # Stream 1 updates f0 using f1
            q1 = self.with_pos_embed(f0, p0)
            k1 = self.with_pos_embed(f1, p1)
            attn_out_1 = self.multihead_attn_1(query=q1, key=k1, value=f1)[0]
            out_1 = self.norm1_1(f0 + self.dropout1_1(attn_out_1))
            ffn_out_1 = self.linear2_1(self.dropout_ffn_1(self.activation_1(self.linear1_1(out_1))))
            out_1 = self.norm2_1(out_1 + self.dropout2_1(ffn_out_1))

            # Stream 2 updates f1 using f0
            q2 = self.with_pos_embed(f1, p1)
            k2 = self.with_pos_embed(f0, p0)
            attn_out_2 = self.multihead_attn_2(query=q2, key=k2, value=f0)[0]
            out_2 = self.norm1_2(f1 + self.dropout1_2(attn_out_2))
            ffn_out_2 = self.linear2_2(self.dropout_ffn_2(self.activation_2(self.linear1_2(out_2))))
            out_2 = self.norm2_2(out_2 + self.dropout2_2(ffn_out_2))

            return out_1, out_2


class FusionBlock(nn.Module):
    """ Stacks multiple FusionLayers """
    def __init__(self, d_model, nhead, fusion_type, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_layers = num_layers
        # Use ModuleList to stack N identical layers
        self.layers = nn.ModuleList([
            FusionLayer(d_model, nhead, fusion_type, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, feat0, feat1, pos0, pos1):
        bs, c, h, w = feat0.shape
        f0 = feat0.flatten(2).permute(2, 0, 1)
        p0 = pos0.flatten(2).permute(2, 0, 1) if pos0 is not None else None
        f1 = feat1.flatten(2).permute(2, 0, 1)
        p1 = pos1.flatten(2).permute(2, 0, 1) if pos1 is not None else None

        # Pass features through all stacked layers iteratively
        for layer in self.layers:
            f0, f1 = layer(f0, f1, p0, p1)

        # Reshape back to image format depending on the fusion type
        if self.fusion_type == 1:
            out = f0.permute(1, 2, 0).view(bs, c, h, w)
            return out, pos0
        elif self.fusion_type == 2:
            out = f1.permute(1, 2, 0).view(bs, c, h, w)
            return out, pos1
        elif self.fusion_type == 3:
            out_1 = f0.permute(1, 2, 0).view(bs, c, h, w)
            out_2 = f1.permute(1, 2, 0).view(bs, c, h, w)
            out = torch.cat([out_1, out_2], axis=3)
            pos = torch.cat([pos0, pos1], axis=3)
            return out, pos

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
    backbones = []
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
        fusion_type=getattr(args, 'fusion_type', 0),
        num_fusion_layers=getattr(args, 'fusion_layers', 1),
        use_qformer=getattr(args, 'use_qformer', False),
        use_implicit_distill=getattr(args, 'use_implicit_distill', False),
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