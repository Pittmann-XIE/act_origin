# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
import numpy as np

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

class DETRVAE(nn.Module):
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
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
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        self.latent_dim = 32 
        self.cls_embed = nn.Embedding(1, hidden_dim) 
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) 
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) 
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) 

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) 
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) 
        
        # DINO cls embed module
        self.cam_cls_pos_embed = nn.Embedding(len(camera_names), hidden_dim) 
        
        # Object detection/Gaze tracking
        self.object_detection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4), nn.Sigmoid()
        )
           
        # Bisimulation Modules
        self.bisim_projection_head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.action_encoder = nn.Sequential(nn.Linear(num_queries * state_dim, 256), nn.ReLU(), nn.Linear(256, 64))
        self.forward_dynamics_model = nn.Sequential(nn.Linear(128 + 64, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, next_qpos=None, next_image=None, valid_next=None):
        is_training = actions is not None 
        bs, _ = qpos.shape
        
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
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        bisim_loss = None
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            all_cls_tokens = []
            
            for cam_id, cam_name in enumerate(self.camera_names):
                out = self.backbones[cam_id](image[:, cam_id])
                if len(out) == 3:
                    features, pos, cls_token = out
                    all_cls_tokens.append(cls_token)
                else:
                    features, pos = out
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            
            cls_tokens = None
            cls_pos_embed = None
            if len(all_cls_tokens) > 0:
                cls_tokens = [self.input_proj(cls.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for cls in all_cls_tokens]
                cls_tokens = torch.stack(cls_tokens, axis=0) 
                cls_pos_embed = self.cam_cls_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

            hs, attn_weights, memory = self.transformer(
                src, None, self.query_embed.weight, pos, 
                latent_input, proprio_input, self.additional_pos_embed.weight, 
                cls_tokens=cls_tokens, cls_pos_embed=cls_pos_embed
            )
            
            # Bisimulation Loss logic
            if actions is not None and next_qpos is not None and next_image is not None:
                phi_global_current = memory.mean(dim=0) 
                z_theta_current = self.bisim_projection_head(phi_global_current) 
                action_flat = self.action_head(hs[-1]).view(bs, -1)
                action_encoded = self.action_encoder(action_flat)
                pred_z_theta_next = self.forward_dynamics_model(torch.cat([z_theta_current, action_encoded], dim=-1))
                
                with torch.no_grad():
                    all_cam_features_next = []
                    all_cam_pos_next = []
                    all_cls_tokens_next = []
                    for cam_id, cam_name in enumerate(self.camera_names):
                        out_next = self.backbones[cam_id](next_image[:, cam_id])
                        if len(out_next) == 3:
                            feat_next, pos_next, cls_token_next = out_next
                            all_cls_tokens_next.append(cls_token_next)
                        else:
                            feat_next, pos_next = out_next
                        all_cam_features_next.append(self.input_proj(feat_next[0]))
                        all_cam_pos_next.append(pos_next[0])
                    
                    proprio_input_next = self.input_proj_robot_state(next_qpos)
                    src_next = torch.cat(all_cam_features_next, axis=3)
                    pos_next = torch.cat(all_cam_pos_next, axis=3)
                    
                    src_next_flat = src_next.flatten(2).permute(2, 0, 1)
                    pos_next_flat = pos_next.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
                    additional_pos_embed_next = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                    pos_next_full = torch.cat([additional_pos_embed_next, pos_next_flat], axis=0)
                    
                    addition_input_next = torch.stack([latent_input, proprio_input_next], axis=0)
                    src_next_full = torch.cat([addition_input_next, src_next_flat], axis=0)

                    if len(all_cls_tokens_next) > 0:
                        cls_tokens_next = [self.input_proj(cls.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for cls in all_cls_tokens_next]
                        cls_tokens_next = torch.stack(cls_tokens_next, axis=0)
                        cls_pos_embed_next = self.cam_cls_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                        src_next_full = torch.cat([cls_tokens_next, src_next_full], axis=0)
                        pos_next_full = torch.cat([cls_pos_embed_next, pos_next_full], axis=0)

                    memory_next = self.transformer.encoder(src_next_full, src_key_padding_mask=None, pos=pos_next_full)
                    phi_global_next = memory_next.mean(dim=0)
                    target_z_theta_next = self.bisim_projection_head(phi_global_next).detach()
                    
                raw_bisim_loss = F.mse_loss(pred_z_theta_next, target_z_theta_next, reduction='none').mean(dim=-1)
                bisim_loss = (raw_bisim_loss * valid_next).mean()
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) 
            hs, attn_weights, memory = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)

        hs = hs[-1] 
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar], attn_weights, bisim_loss

def build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    return encoder

def build(args):
    state_dim = 14 
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(backbones, transformer, encoder, state_dim=state_dim, num_queries=args.num_queries, camera_names=args.camera_names)
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