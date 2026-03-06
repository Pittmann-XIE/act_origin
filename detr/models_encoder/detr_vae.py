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
    def __init__(self, backbones_encoder, backbones_decoder, transformer, encoder, state_dim, num_queries, camera_names_encoder, camera_names_decoder):
        """ Initializes the model.
        Parameters:
            backbones_encoder: torch module of the backbone to be used for encoder. See backbone.py
            backbones_decoder: torch module of the backbone to be used for decoder. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names_encoder = camera_names_encoder
        self.camera_names_decoder = camera_names_decoder
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones_encoder is not None:
            self.input_proj = nn.Conv2d(backbones_encoder[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones_encoder = nn.ModuleList(backbones_encoder)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        if backbones_decoder is not None:
            self.input_proj_decoder = nn.Conv2d(backbones_decoder[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones_decoder = nn.ModuleList(backbones_decoder)
        else:
            raise NotImplementedError
        
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
        
        self.object_detection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
           nn.Sigmoid()
           )
 
    def forward(self, qpos, image_encoder, image_decoder, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image_encoder: batch, num_cam_enc, channel, height, width
        image_decoder: batch, num_cam_dec, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        
        # --- 1. Process Encoder Images ---
        all_cam_features_enc = []
        all_cam_pos_enc = []
        for cam_id, cam_name in enumerate(self.camera_names_encoder):
            features, pos = self.backbones_encoder[cam_id](image_encoder[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features_enc.append(self.input_proj(features)) # Re-using input_proj for encoder
            all_cam_pos_enc.append(pos)
            
        # Flatten spatial dimensions for Transformer encoder: [bs, c, h, w*num_cams] -> [seq_img, bs, hidden_dim]
        src_enc = torch.cat(all_cam_features_enc, axis=3)
        pos_enc = torch.cat(all_cam_pos_enc, axis=3)
        src_enc = src_enc.flatten(2).permute(2, 0, 1) 
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)

        ### Obtain latent z from image + action sequence
        if is_training:
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            
            # Combine image sequence with state/action sequence
            img_embed = src_enc + pos_enc # Add positional encoding to image features
            state_action_embed = torch.cat([cls_embed, qpos_embed, action_embed], axis=1).permute(1, 0, 2)
            encoder_input = torch.cat([img_embed, state_action_embed], axis=0) # (seq_img + seq+1, bs, hidden_dim)

            # Masking
            img_is_pad = torch.full((bs, img_embed.shape[0]), False).to(qpos.device)
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([img_is_pad, cls_joint_is_pad, is_pad], axis=1)  # (bs, seq_img + seq+1)

            # Position embedding for state/action (images already have pos added)
            pos_embed_state = self.pos_table.clone().detach().permute(1, 0, 2).expand(-1, bs, -1)
            pos_embed = torch.cat([torch.zeros_like(img_embed), pos_embed_state], axis=0) 

            # Query encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            
            # Extract CLS token output (it is located right after the image sequence)
            cls_idx = img_embed.shape[0]
            encoder_output = encoder_output[cls_idx] 
            
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # --- 2. Process Decoder Images ---
        all_cam_features_dec = []
        all_cam_pos_dec = []
        for cam_id, cam_name in enumerate(self.camera_names_decoder):
            features, pos = self.backbones_decoder[cam_id](image_decoder[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features_dec.append(self.input_proj_decoder(features))
            all_cam_pos_dec.append(pos)
            
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        
        # fold camera dimension into width dimension
        src_dec = torch.cat(all_cam_features_dec, axis=3)
        pos_dec = torch.cat(all_cam_pos_dec, axis=3)
        
        hs, attn_weights = self.transformer(src_dec, None, self.query_embed.weight, pos_dec, latent_input, proprio_input, self.additional_pos_embed.weight)
        
        hs = hs[0] # Take the output from the last decoder layer
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        box_hat = self.object_detection(hs.mean(dim=1))
        
        return a_hat, is_pad_hat, [mu, logvar], attn_weights, box_hat


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
    backbones_encoder = []
    backbones_decoder = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)
    for _ in args.camera_names_encoder:
        backbone = build_backbone(args)
        backbones_encoder.append(backbone)
    
    for _ in args.camera_names_decoder:
        backbone = build_backbone(args)
        backbones_decoder.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones_encoder,
        backbones_decoder,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names_encoder=args.camera_names_encoder,
        camera_names_decoder=args.camera_names_decoder,
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

