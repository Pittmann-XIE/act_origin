# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) 
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class DINOv3Backbone(nn.Module):
    def __init__(self, name="facebook/dinov3-vits16-pretrain-lvd1689m", downsample=False):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name)
        
        # Freeze DINO backbone
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        
        self.num_channels = self.model.config.hidden_size
        self.downsample = downsample
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)

    # --- ADD THIS METHOD ---
    def train(self, mode=True):
        """Override train() to ensure the DINOv3 model stays in eval mode."""
        super().train(mode)
        self.model.eval()
        return self
    # -----------------------

    def forward(self, tensor):
        batch_size, _, img_height, img_width = tensor.shape
        num_patches_height = img_height // self.patch_size
        num_patches_width = img_width // self.patch_size
        
        with torch.no_grad(): 
            outputs = self.model(pixel_values=tensor)
            
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]
        patch_features_flat = last_hidden_states[:, 1 + self.num_register_tokens:, :]
        patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
        patch_features = patch_features.permute(0, 3, 1, 2).contiguous()
        
        if self.downsample:
            patch_features = F.adaptive_avg_pool2d(patch_features, (num_patches_height // 2, num_patches_width // 2))
            
        return {"0": patch_features, "cls": cls_token}
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        cls_tokens = []
        for name, x in xs.items():
            if name == "cls":
                cls_tokens.append(x)
                continue
            out.append(x)
            pos.append(self[1](x).to(x.dtype))

        if len(cls_tokens) > 0:
            return out, pos, cls_tokens[0]
        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    backbone_name = getattr(args, 'backbone', 'resnet18')
    if backbone_name == 'dinov3':
        downsample = getattr(args, 'dinov3_downsample', False)
        backbone = DINOv3Backbone(downsample=downsample)
    else:
        backbone = Backbone(backbone_name, train_backbone, return_interm_layers, args.dilation)
        
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model