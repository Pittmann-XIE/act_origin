# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
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
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)

#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)
#         out: List[NestedTensor] = []
#         pos = []
#         for name, x in xs.items():
#             out.append(x)
#             # position encoding
#             pos.append(self[1](x).to(x.dtype))

#         return out, pos


# def build_backbone(args):
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model


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

    def forward(self, tensor):
        # tensor shape: [B, C, H, W]
        batch_size, _, img_height, img_width = tensor.shape
        num_patches_height = img_height // self.patch_size
        num_patches_width = img_width // self.patch_size
        
        # DINOv3 forward pass
        with torch.no_grad(): 
            outputs = self.model(pixel_values=tensor)
            
        last_hidden_states = outputs.last_hidden_state
        
        # Extract CLS token
        cls_token = last_hidden_states[:, 0, :]
        
        # Extract patches (skipping 1 CLS token + N register tokens)
        patch_features_flat = last_hidden_states[:, 1 + self.num_register_tokens:, :]
        
        # Elegantly reconstruct the 2D spatial grid
        patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
        
        # Permute from [B, H, W, Channels] to [B, Channels, H, W] to match CNN behavior
        patch_features = patch_features.permute(0, 3, 1, 2).contiguous()
        
        if self.downsample:
            # Downsample from full patch grid to half size (e.g., 40x30 -> 20x15)
            patch_features = F.adaptive_avg_pool2d(
                patch_features, 
                (num_patches_height // 2, num_patches_width // 2)
            )
            
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
            # Apply dynamic spatial position encoding
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