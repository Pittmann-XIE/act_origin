import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        
        # Load weights from config with defaults
        self.kl_weight = args_override.get('kl_weight', 10.0)
        self.bisim_weight = args_override.get('bisim_weight', 1.0)
        self.layer_to_align = args_override.get('layer_to_align', 6)
        self.gaze_weight = args_override.get('gaze_weight', 1.0)
        
        print(f'KL Weight {self.kl_weight} | Bisim Weight: {self.bisim_weight} | Gaze Weight: {self.gaze_weight} | Aligning Layer: {self.layer_to_align}')

    def __call__(self, qpos, image, actions=None, is_pad=None, next_qpos=None, next_image=None, valid_next=None, gaze_data=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if next_image is not None:
            next_image = normalize(next_image)

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Forward pass through unified model
            a_hat, is_pad_hat, (mu, logvar), attn_weights_list, bisim_loss = self.model(
                qpos, image, env_state, actions, is_pad, next_qpos, next_image, valid_next
            )
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['bisim'] = bisim_loss if bisim_loss is not None else torch.tensor(0.0).to(qpos.device)
            
            # Base loss accumulation
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['bisim'] * self.bisim_weight
            
            # --- SPATIOTEMPORAL POOLED GAZE KL DIVERGENCE ---
            if gaze_data is not None and attn_weights_list is not None:
                # 1. Grab specific layer's attention weights
                A = attn_weights_list[self.layer_to_align] 
                
                # 2. Slice out the prefix tokens dynamically (Latent, Proprio, and DINO CLS tokens)
                # By taking the last `gaze_data.shape[-1]` elements, we perfectly isolate the vision grid
                vision_tokens_len = gaze_data.shape[-1]
                A_vision = A[:, :, :, -vision_tokens_len:] 
                batch, num_heads, query_length, spatial_len = A_vision.shape
                
                # Derive number of cameras
                num_cams = spatial_len // (15 * 20)
                
                # 3. Reshape and Permute to match flattened targets
                A_vision = A_vision.view(batch, num_heads, query_length, 15, num_cams, 20)
                A_vision = A_vision.permute(0, 1, 2, 4, 3, 5)
                A_vision = A_vision.reshape(batch, num_heads, query_length, num_cams * 15 * 20)
                
                # 4. Mean pooling across heads to prevent collapse
                A_vision_pooled = A_vision.mean(dim=1) 
                
                # Normalize inputs and targets (adding epsilon to prevent division by zero)
                A_vision_prob = A_vision_pooled / (A_vision_pooled.sum(dim=-1, keepdim=True) + 1e-8)
                gaze_prob = gaze_data / (gaze_data.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Convert model probabilities to Log Space for KL Divergence
                A_vision_logprob = torch.log(A_vision_prob + 1e-8)
                
                # Calculate raw unreduced KL Divergence
                kl_unreduced = F.kl_div(A_vision_logprob, gaze_prob, reduction='none') 
                kl_per_query = kl_unreduced.sum(dim=-1) 
                
                # 5. Combined Padding Mask
                valid_gaze_mask = (gaze_data.sum(dim=-1) > 1e-5) # True if gaze exists
                valid_mask = (~is_pad) & valid_gaze_mask         # True if action is real AND gaze exists
                
                # Apply mask and safely average over valid queries
                kl_masked = kl_per_query * valid_mask
                valid_queries_count = valid_mask.sum()
                loss_gaze = kl_masked.sum() / torch.clamp(valid_queries_count, min=1.0)
                
                loss_dict['gaze'] = loss_gaze
                loss_dict['loss'] += loss_gaze * self.gaze_weight
            # --------------------------------------------------------------
                            
            return loss_dict
        else: # inference time
            a_hat, is_pad_hat, (mu, logvar), attn_weights_list, bisim_loss = self.model(qpos, image, env_state) 
            return a_hat, attn_weights_list 
        
    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) 
            return a_hat

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