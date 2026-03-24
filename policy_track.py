import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model 
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        
        # --- MODIFICATION START ---
        self.layer_to_align = args_override.get('layer_to_align', 0)
        self.gaze_weight = args_override.get('gaze_weight', 1.0)
        print(f'KL Weight {self.kl_weight} | Gaze Weight: {self.gaze_weight} | Aligning Layer: {self.layer_to_align}')
        # --- MODIFICATION END ---

    # Allow passing of gaze_data
    def __call__(self, qpos, image, actions=None, is_pad=None, gaze_data=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # attn_weights is now a LIST of all decoder layers' matrices
            a_hat, is_pad_hat, (mu, logvar), attn_weights_list = self.model(qpos, image, env_state, actions, is_pad)
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            
            # --- MODIFICATION: SPATIOTEMPORAL POOLED GAZE KL DIVERGENCE ---
            if gaze_data is not None:
                # 1. Grab specific layer's attention weights
                A = attn_weights_list[self.layer_to_align] 
                
                # 2. Slice out the 2 auxiliary tokens (latent and proprio)
                A_vision = A[:, :, :, 2:] 
                batch, num_heads, query_length, spatial_len = A_vision.shape
                num_cams = spatial_len // (15 * 20)
                
                # 3. Reshape and Permute to match flattened targets
                A_vision = A_vision.view(batch, num_heads, query_length, 15, num_cams, 20)
                A_vision = A_vision.permute(0, 1, 2, 4, 3, 5)
                A_vision = A_vision.reshape(batch, num_heads, query_length, num_cams * 15 * 20)
                
                # --- FIX 1: MEAN POOLING TO PREVENT COLLAPSE ---
                # Average the spatial attention across all heads. 
                # This allows individual heads to specialize, as long as their combined sum matches human gaze.
                A_vision_pooled = A_vision.mean(dim=1) # Shape: (batch, query_length, 1200)
                
                # Normalize inputs and targets (adding epsilon to prevent division by zero)
                A_vision_prob = A_vision_pooled / (A_vision_pooled.sum(dim=-1, keepdim=True) + 1e-8)
                gaze_prob = gaze_data / (gaze_data.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Convert model probabilities to Log Space for KL Divergence
                A_vision_logprob = torch.log(A_vision_prob + 1e-8)
                
                # --- FIX 2: UNREDUCED LOSS FOR MASKING ---
                # Calculate raw KL Divergence. reduction='none' prevents PyTorch's batchmean inflation.
                kl_unreduced = F.kl_div(A_vision_logprob, gaze_prob, reduction='none') # (batch, query_length, 1200)
                
                # Sum across the spatial dimension to get the total KL penalty per temporal query
                kl_per_query = kl_unreduced.sum(dim=-1) # Shape: (batch, query_length)
                
                # --- FIX 3: COMBINED PADDING MASK ---
                # We must ignore queries where:
                # 1. The action is a padded dummy action (is_pad == True)
                # 2. The human gaze data is entirely missing/empty (sum == 0)
                valid_gaze_mask = (gaze_data.sum(dim=-1) > 1e-5) # True if gaze exists
                valid_mask = (~is_pad) & valid_gaze_mask         # True if action is real AND gaze exists
                
                # Apply mask to zero out invalid queries
                kl_masked = kl_per_query * valid_mask
                
                # Safely average the loss only over the valid queries in the batch
                valid_queries_count = valid_mask.sum()
                loss_gaze = kl_masked.sum() / torch.clamp(valid_queries_count, min=1.0)
                
                # Append to dict and scale by user-defined weight
                loss_dict['gaze'] = loss_gaze
                loss_dict['loss'] += loss_gaze * self.gaze_weight
            # --------------------------------------------------------------
                            
            return loss_dict
        else: # inference time
            a_hat, is_pad_hat, (mu, logvar), attn_weights_list = self.model(qpos, image, env_state) 
            return a_hat, attn_weights_list # Optionally return the list if you want to visualize it
        
    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
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
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
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