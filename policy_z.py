import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.z_weight = args_override.get('z_weight', 1.0)
        print(f'KL Weight {self.kl_weight}, z_weight {self.z_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, gaze_data=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None: 
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Unpack z_gaze from the model
            a_hat, is_pad_hat, (mu, logvar, z_gaze), _ = self.model(qpos, image, env_state, actions, is_pad, gaze_data=gaze_data)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            
            # --- NEW: Gaze-Trajectory Contrastive Loss (InfoNCE) ---
            if z_gaze is not None:
                # 1. L2 Normalize the latents to compute cosine similarity safely
                mu_norm = F.normalize(mu, p=2, dim=1)
                z_gaze_norm = F.normalize(z_gaze, p=2, dim=1)
                
                # 2. Compute similarity matrix: (bs, bs)
                # Diagonal elements are positive pairs (same demo), off-diagonals are negative pairs.
                sim_matrix = torch.matmul(mu_norm, z_gaze_norm.T)
                
                # 3. Temperature scaling
                tau = 0.1 
                
                # 4. Create labels (0 to bs-1) for the diagonal
                labels = torch.arange(mu.shape[0], device=mu.device)
                
                # 5. Cross Entropy automatically applies softmax and computes the NLL
                loss_contrastive = F.cross_entropy(sim_matrix / tau, labels)
                
                loss_dict['contrastive'] = loss_contrastive
                
                # Add to total loss (You can expose this weight to argparse later)
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['contrastive'] * self.z_weight
            else:
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            # -------------------------------------------------------
                
            return loss_dict
        else: 
            # inference time
            a_hat, _, (_, _, _), attn_weights = self.model(qpos, image, env_state) 
            return a_hat, attn_weights

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