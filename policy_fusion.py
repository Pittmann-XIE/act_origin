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
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Catch the 5th return value
            a_hat, is_pad_hat, (mu, logvar), _, distill_loss = self.model(qpos, image, env_state, actions, is_pad)
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['distill'] = distill_loss # Record the distillation loss
            
            # Add distillation loss to the total loss
            # You can add a weight multiplier here (e.g., 10.0 * distill_loss) if you want to force it harder
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['distill']
                
            return loss_dict
        else: # inference time
            # Catch the 5th return value
            a_hat, _, (_, _), attn_weights, _ = self.model(qpos, image, env_state) 
            return a_hat, attn_weights

    def configure_optimizers(self):
        return self.optimizer

class ACTPolicyQFormer(ACTPolicy):
    def __init__(self, args_override):
        # Force the DETRVAE model to use the Q-former logic
        args_override['use_qformer'] = True
        
        # Initialize the base ACTPolicy
        super().__init__(args_override)
        
        # FIX: detr.main.build_ACT_model_and_optimizer often drops unrecognized arguments.
        # If use_qformer was dropped during build, we manually inject the parameters here.
        if not hasattr(self.model, 'learned_feat1'):
            self.model.use_qformer = True
            hidden_dim = self.model.transformer.d_model
            # Dimensions: (1, hidden_dim, H/32, W/32) -> (1, hidden_dim, 15, 20) for 480x640 images
            self.model.register_parameter('learned_feat1', nn.Parameter(torch.randn(1, hidden_dim, 15, 20)))
            self.model.register_parameter('learned_pos1', nn.Parameter(torch.randn(1, hidden_dim, 15, 20)))
        
        # 1. Freeze all parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze ONLY the newly added learnable queries
        self.model.learned_feat1.requires_grad = True
        self.model.learned_pos1.requires_grad = True
        
        # 3. Override the optimizer to only train these two parameters
        self.optimizer = torch.optim.AdamW([
            self.model.learned_feat1,
            self.model.learned_pos1
        ], lr=args_override['lr'], weight_decay=1e-4)


class ACTPolicyImplicitDistill(ACTPolicy):
    def __init__(self, args_override):
        # Force the model to use the Implicit Distillation logic
        args_override['use_implicit_distill'] = True
        
        # Initialize the base ACTPolicy
        super().__init__(args_override)
        
        # 1. Freeze all parameters in the network (The Teacher)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze ONLY the new implicit attention block (The Student)
        for param in self.model.implicit_attention_block.parameters():
            param.requires_grad = True
        
        # 3. Override the optimizer to only train this specific block
        self.optimizer = torch.optim.AdamW(
            self.model.implicit_attention_block.parameters(),
            lr=args_override['lr'], 
            weight_decay=1e-4
        )

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