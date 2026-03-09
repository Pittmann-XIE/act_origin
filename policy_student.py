import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

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
        # Removed box_weight

    def __call__(self, qpos, image_encoder, image_decoder, actions=None, is_pad=None, train_stage=1):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        image_encoder = normalize(image_encoder)
        image_decoder = normalize(image_decoder)
        
        if actions is not None: # Training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            
            if train_stage == 1:
                # Stage 1: Train Teacher
                a_hat, is_pad_hat, (mu, logvar), _ = self.model(qpos, image_encoder, env_state, actions, is_pad, train_stage=1)
                total_kld, _, _ = kl_divergence(mu, logvar)
                
                l1 = (F.l1_loss(actions, a_hat, reduction='none') * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = l1 + total_kld[0] * self.kl_weight
                
            elif train_stage == 2:
                # Stage 2: Train Student (Distillation)
                a_hat, is_pad_hat, (mu, logvar), _, memory_teacher, memory_student = self.model(qpos, image_encoder, env_state, actions, is_pad, train_stage=2)
                
                total_kld, _, _ = kl_divergence(mu, logvar)
                l1 = (F.l1_loss(actions, a_hat, reduction='none') * ~is_pad.unsqueeze(-1)).mean()
                
                # Knowledge Distillation Loss
                distill_loss = F.mse_loss(memory_student, memory_teacher.detach())
                
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['distill_loss'] = distill_loss
                
                # Combine standard behavioral cloning loss with distillation loss
                loss_dict['loss'] = l1 + (total_kld[0] * self.kl_weight) + distill_loss
                
            return loss_dict
        else: 
            # Inference time (Stage 0 for Student, Stage 1 for Teacher)
            a_hat, _, (_, _), attn_weights = self.model(qpos, image_encoder, env_state, train_stage=train_stage)
            return a_hat, attn_weights

    def configure_optimizers(self):
        return self.optimizer

class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None, box_data=None):
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