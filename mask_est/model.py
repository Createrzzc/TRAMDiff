import sys
sys.path.append("./")
import torch
import torch.nn as nn
from ldm.util import instantiate_from_config


class Mask_Est_Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.unet = instantiate_from_config(config.model).to(self.device)
        self.cond_stage_config = config.cond_stage_config
        self.cond_stage_model = self.instantiate_cond_stage(config=self.cond_stage_config, device=self.device)

    def instantiate_cond_stage(self, config, device):
        model = instantiate_from_config(config)
        cond_stage_model = model.eval()
        cond_stage_model.to(device)
        for param in cond_stage_model.parameters():
            param.requires_grad = False
        return cond_stage_model

    def forward(self, x, text):
        c = self.cond_stage_model(text)
        outputs = self.unet(x, context=c)
        return outputs



