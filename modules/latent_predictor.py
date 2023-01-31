import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from einops import rearrange
from diffusers import UNet2DConditionModel

class LatentEdgePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Following section 4.1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )
        
        # init using kaiming uniform
        for name, module in self.layers.named_modules():
            if module.__class__.__name__ == "Linear":
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # x: b, (h w), c
        pos_elem = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_layers)]
        pos_encoding = torch.cat(pos_elem, dim=1)
        
        x = torch.cat((x, t, pos_encoding), dim=1)
        x = rearrange(x, "b c h w -> (b w h) c").to(torch.float16)
        
        return self.layers(x)
    
def hook_unet(unet: UNet2DConditionModel):
    blocks_idx = [0, 1, 2]
    feature_blocks = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        if isinstance(output, torch.TensorType):
            feature = output.float()
            setattr(module, "output", feature)
        elif isinstance(output, dict): 
            feature = output.sample.float()
            setattr(module, "output", feature)
        else: 
            feature = output.float()
            setattr(module, "output", feature)
    
    # 0, 1, 2 -> (ldm-down) 2, 4, 8
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block) 
            
    # ldm-mid 0, 1, 2
    for block in unet.mid_block.attentions + unet.mid_block.resnets:
        block.register_forward_hook(hook)
        feature_blocks.append(block) 
    
    # 0, 1, 2 -> (ldm-up) 2, 4, 8
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)  
            
    return feature_blocks