# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss/blob/master/networks/lora.py#L48

import math
import os
import torch
import modules.safe as _
from safetensors.torch import load_file


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
            
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.enable = False
        
    def resize(self, rank, alpha):
        self.alpha = torch.tensor(alpha)
        self.scale = alpha / rank
        if self.lora_down.__class__.__name__ == "Conv2d":
            in_dim = self.lora_down.in_channels
            out_dim = self.lora_up.out_channels
            self.lora_down = torch.nn.Conv2d(in_dim, rank, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(rank, out_dim, (1, 1), bias=False)
        else:
            in_dim = self.lora_down.in_features
            out_dim = self.lora_up.out_features
            self.lora_down = torch.nn.Linear(in_dim, rank, bias=False)
            self.lora_up = torch.nn.Linear(rank, out_dim, bias=False)
        
    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if self.enable:
            return (
                self.org_forward(x)
                + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            )
        return self.org_forward(x)
    

class LoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4, alpha=1) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha

        # create module instances
        def create_modules(
            prefix, root_module: torch.nn.Module, target_replace_modules
        ) -> list[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == "Linear" or (
                            child_module.__class__.__name__ == "Conv2d"
                            and child_module.kernel_size == (1, 1)
                        ):
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha,)
                            loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(LoRANetwork.LORA_PREFIX_TEXT_ENCODER, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        print(f"Create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, LoRANetwork.UNET_TARGET_REPLACE_MODULE)
        print(f"Create LoRA for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert (lora.lora_name not in names), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
            
            lora.apply()
            self.add_module(lora.lora_name, lora)
            
    def reset(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enable = False

    def load(self, file, scale):
        
        weights = None
        if os.path.splitext(file)[1] == ".safetensors":
            weights = load_file(file)
        else:
            weights = torch.load(file, map_location="cpu")
        
        if not weights:
            return
        
        network_alpha = None
        network_dim = None
        for key, value in weights.items():
            if network_alpha is None and "alpha" in key:
                network_alpha = value
            if network_dim is None and "lora_down" in key and len(value.size()) == 2:
                network_dim = value.size()[0]

        if network_alpha is None:
            network_alpha = network_dim

        weights_has_text_encoder = weights_has_unet = False
        weights_to_modify = []
        
        for key in weights.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                weights_has_text_encoder = True
                
            if key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                weights_has_unet = True

        if weights_has_text_encoder:
            weights_to_modify += self.text_encoder_loras

        if weights_has_unet:
            weights_to_modify += self.unet_loras
            
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.resize(network_dim, network_alpha)
            if lora in weights_to_modify:
                lora.enable = True

        info = self.load_state_dict(weights, False)
        print(f"Weights are loaded. Unexpect keys={info.unexpected_keys}")
