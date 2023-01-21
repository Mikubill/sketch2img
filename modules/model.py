import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention import CrossAttention, BasicTransformerBlock
from einops import rearrange

class SatMixin(torch.nn.Module):
    
    def __init__(self, unet):

        super().__init__()
        self.blocks = []
        prefix = "sketch_attn"
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                module_name = prefix + '.' + name 
                print(f"Injected: {module_name}")
                module_name = module_name.replace('.', '_')
                sat_module = AttnModule(module_name, module)
                self.blocks.append(sat_module)
        
        names = set()
        for block in self.blocks:
            assert block.name not in names, f"duplicated module name: {block.name}"
            names.add(block.name)  
            self.add_module(block.name, block)
            
    def set_res_samples(self, res_samples):
        down_blocks = ()
        up_blocks = ()
        mid_block = (res_samples[-1][-1], )
        for mid_layers in res_samples:
            if len(mid_layers) == 3:
                down_blocks += (mid_layers[0], mid_layers[1])
                up_blocks += (mid_layers[0], mid_layers[1], mid_layers[1])
                
        total_blocks = down_blocks + up_blocks[::-1] + mid_block
        for idx in range(len(self.blocks)):
            self.blocks[idx].set_res_sample(total_blocks[idx])
            
    def set_scale(self, scale):
        for block in self.blocks:
            block.set_scale(scale)

class AttnModule(torch.nn.Module):
    
    def __init__(self, sat_name, base_layer):
        super().__init__()
        self.name = sat_name    
        attn1 = base_layer.attn1
        
        # extract params from inner attn1
        base_dim = attn1.to_q.in_features
        dim = base_dim * 2
        heads = attn1.heads
        dim_head = attn1.to_q.out_features // heads
        attention_bias = hasattr(attn1, "bias") and attn1.bias != None
        dropout = 0.0 # no active dropout layers 
        upcast_attention = attn1.upcast_attention
        
        self.sketch_norm = nn.LayerNorm(dim)
        self.sketch_attn = CrossAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.sketch_conv = nn.Conv1d(base_dim, base_dim, 1)
        self.sketch_scale = 1.0
    
        outer = self
        def forward(self, hidden_states, *args, **kwargs):
            return outer.forward(hidden_states, org_module=self, **kwargs)
            
        # inject to model
        base_layer.forward = forward.__get__(base_layer, BasicTransformerBlock)
        
    def set_res_sample(self, res_sample):
        self.res_sample = rearrange(res_sample, "b c h w -> b (h w) c")
        
    def set_scale(self, scale):
        self.sketch_scale = scale
        
    def forward(
        self,
        hidden_states,
        org_module=None,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        inner = self
        self = org_module
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states
        
        if hasattr(inner, "res_sample") and  inner.res_sample is not None:
            # 1.5. Injected Self-Attention
            norm_hidden_states = inner.sketch_norm(torch.cat([hidden_states, inner.res_sample], dim=2)) 
            sketch_attn_output = inner.sketch_attn(norm_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)
            
            # 1.5. Injected CrossAttention
            # norm_hidden_states = inner.sketch_norm(hidden_states) 
            # sketch_attn_output = inner.sketch_attn(norm_hidden_states, encoder_hidden_states=inner.res_sample, **cross_attention_kwargs)
            
            # TS(w): select non-sketch token only
            sketch_attn_output = sketch_attn_output[:, :attn_output.shape[1], :attn_output.shape[2]].permute(0, 2, 1)
            sketch_attn_output = inner.sketch_scale * inner.sketch_conv(sketch_attn_output)
            hidden_states = sketch_attn_output.permute(0, 2, 1) + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states