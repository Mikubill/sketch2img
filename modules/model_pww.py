import importlib
import inspect
import math
from typing import List, Optional, Union

import k_diffusion
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import math
import torch
from torch import nn, einsum
from torch.autograd.function import Function

from einops import rearrange
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser

from diffusers import DiffusionPipeline
from diffusers.models.cross_attention import CrossAttention
from diffusers.utils import (PIL_INTERPOLATION, is_accelerate_available, logging, randn_tensor)

xformers_available = False
try: 
    import xformers
    xformers_available = True
except ImportError:
    pass

EPSILON = 1e-6
exists = lambda val: val is not None
default = lambda val, d: val if exists(val) else d
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_attention_scores(attn, query, key, attention_mask=None):

    if attn.upcast_attention:
        query = query.float()
        key = key.float()

    attention_scores = torch.baddbmm(
        torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        ),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=attn.scale,
    )

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if attn.upcast_softmax:
        attention_scores = attention_scores.float()

    return attention_scores


def CrossAttnProcessor(
    attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None
):
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)
    
    query = attn.to_q(hidden_states)
    query = attn.head_to_batch_dim(query)

    encoder_states = hidden_states
    is_xattn = False
    if encoder_hidden_states is not None:
        is_xattn = True
        img_state = encoder_hidden_states["img_state"]
        encoder_states = encoder_hidden_states["states"]
        weight_func = encoder_hidden_states["weight_func"]
        sigma = encoder_hidden_states["sigma"]

    key = attn.to_k(encoder_states)
    value = attn.to_v(encoder_states)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    if is_xattn and isinstance(img_state, dict):
        # use torch.baddbmm method (slow)
        attention_scores = get_attention_scores(attn, query, key, attention_mask)
        w = img_state[sequence_length].to(query.device)
        cross_attention_weight = weight_func(w, sigma, attention_scores)
        attention_scores += torch.repeat_interleave(cross_attention_weight, repeats=attn.heads, dim=0)
        
        # calc probs
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)
        hidden_states = torch.bmm(attention_probs, value)
        
    elif xformers_available:
        hidden_states = xformers.ops.memory_efficient_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), attn_bias=attention_mask
        )
        hidden_states = hidden_states.to(query.dtype)
    
    else:
        q_bucket_size = 512
        k_bucket_size = 1024
        
        # use flash-attention
        hidden_states = FlashAttentionFunction.apply(
            query.contiguous(), key.contiguous(), value.contiguous(), 
            attention_mask, causal=False, q_bucket_size=q_bucket_size, k_bucket_size=k_bucket_size
        )
        hidden_states = hidden_states.to(query.dtype)
        
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(
            *args, encoder_hidden_states=encoder_hidden_states, **kwargs
        ).sample


class StableDiffusionPipeline(DiffusionPipeline):

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
    ):
        super().__init__()

        # get correct sigmas from LMS
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        model = ModelWrapper(unet, scheduler.alphas_cumprod)
        if scheduler.prediction_type == "v_prediction":
            self.k_diffusion_model = CompVisVDenoiser(model)
        else:
            self.k_diffusion_model = CompVisDenoiser(model)

    def get_scheduler(self, scheduler_type: str):
        library = importlib.import_module("k_diffusion")
        sampling = getattr(library, "sampling")
        return getattr(sampling, scheduler_type)
    
    def encode_sketchs(self, state, scale_ratio=8, g_strength=1.0, cond=None, uncond=None):
        img_state = []
        if state is None:
            return torch.FloatTensor(0)
        
        for k, v in state.items():
            if v["map"] is None:
                continue

            v_input = self.tokenizer(
                k,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids
            
            dotmap = v["map"] < 255
            arr = torch.from_numpy(dotmap.astype(float) * float(v["weight"]) * g_strength)
            img_state.append((v_input, arr))
            
        if len(img_state) == 0:
            return torch.FloatTensor(0)
            
        w_tensors = dict()
        for layer in self.unet.down_blocks:
            c = int(self.tokenizer.model_max_length)
            w, h = img_state[0][1].shape
            w_r, h_r = w // scale_ratio, h // scale_ratio
            
            cond_token_lis = cond.tolist()
            uncond_token_lis = uncond.tolist()
            
            ret_cond_tensor = torch.zeros((1, int(w_r * h_r), c), dtype=torch.float32)
            ret_uncond_tensor = torch.zeros((1, int(w_r * h_r), c), dtype=torch.float32)

            for v_as_tokens, img_where_color in img_state:
                is_in = 0

                ret = F.interpolate(
                    img_where_color.unsqueeze(0).unsqueeze(1),
                    scale_factor=1 / scale_ratio,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze().reshape(-1, 1).repeat(1, len(v_as_tokens))  
                
                for idx, tok in enumerate(cond_token_lis):
                    if cond_token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                        is_in = 1
                        ret_cond_tensor[0, :, idx : idx + len(v_as_tokens)] += (ret)
                        
                for idx, tok in enumerate(uncond_token_lis):
                    if cond_token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                        is_in = 1                      
                        ret_uncond_tensor[0, :, idx : idx + len(v_as_tokens)] += (ret)

                if not is_in == 1:
                    print(f"tokens {v_as_tokens} not found in text")
                    
            w_tensors[w_r * h_r] = torch.cat([ret_uncond_tensor, ret_cond_tensor])
            scale_ratio *= 2

        return w_tensors

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [
            self.unet,
            self.text_encoder,
            self.vae,
            self.safety_checker,
        ]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="max_length", return_tensors="pt"
        ).input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(
                    shape, generator=generator, device="cpu", dtype=dtype
                ).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            # if latents.shape != shape:
            #     raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [
                np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[
                    None, :
                ]
                for i in image
            ]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        image: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        latents=None,
        strength=1.0,
        pww_state=None,
        pww_attn_weight=1.0,
        sampler_name="",
        sampler_opt={},
        scale_ratio=8.0
    ):
        sampler = self.get_scheduler(sampler_name)
        if image is not None:
            image = self.preprocess(image)
            image = image.to(self.vae.device, dtype=self.vae.dtype)

            init_latents = self.vae.encode(image).latent_dist.sample(generator)
            latents = 0.18215 * init_latents

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True
        if guidance_scale <= 1.0:
            raise ValueError("has to use guidance_scale")

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        
        init_timestep = int(num_inference_steps / min(strength, 0.999)) if strength > 0 else 0
        sigmas = self.get_sigmas(init_timestep, sampler_opt).to(
            text_embeddings.device, dtype=text_embeddings.dtype
        )

        t_start = max(init_timestep - num_inference_steps, 0)
        sigma_sched = sigmas[t_start:]

        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=device,
            dtype=text_embeddings.dtype,
        )
        latents = latents.to(device)
        latents = latents + noise * sigma_sched[0]

        # 5. Prepare latent variables
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        text_input = self.tokenizer(
            [prompt, negative_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        
        img_state = self.encode_sketchs(
            pww_state, 
            g_strength=pww_attn_weight,
            cond=text_input[0],
            uncond=text_input[1],
            scale_ratio=scale_ratio,
        )
        
        def model_fn(x, sigma):
            
            latent_model_input = torch.cat([x] * 2)
            weight_func = (
                lambda w, sigma, qk: w * math.log(1 + sigma) * qk.max()
            )
            encoder_state = {
                "img_state": img_state,
                "states": text_embeddings,
                "sigma": sigma[0],
                "weight_func": weight_func,
            }

            noise_pred = self.k_diffusion_model(
                latent_model_input, sigma, cond=encoder_state
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

        sampler_args = self.get_sampler_extra_args_i2i(sigma_sched, sampler)
        latents = sampler(model_fn, latents, **sampler_args)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return (image,)

    def get_sigmas(self, steps, params):
        discard_next_to_last_sigma = params.get("discard_next_to_last_sigma", False)
        steps += 1 if discard_next_to_last_sigma else 0

        if params.get("scheduler", None) == "karras":
            sigma_min, sigma_max = (
                self.k_diffusion_model.sigmas[0].item(),
                self.k_diffusion_model.sigmas[-1].item(),
            )
            sigmas = k_diffusion.sampling.get_sigmas_karras(
                n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=self.device
            )
        else:
            sigmas = self.k_diffusion_model.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas

    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/48a15821de768fea76e66f26df83df3fddf18f4b/modules/sd_samplers.py#L454
    def get_sampler_extra_args_t2i(self, sigmas, eta, steps, func):
        extra_params_kwargs = {}

        if "eta" in inspect.signature(func).parameters:
            extra_params_kwargs["eta"] = eta

        if "sigma_min" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_min"] = sigmas[0].item()
            extra_params_kwargs["sigma_max"] = sigmas[-1].item()

        if "n" in inspect.signature(func).parameters:
            extra_params_kwargs["n"] = steps
        else:
            extra_params_kwargs["sigmas"] = sigmas

        return extra_params_kwargs

    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/48a15821de768fea76e66f26df83df3fddf18f4b/modules/sd_samplers.py#L454
    def get_sampler_extra_args_i2i(self, sigmas, func):
        extra_params_kwargs = {}

        if "sigma_min" in inspect.signature(func).parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs["sigma_min"] = sigmas[-2]

        if "sigma_max" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_max"] = sigmas[0]

        if "n" in inspect.signature(func).parameters:
            extra_params_kwargs["n"] = len(sigmas) - 1

        if "sigma_sched" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_sched"] = sigmas

        if "sigmas" in inspect.signature(func).parameters:
            extra_params_kwargs["sigmas"] = sigmas

        return extra_params_kwargs

    @torch.no_grad()
    def txt2img(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_steps: Optional[int] = 1,
        upscale=False,
        upscale_x: float = 2.0,
        upscale_method: str = "bicubic",
        upscale_antialias: bool = False,
        upscale_denoising_strength: int = 0.7,
        pww_state=None,
        pww_attn_weight=1.0,
        sampler_name="",
        sampler_opt={},
    ):
        sampler = self.get_scheduler(sampler_name)
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True
        if guidance_scale <= 1.0:
            raise ValueError("has to use guidance_scale")

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        sigmas = self.get_sigmas(num_inference_steps, sampler_opt).to(
            text_embeddings.device, dtype=text_embeddings.dtype
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents = latents * sigmas[0]
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        text_input = self.tokenizer(
            [prompt, negative_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        
        img_state = self.encode_sketchs(
            pww_state, 
            g_strength=pww_attn_weight,
            cond=text_input[0],
            uncond=text_input[1],
        )

        def model_fn(x, sigma):
            
            latent_model_input = torch.cat([x] * 2)
            weight_func = (
                lambda w, sigma, qk: w * math.log(1 + sigma) * qk.max()
            )
            encoder_state = {
                "img_state": img_state,
                "states": text_embeddings,
                "sigma": sigma[0],
                "weight_func": weight_func,
            }

            noise_pred = self.k_diffusion_model(
                latent_model_input, sigma, cond=encoder_state
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

        extra_args = self.get_sampler_extra_args_t2i(
            sigmas, eta, num_inference_steps, sampler
        )
        latents = sampler(model_fn, latents, **extra_args)

        if upscale:
            target_height = height * upscale_x
            target_width = width * upscale_x
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            latents = torch.nn.functional.interpolate(
                latents,
                size=(
                    int(target_height // vae_scale_factor),
                    int(target_width // vae_scale_factor),
                ),
                mode=upscale_method,
                antialias=upscale_antialias,
            )
            return self.img2img(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                latents=latents,
                strength=upscale_denoising_strength,
                sampler_name=sampler_name,
                sampler_opt=sampler_opt,
            )

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return (image,)


class FlashAttentionFunction(Function):

    
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 2 in the paper """

        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim = -1)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 4 in the paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights - lsec)

                if exists(row_mask):
                    p.masked_fill_(~row_mask, 0.)

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim = -1, keepdims = True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None