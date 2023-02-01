
import argparse
import time
from einops import rearrange
import numpy as np
import torch
from modules.latent_predictor import LatentEdgePredictor, hook_unet
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default="/root/workspace/sketch2img/edge_predictor.pt")
parser.add_argument("--input", type=str, default="/root/workspace/nahida/0e17302b9bfa15402f783c29c0d1d34f.jpg")
parser.add_argument("--prompt", type=str, default="1girl, masterpiece")
parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality, low quality")
args = parser.parse_args()

start_time = time.time()
scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
)
pipe = StableDiffusionPipeline.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=scheduler,
)

unet = pipe.unet
feature_blocks = hook_unet(unet)
edge_predictor = LatentEdgePredictor(9320, 4, 9)
edge_predictor.load_state_dict(torch.load(args.weights))

unet.enable_xformers_memory_efficient_attention()
edge_predictor.to(torch.device("cuda"), dtype=unet.dtype)
pipe = pipe.to("cuda")

timesteps = torch.tensor(100, dtype=torch.int64)
encoder_hidden_states = pipe._encode_prompt(args.prompt, unet.device, 1, True, args.negative_prompt)

def get_noise_level(noise, noise_scheduler, timesteps):
    sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
    noise_level = sqrt_one_minus_alpha_prod * noise
    return noise_level

def decode_latents(latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image[image<0.5] = 0
    image = image.squeeze(0) * 255
    
    return image.astype(np.uint8)

transform = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

px = transform(Image.open(args.input).convert("RGB")).to(vae.device, dtype=vae.dtype)
latents = vae.encode(torch.stack([px])).latent_dist.sample() * 0.18215
noise = torch.randn_like(latents, device=latents.device)
bsz = latents.shape[0]

noisy_latents = scheduler.add_noise(latents, noise, timesteps)
noise_level = get_noise_level(noise, scheduler, timesteps)

# Predict the noise residual
noisy_latents = torch.cat([noisy_latents] * 2)
noisy_latents = scheduler.scale_model_input(noisy_latents, timesteps)
unet(noisy_latents, timesteps, encoder_hidden_states)
                
intermidiate_result = []
for block in feature_blocks:
    uncond, cond = block.output.chunk(2)
    resized = torch.nn.functional.interpolate(cond, size=(latents.shape[2], latents.shape[3]), mode="bilinear") 
    intermidiate_result.append(resized)
                    
intermidiate_result = torch.cat(intermidiate_result, dim=1)
result = edge_predictor(intermidiate_result, noise_level)
result = rearrange(result, "(b w h) c -> b c h w", b=bsz, h=latents.shape[2], w=latents.shape[3])
opx = Image.fromarray(decode_latents(result))
opx.convert('L').save("output.png")