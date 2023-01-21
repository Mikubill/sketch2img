import argparse
import contextlib
import itertools
import math
import os
import tempfile

import bitsandbytes as bnb
import torch
import torchtext
import torchvision
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from anime2sketch.model import create_model
from modules.dataset import ImageStore
from modules.model import SatMixin
from modules.unet import SketchEncoder
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTokenizer

import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    return args

def generate_sketch(sketch_generator, img, fixed=512, method=torchvision.transforms.InterpolationMode.BICUBIC):
    org_size = (img.shape[-2], img.shape[-1])
    transformed = torchvision.transforms.Resize((fixed, fixed), method)(img)
    tiled = torch.tile(sketch_generator(transformed), (3, 1, 1))
    resized = torchvision.transforms.Resize(org_size, method)(tiled)
    return resized

def encode_tokens(tokenizer, text_encoder, input_ids):
    z = []
    if input_ids.shape[1] > 77:  
        # todo: Handle end-of-sentence truncation
        while max(map(len, input_ids)) != 0:
            rem_tokens = [x[75:] for x in input_ids]
            tokens = []
            for j in range(len(input_ids)):
                tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [tokenizer.eos_token_id] * 75)

            rebuild = [[tokenizer.bos_token_id] + list(x[:75]) + [tokenizer.eos_token_id] for x in tokens]
            if hasattr(torch, "asarray"):
                z.append(torch.asarray(rebuild))
            else:
                z.append(torch.IntTensor(rebuild))
            input_ids = rem_tokens
    else:
        z.append(input_ids)

    # Get the text embedding for conditioning
    encoder_hidden_states = None
    for tokens in z:
        state = text_encoder(tokens.to(text_encoder.device), output_hidden_states=True)
        state = text_encoder.text_model.final_layer_norm(state['hidden_states'][-1])
        encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)
        
    return encoder_hidden_states

def train():

    args = parse_args()
    config = OmegaConf.load(args.config)
    # get_world_size = lambda: int(os.environ.get("WORLD_SIZE", 1))
    get_local_rank = lambda: int(os.environ.get("LOCAL_RANK", -1))
    set_seed(config.seed)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = ImageStore(
        size=config.resolution,
        seed=config.seed,
        rank=get_local_rank(),
        tokenizer=tokenizer,
        **config.dataset
    )
    
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    metrics = []
    if config.monitor.wandb_id != "" and get_local_rank() in [0, -1]:
        import wandb
        wandb.init(project=config.monitor.wandb_id, reinit=False)
        metrics.append("wandb")
        
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_scaler], log_with=metrics,)

    # モデルを読み込む
    pipe = StableDiffusionPipeline.from_pretrained(config.model_path, tokenizer=None, safety_checker=None)
    text_encoder, vae, unet = pipe.text_encoder, pipe.vae, pipe.unet
    del pipe

    # モデルに xformers とか memory efficient attention を組み込む
    unet.enable_xformers_memory_efficient_attention()

    # prepare network
    sat_model = SatMixin(unet)
    sketch_encoder = SketchEncoder.from_config(".")
    sketch_encoder.enable_xformers_memory_efficient_attention()
        
    # generator
    torchtext.utils.download_from_url("https://huggingface.co/datasets/nyanko7/tmp-public/resolve/main/netG.pth", root="./weights/")
    sketch_generator =  create_model()
    sketch_generator.eval()

    # if args.network_weights is not None:
    #     print("load network weights from:", args.network_weights)
    #     network.load_weights(args.network_weights)

    params_to_optim = itertools.chain.from_iterable([sat_model.parameters(), sketch_encoder.parameters()])
    optimizer = bnb.optim.AdamW8bit(
        params_to_optim,
        **config.optimizer.params
    )

    # dataloaderを準備する
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        num_workers=config.dataset.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        persistent_workers=True,
    )

    # 学習ステップ数を計算する
    max_train_steps = config.train_epochs * len(train_dataloader)
    
    # lr schedulerを用意する
    lr_scheduler = diffusers.optimization.get_scheduler(
        "cosine_with_restarts",
        optimizer,
        num_warmup_steps=150,
        num_training_steps=max_train_steps,
    )

    unet, sketch_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, sketch_encoder, optimizer, train_dataloader, lr_scheduler
    )


    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    sketch_generator.to(accelerator.device)
    sat_model.to(accelerator.device)
    
    unet.eval()
    vae.eval()
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    sat_model.requires_grad_(True)
    sketch_encoder.requires_grad_(True)
    
    if config.monitor.huggingface_repo and get_local_rank() in [0, -1]:
        from huggingface_hub import Repository
        from huggingface_hub.constants import ENDPOINT
        repo = Repository(
            tempfile.TemporaryDirectory().name,
            clone_from=f"{ENDPOINT}/{config.monitor.huggingface_repo}",
            use_auth_token=config.monitor.huggingface_token,
            revision=None, 
        )

    # resumeする
    if args.resume is not None:
        print(f"resume training from state: {args.resume}")
        accelerator.load_state(args.resume)

    # epoch数を計算する
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    progress_bar = tqdm(
        range(max_train_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
    global_step = 0

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("network_train")

    for epoch in range(num_train_epochs):
        print(f"epoch {epoch+1}/{num_train_epochs}")

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([sat_model, sketch_encoder]):
                
                input_ids, px = batch[0], batch[1]
                with torch.no_grad():
                    input_ids = input_ids.to(accelerator.device)
                    encoder_hidden_states = encode_tokens(tokenizer, text_encoder, input_ids)
                    latents = vae.encode(px).latent_dist.sample() * 0.18215
                    sketchs = vae.encode(generate_sketch(sketch_generator, px)).latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents 
                noise = torch.randn_like(latents, device=latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Fixed timestep to avoid multiply forward pass during inference.
                sketch_hidden_state = sketch_encoder(sketchs, 0, None).sample
                sat_model.set_res_samples(sketch_hidden_state)

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")  

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optim, 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            current_loss = loss.detach().item()
            logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)

            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        
    if accelerator.is_main_process:
        ctx = contextlib.nullcontext()
        if config.monitor.huggingface_repo and accelerator.is_main_process:
            ctx = repo.commit(f"add/update model: epoch {epoch}", blocking=False, auto_lfs_prune=True)
            
        with ctx:
            accelerator.save(accelerator.unwrap_model(sketch_encoder).state_dict(), "sketch_encoder_model.pt")
            accelerator.save(accelerator.unwrap_model(sat_model).state_dict(), "sketch_attn_model.pt")

        # end of epoch
    accelerator.end_training()


if __name__ == "__main__":
    train()
