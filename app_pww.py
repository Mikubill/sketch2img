import random
import re
import string
import tempfile
import time
import gradio as gr
import numpy as np
import torch

from gradio import inputs
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from modules.model_pww import (
    CrossAttnProcessor,
    StableDiffusionPipeline,
)
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file
import modules.safe as _
from modules.lora import LoRANetwork

models = [
    ("AbyssOrangeMix_Base", "/root/workspace/storage/models/orangemix"),
    ("Stable Diffuison 1.5", "/root/models/stable-diffusion-v1-5"),
    ("AnimeSFW", "/root/workspace/animesfw"),
]

base_name, base_model = models[0]
clip_skip = 1

samplers_k_diffusion = [
    ("Euler a", "sample_euler_ancestral", {}),
    ("Euler", "sample_euler", {}),
    ("LMS", "sample_lms", {}),
    ("Heun", "sample_heun", {}),
    ("DPM2", "sample_dpm_2", {"discard_next_to_last_sigma": True}),
    ("DPM2 a", "sample_dpm_2_ancestral", {"discard_next_to_last_sigma": True}),
    ("DPM++ 2S a", "sample_dpmpp_2s_ancestral", {}),
    ("DPM++ 2M", "sample_dpmpp_2m", {}),
    ("DPM++ SDE", "sample_dpmpp_sde", {}),
    ("LMS Karras", "sample_lms", {"scheduler": "karras"}),
    ("DPM2 Karras", "sample_dpm_2", {"scheduler": "karras", "discard_next_to_last_sigma": True}),
    ("DPM2 a Karras", "sample_dpm_2_ancestral", {"scheduler": "karras", "discard_next_to_last_sigma": True}),
    ("DPM++ 2S a Karras", "sample_dpmpp_2s_ancestral", {"scheduler": "karras"}),
    ("DPM++ 2M Karras", "sample_dpmpp_2m", {"scheduler": "karras"}),
    ("DPM++ SDE Karras", "sample_dpmpp_sde", {"scheduler": "karras"}),
]

# samplers_diffusers = [
#     ("DDIMScheduler", "diffusers.schedulers.DDIMScheduler", {})
#     ("DDPMScheduler", "diffusers.schedulers.DDPMScheduler", {})
#     ("DEISMultistepScheduler", "diffusers.schedulers.DEISMultistepScheduler", {})
# ]

start_time = time.time()

scheduler = DDIMScheduler.from_pretrained(
    base_model,
    subfolder="scheduler",
)
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema", 
    torch_dtype=torch.float16
)
text_encoder = CLIPTextModel.from_pretrained(
    base_model,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
tokenizer = CLIPTokenizer.from_pretrained(
    base_model,
    subfolder="tokenizer",
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    base_model,
    subfolder="unet",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    vae=vae,
    scheduler=scheduler,
)

unet.set_attn_processor(CrossAttnProcessor)
pipe.set_clip_skip(clip_skip)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")


def get_model_list():
    model_available = []
    for model in models:
        if Path(model[1]).is_dir():
            model_available.append(model)
    return model_available

unet_cache = {
    base_name: unet
}

lora_cache = {
    base_name: LoRANetwork(text_encoder, unet)
}

def get_model(name):
    keys = [k[0] for k in models]
    if name not in unet_cache:
        if name not in keys:
            raise ValueError(name)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                models[keys.index(name)][1],
                subfolder="unet",
                torch_dtype=torch.float16,
            )
            if torch.cuda.is_available():
                unet.to("cuda")
                
            unet_cache[name] = unet
            lora_cache[name] = LoRANetwork(lora_cache[base_name].text_encoder_loras, unet)

    g_unet = unet_cache[name]
    g_lora = lora_cache[name]
    g_unet.set_attn_processor(CrossAttnProcessor())
    g_lora.reset()
    return g_unet, g_lora

def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )

te_base_weight_length = text_encoder.get_input_embeddings().weight.data.shape[0]
original_prepare_for_tokenization = tokenizer.prepare_for_tokenization

def make_token_names(embs):
    all_tokens = []
    for name, vec in embs.items():
        tokens = [f'emb-{name}-{i}' for i in range(len(vec))]
        all_tokens.append(tokens)
    return all_tokens

def setup_tokenizer(embs):
    reg_match = [re.compile(fr"(?:^|(?<=\s|,)){k}(?=,|\s|$)") for k in embs.keys()]
    clip_keywords = [' '.join(s) for s in make_token_names(embs)]

    def parse_prompt(prompt: str):
        for m, v in zip(reg_match, clip_keywords):
            prompt = m.sub(v, prompt)
        return prompt
        
    def prepare_for_tokenization(self, text: str, is_split_into_words: bool = False, **kwargs):
        text = parse_prompt(text)
        r = original_prepare_for_tokenization(text, is_split_into_words, **kwargs)
        return r

    tokenizer.prepare_for_tokenization = prepare_for_tokenization.__get__(tokenizer, CLIPTokenizer)
    return [t for sublist in make_token_names(embs) for t in sublist]

def restore_all():
    global te_base_weight, tokenizer
    tokenizer.prepare_for_tokenization = original_prepare_for_tokenization
    
    embeddings = text_encoder.get_input_embeddings()
    text_encoder.get_input_embeddings().weight.data = embeddings.weight.data[:te_base_weight_length]
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
        torch_dtype=torch.float16,
    )


def inference(
    prompt,
    guidance,
    steps,
    width=512,
    height=512,
    seed=0,
    neg_prompt="",
    state=None,
    g_strength=0.4,
    img_input=None,
    i2i_scale=0.5,
    hr_enabled=False,
    hr_method="Latent",
    hr_scale=1.5,
    hr_denoise=0.8,
    sampler="DPM++ 2M Karras",
    embs=None,
    model=None,
    lora_state=None,
    lora_scale=None,
):
    global pipe, unet, tokenizer, text_encoder
    if seed is None or seed == 0:
        seed = random.randint(0, 2147483647)
    generator = torch.Generator("cuda").manual_seed(int(seed))

    local_unet, local_lora = get_model(model)
    if lora_state is not None and lora_state != "":
        local_lora.load(lora_state, lora_scale)
        local_lora.to(local_unet.device, dtype=local_unet.dtype)

    pipe.setup_unet(local_unet)
    sampler_name, sampler_opt = None, None
    for label, funcname, options in samplers_k_diffusion:
        if label == sampler:
            sampler_name, sampler_opt = funcname, options

    if embs is not None and len(embs) > 0:
        delta_weight = []
        ti_embs = {}
        for name, file in embs.items():
            if str(file).endswith(".pt"):
                loaded_learned_embeds = torch.load(file, map_location="cpu")
            else:
                loaded_learned_embeds = load_file(file, device="cpu")
            loaded_learned_embeds = loaded_learned_embeds["string_to_param"]["*"]
            ti_embs[name] = loaded_learned_embeds
            
        if len(ti_embs) > 0:
            tokens = setup_tokenizer(ti_embs)
            added_tokens = tokenizer.add_tokens(tokens)
            delta_weight = torch.cat([val for val in ti_embs.values()], dim=0)
            
            assert added_tokens == delta_weight.shape[0]
            text_encoder.resize_token_embeddings(len(tokenizer))
            text_encoder.get_input_embeddings().weight.data[-delta_weight.shape[0]:] = delta_weight

    config = {
        "negative_prompt": neg_prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": guidance,
        "generator": generator,
        "sampler_name": sampler_name,
        "sampler_opt": sampler_opt,
        "pww_state": state,
        "pww_attn_weight": g_strength,
    }

    if img_input is not None:
        ratio = min(height / img_input.height, width / img_input.width)
        img_input = img_input.resize(
            (int(img_input.width * ratio), int(img_input.height * ratio)), Image.LANCZOS
        )
        result = pipe.img2img(prompt, image=img_input, strength=i2i_scale, **config)
    elif hr_enabled:
        result = pipe.txt2img(
            prompt,
            width=width,
            height=height,
            upscale=True,
            upscale_x=hr_scale,
            upscale_denoising_strength=hr_denoise,
            **config,
            **latent_upscale_modes[hr_method],
        )
    else:
        result = pipe.txt2img(prompt, width=width, height=height, **config)

    # restore
    if embs is not None and len(embs) > 0:
        restore_all()
    return gr.Image.update(result[0][0], label=f"Initial Seed: {seed}")


color_list = []


def get_color(n):
    for _ in range(n - len(color_list)):
        color_list.append(tuple(np.random.random(size=3) * 256))
    return color_list


def create_mixed_img(current, state, w=512, h=512):
    w, h = int(w), int(h)
    image_np = np.full([h, w, 4], 255)
    colors = get_color(len(state))
    idx = 0

    for key, item in state.items():
        if item["map"] is not None:
            m = item["map"] < 255
            alpha = 150
            if current == key:
                alpha = 200
            image_np[m] = colors[idx] + (alpha,)
        idx += 1

    return image_np


# width.change(apply_new_res, inputs=[width, height, global_stats], outputs=[global_stats, sp, rendered])
def apply_new_res(w, h, state):
    w, h = int(w), int(h)

    for key, item in state.items():
        if item["map"] is not None:
            item["map"] = resize(item["map"], w, h)

    update_img = gr.Image.update(value=create_mixed_img("", state, w, h))
    return state, update_img


def detect_text(text, state, width, height):
    
    if text is None or text == "":
        return None, None, None, None

    t = text.split(",")
    new_state = {}

    for item in t:
        item = item.strip()
        if item == "":
            continue
        if item in state:
            new_state[item] = {
                "map": state[item]["map"],
                "weight": state[item]["weight"],
                "mask_outsides": state[item]["mask_outsides"],
            }
        else:
            new_state[item] = {
                "map": None,
                "weight": 0.5,
                "mask_outsides": False
            }
    update = gr.Radio.update(choices=[key for key in new_state.keys()], value=None)
    update_img = gr.update(value=create_mixed_img("", new_state, width, height))
    update_sketch = gr.update(value=None, interactive=False)
    return new_state, update_sketch, update, update_img


def resize(img, w, h):
    trs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(min(h, w)),
            transforms.CenterCrop((h, w)),
        ]
    )
    result = np.array(trs(img), dtype=np.uint8)
    return result


def switch_canvas(entry, state, width, height):
    if entry == None:
        return None, 0.5, create_mixed_img("", state, width, height)

    return (
        gr.update(value=None, interactive=True),
        gr.update(value=state[entry]["weight"] if entry in state else 0.5),
        gr.update(value=state[entry]["mask_outsides"] if entry in state else False),
        create_mixed_img(entry, state, width, height),
    )


def apply_canvas(selected, draw, state, w, h):
    if selected in state:
        w, h = int(w), int(h)
        state[selected]["map"] = resize(draw, w, h)
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


def apply_weight(selected, weight, state):
    if selected in state:
        state[selected]["weight"] = weight
    return state


def apply_option(selected, mask, state):
    if selected in state:
        state[selected]["mask_outsides"] = mask
    return state


# sp2, radio, width, height, global_stats
def apply_image(image, selected, w, h, strgength, mask, state):
    if selected in state:
        state[selected] = {
            "map": resize(image, w, h), 
            "weight": strgength, 
            "mask_outsides": mask
        }
        
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))

# [ti_state, lora_state, ti_vals, lora_vals, uploads]
def add_net(files, ti_state, lora_state):
    if files is None:
        return ti_state, "", lora_state, None

    for file in files:
        item = Path(file.name)
        stripedname = str(item.stem).strip()
        if item.suffix == ".pt":
            state_dict = torch.load(file.name, map_location="cpu")
        else:
            state_dict = load_file(file.name, device="cpu")
        if any("lora" in k for k in state_dict.keys()):
            lora_state = file.name
        else:
            ti_state[stripedname] = file.name

    return (
        ti_state,
        lora_state,
        gr.Text.update(f"{[key for key in ti_state.keys()]}"),
        gr.Text.update(f"{lora_state}"),
        gr.Files.update(value=None),
    )


# [ti_state, lora_state, ti_vals, lora_vals, uploads]
def clean_states(ti_state, lora_state):
    return (
        dict(),
        None,
        gr.Text.update(f""),
        gr.Text.update(f""),
        gr.File.update(value=None),
    )


latent_upscale_modes = {
    "Latent": {"upscale_method": "bilinear", "upscale_antialias": False},
    "Latent (antialiased)": {"upscale_method": "bilinear", "upscale_antialias": True},
    "Latent (bicubic)": {"upscale_method": "bicubic", "upscale_antialias": False},
    "Latent (bicubic antialiased)": {
        "upscale_method": "bicubic",
        "upscale_antialias": True,
    },
    "Latent (nearest)": {"upscale_method": "nearest", "upscale_antialias": False},
    "Latent (nearest-exact)": {
        "upscale_method": "nearest-exact",
        "upscale_antialias": False,
    },
}

css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem;
    padding-top:2rem;
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
.box {
  float: left;
  height: 20px;
  width: 20px;
  margin-bottom: 15px;
  border: 1px solid black;
  clear: both;
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
.no-border {
    border: none !important;
}
 """
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for diffusion models</h1>
              </div>
              <p>Hso @ nyanko.sketch2img.gradio</p>
            </div>
        """
    )
    global_stats = gr.State(value={})

    with gr.Row():

        with gr.Column(scale=55):
            model = gr.Dropdown(
                choices=[k[0] for k in get_model_list()],
                label="Model",
                value=base_name,
            )
            image_out = gr.Image(height=512)
        # gallery = gr.Gallery(
        #     label="Generated images", show_label=False, elem_id="gallery"
        # ).style(grid=[1], height="auto")

        with gr.Column(scale=45):

            with gr.Group():

                with gr.Row():
                    with gr.Column(scale=70):

                        prompt = gr.Textbox(
                            label="Prompt",
                            value="loli cat girl, blue eyes, flat chest, solo, long messy silver hair, blue capelet, cat ears, cat tail, upper body",
                            show_label=True,
                            max_lines=4,
                            placeholder="Enter prompt.",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="bad quality, low quality, jpeg artifact, cropped",
                            show_label=True,
                            max_lines=4,
                            placeholder="Enter negative prompt.",
                        )

                    generate = gr.Button(value="Generate").style(
                        rounded=(False, True, True, False)
                    )

            with gr.Tab("Options"):

                with gr.Group():

                    # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps", value=25, minimum=2, maximum=75, step=1
                        )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width", value=512, minimum=64, maximum=2048, step=64
                        )
                        height = gr.Slider(
                            label="Height", value=512, minimum=64, maximum=2048, step=64
                        )

                    sampler = gr.Dropdown(
                        value="DPM++ 2M Karras",
                        label="Sampler",
                        choices=[s[0] for s in samplers_k_diffusion],
                    )
                    seed = gr.Number(label="Seed (0 = random)", value=0)

            with gr.Tab("Image to image"):
                with gr.Group():

                    inf_image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    inf_strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )

            def res_cap(g, w, h, x):
                if g:
                    return f"Enable upscaler: {w}x{h} to {int(w*x)}x{int(h*x)}"
                else:
                    return "Enable upscaler"

            with gr.Tab("Hires fix"):
                with gr.Group():

                    hr_enabled = gr.Checkbox(label="Enable upscaler", value=False)
                    hr_method = gr.Dropdown(
                        [key for key in latent_upscale_modes.keys()],
                        value="Latent",
                        label="Upscale method",
                    )
                    hr_scale = gr.Slider(
                        label="Upscale factor",
                        minimum=1.0,
                        maximum=3,
                        step=0.1,
                        value=1.5,
                    )
                    hr_denoise = gr.Slider(
                        label="Denoising strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                    )

                    hr_scale.change(
                        lambda g, x, w, h: gr.Checkbox.update(
                            label=res_cap(g, w, h, x)
                        ),
                        inputs=[hr_enabled, hr_scale, width, height],
                        outputs=hr_enabled,
                    )
                    hr_enabled.change(
                        lambda g, x, w, h: gr.Checkbox.update(
                            label=res_cap(g, w, h, x)
                        ),
                        inputs=[hr_enabled, hr_scale, width, height],
                        outputs=hr_enabled,
                    )

            with gr.Tab("Embeddings/Loras"):

                ti_state = gr.State(dict())
                lora_state = gr.State()

                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=90):
                            ti_vals = gr.Text(label="Loaded embeddings")

                    with gr.Row():
                        with gr.Column(scale=90):
                            lora_vals = gr.Text(label="Loaded loras")

                with gr.Row():

                    uploads = gr.Files(label="Upload new embeddings/lora")

                    with gr.Column():
                        lora_scale = gr.Slider(
                            label="Lora scale",
                            minimum=0,
                            maximum=2,
                            step=0.01,
                            value=1.0,
                        )
                        btn = gr.Button(value="Upload")
                        btn_del = gr.Button(value="Reset")

                btn.click(
                    add_net,
                    inputs=[uploads, ti_state, lora_state],
                    outputs=[ti_state, lora_state, ti_vals, lora_vals, uploads],
                )
                btn_del.click(
                    clean_states,
                    inputs=[ti_state, lora_state],
                    outputs=[ti_state, lora_state, ti_vals, lora_vals, uploads],
                )

        # error_output = gr.Markdown()

    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Paint with words</h1>
              </div>
              <p>
                Will use the following formula: w = scale * token_weight_martix * log(1 + sigma) * max(qk).
              </p>
            </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=55):

            rendered = gr.Image(
                invert_colors=True,
                source="canvas",
                interactive=False,
                image_mode="RGBA",
            )

        with gr.Column(scale=45):

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=70):
                        g_strength = gr.Slider(
                            label="Weight scaling",
                            minimum=0,
                            maximum=0.8,
                            step=0.01,
                            value=0.4,
                        )

                        text = gr.Textbox(
                            lines=2,
                            interactive=True,
                            label="Token to Draw: (Separate by comma)",
                        )

                        radio = gr.Radio([], label="Tokens")

                    sk_update = gr.Button(value="Update").style(
                        rounded=(False, True, True, False)
                    )

                # g_strength.change(lambda b: gr.update(f"Scaled additional attn: $w = {b} \log (1 + \sigma) \std (Q^T K)$."), inputs=g_strength, outputs=[g_output])

            with gr.Tab("SketchPad"):

                sp = gr.Image(
                    image_mode="L",
                    tool="sketch",
                    source="canvas",
                    interactive=False,
                )

                mask_outsides = gr.Checkbox(
                    label="Mask other areas", 
                    value=False
                )

                strength = gr.Slider(
                    label="Token strength",
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.5,
                )
                

                sk_update.click(
                    detect_text,
                    inputs=[text, global_stats, width, height],
                    outputs=[global_stats, sp, radio, rendered],
                )
                radio.change(
                    switch_canvas,
                    inputs=[radio, global_stats, width, height],
                    outputs=[sp, strength, mask_outsides, rendered],
                )
                sp.edit(
                    apply_canvas,
                    inputs=[radio, sp, global_stats, width, height],
                    outputs=[global_stats, rendered],
                )
                strength.change(
                    apply_weight,
                    inputs=[radio, strength, global_stats],
                    outputs=[global_stats],
                )
                mask_outsides.change(
                    apply_option,
                    inputs=[radio, mask_outsides, global_stats],
                    outputs=[global_stats],
                )

            with gr.Tab("UploadFile"):

                sp2 = gr.Image(
                    image_mode="L",
                    source="upload",
                    shape=(512, 512),
                )
                
                mask_outsides2 = gr.Checkbox(
                    label="Mask other areas", 
                    value=False
                )

                strength2 = gr.Slider(
                    label="Token strength",
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.5,
                )

                apply_style = gr.Button(value="Apply")
                apply_style.click(
                    apply_image,
                    inputs=[sp2, radio, width, height, strength2, mask_outsides2, global_stats],
                    outputs=[global_stats, rendered],
                )

            width.change(
                apply_new_res,
                inputs=[width, height, global_stats],
                outputs=[global_stats, rendered],
            )
            height.change(
                apply_new_res,
                inputs=[width, height, global_stats],
                outputs=[global_stats, rendered],
            )

    # color_stats = gr.State(value={})
    # text.change(detect_color, inputs=[sp, text, color_stats], outputs=[color_stats, rendered])
    # sp.change(detect_color, inputs=[sp, text, color_stats], outputs=[color_stats, rendered])

    inputs = [
        prompt,
        guidance,
        steps,
        width,
        height,
        seed,
        neg_prompt,
        global_stats,
        g_strength,
        inf_image,
        inf_strength,
        hr_enabled,
        hr_method,
        hr_scale,
        hr_denoise,
        sampler,
        ti_state,
        model,
        lora_state,
        lora_scale,
    ]
    outputs = [image_out]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
# demo.launch(share=True)
demo.launch()
