import time
import gradio as gr
import numpy as np
import torch

from gradio import inputs
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)
from modules.model_pww import CrossAttnProcessor, hook_unet, set_state
from torchvision import transforms
from PIL import Image

start_time = time.time()
scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    prediction_type="epsilon",
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
)
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=scheduler,
)

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    vae=vae,
    unet=pipe_t2i.unet,
    torch_dtype=torch.float16,
    scheduler=scheduler,
)

pipe = pipe_t2i
unet = pipe.unet
pipe.unet.set_attn_processor(CrossAttnProcessor)
hook_unet(pipe.tokenizer, pipe.unet)

if torch.cuda.is_available():
    pipe_t2i = pipe_t2i.to("cuda")
    pipe_i2i = pipe_i2i.to("cuda")

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
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
):
    global pipe_t2i, pipe_i2i
    generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
    set_state(pipe.unet, prompt, state, g_strength, guidance > 1)

    config = {
        "negative_prompt": neg_prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": guidance,
        "generator": generator,
    }

    if img_input is not None:
        ratio = min(height / img_input.height, width / img_input.width)
        img_input = img_input.resize(
            (int(img_input.width * ratio), int(img_input.height * ratio)), Image.LANCZOS
        )
        result = pipe_i2i(prompt, image=img_input, strength=i2i_scale, **config)
    else:
        result = pipe_t2i(prompt, width=width, height=height, **config)
    return result[0][0]


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
            }
        else:
            new_state[item] = {
                "map": None,
                "weight": 1.0,
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


def switch_canvas(entry, state):
    if entry == None:
        return None, 1.0, create_mixed_img("", state)
    return (
        gr.update(value=None, interactive=True),
        gr.update(value=state[entry]["weight"]),
        create_mixed_img(entry, state),
    )


def apply_canvas(selected, draw, state, w, h):
    w, h = int(w), int(h)
    state[selected]["map"] = resize(draw, w, h)
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


def apply_weight(selected, weight, state):
    state[selected]["weight"] = weight
    return state


# sp2, radio, width, height, global_stats
def apply_image(image, selected, w, h, strgength, state):
    if selected is not None:
        state[selected] = {"map": resize(image, w, h), "weight": strgength}
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


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
 """
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for orangemix</h1>
              </div>
              <p>
                <br />
                <a style="display:inline-block" href="https://huggingface.co/spaces/akhaliq/anything-v3.0?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>       </p>
              </p>
            </div>
        """
    )
    global_stats = gr.State(value={})

    with gr.Row():

        with gr.Column(scale=55):

                image_out = gr.Image(height=512)
                # gallery = gr.Gallery(
                #     label="Generated images", show_label=False, elem_id="gallery"
                # ).style(grid=[1], height="auto")

        with gr.Column(scale=45):
            
                with gr.Group():
                
                    model = gr.Textbox(
                        interactive=False,
                        label="Model",
                        placeholder="Worangemix-Modified",
                    )
                
                    with gr.Row():
                        with gr.Column(scale=70):
                            prompt = gr.Textbox(
                                    label="Prompt",
                                    show_label=True,
                                    max_lines=2,
                                    placeholder="Enter prompt.",
                            )
                            neg_prompt = gr.Textbox(
                                    label="Negative Prompt",
                                    show_label=True,
                                    max_lines=2,
                                    placeholder="Enter negative prompt.",
                            )
                            
                        generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))
            
                        
                with gr.Tab("Options"):
                    
                    with gr.Group():

                    # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)
                        with gr.Row():
                            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                            steps = gr.Slider(
                                label="Steps", value=25, minimum=2, maximum=75, step=1
                            )

                        with gr.Row():
                            width = gr.Slider(
                                label="Width", value=512, minimum=64, maximum=1024, step=8
                            )
                            height = gr.Slider(
                                label="Height", value=512, minimum=64, maximum=1024, step=8
                            )

                        seed = gr.Slider(
                            0, 2147483647, label="Seed (0 = random)", value=0, step=1
                        )
                        

                with gr.Tab("Image to image"):
                    with gr.Group():
                        
                        inf_image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                        inf_strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

            
            # error_output = gr.Markdown()

    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Sketch Injection</h1>
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

                g_strength = gr.Slider(
                    label="Weight scaling",
                    minimum=0,
                    maximum=2,
                    step=0.01,
                    value=0.6,
                )
                
                text = gr.Textbox(
                    lines=2,
                    interactive=True,
                    label="Token to Draw: (Separate by comma)",
                )

                radio = gr.Radio([], label="Tokens")

                # g_strength.change(lambda b: gr.update(f"Scaled additional attn: $w = {b} \log (1 + \sigma) \std (Q^T K)$."), inputs=g_strength, outputs=[g_output])

            with gr.Tab("SketchPad"):

                sp = gr.Image(
                    image_mode="L",
                    tool="sketch",
                    source="canvas",
                    interactive=False,
                )

                strength = gr.Slider(
                    label="Token weight",
                    minimum=0,
                    maximum=2,
                    step=0.01,
                    value=1.0,
                )

                text.change(
                    detect_text,
                    inputs=[text, global_stats, width, height],
                    outputs=[global_stats, sp, radio, rendered],
                )
                radio.change(
                    switch_canvas,
                    inputs=[radio, global_stats],
                    outputs=[sp, strength, rendered],
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

            with gr.Tab("UploadFile"):

                sp2 = gr.Image(
                    image_mode="L",
                    source="upload",
                    shape=(512, 512),
                )

                strength2 = gr.Slider(
                    label="Token strength",
                    minimum=0,
                    maximum=2,
                    step=0.01,
                    value=1.0,
                )

                apply_style = gr.Button(value="Apply")
                apply_style.click(
                    apply_image,
                    inputs=[sp2, radio, width, height, strength2, global_stats],
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
    ]
    outputs = [image_out]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
demo.launch(debug=True, share=False)
