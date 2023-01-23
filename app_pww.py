import time
import gradio as gr
import numpy as np
import torch

from gradio import inputs
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from modules.model_pww import CrossAttnProcessor, hook_unet, set_state

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

last_mode = "txt2img"

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
)
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=scheduler,
)
pipe = pipe_t2i
unet = pipe.unet
pipe.unet.set_attn_processor(CrossAttnProcessor)
hook_unet(pipe.tokenizer, pipe.unet)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

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
    strength=0.5,
    neg_prompt="",
    state=None,
    g_strength=0.4,
):
    global current_model
    generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None

    global last_mode
    global pipe
    global current_model_path
    global vae
    global sketch_encoder
    global sat_model

    set_state(pipe.unet, prompt, state, g_strength, guidance > 1)
    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    return result[0][0], None


color_list = []


def get_color(n):
    for _ in range(n - len(color_list)):
        color_list.append(tuple(np.random.random(size=3) * 256))
    return color_list


def create_mixed_img(current, state, w=512, h=512):
    w, h = int(w), int(h)
    image_np = np.full([w, h, 4], 255)
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
    trs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(max(w, h)),
            transforms.CenterCrop((w, h)),
        ]
    )

    for key, item in state.items():
        if item["map"] is not None:
            item["map"] = np.array(trs(item["map"]))

    update_img = gr.update(value=create_mixed_img("", state, w, h))
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


def switch_canvas(entry, state):
    if entry == None:
        return None, 1.0, create_mixed_img("", state)
    return (
        gr.update(value=None, interactive=True),
        gr.update(value=state[entry]["weight"]),
        create_mixed_img(entry, state),
    )


from torchvision import transforms

# sp.edit(apply_canvas, inputs=[radio, sp, global_stats], outputs=[global_stats, rendered])
def apply_canvas(selected, draw, state, w, h):
    w, h = int(w), int(h)
    trs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(max(w, h)),
            transforms.CenterCrop((w, h)),
        ]
    )

    state[selected]["map"] = np.array(trs(draw))
    ren = create_mixed_img(selected, state, w, h)
    return state, ren


def apply_weight(selected, weight, state):
    state[selected]["weight"] = weight
    return state


# def apply_rgb_image(image):


css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem
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
            with gr.Group():

                image_out = gr.Image(height=512)
                # gallery = gr.Gallery(
                #     label="Generated images", show_label=False, elem_id="gallery"
                # ).style(grid=[1], height="auto")

                
        with gr.Column(scale=45):
            

            model = gr.Textbox(
                interactive=False,
                label="Model",
                placeholder="Worangemix-Modified",
            )

            # with gr.Tab("Options"):
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

            generate = gr.Button(value="Generate")
            error_output = gr.Markdown()
        

    with gr.Row():

        with gr.Column(scale=55):

            rendered = gr.Image(
                invert_colors=True,
                source="canvas",
                interactive=False,
                shape=(512, 512),
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

                # g_output = gr.Markdown(r"Scaled additional attn: $w = 0.4 * \log (1 + \sigma) \std (Q^T K)$.")

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
                    label="Transformation strength",
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
                pass
                # sp2 = gr.Image(
                #     image_mode="L",
                #     source="upload",
                #     shape=(512, 512),
                # )

                # radio2 = gr.Textbox(label="Apply to...(Sperate by comma)")
                # apply_style = gr.Button(value="Submit")

                # apply_style.click(apply_rgb_image, input=[sp2], output=[global_stats, sp, radio, rendered])

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
        strength,
        neg_prompt,
        global_stats,
        g_strength,
    ]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
demo.launch(debug=True, share=False)
