
import time
import gradio as gr
import torch

from gradio import inputs
from modules.model_unet import SatMixin
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from modules.unet import SketchEncoder

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


# inject
unet = pipe.unet

sat_model = SatMixin(unet)
sat_model.load_state_dict(torch.load("/root/workspace/sketch2img/sketch_attn_model.pt"))

sketch_encoder = SketchEncoder.from_config(".")
sketch_encoder.load_state_dict(torch.load("/root/workspace/sketch2img/sketch_encoder_model.pt"))
sketch_encoder.enable_xformers_memory_efficient_attention()
unet.enable_xformers_memory_efficient_attention()

sketch_encoder.to(torch.device("cuda"), dtype=unet.dtype)
sat_model.to(torch.device("cuda"), dtype=unet.dtype)

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


transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
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
    spimg=None,
):
    global current_model
    generator = torch.Generator("cuda").manual_seed(
        seed) if seed != 0 else None

    global last_mode
    global pipe
    global current_model_path
    global vae
    global sketch_encoder
    global sat_model

    if spimg is not None:
        print(spimg)
        gsimg = Image.fromarray(spimg)
        tensor_img = torch.tile(transforms(gsimg), (3, 1, 1)).unsqueeze(0)
        stacked_img = (
            torch.stack([torch.zeros_like(tensor_img), tensor_img])
            .squeeze(1)
            .to(torch.device("cuda"), dtype=vae.dtype)
        )  # for uncond
        sketchs = vae.encode(stacked_img).latent_dist.sample() * 0.18215
        sketch_hidden_state = sketch_encoder(sketchs, 0, None).sample
        sat_model.set_res_samples(sketch_hidden_state)
        sat_model.set_scale(strength)

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


css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for orangemix</h1>
              </div>
              <p>Duplicating this space: <a style="display:inline-block" href="https://huggingface.co/spaces/akhaliq/anything-v3.0?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>       </p>
              </p>
            </div>
        """
    )
    with gr.Row():

        with gr.Column(scale=55):
            with gr.Group():
                with gr.Row():
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

                with gr.Row():
                    generate = gr.Button(value="Generate")

                image_out = gr.Image(height=512)
                # gallery = gr.Gallery(
                #     label="Generated images", show_label=False, elem_id="gallery"
                # ).style(grid=[1], height="auto")
            error_output = gr.Markdown()

        with gr.Column(scale=45):
        
    # with gr.Row():
        
            with gr.Tab("Options"):
                with gr.Group():
                    model = gr.Textbox(
                        interactive=False,
                        label="Model",
                        placeholder="Worangemix-Modified",
                    )

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
                            label="Width", value=512, minimum=64, maximum=1024, step=8
                        )
                        height = gr.Slider(
                            label="Height", value=512, minimum=64, maximum=1024, step=8
                        )

                    seed = gr.Slider(
                        0, 2147483647, label="Seed (0 = random)", value=0, step=1
                    )

            with gr.Tab("SketchPad"):
                with gr.Group():
                    sp = gr.Sketchpad(shape=(512, 512), tool="sketch")

                    strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )

    inputs = [
        prompt,
        guidance,
        steps,
        width,
        height,
        seed,
        strength,
        neg_prompt,
        sp,
    ]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
demo.launch(debug=True, share=False)
