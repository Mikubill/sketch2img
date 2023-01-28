# sketch2img

Sketch-to-Image Generation without retraining diffusion models.

Currently supported method: 

* paint with words and sketch (ediff-i)
* inject with additional pretrained self-attn layer or cross-attn layer (gligen?)

## paint-with-words

[Paper](https://arxiv.org/abs/2211.01324) | [Demo](https://huggingface.co/spaces/nyanko7/sd-diffusers-webui)

Paint-with-words is a method proposed by researchers from NVIDIA that allows users to control the location of objects by selecting phrases and painting them on the canvas. The user-specified masks increase the value of corresponding entries of the attention matrix in the cross-attention layers. 

Inspired by this method, we created a simple a1111-style sketching UI that allows multi-mask input to address same area on different tokens. Also, textual-inversion and LoRA support are fully functional*, you can add them to the generation process and adjust the strength and area they are applied to.

**Config and Run**

1. Set your model path in https://github.com/Mikubill/sketch2img/blob/660699bcb3eee8b34724f8c1c011fadb9de07b1a/app_pww.py#L28-L35
2. `python app_pww.py`

**Some samples**

| Sketch | Image |
|:-------------------------:|:-------------------------:|
|<img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-3-1.png">  |  <img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-3-2.png"> |
|<img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-1-compressed.png">  |  <img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-1-output-compressed.png"> |
