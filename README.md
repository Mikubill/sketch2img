# sketch2img

Sketch-to-Image Generation without retraining diffusion models.

Currently supported method: 

* paint with words and sketch (ediff-i)
* inject with additional pretrained self-attn layer or cross-attn layer (gligen?)

## paint-with-words

1. Set your model path in https://github.com/Mikubill/sketch2img/blob/eb886f1f6cbcc92fa58fc519fa0466cb50b05e55/app_pww.py#L26-L33
2. `python app_pww.py`

Some samples

| Sketch | Image |
|:-------------------------:|:-------------------------:|
|<img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-3-1.png">  |  <img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-3-2.png"> |
|<img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-1-compressed.png">  |  <img width="256" alt="" src="https://raw.githubusercontent.com/Mikubill/sketch2img/main/samples/sample-1-output-compressed.png"> |
