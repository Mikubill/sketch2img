# sketch2img

Sketch-to-Image Generation without retraining diffusion models.

Currently supported method: 

* paint with words and sketch (ediff-i)
* inject with additional pretrained self-attn layer or cross-attn layer (gligen)

## paint-with-words

1. Set your model path in https://github.com/Mikubill/sketch2img/blob/95419902e23187d91175489786b7bf0284b14574/app_pww.py#L33-L38
2. `python app_pww.py`

Some samples

| Sketch | Image |
|:-------------------------:|:-------------------------:|
