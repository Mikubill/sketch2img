## Sketch-Guided Text-to-Image Diffusion 

[Paper](https://sketch-guided-diffusion.github.io/files/sketch-guided-preprint.pdf) | Demo

![intro](https://sketch-guided-diffusion.github.io/files/scheme_inference.jpg)

Sketch-Guided Text-to-Image is a method preposeed by researchers in Google Research to guide the inference process of a pretrained text-to-image diffusion model with an edge predictor that operates on the internal activations of the core network of the diffusion model, encouraging the edge of the synthesized image to follow a reference sketch.

Note: Paint-With-Words already moved to https://github.com/Mikubill/sd-paint-with-words

Pretrained LGP Weights: https://huggingface.co/nyanko7/sketch2img-edge-predictor-train/blob/main/edge_predictor.pt