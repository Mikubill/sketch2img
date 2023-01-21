from model import create_model
from data import read_img_path, tensor_to_img
import torchtext
from PIL import Image
import torch

def pic2sketch(model, img, load_size=768):
    img, aus_resize = read_img_path(img, load_size)
    aus_tensor = model(img)
    print(aus_tensor.data.shape, aus_resize)
    aus_img = tensor_to_img(aus_tensor)
    image_pil = Image.fromarray(aus_img)
    image_pil = image_pil.resize(aus_resize, Image.LANCZOS)
    return image_pil

if __name__ == "__main__":
    torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2020/10/02/13/49/bridge-5621201_1280.jpg', 'building.jpg')
    torchtext.utils.download_from_url("https://huggingface.co/datasets/nyanko7/tmp-public/resolve/main/netG.pth", root="./weights/")
    model = create_model()
    model.eval()
    img = pic2sketch(model, 'building.jpg', 1024)
    img.save("output.png")