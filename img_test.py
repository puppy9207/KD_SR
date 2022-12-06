import argparse
import hydra

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

import time
from models import Generator
from utils import preprocess, postprocess, get_concat_h

@hydra.main(config_path="./configs", config_name="test")
def main(cfg):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Generator(cfg.test.generator).to(device)
    try:
        model.load_state_dict(torch.load(cfg.test.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(cfg.test.weights_file, map_location=device)["g"].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise RuntimeError("Error when loading a model")

    model.eval()

    image = pil_image.open(cfg.test.image_file).convert("RGB")

    bicubic = image.resize(
        (image.width * cfg.test.generator.scale, image.height * cfg.test.generator.scale),
        resample=pil_image.BICUBIC,
    )
    bicubic.save(cfg.test.image_file.replace(".", "_bicubic_x{}.".format(cfg.test.generator.scale)))

    lr = preprocess(image).to(device)
    bic = preprocess(bicubic).to(device)

    with torch.no_grad():
        print("here jinaga")
        start = time.time()
        preds = model(lr)
        end = time.time() - start
        print(end, "<<<<<<<<")
    output = postprocess(preds)
    output = pil_image.fromarray(output)
    print(cfg.test.image_file)
    output.save(
        cfg.test.image_file.replace(".", f"_{model.__class__.__name__}_x{cfg.test.generator.scale}.")
    )

    if cfg.test.merge:
        merge = get_concat_h(bicubic, output).save(
            cfg.test.image_file.replace(".", "_hconcat_.")
        )

if __name__ == "__main__":
    main()

    