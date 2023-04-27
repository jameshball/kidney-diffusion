from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from train_ultra_res import unet_generator, init_imagen

import os
import gc
import pandas as pd
from glob import glob

import re

def generate_images_with_unet(unet_number, mag_level, args, lowres_images=None):
    imagen = init_imagen(unet_number)
    trainer = ImagenTrainer(imagen=imagen)

    if unet_number == 1:
        path = args.unet1_checkpoint
    elif unet_number == 2:
        path = args.unet2_checkpoint
    else:
        path = args.unet3_checkpoint

    trainer.load(path)

    conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).repeat_interleave(args.num_images, dim=0).float().cuda()

    images = trainer.sample(
        batch_size=args.num_images,
        return_pil_images=(unet_number==3),
        text_embeds=conds,
        cond_images=torch.zeros((args.num_images, 4, 1024, 1024)).cuda(),
        start_image_or_video=lowres_images,
        start_at_unet_number=unet_number,
        stop_at_unet_number=unet_number,
    )

    del trainer
    del imagen
    del conds
    gc.collect()
    torch.cuda.empty_cache()

    return images


def generate_images(mag_level, args):
    lowres_images = generate_images(1, mag_level, args)
    medres_images = generate_images(2, mag_level, args, lowres_images=lowres_images)
    highres_images = generate_images(3, mag_level, args, lowres_images=medres_images)

    return highres_images


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    images = generate_images(0, args)

    for image in images:
        image.save(f'samples/inference-{uuid4()}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_mag0', type=str)
    parser.add_argument('--unet1_mag1', type=str)
    parser.add_argument('--unet1_mag2', type=str)
    parser.add_argument('--unet2_mag0', type=str)
    parser.add_argument('--unet2_mag1', type=str)
    parser.add_argument('--unet2_mag2', type=str)
    parser.add_argument('--unet3_mag0', type=str)
    parser.add_argument('--unet3_mag1', type=str)
    parser.add_argument('--unet3_mag2', type=str)
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    return parser.parse_args()


if __name__ == '__main__':
    main()
