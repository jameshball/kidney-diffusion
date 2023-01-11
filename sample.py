from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from train import unet_generator, init_imagen

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    unet1_imagen = init_imagen(1)
    unet2_imagen = init_imagen(2)

    unet1_trainer = ImagenTrainer(
        imagen=unet1_imagen,
    )

    unet2_trainer = ImagenTrainer(
        imagen=unet2_imagen,
    )

    unet1_trainer.load(args.unet1_checkpoint)
    unet2_trainer.load(args.unet2_checkpoint)

    conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).repeat_interleave(args.num_images, dim=0).float().cuda()

    lowres_images = unet1_trainer.sample(
        batch_size=args.num_images,
        return_pil_images=False,
        text_embeds=conds,
        start_at_unet_number=1,
        stop_at_unet_number=1,
    )

    highres_images = unet2_trainer.sample(
        batch_size=args.num_images,
        return_pil_images=True,
        text_embeds=conds,
        start_image_or_video=lowres_images,
        start_at_unet_number=2,
        stop_at_unet_number=2,
    )

    for image in highres_images:
        image.save(f'samples/inference-{uuid4()}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    return parser.parse_args()


if __name__ == '__main__':
    main()
