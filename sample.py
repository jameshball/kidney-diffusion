import random
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
import argparse

from ema_pytorch import EMA
from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from imagen_pytorch.trainer import restore_parts, exists
from torch import nn

from imagen_pytorch.version import __version__
from packaging import version

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re


CHECKPOINT_PATH = "./checkpoint.pt"
TEXT_EMBED_DIM = 3


def unet_generator(unet_number):
    if unet_number == 1:
        return Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            cond_dim=32,
            text_embed_dim=3,
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
        )

    if unet_number == 2:
        return Unet(
            dim=64,
            cond_dim=32,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
            text_embed_dim=TEXT_EMBED_DIM,
            lowres_cond=True,
        )

    return None


def init_imagen():
    imagen = Imagen(
        unets=(unet_generator(1), unet_generator(2)),
        image_sizes=(64, 256),
        timesteps=1000,
        text_embed_dim=TEXT_EMBED_DIM,
    )

    return imagen


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    imagen = init_imagen()
    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True,
    ).cuda()

    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)

    conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).repeat_interleave(args.num_images, dim=0).float().cuda()
    images = trainer.sample(
        batch_size=args.num_images,
        return_pil_images=True,
        text_embeds=conds,
    )

    for index in range(len(images)):
        images[index].save(f'samples/inference-{uuid4()}-{index}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to checkpoint')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    return parser.parse_args()


if __name__ == '__main__':
    main()
