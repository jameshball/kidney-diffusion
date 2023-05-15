from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from train import unet_generator, init_imagen

from patient_dataset import PatientDataset
import os
import gc
import pandas as pd
from glob import glob

import re

BATCH_SIZES = [128, 64, 6]

def generate_images(unet_number, args, lowres_images=None):
    imagen = init_imagen(unet_number)
    trainer = ImagenTrainer(imagen=imagen)

    if unet_number == 1:
        path = args.unet1_checkpoint
    elif unet_number == 2:
        path = args.unet2_checkpoint
    else:
        path = args.unet3_checkpoint

    trainer.load(path)

    batch_size = BATCH_SIZES[unet_number - 1]
    all_images = []

    for start_idx in range(0, args.num_images, batch_size):
        end_idx = min(start_idx + batch_size, args.num_images)
        actual_batch_size = end_idx - start_idx

        print(start_idx, end_idx)
        print("batch size", actual_batch_size)
        conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).repeat_interleave(actual_batch_size, dim=0).float().cuda()

        batch_lowres_images = None if lowres_images is None else lowres_images[start_idx:end_idx]

        print("conds", conds.shape)
        if batch_lowres_images is not None:
            print("batch_lowres_image", batch_lowres_images.shape)

        images = trainer.sample(
            batch_size=actual_batch_size,
            return_pil_images=(unet_number==3),
            text_embeds=conds,
            cond_images=torch.zeros((actual_batch_size, 4, 1024, 1024)).cuda(),
            start_image_or_video=batch_lowres_images,
            start_at_unet_number=unet_number,
            stop_at_unet_number=unet_number,
            cond_scale=args.cond_scale,
        )
        
        if unet_number != 3:
            images = images.cpu()
            all_images.append(images)
        else:
            for image in images:
                image.save(f'{args.folder_name}/inference-{uuid4()}.png')

    del trainer
    del imagen
    del conds

    if unet_number == 3:
        del images

    gc.collect()
    torch.cuda.empty_cache()

    if unet_number != 3:
        return torch.cat(all_images, dim=0)


def main():
    args = parse_args()

    try:
        os.makedirs(f"{args.folder_name}")
    except FileExistsError:
        pass

    lowres_images = generate_images(1, args)
    print(f"{lowres_images.shape} images generated for unet 1")
    medres_images = generate_images(2, args, lowres_images=lowres_images)
    print(f"{medres_images.shape} images generated for unet 2")
    generate_images(3, args, lowres_images=medres_images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='./unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--cond_scale', type=float, default=1, help='Conditioning scale (0 for unconditional)')
    parser.add_argument('--folder_name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    main()
