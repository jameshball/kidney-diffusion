from uuid import uuid4

import torch
import torch.multiprocessing as mp
import argparse
import math

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from imagen_pytorch.trainer import restore_parts
from imagen_pytorch.version import __version__
from packaging import version
from torch import nn
from torchvision.utils import save_image

from train_ultra_res import unet_generator, init_imagen
from ultra_res_patient_dataset import MAG_LEVEL_SIZES

import os
import gc
import pandas as pd
from glob import glob

from fsspec.core import url_to_fs

import re

PATCH_SIZE = 1024
BATCH_SIZES = [64, 48, 4]
FILL_COLOR = 0.95


def load_model(mag_level, unet_number, device, args):
    imagen = init_imagen(mag_level, unet_number, device=device)

    checkpoint_name = f"unet{unet_number}_mag{mag_level}"
    path = vars(args)[checkpoint_name]

    print("loading checkpoint...")
    fs, _ = url_to_fs(path)

    with fs.open(path) as f:
        loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

    try:
        imagen.load_state_dict(loaded_obj['model'], strict=True)
    except RuntimeError:
        print("Failed loading state dict. Trying partial load")
        imagen.load_state_dict(restore_parts(self.imagen.state_dict(), loaded_obj['model']))
    
    print("loaded checkpoint for", checkpoint_name)
    
    return imagen


def generate_image_distributed(rank, mag_level, unet_number, args, in_queue, out_queue):
    device = torch.device(f"cuda:{rank}")
    print("Starting process on ", device)

    imagen = load_model(mag_level, unet_number, device, args)

    while True:
        item = in_queue.get()
        if item is None:
            break
        start_index, end_index, batch_lowres_image, batch_cond_image = item

        if batch_cond_image != None:
            batch_cond_image = batch_cond_image.to(device)
        if batch_lowres_image != None:
            batch_lowres_image = batch_lowres_image.to(device)

        batch_image = imagen.sample(
            batch_size=end_index - start_index,
            return_pil_images=False,
            cond_images=batch_cond_image,
            start_image_or_video=batch_lowres_image,
            start_at_unet_number=unet_number,
            stop_at_unet_number=unet_number,
            device=device,
        )

        out_queue.put((start_index, batch_image))


    del imagen
    del batch_cond_image
    del batch_lowres_image
    gc.collect()
    torch.cuda.empty_cache()


def generate_image_with_unet(mag_level, unet_number, args, lowres_image=None, cond_image=None):
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    processes = []

    batch_size = BATCH_SIZES[unet_number - 1]

    if cond_image is not None:
        num_cond_images = cond_image.shape[0]
        num_batches = int(math.ceil(num_cond_images / batch_size))
    else:
        num_cond_images = 1
        num_batches = 1

    images = []
    for batch_idx in range(num_batches):
        # Determine the start and end index for the current batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_cond_images)

        # Extract the corresponding batch of cond_images and call trainer.sample()
        batch_cond_image = None if cond_image is None else cond_image[start_idx:end_idx]
        batch_lowres_image = None if lowres_image is None else lowres_image[start_idx:end_idx]

        in_queue.put((start_idx, end_idx, batch_lowres_image, batch_cond_image))

    num_processes = min(args.num_gpus, num_batches)

    for rank in range(num_processes):
        p = mp.Process(target=generate_image_distributed, args=(rank, mag_level, unet_number, args, in_queue, out_queue))
        p.start()
        processes.append(p)

    for _ in range(num_processes):
        in_queue.put(None)

    results = [out_queue.get() for _ in range(num_batches)]
    images = [image_batch.cuda() for start_idx, image_batch in sorted(results, key=lambda x: x[0])]

    for p in processes:
        p.join()

    # Concatenate the resulting batch of tensors into a single tensor using torch.cat()
    all_images = torch.cat(images, dim=0)

    return all_images


def generate_image(mag_level, args, cond_image=None):
    lowres_image = generate_image_with_unet(mag_level, 1, args, cond_image=cond_image)
    medres_image = generate_image_with_unet(mag_level, 2, args, lowres_image=lowres_image, cond_image=cond_image)
    highres_image = generate_image_with_unet(mag_level, 3, args, lowres_image=medres_image, cond_image=cond_image)

    return highres_image


# mag0 images represent 40000x40000 patches, but are 1024x1024
# We need to get the positions of the centers of all patches
# that are 6500x6500 in this image, and use these as the conditioning
# images to generate the mag1 images.
#
# Each pixel in this image is 40000/1024 pixels in the original, and
# eaxh pixel in the original is 1024/40000 pixels in this image.
#
# So a 6500x6500 patch is 6500 * 1024/40000 pixels in this image.
#
# So split the image into these patches, and move each image around
# so that the patch is at the center.
def get_cond_images(zoomed_image, mag_level):
    # patch size of a mag1 image within the generated mag0 image
    mag_patch_size = int(MAG_LEVEL_SIZES[mag_level] * PATCH_SIZE / MAG_LEVEL_SIZES[mag_level - 1])
    print("mag_patch_size", mag_level, mag_patch_size)

    num_mag_images_width = math.ceil(PATCH_SIZE / mag_patch_size)
    num_mag_images = num_mag_images_width * num_mag_images_width
    print("num mag images", mag_level, num_mag_images)

    mag_cond_images = torch.zeros(num_mag_images, zoomed_image.shape[1], zoomed_image.shape[2], zoomed_image.shape[3])

    for i in range(num_mag_images_width):
        for j in range(num_mag_images_width):
            y = i * mag_patch_size
            x = j * mag_patch_size

            center_y = y + mag_patch_size // 2
            center_x = x + mag_patch_size // 2

            # need to move the mag0_image so that the center of it
            # aligns with the center of this patch. So PATCH_SIZE // 2
            # in the generated image should be aligned with center_y and center_x

            shift_y = PATCH_SIZE // 2 - center_y
            shift_x = PATCH_SIZE // 2 - center_x

            # Shift the image horizontally and vertically
            shifted_img = torch.roll(zoomed_image[0], shifts=(shift_y, shift_x), dims=(1, 2))

            # Fill any gaps with the fill_color
            if shift_y > 0:
                shifted_img[:, :shift_y, :] = FILL_COLOR
            else:
                shifted_img[:, shift_y:, :] = FILL_COLOR

            if shift_x > 0:
                shifted_img[:, :, :shift_x] = FILL_COLOR
            else:
                shifted_img[:, :, shift_x:] = FILL_COLOR

            mag_cond_images[i * num_mag_images_width + j] = shifted_img

    return mag_cond_images


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    sample_id = uuid4()

    mag0_images = generate_image(0, args)
    save_image(mag0_images[0], f'samples/MAG0-{sample_id}.png')
    mag1_cond_images = get_cond_images(mag0_images, 1)
    mag1_images = generate_image(1, args, cond_image=mag1_cond_images)

    mag1_num_images_width = int(math.sqrt(mag1_images.shape[0]))
    mag1_full_image_width = mag1_num_images_width * PATCH_SIZE
    mag1_full_image = torch.zeros(1, 3, mag1_full_image_width, mag1_full_image_width)

    for i in range(mag1_num_images_width):
        for j in range(mag1_num_images_width):
            y = i * PATCH_SIZE
            x = j * PATCH_SIZE

            mag1_full_image[0, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = mag1_images[i * mag1_num_images_width + j]

    
    save_image(mag1_full_image[0], f'samples/MAG1-{sample_id}.png')


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
    parser.add_argument('--num_gpus', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
