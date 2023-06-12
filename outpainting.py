from uuid import uuid4

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import argparse
import math
from skimage import color
import cv2
import numpy as np

from imagen_pytorch.trainer import restore_parts
from imagen_pytorch.version import __version__
from packaging import version
from torchvision.utils import save_image
import torchvision.transforms as transforms

from train_uncond import init_imagen

import os
import gc

from fsspec.core import url_to_fs
import warnings

# used to ignore CUDA warnings that clog stdout
# REMOVE if there are CUDA errors other than those expected
warnings.filterwarnings("ignore", category=UserWarning)

PATCH_SIZE = 1024
PATCH_SIZES = {1: 64, 2: 256, 3: 1024}
BATCH_SIZES = [128, 64, 6]
FILL_COLOR = 0.95


def load_model(unet_number, device, args):
    imagen = init_imagen(unet_number).to(device)

    checkpoint_name = f"unet{unet_number}"
    path = vars(args)[checkpoint_name]

    fs, _ = url_to_fs(path)

    with fs.open(path) as f:
        loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

    try:
        imagen.load_state_dict(loaded_obj['model'], strict=True)
    except RuntimeError:
        print("Failed loading state dict. Trying partial load")
        imagen.load_state_dict(restore_parts(imagen.state_dict(), loaded_obj['model']))
    
    return imagen


def print_memory_usage(rank):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"cuda:{rank} total memory: {t}, reserverd memory: {r}, allocated memory: {a}, free memory: {r-a}")


def generate_image_distributed(rank, unet_number, args, in_queue, out_queue, patches_generated, overlap, orientation, patch_pos, num_patches_width):
    device = torch.device(f"cuda:{rank}")
    print(f"started process on {device}")

    imagen = load_model(unet_number, device, args)

    while True:
        item = in_queue.get()
        if item is None:
            break
        idx, batch_lowres_image, pos = item
        
        inpaint_patch = None
        inpaint_mask = None

        # need to check if patch above, next to, and above and next to this patch have been generated
        # if they have, then we can generate this patch
        if pos is not None:
            i, j = pos

            above_patch = None
            next_to_patch = None
            above_next_to_patch = None

            above = (i - 1, j)
            above_idx = -1 if above not in patch_pos else patch_pos.index(above)
            next_to = (i, j + orientation)
            next_to_idx = -1 if next_to not in patch_pos else patch_pos.index(next_to)
            above_next_to = (i - 1, j + orientation)
            above_next_to_idx = -1 if above_next_to not in patch_pos else patch_pos.index(above_next_to)
            above_exists = above_idx in patches_generated or above not in patch_pos
            next_to_exists = next_to_idx in patches_generated or next_to not in patch_pos
            above_next_to_exists = above_next_to_idx in patches_generated or above_next_to not in patch_pos

            unet_patch_size = PATCH_SIZES[unet_number]

            if above_exists and next_to_exists and above_next_to_exists:
                if above_idx in patches_generated:
                    above_patch = patches_generated[above_idx][0]
                if next_to_idx in patches_generated:
                    next_to_patch = patches_generated[next_to_idx][0]
                if above_next_to_idx in patches_generated:
                    above_next_to_patch = patches_generated[above_next_to_idx][0]
            else:
                in_queue.put((idx, batch_lowres_image, pos))
                continue

            print(f"generating patch at {pos} which is index {idx}", flush=True)

            # inpaint_patch is the patch that will be generated with above, next_to, and above_next_to patches
            # already generated. They need to be added to the inpaint_patch in the correct positions
            inpaint_patch = torch.zeros(3, unet_patch_size, unet_patch_size)
            inpaint_mask = torch.zeros(unet_patch_size, unet_patch_size)
            overlap_pos = int(overlap * unet_patch_size)
            
            # if we are at the top of the image, then above_patch is None
            # if we are at the left/right of the image, then next_to_patch is None
            # if we are at the top left/right of the image, then above_next_to_patch is None#
            if above_patch is not None:
                inpaint_patch[:, :overlap_pos, :] = above_patch[:, -overlap_pos:, :]
                inpaint_mask[:overlap_pos, :] = 1
            if next_to_patch is not None:
                if orientation == -1:
                    inpaint_patch[:, :, :overlap_pos] = next_to_patch[:, :, -overlap_pos:]
                    inpaint_mask[:, :overlap_pos] = 1
                else:
                    inpaint_patch[:, :, -overlap_pos:] = next_to_patch[:, :, :overlap_pos]
                    inpaint_mask[:, -overlap_pos:] = 1
            if above_next_to_patch is not None:
                if orientation == -1:
                    inpaint_patch[:, :overlap_pos, :overlap_pos] = above_next_to_patch[:, -overlap_pos:, -overlap_pos:]
                else:
                    inpaint_patch[:, :overlap_pos, -overlap_pos:] = above_next_to_patch[:, -overlap_pos:, :overlap_pos]

            inpaint_patch = inpaint_patch.unsqueeze(0).to(device)
            inpaint_mask = inpaint_mask.unsqueeze(0).to(device)
            
        if batch_lowres_image != None:
            batch_lowres_image = batch_lowres_image.to(device)

        batch_image = imagen.sample(
            batch_size=1,
            return_pil_images=False,
            start_image_or_video=batch_lowres_image,
            start_at_unet_number=unet_number,
            stop_at_unet_number=unet_number,
            inpaint_images=inpaint_patch,
            inpaint_masks=inpaint_mask,
            inpaint_resample_times=args.inpaint_resample,
            use_tqdm=False,
            device=device,
        )
            

        patches_generated[idx] = batch_image.cpu()
        out_queue.put((idx,))

        if pos is not None:
            print(f"{len(patches_generated)}/{len(patch_pos)} patches generated", flush=True)


    del imagen
    del batch_lowres_image
    gc.collect()
    torch.cuda.empty_cache()


def generate_image_with_unet(unet_number, args, lowres_image, overlap, orientation, num_patches_width):
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    patches_generated = mp.Manager().dict()
    processes = []

    num_images = num_patches_width * num_patches_width
    patch_pos = [(i, j) for i in range(num_patches_width) for j in range(num_patches_width)]

    print(f"Generating {num_images} images for unet {unet_number}")

    images = []
    for idx in range(num_images):
        # Extract the corresponding batch and call trainer.sample()
        idx_lowres_image = None if lowres_image is None else lowres_image[idx]
        pos = None if patch_pos is None else patch_pos[idx]

        in_queue.put((idx, idx_lowres_image, pos))

    num_processes = min(args.num_gpus, num_images)

    for rank in range(num_processes):
        p = mp.Process(target=generate_image_distributed, args=(rank, unet_number, args, in_queue, out_queue, patches_generated, overlap, orientation, patch_pos, num_patches_width))
        p.start()
        processes.append(p)

    for _ in range(num_images):
        out_queue.get()

    for _ in range(num_processes):
        in_queue.put(None)

    for p in processes:
        p.join()

    images = [patches_generated[idx] for idx in range(num_images)]

    if lowres_image is not None:
        del lowres_image

    gc.collect()
    torch.cuda.empty_cache()

    return images


def generate_image(args, overlap=0.25, orientation=-1, num_patches_width=1):
    lowres_image = generate_image_with_unet(1, args, None, overlap, orientation, num_patches_width)
    medres_image = generate_image_with_unet(2, args, lowres_image, overlap, orientation, num_patches_width)
    highres_image = generate_image_with_unet(3, args, medres_image, overlap, orientation, num_patches_width)

    return highres_image


def generate_high_res_image(args):
    num_patches_width = args.num_patches_width
    orientation = -1
    images = generate_image(args, overlap=args.overlap, orientation=orientation, num_patches_width=num_patches_width)

    patch_dist = int(PATCH_SIZE * (1 - args.overlap))
    full_image_width = PATCH_SIZE + (num_patches_width - 1) * patch_dist
    full_image = torch.zeros(1, 3, full_image_width, full_image_width)
    patch_pos = [(i, j) for i in range(num_patches_width) for j in range(num_patches_width)]

    for index, (i, j) in enumerate(patch_pos):
        y = i * patch_dist
        x = j * patch_dist

        full_image[0, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = images[index][0]

    return full_image


def main():
    args = parse_args()

    try:
        os.makedirs(args.sample_dir)
    except FileExistsError:
        pass

    sample_id = uuid4()

    full_image = generate_high_res_image(args)
    save_image(full_image[0], f'{args.sample_dir}/outpainting-{sample_id}.jpg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1', type=str)
    parser.add_argument('--unet2', type=str)
    parser.add_argument('--unet3', type=str)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--inpaint_resample', type=int)
    parser.add_argument('--num_patches_width', type=int)
    parser.add_argument('--overlap', type=float)
    parser.add_argument('--sample_dir', default="samples", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
