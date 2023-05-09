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

from train_ultra_res import init_imagen
from ultra_res_patient_dataset import MAG_LEVEL_SIZES

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


def load_model(mag_level, unet_number, device, args):
    imagen = init_imagen(mag_level, unet_number, device=device)

    checkpoint_name = f"unet{unet_number}_mag{mag_level}"
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


def generate_image_distributed(rank, mag_level, unet_number, args, in_queue, out_queue, patches_generated, overlap=0.25, orientation=-1, patch_pos=None):
    device = torch.device(f"cuda:{rank}")
    print(f"started process on {device}")

    imagen = load_model(mag_level, unet_number, device, args)

    while True:
        item = in_queue.get()
        if item is None:
            break
        idx, batch_lowres_image, batch_cond_image, pos = item
        
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

            if above_exists and next_to_exists and above_next_to_exists:
                if above_idx in patches_generated:
                    above_patch = patches_generated[above_idx][0]
                if next_to_idx in patches_generated:
                    next_to_patch = patches_generated[next_to_idx][0]
                if above_next_to_idx in patches_generated:
                    above_next_to_patch = patches_generated[above_next_to_idx][0]
            else:
                in_queue.put((idx, batch_lowres_image, batch_cond_image, pos))
                continue

            print(f"generating patch at {pos} which is index {idx}", flush=True)

            # inpaint_patch is the patch that will be generated with above, next_to, and above_next_to patches
            # already generated. They need to be added to the inpaint_patch in the correct positions
            unet_patch_size = PATCH_SIZES[unet_number]
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

            # save images to disk
            #save_image(inpaint_patch, f'samples/inpaint_patch-{idx}.png')
            #save_image(inpaint_mask, f'samples/inpaint_mask-{idx}.png')
            #save_image(batch_cond_image[0], f'samples/batch_cond_image-{idx}.png')
            #if batch_lowres_image is not None:
            #    save_image(batch_lowres_image, f'samples/batch_lowres_image-{idx}.png')
            #if above_patch is not None:
            #    save_image(above_patch, f'samples/above_patch-{idx}.png')
            #if next_to_patch is not None:
            #    save_image(next_to_patch, f'samples/next_to_patch-{idx}.png')
            #if above_next_to_patch is not None:
            #    save_image(above_next_to_patch, f'samples/above_next_to_patch-{idx}.png')
            
            inpaint_patch = inpaint_patch.unsqueeze(0).to(device)
            inpaint_mask = inpaint_mask.unsqueeze(0).to(device)
            
        if batch_cond_image != None:
            batch_cond_image = batch_cond_image.unsqueeze(0).to(device)
        if batch_lowres_image != None:
            batch_lowres_image = batch_lowres_image.to(device)

        batch_image = imagen.sample(
            batch_size=1,
            return_pil_images=False,
            cond_images=batch_cond_image,
            start_image_or_video=batch_lowres_image,
            start_at_unet_number=unet_number,
            stop_at_unet_number=unet_number,
            inpaint_images=inpaint_patch,
            inpaint_masks=inpaint_mask,
            inpaint_resample_times=args.inpaint_resample,
            use_tqdm=False,
            device=device,
        )
            
        if pos is not None:
            print(f"{len(patches_generated)}/{len(patch_pos)} patches generated", flush=True)

        patches_generated[idx] = batch_image.cpu()
        out_queue.put((idx,))


    del imagen
    del batch_cond_image
    del batch_lowres_image
    gc.collect()
    torch.cuda.empty_cache()


def generate_image_with_unet(mag_level, unet_number, args, lowres_image=None, cond_image=None, patch_pos=None, overlap=0.25, orientation=-1):
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    patches_generated = mp.Manager().dict()
    processes = []

    if cond_image is not None:
        num_cond_images = cond_image.shape[0]
    else:
        num_cond_images = 1

    print(f"Generating {num_cond_images} images for mag {mag_level} and unet {unet_number}")

    images = []
    for idx in range(num_cond_images):
        # Extract the corresponding batch of cond_images and call trainer.sample()
        idx_cond_image = None if cond_image is None else cond_image[idx]
        idx_lowres_image = None if lowres_image is None else lowres_image[idx]
        pos = None if patch_pos is None else patch_pos[idx]

        in_queue.put((idx, idx_lowres_image, idx_cond_image, pos))

    num_processes = min(args.num_gpus, num_cond_images)

    for rank in range(num_processes):
        p = mp.Process(target=generate_image_distributed, args=(rank, mag_level, unet_number, args, in_queue, out_queue, patches_generated, overlap, orientation, patch_pos))
        p.start()
        processes.append(p)

    for _ in range(num_cond_images):
        out_queue.get()

    for _ in range(num_processes):
        in_queue.put(None)

    for p in processes:
        p.join()

    images = [patches_generated[idx] for idx in range(num_cond_images)]

    if cond_image is not None:
        del cond_image
    if lowres_image is not None:
        del lowres_image

    gc.collect()
    torch.cuda.empty_cache()

    return images


def generate_image(mag_level, args, cond_image=None, patch_pos=None, overlap=0.25, orientation=-1):
    lowres_image = generate_image_with_unet(mag_level, 1, args, cond_image=cond_image, patch_pos=patch_pos, overlap=overlap, orientation=orientation)
    medres_image = generate_image_with_unet(mag_level, 2, args, lowres_image=lowres_image, cond_image=cond_image, patch_pos=patch_pos, overlap=overlap, orientation=orientation)
    highres_image = generate_image_with_unet(mag_level, 3, args, lowres_image=medres_image, cond_image=cond_image, patch_pos=patch_pos, overlap=overlap, orientation=orientation)

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
#
# FOR MAG2
# Zoomed image is now much larger than patch_size. Each PATCH_SIZE
# patch in the image is the correct scale for a 6500x6500 patch that
# will be used to condition mag2 generation.
#
# So basically, we just need to find all the patches we need to
# generate for mag2, then get a PATCH_SIZE crop around that area in
# the mag1 full scale image
def get_cond_images(zoomed_image, mag_level, overlap=0.25):
    # patch size of a mag1 image within the generated mag0 image
    patch_width = int(MAG_LEVEL_SIZES[mag_level] * PATCH_SIZE / MAG_LEVEL_SIZES[mag_level - 1])

    patch_dist = int(patch_width * (1 - overlap))
    zoomed_image_width = zoomed_image.shape[3]
    # This takes into account the overlap
    num_patches_width = 1 + math.ceil((zoomed_image_width - patch_width) / patch_dist)

    # we want to filter out white patches to save time
    if mag_level == 2:
        zoomed_image_np = zoomed_image[0].permute(1, 2, 0).cpu().numpy()

        # Mask out the background
        img_hs = color.rgb2hsv(zoomed_image_np)
        img_hs = np.logical_and(img_hs[:, :, 0] > 0.5, img_hs[:, :, 1] > 0.02)

        # remove small objects
        img_hs = cv2.erode(img_hs.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # grow the mask
        kernel = np.ones((51, 51), np.uint8)
        img_hs = cv2.dilate(img_hs.astype(np.uint8), kernel, iterations=1)

        print("cond image details:")
        print("patch_width", patch_width)
        print("patch_dist", patch_dist)
        print("zoomed_image_width", zoomed_image_width)
        print("num_patches_width", num_patches_width)
        print("", flush=True)

        # find patches of 161x161 that have mask > 0.5
        # iterate over patch positions and check if the mask is > 0.5
        patch_pos = []
        for i in range(num_patches_width):
            for j in range(num_patches_width):
                y = i * patch_dist
                x = j * patch_dist

                patch = img_hs[y:y + patch_width, x:x + patch_width]
                # if any of the pixels in the patch are > 0.5, then add the patch to the list
                if np.any(patch > 0.5):
                    patch_pos.append((i, j))
    else:
        patch_pos = [(i, j) for i in range(num_patches_width) for j in range(num_patches_width)]
    
    cond_images = []

    for i, j in patch_pos:
        y = i * patch_dist
        x = j * patch_dist

        center_y = y + patch_width // 2
        center_x = x + patch_width // 2

        # need to move the mag0_image so that the center of it
        # aligns with the center of this patch. So zoomed_image_width // 2
        # in the generated image should be aligned with center_y and center_x
        shift_y = zoomed_image_width // 2 - center_y
        shift_x = zoomed_image_width // 2 - center_x

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

        # This shouldn't do anything for mag1 since zoomed_image is 1024x1024
        shifted_img = transforms.CenterCrop(PATCH_SIZE)(shifted_img)

        cond_images.append(shifted_img)

    return torch.stack(cond_images), patch_pos, num_patches_width


def get_next_patches(patches, orientation):
    processed_patches = []
    waiting_patches = []
    for i, j in patches:
        if (i - 1, j) not in patches and (i, j + orientation) not in patches and (i - 1, j + orientation) not in patches:
            processed_patches.append((i, j))
        else:
            waiting_patches.append((i, j))
    
    return processed_patches, waiting_patches


def generate_high_res_image(zoomed_image, mag_level, args):
    cond_images, patch_pos, num_patches_width = get_cond_images(zoomed_image, mag_level, overlap=args.overlap)

    num_top_left_patches = len(get_next_patches(patch_pos, -1)[0])
    num_top_right_patches = len(get_next_patches(patch_pos, 1)[0])

    orientation = -1 if num_top_left_patches > num_top_right_patches else 1

    mag_images = generate_image(mag_level, args, cond_image=cond_images, patch_pos=patch_pos, overlap=args.overlap, orientation=orientation)

    patch_dist = int(PATCH_SIZE * (1 - args.overlap))
    full_image_width = PATCH_SIZE + (num_patches_width - 1) * patch_dist
    print("generate high res image details:")
    print("full_image_width", full_image_width)
    print("patch_dist", patch_dist)
    print("num_patches_width", num_patches_width)
    print("mag_images size", len(mag_images))
    print("", flush=True)
    # Initially, mag_full_image is zoomed_image resized to the correct size. We
    # then replace some of the patches with the generated images
    full_image = F.interpolate(zoomed_image, size=(full_image_width, full_image_width), mode='bilinear', align_corners=False)

    for index, (i, j) in enumerate(patch_pos):
        y = i * patch_dist
        x = j * patch_dist

        full_image[0, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = mag_images[index][0]

    return full_image


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    sample_id = uuid4()

    mag0_images = generate_image(0, args)
    save_image(mag0_images[0][0], f'samples/MAG0-{sample_id}.png')
    
    mag1_full_image = generate_high_res_image(mag0_images[0], 1, args)
    save_image(mag1_full_image[0], f'samples/MAG1-{sample_id}.png')

    mag2_full_image = generate_high_res_image(mag1_full_image, 2, args)
    save_image(mag2_full_image[0], f'samples/MAG2-{sample_id}.png')

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
    parser.add_argument('--inpaint_resample', type=int)
    parser.add_argument('--overlap', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
