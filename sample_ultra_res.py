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

PATCH_SIZE = 1024
BATCH_SIZES = [128, 64, 6]
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
        imagen.load_state_dict(restore_parts(imagen.state_dict(), loaded_obj['model']))
    
    print("loaded checkpoint for", checkpoint_name)
    
    return imagen


def print_memory_usage(rank):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"cuda:{rank} total memory: {t}, reserverd memory: {r}, allocated memory: {a}, free memory: {r-a}")


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

        print_memory_usage(rank)

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
        # divide batches across all gpus if we can to save time, even
        # if this results in decreasing the batch size
        if batch_size * args.num_gpus > num_cond_images:
            batch_size = math.ceil(num_cond_images / args.num_gpus)
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
    images = [image_batch.cpu() for start_idx, image_batch in sorted(results, key=lambda x: x[0])]

    for p in processes:
        p.join()

    # Concatenate the resulting batch of tensors into a single tensor using torch.cat()
    # Need this on cpu because they are massive tensors
    all_images = torch.cat(images, dim=0).cpu()

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
#
# FOR MAG2
# Zoomed image is now much larger than patch_size. Each PATCH_SIZE
# patch in the image is the correct scale for a 6500x6500 patch that
# will be used to condition mag2 generation.
#
# So basically, we just need to find all the patches we need to
# generate for mag2, then get a PATCH_SIZE crop around that area in
# the mag1 full scale image
def get_cond_images(zoomed_image, mag_level):
    # patch size of a mag1 image within the generated mag0 image
    mag_patch_size = int(MAG_LEVEL_SIZES[mag_level] * PATCH_SIZE / MAG_LEVEL_SIZES[mag_level - 1])
    print("mag_patch_size", mag_level, mag_patch_size)

    zoomed_image_width = zoomed_image.shape[3]
    num_mag_images_width = math.ceil(zoomed_image_width / mag_patch_size)

    # we want to filter out white patches to save time
    if mag_level == 2:
        zoomed_image_np = zoomed_image.cpu().numpy()

        # Mask out the background
        img_hs = color.rgb2hsv(zoomed_image_np)
        img_hs = np.logical_and(img_hs[:, :, 0] > 0.8, img_hs[:, :, 1] > 0.05)

        # remove small objects
        img_hs = cv2.erode(img_hs.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # grow the mask
        kernel = np.ones((51, 51), np.uint8)
        img_hs = cv2.dilate(img_hs.astype(np.uint8), kernel, iterations=1)

        # find patches of 161x161 that have mask > 0.5
        # iterate over patch positions and check if the mask is > 0.5
        patch_pos = []
        for i in range(num_mag_images_width):
            for j in range(num_mag_images_width):
                patch = img_hs[i * mag_patch_size:(i + 1) * mag_patch_size, j * mag_patch_size:(j + 1) * mag_patch_size]
                # if any of the pixels in the patch are > 0.5, then add the patch to the list
                if np.any(patch > 0.5):
                    patch_pos.append((i, j))
    else:
        patch_pos = [(i, j) for i in range(num_mag_images_width) for j in range(num_mag_images_width)]
    
    print("Generating", len(patch_pos), "images for mag", mag_level)

    mag_cond_images = []

    for i, j in patch_pos:
        y = i * mag_patch_size
        x = j * mag_patch_size

        center_y = y + mag_patch_size // 2
        center_x = x + mag_patch_size // 2

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

        mag_cond_images.append(shifted_img)

    return torch.stack(mag_cond_images), patch_pos, num_mag_images_width


def generate_high_res_image(zoomed_image, mag_level, args):
    mag_cond_images, mag_cond_images_pos, num_mag_images_width = get_cond_images(zoomed_image, mag_level)
    mag_images = generate_image(mag_level, args, cond_image=mag_cond_images)

    print(mag_images.shape[0])
    mag_full_image_width = num_mag_images_width * PATCH_SIZE
    # Initially, mag_full_image is zoomed_image resized to the correct size. We
    # then replace some of the patches with the generated images
    mag_full_image = F.interpolate(zoomed_image, size=(mag_full_image_width, mag_full_image_width), mode='bilinear', align_corners=False)

    for index, (i, j) in enumerate(mag_cond_images_pos):
        y = i * PATCH_SIZE
        x = j * PATCH_SIZE

        mag_full_image[0, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = mag_images[index]

    return mag_full_image


def main():
    args = parse_args()

    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass

    sample_id = uuid4()

    mag0_images = generate_image(0, args)
    save_image(mag0_images[0], f'samples/MAG0-{sample_id}.png')
    
    mag1_full_image = generate_high_res_image(mag0_images, 1, args)
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
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
