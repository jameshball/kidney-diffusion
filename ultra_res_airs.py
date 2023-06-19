from collections import Counter

import time
import h5py
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from skimage import color
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
import os
import glob

Image.MAX_IMAGE_PIXELS = 9999999999999999

NUM_FLIPS_ROTATIONS = 8
MAG_LEVEL_SIZES = [10000, 3328, 1024]
FILL_COLOR = (0, 0, 0)


def read_block_mag_zero(image):
    width = MAG_LEVEL_SIZES[0]
    height = width

    center_x = width // 2
    center_y = height // 2
    zoomed_size = MAG_LEVEL_SIZES[0]
    x = center_x - zoomed_size // 2
    y = center_y - zoomed_size // 2

    return read_block(0, x, y, image)


# x y is the coordinate of the top-left corner of the patch to read in the overall image
# mag_level controls the magnification of the patch
def read_block(mag_level, x, y, image):
    width = MAG_LEVEL_SIZES[0]
    height = width

    image_size = MAG_LEVEL_SIZES[mag_level]

    patch_size = MAG_LEVEL_SIZES[2]
    patch = np.full((patch_size, patch_size, 3), FILL_COLOR, dtype=np.single)
    patch = torch.from_numpy(patch).permute(2, 0, 1)

    # if coords are negative, cap to 0
    cropped_x = max(x, 0)
    cropped_y = max(y, 0)

    # if coords are negative, then the section that is out of bounds
    # should count towards the image_size so we should trim this off
    x_trim = max(-x, 0)
    y_trim = max(-y, 0)

    cropped_width = min(width - cropped_x, image_size - x_trim)
    cropped_height = min(height - cropped_y, image_size - y_trim)

    patch_width = int(cropped_width * (patch_size / image_size))
    patch_height = int(cropped_height * (patch_size / image_size))

    cropped_patch = image[:, cropped_y:cropped_y + cropped_height, cropped_x:cropped_x + cropped_width]
    cropped_patch = F.interpolate(cropped_patch.unsqueeze(0), size=(patch_height, patch_width), mode='nearest').squeeze(0)

    # x and y are relative to the actual kidney image, and we need coordinates
    # relative to the patch we are returning. x and y define the top-left corner
    # of the patch, which is coordinate [0,0] so by subtracting x and y from a set
    # of coordinates, it now is relative to the patch. So we subtract x and y from
    # cropped_x and cropped_y to get the right coordinates.

    patch_x = cropped_x - x
    patch_y = cropped_y - y

    # need to multiply by (self.patch_size / image_size) to change coordinates into
    # the same magnification as the patch, rather than the whole slide.
    patch_x = int(patch_x * (patch_size / image_size))
    patch_y = int(patch_y * (patch_size / image_size))

    patch[:, patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] = cropped_patch

    return patch


def read_block_and_zoomed(image, y, x, mag_level):
    image_size = MAG_LEVEL_SIZES[mag_level]
    center_x = x + image_size // 2
    center_y = y + image_size // 2
    zoomed_size = MAG_LEVEL_SIZES[mag_level - 1]
    zoomed_x = center_x - zoomed_size // 2
    zoomed_y = center_y - zoomed_size // 2

    patch = read_block(mag_level, x, y, image) 
    zoomed_patch = read_block(mag_level - 1, zoomed_x, zoomed_y, image)

    return patch, zoomed_patch


def process_image(path, mag_level):
    image = Image.open(path)
    image = T.ToTensor()(image)

    if mag_level > 0:
        for y in range(0, MAG_LEVEL_SIZES[0] - MAG_LEVEL_SIZES[2], MAG_LEVEL_SIZES[2] // 2):
            for x in range(0, MAG_LEVEL_SIZES[0] - MAG_LEVEL_SIZES[2], MAG_LEVEL_SIZES[2] // 2):
                patch, cond_image = read_block_and_zoomed(image, y, x, mag_level)
                return patch.to(torch.half), cond_image.to(torch.half)
    else:
        return read_block_mag_zero(image).to(torch.half)


class AirsDataset(Dataset):
    def __init__(self, image_dir, ignore_list, magnification_level, verbose=False, center_cond=False):
        super().__init__()

        self.patch_size = 1024
        self.center_cond = center_cond
        self.magnification_level = magnification_level

        image_paths = glob.glob(image_dir + "/*")
        image_paths = [path for path in image_paths if os.path.basename(path) not in ignore_list]

        self.data = Parallel(n_jobs=8)(delayed(process_image)(path, self.magnification_level) for path in tqdm(image_paths))

        if verbose:
            print(f"{len(image_paths)} images in dataset")
    


    def __len__(self):
        return NUM_FLIPS_ROTATIONS * len(self.data)



    def flip_rotate_patch(self, index, patch):
        if index % NUM_FLIPS_ROTATIONS == 0:
            return patch
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1)
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2)
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1)
        else:
            return patch.transpose(1, 2).flip(1).flip(2)


    def __getitem__(self, index):
        # size of higher mag patch within the zoomed_patch (once a center crop is made)
        patch_width = int(MAG_LEVEL_SIZES[self.magnification_level] * self.patch_size / MAG_LEVEL_SIZES[self.magnification_level - 1])

        data_index = index // NUM_FLIPS_ROTATIONS
        
        if self.magnification_level > 0:
            patch, zoomed_patch = self.data[data_index]
            patch, zoomed_patch = patch.float(), zoomed_patch.float()
            patch = self.flip_rotate_patch(index, patch)
            zoomed_patch = self.flip_rotate_patch(index, zoomed_patch)
            if self.center_cond:
                center_patch = T.CenterCrop(patch_width)(zoomed_patch)
                center_patch = F.interpolate(center_patch.unsqueeze(0), zoomed_patch.shape[-1], mode='nearest').squeeze(0)
                cond_image = torch.cat((zoomed_patch, center_patch), 0)
                return patch, cond_image
            else:
                return patch, zoomed_patch
        else:
            patch = self.data[data_index].float()
            return self.flip_rotate_patch(index, patch)

