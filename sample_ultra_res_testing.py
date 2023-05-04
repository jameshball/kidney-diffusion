from uuid import uuid4

import numpy as np
import torch
import argparse
import slideio
import math

from matplotlib import pyplot as plt
from skimage import color

import os
import pandas as pd
from glob import glob
import cv2

import re


def main():
    args = parse_args()
    
    # Load the patient outcomes
    patient_outcomes = pd.read_excel(f'{args.data_path}/outcomes.xlsx', 'Sheet1')

    # Filter any patients that don't have an SVS file
    slide_ids = [re.sub(r'\.svs', '', os.path.basename(slide)) for slide in glob(f'{args.data_path}/svs/*.svs')]
    patient_outcomes = patient_outcomes[patient_outcomes['slide_UUID'].isin(slide_ids)]

    print(f'Found {len(patient_outcomes)} patients with SVS files')

    slide_ids = patient_outcomes["slide_UUID"]
    for slide_id in slide_ids:
        slide = slideio.open_slide(f'{args.data_path}/svs/' + slide_id + ".svs", "SVS")
        image = slide.get_scene(0)

        # Resize the image to blocks of the patch size
        small_img = image.read_block(image.rect, size=(7168, 7168))

        # Mask out the background
        img_hs = color.rgb2hsv(small_img)
        img_hs = np.logical_and(img_hs[:, :, 0] > 0.8, img_hs[:, :, 1] > 0.05)

        # remove small objects
        img_hs = cv2.erode(img_hs.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # grow the mask
        kernel = np.ones((51, 51), np.uint8)
        img_hs = cv2.dilate(img_hs.astype(np.uint8), kernel, iterations=1)

        # find patches of 161x161 that have mask > 0.5
        # iterate over patch positions and check if the mask is > 0.5
        num_patches = math.ceil(img_hs.shape[0] / 161)
        patches = []
        for i in range(num_patches):
            for j in range(num_patches):
                patch = img_hs[i * 161:(i + 1) * 161, j * 161:(j + 1) * 161]
                # if any of the pixels in the patch are > 0.5, then add the patch to the list
                if np.any(patch > 0.5):
                    patches.append((i, j))
        

        # display the image with the img_hs mask and the patches
        plt.imshow(small_img)
        plt.imshow(img_hs, alpha=0.5)
        for i, j in patches:
            plt.plot([j * 161, j * 161 + 161, j * 161 + 161, j * 161, j * 161],
                     [i * 161, i * 161, i * 161 + 161, i * 161 + 161, i * 161], 'r')
        plt.show()

        print(f'Found {len(patches)} patches for {slide_id}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
