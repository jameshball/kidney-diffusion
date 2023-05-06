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


def get_next_patches(patches, patch_dist, patch_width, orientation):
    processed_patches = []
    waiting_patches = []
    for i, j in patches:
        y = i * patch_dist
        x = j * patch_dist

        if (i - 1, j) not in patches and (i, j + orientation) not in patches and (i - 1, j + orientation) not in patches:
            processed_patches.append((i, j))
            plt.plot([x, x + patch_width, x + patch_width, x, x],
                        [y, y, y + patch_width, y + patch_width, y], 'g')
        else:
            waiting_patches.append((i, j))
            plt.plot([x, x + patch_width, x + patch_width, x, x],
                        [y, y, y + patch_width, y + patch_width, y], 'r')
    
    return processed_patches, waiting_patches


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
        img_hs = np.logical_and(img_hs[:, :, 0] > 0.5, img_hs[:, :, 1] > 0.02)

        # remove small objects
        img_hs = cv2.erode(img_hs.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # grow the mask
        kernel = np.ones((51, 51), np.uint8)
        img_hs = cv2.dilate(img_hs.astype(np.uint8), kernel, iterations=1)

        # find patches of 161x161 that have mask > 0.5
        # iterate over patch positions and check if the mask is > 0.5
        patch_width = 161
        patch_overlap = 0.25
        patch_dist = int(patch_width * (1 - patch_overlap))
        num_patches = math.ceil(img_hs.shape[0] / (patch_width * (1 - patch_overlap)))
        patches = []
        for i in range(num_patches):
            for j in range(num_patches):
                y = i * patch_dist
                x = j * patch_dist

                patch = img_hs[y:y + patch_width, x:x + patch_width]
                # if any of the pixels in the patch are > 0.5, then add the patch to the list
                if np.any(patch > 0.5):
                    patches.append((i, j))
        

        # display the image with the img_hs mask and the patches
        plt.imshow(small_img)
        plt.imshow(img_hs, alpha=0.5)
        for i, j in patches:
            y = i * patch_dist
            x = j * patch_dist

            plt.plot([x, x + patch_width, x + patch_width, x, x],
                     [y, y, y + patch_width, y + patch_width, y], 'r')
            
        
        print(f'Found {len(patches)} patches for {slide_id}')

        num_top_left_patches = len(get_next_patches(patches, patch_dist, patch_width, -1)[0])
        num_top_right_patches = len(get_next_patches(patches, patch_dist, patch_width, 1)[0])

        if num_top_left_patches > num_top_right_patches:
            orientation = -1
        else:
            orientation = 1

        while len(patches) != 0:
            print("in loop, " + str(len(patches)) + " patches left")
            processed_patches, waiting_patches = get_next_patches(patches, patch_dist, patch_width, orientation)
            
            print("processed", len(processed_patches), "patches")
            patches = waiting_patches
            
            plt.pause(1)
        
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
