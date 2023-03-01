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
import numpy as np
import matplotlib.pyplot as plt

import re

labels_path = '/vol/biomedic2/sc7718/Unet_renal_segmentation/James_images/generated-uncond-labels'
labels = {'Tubuli': 1, 'Vein': 2,'Vessel_indeterminate': 2,  'Artery': 3, 'Glomerui': 4}

def generate_images(unet_number, args, deep_labelmap, num_variants, lowres_images=None):
    imagen = init_imagen(unet_number)
    trainer = ImagenTrainer(imagen=imagen)

    if unet_number == 1:
        path = args.unet1_checkpoint
    elif unet_number == 2:
        path = args.unet2_checkpoint
    else:
        path = args.unet3_checkpoint

    trainer.load(path)

    conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).repeat_interleave(num_variants, dim=0).float().cuda()
    deep_labelmap = torch.tensor(deep_labelmap).unsqueeze(0).repeat_interleave(num_variants, dim=0).float().cuda()

    images = trainer.sample(
        batch_size=num_variants,
        return_pil_images=(unet_number==3),
        text_embeds=conds,
        start_image_or_video=lowres_images,
        cond_images=deep_labelmap,
        start_at_unet_number=unet_number,
        stop_at_unet_number=unet_number,
    )

    del trainer
    del imagen
    del conds
    gc.collect()
    torch.cuda.empty_cache()

    return images


def main():

    args = parse_args()
    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass
        
    #Load the labelmasks:
    ids = []
    for images in os.listdir(labels_path):
        if (images.endswith(".npy")):
            ids += [images]
    print(len(ids))
    label_indices = range(args.start_index, args.end_index)
    for i in label_indices:
        labelmap = np.load(labels_path+'/'+ids[i])
        deep_labelmap = np.zeros((len(set(labels.values())), 1024, 1024))

        for label in set(labels.values()):
            deep_labelmap[label - 1, labelmap == label] = 1

        plt.imshow(labelmap)
        plt.savefig(f"labelmap_{i}.png")
        
        lowres_images = generate_images(1, args, deep_labelmap, 3)
        medres_images = generate_images(2, args, deep_labelmap, 3, lowres_images=lowres_images)
        highres_images = generate_images(3, args, deep_labelmap, 3, lowres_images=medres_images)
        for j, image in enumerate(highres_images):
            image.save(f'samples/inference-label-index{i}-gen-{j}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='./unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for segmentation mask')
    parser.add_argument('--end_index', type=int, default=0, help='End index for segmentation mask')
    return parser.parse_args()


if __name__ == '__main__':
    main()
