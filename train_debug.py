from uuid import uuid4

import matplotlib
import numpy as np
import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet, SRUnet1024, ElucidatedImagen
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T

from ultra_res_patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob
import wandb

import re
import gc


SPLIT_VALID_FRACTION = 0.025


def log_wandb(cur_step, loss, validation=False):
    wandb.log({
        "loss" if not validation else "val_loss" : loss,
        "step": cur_step,
    })

def main():
    args = parse_args()
    
    # Load the patient outcomes
    patient_outcomes = pd.read_excel(f'{args.data_path}/outcomes.xlsx', 'Sheet1')

    # Filter any patients that don't have an SVS file
    slide_ids = [re.sub(r'\.svs', '', os.path.basename(slide)) for slide in glob(f'{args.data_path}/svs/*.svs')]
    patient_outcomes = patient_outcomes[patient_outcomes['slide_UUID'].isin(slide_ids)]

    # Load all patient creatinine files
    creatinine_files = glob(f'{args.data_path}/creatinine/*.xlsx')
    patient_creatinine = {}
    for file in creatinine_files:
        df = pd.read_excel(file, 'Sheet1')
        file_name = os.path.basename(file)
        patient_id = re.sub(r'\.xlsx$', '', file_name)
        patient_creatinine[patient_id] = df

    # Filter any creatinine files that don't have an outcome
    patient_creatinine = {k: v for k, v in patient_creatinine.items() if k in patient_outcomes['patient_UUID'].values}

    print(f'Found {len(patient_outcomes)} patients with SVS files')

    # Load the labelled data from the h5 labelbox download
    patient_labelled_dir = f'{args.data_path}/results.h5'

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patient_labelled_dir, args.magnification_level)
    print('Using UNANNOTATED dataset for magnification level ' + str(args.magnification_level))


    train_size = int((1 - SPLIT_VALID_FRACTION) * len(dataset))
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, np.random.permutation(indices[:train_size]))
    valid_dataset = Subset(dataset, np.random.permutation(indices[train_size:]))

    print(f'training with dataset of {len(train_dataset)} samples and validating with {len(valid_dataset)} samples')


    index = 0
    for data in train_dataset:
        zoomed_patch = None
        if args.magnification_level == 0:
            patch = data
        else:
            patch, zoomed_patch = data

        print(f"saving image {index}")
        save_image(patch.cpu(), f"test_img_{index}_mag_level_{args.magnification_level}.png")

        if zoomed_patch != None:
            print(f"saving zoomed image {index}")
            save_image(zoomed_patch.cpu(), f"test_img_{index}_zoomed_mag_level_{args.magnification_level}.png")

        index += 1
        if index > 100:
            break
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='./unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--unet_number', type=int, choices=range(1, 4), help='Unet to train')
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    parser.add_argument('--sample_freq', type=int, default=500, help='How many epochs between sampling and checkpoint.pt saves')
    parser.add_argument('--resume', action='store_true', help='Resume previous run using wandb')
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--magnification_level", type=int, choices=range(0, 3))
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
