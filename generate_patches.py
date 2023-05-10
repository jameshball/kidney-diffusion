import numpy as np

from matplotlib import pyplot as plt, cm
import torchvision.transforms as T
from torch.utils.data import DataLoader

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re
import gc
import argparse

from tqdm import tqdm
from joblib import Parallel, delayed


def save_file(args, dataset, i):
    patch, _, _ = dataset[i]
    T.ToPILImage()(patch).save(f'{args.output_path}/real/{i}.png')


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
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patient_labelled_dir, patch_size=1024, image_size=1024, annotated_dataset=args.annotated_dataset, transformations=False)
    if args.annotated_dataset:
        print('Using ANNOTATED dataset')
    else:
        print('Using UNANNOTATED dataset')

    try:
        os.makedirs(f"{args.output_path}/real")
    except FileExistsError:
        pass

    result = Parallel(n_jobs=64)(delayed(save_file)(args, dataset, i) for i in tqdm(range(len(dataset))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    parser.add_argument('--output_path', type=str, help='Path where patches will be saved')
    parser.add_argument('--annotated_dataset', action='store_true', help='Use annotated dataset')
    return parser.parse_args()


if __name__ == '__main__':
    main()
