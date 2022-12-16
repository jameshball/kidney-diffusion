from uuid import uuid4

import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from matplotlib import pyplot as plt
from torchvision import datasets, transforms as T
from patient_dataset import PatientDataset
import torch.nn.functional as F
import os
import pandas as pd
from glob import glob

import re


CHECKPOINT_PATH = "./checkpoint.pt"
DATA_PATH = "E:/kidney_data"
IMAGE_SIZE = 64


def main():
    # Load the patient outcomes
    patient_outcomes = pd.read_excel(f'{DATA_PATH}/outcomes.xlsx', 'Sheet1')

    # Filter any patients that don't have an SVS file
    slide_ids = [re.sub(r'\.svs', '', os.path.basename(slide)) for slide in glob(f'{DATA_PATH}/svs/*.svs')]
    patient_outcomes = patient_outcomes[patient_outcomes['slide_UUID'].isin(slide_ids)]

    # Load all patient creatinine files
    creatinine_files = glob(f'{DATA_PATH}/creatinine/*.xlsx')
    patient_creatinine = {}
    for file in creatinine_files:
        df = pd.read_excel(file, 'Sheet1')
        file_name = os.path.basename(file)
        patient_id = re.sub(r'\.xlsx$', '', file_name)
        patient_creatinine[patient_id] = df

    # Filter any creatinine files that don't have an outcome
    patient_creatinine = {k: v for k, v in patient_creatinine.items() if k in patient_outcomes['patient_UUID'].values}

    print(f'Found {len(patient_outcomes)} patients with SVS files')

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{DATA_PATH}/svs/', patch_size=256, image_size=IMAGE_SIZE)
    print(f'Found {len(dataset)} patches')

    patch, outcome = dataset[0]
    print(f'Patch shape: {patch.shape}')


    unet = Unet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        cond_dim=512,
        text_embed_dim=3,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=False
    )

    imagen = Imagen(
        unets=unet,
        image_sizes=IMAGE_SIZE,
        timesteps=1000,
        text_embed_dim=3,
    )

    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True  # whether to split the validation dataset from the training
    ).cuda()

    trainer.add_train_dataset(dataset, batch_size=16)

    run_name = uuid4()

    if os.path.exists(CHECKPOINT_PATH):
        trainer.load(CHECKPOINT_PATH)

    # working training loop

    for i in range(200000):
        loss = trainer.train_step(unet_number=1, max_batch_size=4)
        print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number=1, max_batch_size=4)
            print(f'valid loss: {valid_loss}')

        if not (i % 500) and trainer.is_main:  # is_main makes sure this can run in distributed
            conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).float().cuda()
            images = trainer.sample(batch_size=1, return_pil_images=True, text_embeds=conds)
            for index in range(len(images)):
                images[index].save(f'samples/sample-{i // 100}-{run_name}.png')
            trainer.save(CHECKPOINT_PATH)


if __name__ == '__main__':
    main()
