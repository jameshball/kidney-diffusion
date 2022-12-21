from uuid import uuid4

import torch
import argparse
from imagen_pytorch import Unet, ImagenTrainer, Imagen
from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re


CHECKPOINT_PATH = "./checkpoint.pt"


def init_imagen():
    unet1 = Unet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        cond_dim=32,
        text_embed_dim=3,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True)
    )

    unet2 = Unet(
        dim=128,
        cond_dim=32,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

    imagen = Imagen(
        unets=(unet1, unet2),
        image_sizes=(64, 256),
        timesteps=1000,
        text_embed_dim=3,
    )

    return imagen


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

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patch_size=1024, image_size=256)
    print(f'Found {len(dataset)} patches')

    patch, outcome = dataset[0]
    print(f'Patch shape: {patch.shape}')

    try:
        os.makedirs("samples")
    except FileExistsError:
        pass

    imagen = init_imagen()
    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True  # whether to split the validation dataset from the training
    ).cuda()

    trainer.add_train_dataset(dataset, batch_size=16)

    run_name = uuid4()

    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)

    trainer.print_unet_devices()

    for i in range(200000):
        loss = trainer.train_step(unet_number=args.unet_number, max_batch_size=4)
        print(f'unet{args.unet_number} loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number, max_batch_size=4)
            print(f'unet{args.unet_number} validation loss: {valid_loss}')

        if not (i % 100) and trainer.is_main:  # is_main makes sure this can run in distributed
            conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).float().cuda()
            images = trainer.sample(
                batch_size=1,
                return_pil_images=True,
                text_embeds=conds,
                cond_scale=5.,
                start_at_unet_number=args.unet_number,
                stop_at_unet_number=args.unet_number,
            )
            for index in range(len(images)):
                images[index].save(f'samples/sample-{i // 100}-{run_name}.png')
            trainer.save(args.checkpoint)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to checkpoint')
    parser.add_argument('--unet_number', type=int, choices=range(1, 3), help='Unet to train')
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    return parser.parse_args()


if __name__ == '__main__':
    main()
