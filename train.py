from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re


TEXT_EMBED_DIM = 3


def unet_generator(unet_number):
    if unet_number == 1:
        return Unet(
            dim=256,
            dim_mults=(1, 2, 3, 4),
            cond_dim=512,
            text_embed_dim=3,
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
        )

    if unet_number == 2:
        return Unet(
            dim=128,
            cond_dim=512,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
        )
    
    if unet_number == 3:
        return Unet(
            dim=128,
            cond_dim=512,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
        )


    return None


class FixedNullUnet(NullUnet):
    def __init__(self, lowres_cond=False, *args, **kwargs):
        super().__init__()
        self.lowres_cond = lowres_cond
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


def init_imagen(unet_number):
    imagen = Imagen(
        unets=(
            unet_generator(1) if unet_number == 1 else FixedNullUnet(),
            unet_generator(2) if unet_number == 2 else FixedNullUnet(lowres_cond=True),
            unet_generator(3) if unet_number == 3 else FixedNullUnet(lowres_cond=True),
        ),
        image_sizes=(64, 256, 1024),
        timesteps=1000,
        text_embed_dim=TEXT_EMBED_DIM,
    ).cuda()

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
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patch_size=1024, image_size=1024)
    print(f'Found {len(dataset) // 8} patches')

    lowres_image = dataset[0][0]

    run_name = uuid4()

    try:
        os.makedirs(f"samples/{run_name}")
    except FileExistsError:
        pass

    imagen = init_imagen(args.unet_number)
    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True,
    )

    trainer.add_train_dataset(dataset, batch_size=16)

    checkpoint_path = args.unet1_checkpoint if args.unet_number == 1 else args.unet2_checkpoint

    trainer.load(checkpoint_path, noop_if_not_exist=True)

    for i in range(200000):
        loss = trainer.train_step(unet_number=args.unet_number, max_batch_size=4)
        print(f'step {trainer.num_steps_taken(args.unet_number)}: unet{args.unet_number} loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number, max_batch_size=4)
            print(f'step {trainer.num_steps_taken(args.unet_number)}: unet{args.unet_number} validation loss: {valid_loss}')

        if not (i % args.sample_freq) and trainer.is_main:  # is_main makes sure this can run in distributed
            conds = torch.tensor([0.0, 0.5, 0.2]).reshape(1, 1, 3).float().cuda()
            images = trainer.sample(
                batch_size=1,
                return_pil_images=True,
                text_embeds=conds,
                start_image_or_video=lowres_image.unsqueeze(0),
                start_at_unet_number=args.unet_number,
                stop_at_unet_number=args.unet_number,
            )
            for index in range(len(images)):
                images[index].save(f'samples/{run_name}/sample-{i}-{run_name}.png')
            trainer.save(checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='./unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--unet_number', type=int, choices=range(1, 3), help='Unet to train')
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    parser.add_argument('--sample_freq', type=int, default=500, help='How many epochs between sampling and checkpoint.pt saves')
    return parser.parse_args()


if __name__ == '__main__':
    main()
