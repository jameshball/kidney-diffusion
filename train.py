from uuid import uuid4

import matplotlib
import numpy as np
import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet, SRUnet1024, ElucidatedImagen
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re
import gc


TEXT_EMBED_DIM = 3
SPLIT_VALID_FRACTION = 0.025


def unet_generator(unet_number):
    if unet_number == 1:
        return Unet(
            dim=256,
            dim_mults=(1, 2, 3, 4),
            cond_dim=512,
            text_embed_dim=3,
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            cond_images_channels=4,
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
            cond_images_channels=4,
        )
    
    if unet_number == 3:
        return Unet(
            dim=128,
            cond_dim=512,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 4, 4),
            memory_efficient=True,
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            init_conv_to_final_conv_residual=True,
            cond_images_channels=4,
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
        timesteps=(1024, 256, 256),
        pred_objectives=("noise", "v", "v"),
        text_embed_dim=TEXT_EMBED_DIM,
        random_crop_sizes=(None, None, 256),
    ).cuda()

    #imagen = ElucidatedImagen(
    #    unets=(
    #        unet_generator(1) if unet_number == 1 else FixedNullUnet(),
    #        unet_generator(2) if unet_number == 2 else FixedNullUnet(lowres_cond=True),
    #        unet_generator(3) if unet_number == 3 else FixedNullUnet(lowres_cond=True),
    #    ),
    #    image_sizes=(64, 256, 1024),
    #    cond_drop_prob=0.1,
    #    num_sample_steps=(32, 128, 128),
    #    text_embed_dim=TEXT_EMBED_DIM,
    #    random_crop_sizes=(None, None, 256),
    #    sigma_min=0.002,           # min noise level
    #    sigma_max=(80, 320, 1280), # max noise level, @crowsonkb recommends double the max noise level for upsampler
    #).cuda()

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

    # Load the labelled data from the h5 labelbox download
    patient_labelled_dir = f'{args.data_path}/results.h5'

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patient_labelled_dir, patch_size=1024, image_size=1024, annotated_dataset=args.annotated_dataset)
    if args.annotated_dataset:
        print('Using ANNOTATED dataset for finetuning')
    else:
        print('Using UNANNOTATED dataset for initial training')

    run_name = uuid4()

    try:
        os.makedirs(f"samples/{run_name}")
    except FileExistsError:
        pass

    train_size = int((1 - SPLIT_VALID_FRACTION) * len(dataset))
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, np.random.permutation(indices[:train_size]))
    valid_dataset = Subset(dataset, np.random.permutation(indices[train_size:]))

    for i in range(10):
        patch, conds, labelmap = train_dataset[i]
        plt.imshow(patch.permute(1, 2, 0).cpu().numpy())
        for j in range(labelmap.shape[0]):
            data_masked = np.ma.masked_where(labelmap[j].cpu().numpy() == 0, labelmap[j].cpu().numpy())
            plt.imshow(data_masked, alpha=0.5, cmap=matplotlib.colors.ListedColormap(np.random.rand(256, 3)))
        plt.show()

    print(f'training with dataset of {len(train_dataset)} samples and validating with {len(valid_dataset)} samples')

    imagen = init_imagen(args.unet_number)
    trainer = ImagenTrainer(
        imagen=imagen,
        dl_tuple_output_keywords_names=('images', 'text_embeds', 'cond_images'),
        fp16=True,
    )

    trainer.add_train_dataset(dataset, batch_size=16)
    trainer.add_valid_dataset(valid_dataset, batch_size=16)

    if args.unet_number == 1:
        checkpoint_path = args.unet1_checkpoint
    elif args.unet_number == 2:
        checkpoint_path = args.unet2_checkpoint
    else:
        checkpoint_path = args.unet3_checkpoint

    trainer.load(checkpoint_path, noop_if_not_exist=True)

    if args.log_to_wandb:
        import wandb
        wandb.init(project=f"training_unet{args.unet_number}")

    for i in range(200000):
        loss = trainer.train_step(unet_number=args.unet_number, max_batch_size=4)
        print(f'step {trainer.num_steps_taken(args.unet_number)}: unet{args.unet_number} loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number, max_batch_size=4)
            print(f'step {trainer.num_steps_taken(args.unet_number)}: unet{args.unet_number} validation loss: {valid_loss}')
            if args.log_to_wandb:
                wandb.log({"loss": loss})
                wandb.log({"valid_loss": valid_loss})

        if not (i % args.sample_freq) and trainer.is_main:  # is_main makes sure this can run in distributed
            lowres_image, conds, labelmap = dataset[0]
            rand_image, rand_conds, rand_labelmap = dataset[np.random.randint(len(dataset))]
            images = trainer.sample(
                batch_size=2,
                return_pil_images=False,
                text_embeds=torch.stack([conds, rand_conds]),
                start_image_or_video=torch.stack([lowres_image, rand_image]),
                start_at_unet_number=args.unet_number,
                stop_at_unet_number=args.unet_number,
                cond_images=torch.stack([labelmap, rand_labelmap]),
            )
            for index in range(len(images)):
                T.ToPILImage()(images[index]).save(f'samples/{run_name}/sample-{i}-{run_name}-{index}.png')
                if args.log_to_wandb:
                    wandb.log({f"sample{'' if index == 0 else f'-{index}'}": wandb.Image(images[index])})

            trainer.save(checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='./unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='./unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='./unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--unet_number', type=int, choices=range(1, 4), help='Unet to train')
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    parser.add_argument('--sample_freq', type=int, default=500, help='How many epochs between sampling and checkpoint.pt saves')
    parser.add_argument('--annotated_dataset', action='store_true', help='Train with an annotated dataset')
    parser.add_argument('--log_to_wandb', action='store_true', help='Log loss and samples to weights & biases')
    return parser.parse_args()


if __name__ == '__main__':
    main()
