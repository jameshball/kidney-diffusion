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

from ultra_res_patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob
import wandb

import re
import gc


SPLIT_VALID_FRACTION = 0.025


def unet_generator(magnification_level, unet_number):
    if unet_number == 1:
        return Unet(
            dim=256,
            dim_mults=(1, 2, 3, 4),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            cond_images_channels=3 if magnification_level > 0 else 0,
        )

    if unet_number == 2:
        return Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
            cond_images_channels=3 if magnification_level > 0 else 0,
        )
    
    if unet_number == 3:
        return Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 6, 8),
            memory_efficient=True,
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            init_conv_to_final_conv_residual=True,
            cond_images_channels=3 if magnification_level > 0 else 0,
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


def init_imagen(magnification_level, unet_number, device=torch.device("cuda")):
    imagen = Imagen(
        unets=(
            unet_generator(magnification_level, 1) if unet_number == 1 else FixedNullUnet(),
            unet_generator(magnification_level, 2) if unet_number == 2 else FixedNullUnet(lowres_cond=True),
            unet_generator(magnification_level, 3) if unet_number == 3 else FixedNullUnet(lowres_cond=True),
        ),
        image_sizes=(64, 256, 1024),
        timesteps=(1024, 256, 256),
        pred_objectives=("noise", "v", "v"),
        random_crop_sizes=(None, None, 256),
        condition_on_text=False,
    ).to(device)

    return imagen

def log_wandb(cur_step, loss, validation=False):
    wandb.log({
        "loss" if not validation else "val_loss" : loss,
        "step": cur_step,
    })

def main():
    args = parse_args()
    
    imagen = init_imagen(args.magnification_level, args.unet_number)
    dl_keywords = ('images',) if args.magnification_level == 0 else ('images', 'cond_images')
    trainer = ImagenTrainer(
        imagen=imagen,
        dl_tuple_output_keywords_names=dl_keywords,
        fp16=True,
    )

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

    trainer.accelerator.print(f'Found {len(patient_outcomes)} patients with SVS files')

    # Load the labelled data from the h5 labelbox download
    patient_labelled_dir = f'{args.data_path}/results.h5'

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patient_labelled_dir, args.magnification_level)
    trainer.accelerator.print('Using UNANNOTATED dataset for magnification level ' + str(args.magnification_level))


    train_size = int((1 - SPLIT_VALID_FRACTION) * len(dataset))
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, np.random.permutation(indices[:train_size]))
    valid_dataset = Subset(dataset, np.random.permutation(indices[train_size:]))

    trainer.accelerator.print(f'training with dataset of {len(train_dataset)} samples and validating with {len(valid_dataset)} samples')


    trainer.add_train_dataset(train_dataset, batch_size=8, num_workers=args.num_workers)
    trainer.add_valid_dataset(valid_dataset, batch_size=8, num_workers=args.num_workers)

    if args.unet_number == 1:
        checkpoint_path = args.unet1_checkpoint
    elif args.unet_number == 2:
        checkpoint_path = args.unet2_checkpoint
    else:
        checkpoint_path = args.unet3_checkpoint

    trainer.load(checkpoint_path, noop_if_not_exist=True)

    run_id = None

    if trainer.is_main:
        run_id = wandb.util.generate_id()
        if args.run_id is not None:
            run_id = args.run_id
        trainer.accelerator.print(f"Run ID: {run_id}")

        try:
            os.makedirs(f"samples/{run_id}")
        except FileExistsError:
            pass

        wandb.init(project=f"training_unet{args.unet_number}", resume=args.resume, id=run_id)

    trainer.accelerator.wait_for_everyone()
    while True:
        step_num = trainer.num_steps_taken(args.unet_number)
        loss = trainer.train_step(unet_number=args.unet_number)
        trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} loss: {loss}')

        if trainer.is_main:
            log_wandb(step_num, loss)

        if not (step_num % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number)
            trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} validation loss: {valid_loss}')
            if trainer.is_main:
                log_wandb(step_num, loss, validation=True)

        if not (step_num % args.sample_freq):
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print()
            trainer.accelerator.print("Saving model and sampling")

            if trainer.is_main:
                lowres_zoomed_image = None
                rand_zoomed_image = None
                if args.magnification_level == 0:
                    lowres_image = dataset[0]
                    rand_image = dataset[np.random.randint(len(dataset))]
                else:
                    lowres_image, lowres_zoomed_image = dataset[0]
                    rand_image, rand_zoomed_image = dataset[np.random.randint(len(dataset))]

                with torch.no_grad():
                    if lowres_zoomed_image == None:
                        images = trainer.sample(
                            batch_size=2,
                            return_pil_images=False,
                            start_image_or_video=torch.stack([lowres_image, rand_image]),
                            start_at_unet_number=args.unet_number,
                            stop_at_unet_number=args.unet_number,
                        )
                    else:
                        images = trainer.sample(
                            batch_size=2,
                            return_pil_images=False,
                            start_image_or_video=torch.stack([lowres_image, rand_image]),
                            start_at_unet_number=args.unet_number,
                            stop_at_unet_number=args.unet_number,
                            cond_images=torch.stack([lowres_zoomed_image, rand_zoomed_image]),
                        )


                for index in range(len(images)):
                    T.ToPILImage()(images[index]).save(f'samples/{run_id}/sample-{step_num}-{run_id}-{index}.png')
                    wandb.log({f"sample{'' if index == 0 else f'-{index}'}": wandb.Image(images[index])})
    
            trainer.accelerator.wait_for_everyone()
            trainer.save(checkpoint_path)
            trainer.accelerator.print("Finished sampling and saving model!")


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
