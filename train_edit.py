import random
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
import argparse

from ema_pytorch import EMA
from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from imagen_pytorch.trainer import restore_parts, exists
from torch import nn

from imagen_pytorch.version import __version__
from packaging import version

from patient_dataset import PatientDataset
import os
import pandas as pd
from glob import glob

import re


CHECKPOINT_PATH = "./checkpoint.pt"
TEXT_EMBED_DIM = 3


def unet_generator(unet_number):
    if unet_number == 1:
        return Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            cond_dim=32,
            text_embed_dim=3,
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
        )

    if unet_number == 2:
        return Unet(
            dim=64,
            cond_dim=32,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
            text_embed_dim=TEXT_EMBED_DIM,
            lowres_cond=True,
        )

    return None


class FixedNullUnet(NullUnet):
    def __init__(self, low_res_cond=False, *args, **kwargs):
        super().__init__()
        self.lowres_cond = low_res_cond
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


class ResettableImagenTrainer(ImagenTrainer):
    def load(self, path, only_model=False, strict=True, noop_if_not_exist=False, reset_unet=None):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            self.print(f'trainer checkpoint not found at {str(path)}')
            return

        assert fs.exists(path), f'{path} does not exist'

        self.reset_ema_unets_all_one_device()

        # to avoid extra GPU memory usage in main process when using Accelerate

        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

        try:
            self.imagen.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict. Trying partial load")
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(),
                                                      loaded_obj['model']))

        if only_model:
            return loaded_obj

        self.steps.copy_(loaded_obj['steps'])
        if exists(reset_unet):
            self.steps[reset_unet-1] = 0

        for ind in range(0, self.num_unets):
            if exists(reset_unet) and ind == reset_unet-1:
                continue

            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler) and warmup_scheduler_key in loaded_obj:
                warmup_scheduler.load_state_dict(loaded_obj[warmup_scheduler_key])

            if exists(optimizer):
                try:
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        if self.use_ema:
            assert 'ema' in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj['ema'], strict=strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(),
                                                             loaded_obj['ema']))

            if exists(reset_unet):
                self.imagen.unets[reset_unet-1] = unet_generator(reset_unet)
                self.ema_unets[reset_unet-1] = EMA(self.imagen.unets[reset_unet-1])

        self.print(f'checkpoint loaded from {path}')
        return loaded_obj


def init_imagen():
    imagen = Imagen(
        unets=(unet_generator(1), unet_generator(2)),
        image_sizes=(64, 256),
        timesteps=1000,
        text_embed_dim=TEXT_EMBED_DIM,
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

    #Load the labelled data from the h5 labelbox download
    patient_labelled_datapath =  f'{args.data_path}/results.h5'

    print(f'Found {len(patient_outcomes)} patients with SVS files')

    # Initialise PatientDataset
    dataset = PatientDataset(patient_outcomes, patient_creatinine, f'{args.data_path}/svs/', patient_labelled_dir, patch_size=1024, image_size=256)
    print(f'Found {len(dataset) // 8} patches')

    run_name = uuid4()

    try:
        os.makedirs(f"samples/{run_name}")
    except FileExistsError:
        pass

    imagen = init_imagen()
    trainer = ResettableImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True,
    ).cuda()

    trainer.add_train_dataset(dataset, batch_size=16)

    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint, reset_unet=args.unet_number if args.reset_unet else None)

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
                stop_at_unet_number=args.unet_number,
            )
            for index in range(len(images)):
                images[index].save(f'samples/{run_name}/sample-{i}-{run_name}.png')
            trainer.save(args.checkpoint)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to checkpoint')
    parser.add_argument('--unet_number', type=int, choices=range(1, 3), help='Unet to train')
    parser.add_argument('--data_path', type=str, help='Path of training dataset')
    parser.add_argument('--sample_freq', type=int, default=500, help='How many epochs between sampling and checkpoint.pt saves')
    parser.add_argument('--reset_unet', action='store_true', help='Reset the unet weights')
    return parser.parse_args()


if __name__ == '__main__':
    main()
