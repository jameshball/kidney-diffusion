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

from ultra_res_airs import AirsDataset
import os
import pandas as pd
from glob import glob
from uuid import uuid4

import re
import gc

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
        pred_objectives=("v", "v", "v"),
        random_crop_sizes=(None, None, 256),
        condition_on_text=False,
    ).to(device)

    return imagen


def log_wandb(args, cur_step, loss, validation=False):
    if args.wandb:
        wandb.log({
            "loss" if not validation else "val_loss" : loss,
            "step": cur_step,
        })


def main():
    args = parse_args()

    if args.wandb:
        import wandb
    
    imagen = init_imagen(args.magnification_level, args.unet_number)
    dl_keywords = ('images',) if args.magnification_level == 0 else ('images', 'cond_images')
    trainer = ImagenTrainer(
        imagen=imagen,
        dl_tuple_output_keywords_names=dl_keywords,
        fp16=False,
        # doing this to try and avoid nan
        max_grad_norm=1,
    )

    ignore_list = ['christchurch_1003.tif', 'christchurch_1020.tif', 'christchurch_374.tif', 'christchurch_405.tif', 'christchurch_436.tif', 'christchurch_467.tif', 'christchurch_530.tif', 'christchurch_562.tif', 'christchurch_595.tif', 'christchurch_628.tif', 'christchurch_662.tif', 'christchurch_731.tif', 'christchurch_806.tif', 'christchurch_844.tif', 'christchurch_882.tif', 'christchurch_920.tif', 'christchurch_957.tif', 'christchurch_982.tif', 'christchurch_1067.tif', 'christchurch_498.tif', 'christchurch_696.tif']

    train_dataset = AirsDataset(f'{args.data_path}/train/image', ignore_list, args.magnification_level, verbose=True)
    valid_dataset = AirsDataset(f'{args.data_path}/val/image', ignore_list, args.magnification_level, verbose=True)
    test_dataset = AirsDataset(f'{args.data_path}/test/image', ignore_list, args.magnification_level, verbose=True)

    trainer.accelerator.print(f'training with dataset of {len(train_dataset)} samples and validating with {len(valid_dataset)} samples')


    trainer.add_train_dataset(train_dataset, batch_size=8, num_workers=args.num_workers, shuffle=True)
    trainer.add_valid_dataset(valid_dataset, batch_size=8, num_workers=args.num_workers, shuffle=True)

    if args.unet_number == 1:
        checkpoint_path = args.unet1_checkpoint
    elif args.unet_number == 2:
        checkpoint_path = args.unet2_checkpoint
    else:
        checkpoint_path = args.unet3_checkpoint

    trainer.load(checkpoint_path, noop_if_not_exist=True)

    run_id = None

    if trainer.is_main:
        if args.wandb:
            run_id = wandb.util.generate_id()
        else:
            run_id = uuid4()

        if args.run_id is not None:
            run_id = args.run_id
        trainer.accelerator.print(f"Run ID: {run_id}")

        try:
            os.makedirs(f"samples/{run_id}")
        except FileExistsError:
            pass

        if args.wandb:
            wandb.init(project=f"training_unet{args.unet_number}", resume=args.resume, id=run_id)

    trainer.accelerator.wait_for_everyone()
    while True:
        step_num = trainer.num_steps_taken(args.unet_number)
        loss = trainer.train_step(unet_number=args.unet_number)
        trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} loss: {loss}')

        if trainer.is_main:
            log_wandb(args, step_num, loss)

        if not (step_num % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number)
            trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} validation loss: {valid_loss}')
            if trainer.is_main:
                log_wandb(args, step_num, loss, validation=True)

        if not (step_num % args.save_freq) and step_num > 0:
            trainer.accelerator.wait_for_everyone()
            unique_path = f"{re.sub(r'.pt$', '', checkpoint_path)}_{step_num}.pt"
            trainer.accelerator.print("Saving model...")
            trainer.save(unique_path)
            trainer.accelerator.print("Saved model under unique name:")
            

        if not (step_num % args.sample_freq):
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print()
            trainer.accelerator.print("Saving model and sampling")

            if trainer.is_main:
                lowres_zoomed_image = None
                rand_zoomed_image = None
                if args.magnification_level == 0:
                    lowres_image = test_dataset[0]
                    rand_image = train_dataset[np.random.randint(len(train_dataset))]
                else:
                    lowres_image, lowres_zoomed_image = test_dataset[0]
                    rand_image, rand_zoomed_image = train_dataset[np.random.randint(len(train_dataset))]

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
                    if args.wandb:
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
    parser.add_argument('--save_freq', type=int, default=50000, help='How many steps between saving a checkpoint under a unique name')
    parser.add_argument('--resume', action='store_true', help='Resume previous run using wandb')
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--magnification_level", type=int, choices=range(0, 3))
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
