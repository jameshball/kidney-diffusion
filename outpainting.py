from uuid import uuid4
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from imagen_pytorch import ImagenTrainer
from torchvision.transforms import transforms
from tqdm import tqdm

from train import init_imagen, IMAGE_SIZE, CHECKPOINT_PATH
from patient_dataset import normalize_patient_outcomes, normalize_time_post_transplant, normalize_creatinine, OUTCOMES


def main():
    args = parse_args()
    print(args.height, args.width)

    trainer = ImagenTrainer(
        imagen=init_imagen(),
    ).cuda()

    trainer.load(args.checkpoint)

    # Normalize the outcomes
    outcomes = normalize_patient_outcomes(args.outcomes)
    time_post_transplant = normalize_time_post_transplant(args.time_post_transplant)
    creatinine = normalize_creatinine(args.creatinine)

    conds = torch.tensor([outcomes, time_post_transplant, creatinine]).reshape(1, 1, 3).float().cuda()

    # Initialise blank image
    image = torch.full((3, args.height, args.width), -1).float().cuda()
    image_name = uuid4()

    ys = range(0, args.height - IMAGE_SIZE // 2, IMAGE_SIZE // 2)
    xs = range(0, args.width - IMAGE_SIZE // 2, IMAGE_SIZE // 2)
    patch_positions = [(y, x) for y in ys for x in xs]

    for y, x in tqdm(patch_positions, desc="Generating image"):
        inpaint_patch = image[:, y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
        inpaint_mask = (inpaint_patch != -1).float()
        inpaint_patch[inpaint_patch == -1] = 0
        plt.imshow(inpaint_patch.permute(1, 2, 0).cpu().numpy())
        plt.show()
        plt.imshow(inpaint_mask.permute(1, 2, 0).cpu().numpy())
        plt.show()

        inpaint_resample_times = 1 if y == 0 and x == 0 else args.resample_times

        patch = trainer.sample(
            batch_size=1,
            text_embeds=conds,
            inpaint_images=inpaint_patch.unsqueeze(0),
            inpaint_masks=inpaint_mask.unsqueeze(0),
            inpaint_resample_times=inpaint_resample_times,
            cond_scale=5.,
            use_tqdm=False,
        )
        plt.imshow(patch[0].permute(1, 2, 0).cpu().numpy())
        plt.show()
        image[:, y:y+IMAGE_SIZE, x:x+IMAGE_SIZE] = patch

    image = transforms.ToPILImage()(image)
    image.save(f'samples/inference-{image_name}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to checkpoint')
    parser.add_argument('--width', type=int, default=64, choices=range(64, 1025), help='Width of image')
    parser.add_argument('--height', type=int, default=64, choices=range(64, 1025), help='Height of image')
    parser.add_argument('--outcomes', type=str, default='Functioning', choices=OUTCOMES, help='Outcome to predict')
    parser.add_argument('--time_post_transplant', type=int, default=90, choices=range(90, 456), help='Number of days the biopsy was taken after transplant')
    parser.add_argument('--creatinine', type=float, default=70, choices=range(30, 2080), help='Average creatinine level between transplant and biopsy')
    parser.add_argument('--resample_times', type=int, default=1, choices=range(1, 11), help='Number of times to resample the image when inpainting')
    return parser.parse_args()


if __name__ == '__main__':
    main()
