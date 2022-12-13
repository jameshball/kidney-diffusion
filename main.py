from uuid import uuid4

import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torchvision import datasets, transforms as T
from data_wrapper import Dataset
import torch.nn.functional as F
import os


CHECKPOINT_PATH = "./checkpoint.pt"
IMAGE_SIZE = 32


def main():
    # unets for unconditional imagen

    unet = Unet(
        dim=IMAGE_SIZE,
        dim_mults=(1, 2, 4, 8),
        cond_dim=512,
        text_embed_dim=10,
        num_resnet_blocks=1,
        layer_attns=(False, False, False, True),
        layer_cross_attns=False
    )

    # imagen, which contains the unet above

    imagen = Imagen(
        unets=unet,
        image_sizes=IMAGE_SIZE,
        timesteps=1000,
        text_embed_dim=10,
    )

    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True  # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.CenterCrop(IMAGE_SIZE),
        T.Grayscale(3),
        T.ToTensor()
    ])

    dataset = Dataset(datasets.MNIST('./mnist/auto', train=True, download=True, transform=transform))

    trainer.add_train_dataset(dataset, batch_size=16)

    run_name = uuid4()

    if os.path.exists(CHECKPOINT_PATH):
        trainer.load(CHECKPOINT_PATH)

    # Test current images

    nums = [5 for i in range(100)]
    nums_one_hot = F.one_hot(torch.tensor(nums), 10).float().reshape(len(nums), 1, 10).cuda()
    images = trainer.sample(batch_size=len(nums), return_pil_images=True, text_embeds=nums_one_hot)
    for index in range(len(nums)):
        images[index].save(f'./test-{index}.png')

    # working training loop

    for i in range(200000):
        loss = trainer.train_step(unet_number=1, max_batch_size=4)
        print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number=1, max_batch_size=4)
            print(f'valid loss: {valid_loss}')

        if not (i % 100) and trainer.is_main:  # is_main makes sure this can run in distributed
            nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            nums_one_hot = F.one_hot(torch.tensor(nums), 10).float().reshape(len(nums), 1, 10).cuda()
            images = trainer.sample(batch_size=len(nums), return_pil_images=True, text_embeds=nums_one_hot)
            for index in range(len(nums)):
                images[index].save(f'./sample-{i // 100}-{nums[index]}-{run_name}.png')
            trainer.save(CHECKPOINT_PATH)


if __name__ == '__main__':
    main()
