from tqdm import tqdm
import numpy as np
import pathlib
import slideio
import argparse
from numpy import random
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2


def sample_patch(img, min_dim, min_scale, max_scale, patch_size, transformation=None):
    new_transformation = {}

    width = img.size[0]
    height = img.size[1]
    print(width, height)

    # real and fake images can have different sizes so we can't do the same
    # random crop
    x_size_diff = width - min_dim
    y_size_diff = height - min_dim
    if x_size_diff <= 0:
        x = 0
    else:
        x = random.randint(x_size_diff)
    if y_size_diff <= 0:
        y = 0
    else:
        y = random.randint(y_size_diff)

    if transformation is None:
        min_scale = max(min_scale, patch_size / width)

        max_size = patch_size / min_scale
        min_size = patch_size / max_scale
        random_size = random.uniform(min_size, max_size)
        scale = patch_size / random_size

        rand_size = int(np.round(patch_size / scale))
        size_diff = rand_size - patch_size
        if size_diff <= 0:
            crop_x = 0
            crop_y = 0
        else:
            crop_x = random.randint(size_diff)
            crop_y = random.randint(size_diff)
    else:
        rand_size = transformation['rand_size']
        crop_x = transformation['crop_x']
        crop_y = transformation['crop_y']

    #if transformation is None:
    print(x, y, min_dim, rand_size)
    img = img.read_block((x, y, x+min_dim, y+min_dim), size=(rand_size, rand_size))
    #else:
    #    img = img[x:x+min_dim, y:y+min_dim, :]
    #    img = cv2.resize(img, dsize=(rand_size, rand_size), interpolation=cv2.INTER_LANCZOS4)

    img = img[:, crop_x:crop_x+patch_size, crop_y:crop_y+patch_size]

    new_transformation['rand_size'] = rand_size
    new_transformation['crop_x'] = crop_x
    new_transformation['crop_y'] = crop_y

    return img, new_transformation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str)
    parser.add_argument('--fake_path', type=str)
    parser.add_argument('--real_output', type=str)
    parser.add_argument('--fake_output', type=str)
    parser.add_argument('--num_files', type=int)
    parser.add_argument('--size_max', type=int)
    parser.add_argument('--size_min', type=int)
    parser.add_argument('--patch_size', type=int)
    args = parser.parse_args()

    scale_min = args.patch_size / args.size_max if args.size_max > 0 else 1.0
    scale_max = args.patch_size / args.size_min
    
    real_path = pathlib.Path(args.real_path)
    real_files = sorted(real_path.glob(f"*.svs"))
    fake_path = pathlib.Path(args.fake_path)
    fake_files = sorted(fake_path.glob(f"*.png"))

    for i in tqdm(range(args.num_files)):
        real_index = random.randint(len(real_files))
        fake_index = random.randint(len(fake_files))

        real_file = str(real_files[real_index])
        fake_file = str(fake_files[fake_index])

        real_slide = slideio.open_slide(real_file, "SVS")
        real_image = real_slide.get_scene(0)

        fake_slide = slideio.open_slide(fake_file, "GDAL")
        fake_image = fake_slide.get_scene(0)

        min_dim = min(real_image.size[0], real_image.size[1], fake_image.size[0], fake_image.size[1])
        real_patch, transformation = sample_patch(real_image, min_dim, scale_min, scale_max, args.patch_size)
        fake_patch, transformation = sample_patch(fake_image, min_dim, scale_min, scale_max, args.patch_size, transformation=transformation)

        cv2.imwrite(args.real_output + f"/{i}.png", real_patch)
        cv2.imwrite(args.fake_output + f"/{i}.png", fake_patch)


if __name__ == '__main__':
    main()
