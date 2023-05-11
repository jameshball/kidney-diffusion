import os
import argparse
import pickle
from metrics import metric_utils
from metrics import frechet_inception_distance
import json
from easydict import EasyDict
import numpy as np
import torch


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, required=True, help='G pkl')
    parser.add_argument('--metrics', type=str, required=True, help='which metrics to calculate')
    parser.add_argument('--add_transform', action='store_true',
                        help='add transformation input to the model '
                        '(useful for models downloaded from original sgan3 repo')
    parser.add_argument('--training_options', type=str)

    args = parser.parse_args()
    metrics_list = args.metrics.split(',')
    output_folder = args.input.replace('.pkl', '_metrics')
    os.makedirs(output_folder, exist_ok=True)

    # check which evaluations have not been computed
    prev_expts = set(os.listdir(output_folder))
    if all([m + '.npz' in prev_expts for m in metrics_list]):
        print("All metrics computed!")
        exit()
    new_expts = [m for m in metrics_list if m + '.npz' not in prev_expts]
    print("Running new metrics")
    print(new_expts)
    assert(len(set(new_expts)) == len(new_expts)) # check no duplicates

    # required for multiple workers loading from same dataset.zip file
    torch.multiprocessing.set_start_method('spawn')

    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with open(args.input, 'rb') as f:
        G_ema = pickle.load(f)['G_ema']

    if args.training_options is not None:
        print("Using specified training options")
        with open(args.training_options) as f:
            training_options = json.load(f)
    else:
        # use the training_options.json file saved 
        # in the snapshot directory
        with open(os.path.join(os.path.dirname(args.input), 'training_options.json')) as f:
            training_options = json.load(f)

    for metric in new_expts:
        if metric.startswith('fid-'):
            if metric.startswith('fid-full'):
                splits = metric.split('-') # fid-full1024
                target_resolution = int(remove_prefix(splits[1], 'full'))
                print("Computing metric %s at resolution %d" % (metric, target_resolution))
                dataset_kwargs = EasyDict(training_options['training_set_kwargs'])
                if dataset_kwargs.resolution < target_resolution:
                    # evaluate at a higher resolution using smaller HR
                    # dataset, resizing everything to the same size
                    dataset_kwargs.resolution = target_resolution
                    dataset_kwargs.max_size = training_options['patch_kwargs']['max_size']
                    dataset_kwargs.path = training_options['patch_kwargs']['path']
                    dataset_kwargs.crop_image = True
                print(dataset_kwargs)
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                mode = 'full'
                fid = frechet_inception_distance.compute_fid_full(
                    opts, target_resolution, None, 50000, mode=mode)
            elif metric.startswith('fid-up'): # fid-up1024
                target_resolution = int(remove_prefix(metric, 'fid-up'))
                print("Computing metric %s at resolution %d" % (metric, target_resolution))
                dataset_kwargs = EasyDict(training_options['training_set_kwargs'])
                if dataset_kwargs.resolution < target_resolution:
                    # evaluate at a higher resolution using smaller HR
                    # dataset, resizing everything to the same size
                    dataset_kwargs.resolution = target_resolution
                    dataset_kwargs.max_size = training_options['patch_kwargs']['max_size']
                    dataset_kwargs.path = training_options['patch_kwargs']['path']
                    dataset_kwargs.crop_image = True
                print(dataset_kwargs)
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                fid = frechet_inception_distance.compute_fid_full(
                    opts, target_resolution, None, 50000, mode='up')
            elif metric.startswith('fid-patch'): # fid-patch256-minXmaxY
                patch_options = training_options
                dataset_kwargs = EasyDict(patch_options['patch_kwargs'])
                patch_size = int(remove_prefix(metric.split('-')[1], 'patch'))
                assert(patch_size == dataset_kwargs.resolution)
                size_min = int(remove_prefix(metric.split('-')[2].split('max')[0], 'min'))
                size_max = int(metric.split('-')[2].split('max')[1])
                scale_min = patch_size / size_max if size_max > 0 else None
                scale_max = patch_size / size_min
                # adjust dataset kwargs using desired scale min and max
                dataset_kwargs.scale_min = scale_min
                dataset_kwargs.scale_max = scale_max
                dataset_kwargs.scale_anneal = -1
                print("Computing metric %s" % (metric))
                print("Patch size %d, size_min: %d size_max: %d scale_min: %s scale_max %f"
                      % (patch_size, size_min, size_max, str(scale_min), scale_max))
                print(dataset_kwargs)
                target_resolution = G_ema.init_kwargs.img_resolution
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                splits = metric.split('-')
                mode = 'patch'
                print("FID mode: %s" % mode)
                fid = frechet_inception_distance.compute_fid_patch(
                    opts, target_resolution, 50000, 50000, mode=mode)
                np.savez(os.path.join(output_folder, metric + '.npz'), value=fid)
            elif metric.startswith('fid-subpatch'): # fid-subpatch1024-minXmaxY
                patch_options = training_options
                dataset_kwargs = EasyDict(patch_options['patch_kwargs'])
                patch_size = int(remove_prefix(metric.split('-')[1], 'subpatch'))
                assert(patch_size == dataset_kwargs.resolution)
                size_min = int(remove_prefix(metric.split('-')[2].split('max')[0], 'min'))
                size_max = int(metric.split('-')[2].split('max')[1])
                scale_min = patch_size / size_max if size_max > 0 else None
                scale_max = patch_size / size_min
                # adjust dataset kwargs using desired scale min and max
                dataset_kwargs.scale_min = scale_min
                dataset_kwargs.scale_max = scale_max
                dataset_kwargs.scale_anneal = -1
                print("Computing metric %s" % (metric))
                print("Patch size %d, size_min: %d size_max: %d scale_min: %s scale_max %f"
                      % (patch_size, size_min, size_max, str(scale_min), scale_max))
                print(dataset_kwargs)
                target_resolution = G_ema.init_kwargs.img_resolution
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                splits = metric.split('-')
                mode = 'subpatch'
                print("FID mode: %s" % mode)
                fid = frechet_inception_distance.compute_fid_patch(
                    opts, target_resolution, 50000, 50000, mode=mode)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError