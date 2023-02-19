import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import color
import numpy as np
import torchvision.transforms as T

NUM_FLIPS_ROTATIONS = 8
NUM_RANDOMCROPS =  4

class PatientDataset(Dataset):
    def __init__(self, data_path, patch_size=256, image_size=64):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size

        self.data_path = data_path

        # Iterate directory
        self.slide_ids = []
        for file in os.listdir(data_path+'/Patches'):
            if file.endswith('.npy'):
                self.slide_ids.append(file)  
        self.num_patches=len(self.slide_ids)
        print(self.num_patches)

    def __len__(self):
        return NUM_FLIPS_ROTATIONS * NUM_RANDOMCROPS * self.num_patches

    def __getitem__(self, original_index):
    
        index = original_index
        patch_index= original_index // (NUM_FLIPS_ROTATIONS * NUM_RANDOMCROPS)      
        labelmap = np.load(self.data_path+'/Labels/'+self.slide_ids[patch_index][:-4]+'1binary_mask.npy')
        labelmap = labelmap.reshape((np.shape(labelmap)[0],np.shape(labelmap)[1],1)) #This is actually only necessary if we were to use multiple labels, but keep if for simplicity
        patch = np.load(self.data_path+'/Patches/'+self.slide_ids[patch_index])

        # Convert the patch to a tensor
        patch = torch.from_numpy(patch / 255).permute((2, 0, 1)).float().cuda()
        labelmap = torch.from_numpy(labelmap).permute((2, 0, 1)).float().cuda()
        #Now apply a random crop
        patch_size = 256
        img_size = list(patch.size())[1]
        pos = np.random.uniform(size=2)*(img_size-self.patch_size)
        patch = T.functional.crop(patch,int(pos[0]),int(pos[1]),self.patch_size,self.patch_size)
        labelmap = T.functional.crop(labelmap,int(pos[0]),int(pos[1]),self.patch_size,self.patch_size)

        # Rotate and flip the patch
        if index % NUM_FLIPS_ROTATIONS == 0:
            return patch, labelmap
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2), labelmap.flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1), labelmap.flip(1)
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2), labelmap.flip(1).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2), labelmap.transpose(1, 2)
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2), labelmap.transpose(1, 2).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1), labelmap.transpose(1, 2).flip(1)
        else:
            return patch.transpose(1, 2).flip(1).flip(2), labelmap.transpose(1, 2).flip(1).flip(2)

