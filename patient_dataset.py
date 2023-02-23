import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import color
import numpy as np
import torchvision.transforms as T
from pathlib import Path

NUM_FLIPS_ROTATIONS = 8
NUM_RANDOMCROPS =  4
TYPE = ['Breast', 'Kidney', 'Liver', 'Prostate', 'Bladder', 'Colon', 'Stomach']
DISEASES = ['Breast invasive carcinoma', 'Kidney renal clear cell carcinoma', 'Kidney renal papillary cell carcinoma', 'Lung squamous cell carcinoma','Lung adenocarcinoma', 'Prostate adenocarcinoma', 'Bladder Urothelial Carcinoma', 'Colon adenocarcinoma', 'Stomach adenocarcinoma']

def normalise_type(x):
    return TYPE.index(x) / len(TYPE)

def normalise_diseases(x):
    return DISEASES.index(x) / len(DISEASES)

class PatientDataset(Dataset):
    def __init__(self, data_path, patch_size=256, image_size=64):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size

        self.data_path = data_path
       
        #Now load in the patient information: 
        patient_data = pd.read_csv(data_path+'/Supplementary/supplementary.csv', delimiter=';')
        self.num_patches=len(patient_data['ID'])
        #Check that all patches are actually found: 
        for id in patient_data['ID']:
            if not Path(self.data_path+'/Patches/'+id[:-1]+'.npy').is_file(): print('Patch '+id+' missing')
            if not Path(self.data_path+'/Labels/'+id+'binary_mask.npy').is_file(): print('Label '+id+' missing')
        patient_data['Type'] = patient_data['Type'].apply(normalise_type)
        patient_data['Disease'] = patient_data['Disease'].apply(normalise_diseases)
        self.patient_data = patient_data
        print(self.num_patches)
        

    def __len__(self):
        return NUM_FLIPS_ROTATIONS * NUM_RANDOMCROPS * self.num_patches

    def __getitem__(self, original_index):
    
        index = original_index
        patch_index= original_index // (NUM_FLIPS_ROTATIONS * NUM_RANDOMCROPS)      
        labelmap = np.load(self.data_path+'/Labels/'+self.patient_data['ID'].iloc[patch_index]+'binary_mask.npy')
        labelmap = labelmap.reshape((np.shape(labelmap)[0],np.shape(labelmap)[1],1)) #This is actually only necessary if we were to use multiple labels, but keep if for simplicity
        patch = np.load(self.data_path+'/Patches/'+self.patient_data['ID'].iloc[patch_index][:-1]+'.npy')

        # Convert the patch to a tensor
        patch = torch.from_numpy(patch / 255).permute((2, 0, 1)).float().cuda()
        labelmap = torch.from_numpy(labelmap).permute((2, 0, 1)).float().cuda()
        #Now apply a random crop
        patch_size = 256
        img_size = list(patch.size())[1]
        pos = np.random.uniform(size=2)*(img_size-self.patch_size)
        patch = T.functional.crop(patch,int(pos[0]),int(pos[1]),self.patch_size,self.patch_size)
        labelmap = T.functional.crop(labelmap,int(pos[0]),int(pos[1]),self.patch_size,self.patch_size)
        
        typ = self.patient_data['Type'].iloc[patch_index]
        disease = self.patient_data['Disease'].iloc[patch_index]
        conds = torch.Tensor([typ,disease]).reshape(1,2).float().cuda()

        # Rotate and flip the patch
        if index % NUM_FLIPS_ROTATIONS == 0:
            return patch, conds, labelmap
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2), conds, labelmap.flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1), conds, labelmap.flip(1)
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2), conds, labelmap.flip(1).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2), conds, labelmap.transpose(1, 2)
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2), conds, labelmap.transpose(1, 2).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1), conds, labelmap.transpose(1, 2).flip(1)
        else:
            return patch.transpose(1, 2).flip(1).flip(2), conds, labelmap.transpose(1, 2).flip(1).flip(2)

