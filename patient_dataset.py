import h5py
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import slideio
from tqdm import tqdm
from skimage import color
import numpy as np

NUM_FLIPS_ROTATIONS = 8
NUM_TRANSLATIONS =  4

# Possible kidney outcomes, ordered by severity. DWFG is ignored here as it does not indicate a poor outcome.
OUTCOMES = ["Functioning", "25%", "50%", "Graft_Loss", "DWGL"]


def normalize_patient_outcomes(x):
    return OUTCOMES.index(x) / len(OUTCOMES) if x in OUTCOMES else 0


def normalize_time_post_transplant(x):
    return (x - 90) / 365


def normalize_creatinine(x):
    return (x - 30) / 2050


class PatientDataset(Dataset):
    def __init__(self, patient_outcomes, patient_creatinine, svs_dir, h5_path, patch_size=256, image_size=64):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.labels = {'Tubuli': 1, 'Vein': 2, 'Vessel_indeterminate': 2, 'Artery': 3, 'Glomerui': 4}
        self.h5_path = h5_path

        # Normalise the patient outcomes
        patient_outcomes["final_outcome"] = patient_outcomes["final_outcome"].apply(normalize_patient_outcomes)

        # Normalise the number of days post transplant
        patient_outcomes["time_post_transplant"] = patient_outcomes["time post tx of biopsy (days)"].apply(
            normalize_time_post_transplant)

        # Get the date of biopsy
        patient_outcomes["date_of_biopsy"] = patient_outcomes["Date of transplantation"] + pd.to_timedelta(
            patient_outcomes["time post tx of biopsy (days)"], unit='d')

        self.creatinine_avg = {}
        # Average the creatinine levels between the transplant and biopsy
        for patient_id, creatinine in tqdm(patient_creatinine.items(), desc="Normalising data"):
            # Normalise the creatinine values
            creatinine["creatinine"] = creatinine["Value"].apply(normalize_creatinine)

            # Get the date of the transplant and biopsy
            transplant_date = \
            patient_outcomes[patient_outcomes["patient_UUID"] == patient_id]["Date of transplantation"].iloc[0]
            biopsy_date = patient_outcomes[patient_outcomes["patient_UUID"] == patient_id]["date_of_biopsy"].iloc[0]

            # Get the creatinine values between the transplant and biopsy and average them
            biopsy_transplant_creatinine = creatinine[(creatinine["Sample Collected Date"] >= transplant_date) & (
                        creatinine["Sample Collected Date"] <= biopsy_date)]
            if len(biopsy_transplant_creatinine) > 0:
                self.creatinine_avg[patient_id] = biopsy_transplant_creatinine["creatinine"].mean()
            else:
                self.creatinine_avg[patient_id] = creatinine["creatinine"].mean()

        self.slide_ids = patient_outcomes["slide_UUID"]

        self.patient_outcomes = patient_outcomes
        self.svs_dir = svs_dir

        self.patch_positions = []
        self.num_patches = 0
        for slide_id in tqdm(self.slide_ids, desc="Processing slides"):
            image = slideio.open_slide(self.svs_dir + slide_id + ".svs", "SVS").get_scene(0)

            # Resize the image to blocks of the patch size
            small_img = image.read_block(image.rect,
                                         size=(image.size[0] // self.patch_size, image.size[1] // self.patch_size))

            # Mask out the background
            img_hs = color.rgb2hsv(small_img)
            img_hs = np.logical_and(img_hs[:, :, 0] > 0.8, img_hs[:, :, 1] > 0.05)

            # Get the positions of the patches that are not background
            patch_positions = np.argwhere(img_hs)

            # Scale the positions to the original image size
            patch_positions = patch_positions * self.patch_size

            self.patch_positions.append(patch_positions)
            self.num_patches += len(patch_positions)

        # Add the annotated data from the h5file:
        self.h5_ids = list()
        with h5py.File(self.h5_path, 'r') as h5:
            for name, cut in h5.items():
                if any([x in cut.keys() for x in self.labels.keys()]):
                    if not name.endswith('_0'):  # Omit repeated annotations
                        self.h5_ids.append(name)
                else:
                    print(f'No label data for:{name}')
        print("Images IDs:\n{}".format('\n'.join(self.h5_ids)))

    def __len__(self):
        return NUM_FLIPS_ROTATIONS * NUM_TRANSLATIONS * self.num_patches

    def index_to_slide(self, index):
        for i in range(len(self.slide_ids)):
            if index < len(self.patch_positions[i]):
                patch_position = self.patch_positions[i][index]
                return i, (patch_position[1], patch_position[0])
            else:
                index -= len(self.patch_positions[i])

    def __getitem__(self, index):
        slide_index, patch_position = self.index_to_slide(index // (NUM_FLIPS_ROTATIONS * NUM_TRANSLATIONS))

        patient_id = self.patient_outcomes.iloc[slide_index]["patient_UUID"]

        # Get data about the patient's outcome
        num_days_post_transplant = self.patient_outcomes.iloc[slide_index]["time_post_transplant"]
        final_outcome = self.patient_outcomes.iloc[slide_index]["final_outcome"]
        if patient_id in self.creatinine_avg:
            avg_creatinine = self.creatinine_avg[patient_id]
        else:
            avg_creatinine = 0

        slide = slideio.open_slide(self.svs_dir + self.slide_ids.iloc[slide_index] + ".svs", "SVS").get_scene(0)

        translation_index = index // NUM_FLIPS_ROTATIONS
        if translation_index % NUM_TRANSLATIONS == 0:
            x, y = (patch_position[0], patch_position[1])
        elif translation_index % NUM_TRANSLATIONS == 1:
            x, y = (patch_position[0] + self.patch_size // 2, patch_position[1])
        elif translation_index % NUM_TRANSLATIONS == 2:
            x, y = (patch_position[0] + self.patch_size // 2, patch_position[1] + self.patch_size // 2)
        else:
            x, y = (patch_position[0], patch_position[1] + self.patch_size // 2)
            
        patch = slide.read_block((x, y, self.patch_size, self.patch_size), size=(self.image_size, self.image_size))

        # Convert the patch to a tensor
        patch = torch.from_numpy(patch / 255).permute((2, 0, 1)).float().cuda()

        # Convert conditions to tensor
        conds = torch.tensor([final_outcome, num_days_post_transplant, avg_creatinine]).reshape(1, 3).float().cuda()

        # Rotate and flip the patch
        if index % NUM_FLIPS_ROTATIONS == 0:
            return patch, conds
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2), conds
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1), conds
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2), conds
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2), conds
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2), conds
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1), conds
        else:
            return patch.transpose(1, 2).flip(1).flip(2), conds

