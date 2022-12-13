import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# dataset and dataloader

class Dataset(Dataset):
    def __init__(
            self,
            dataset,
    ):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (img, idx) = self.dataset[index]

        return img, F.one_hot(torch.tensor([idx]), 10).float().cuda()
