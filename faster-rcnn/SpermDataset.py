from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from skimage import io, transform
import numpy as np
class SpermDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sperm_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sperm_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.sperm_frame.iloc[idx, 0])
        image = io.imread(img_name)
        sperms = self.sperm_frame.iloc[idx, 1:5].as_matrix()
        sperms = sperms.astype('float').reshape(-1, 2)
        sample = {'image': image, 'sperms': sperms}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, sperms = sample['image'], sample['sperms']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'sperms': torch.from_numpy(sperms)}