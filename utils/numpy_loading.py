import logging

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', mapping={}, train=False):
        if train:
            self.images = np.load(str(images_dir) + "/train_X.npy")
            self.masks = np.load(str(masks_dir) + "/train_seg.npy")
        else:
            self.images = np.load(str(images_dir) + "/valid_X.npy")
            self.masks = np.load(str(masks_dir) + "/valid_seg.npy")
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mapping = mapping

        logging.info(f'Creating dataset with {self.images.shape[0]} examples')

    def __len__(self):
        return self.images.shape[0]


    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        img = img.reshape((64, 64, 3)).transpose((2, 0, 1)) / 255
        mask = mask.reshape((64,64))

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

