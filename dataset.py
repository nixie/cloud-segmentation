import os
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class CloudDataset(Dataset):
    def __init__(self, root_dir='/data/zaitra/clouds/'):
        self.img_dir = Path(root_dir) / 'subscenes'
        self.mask_dir = Path(root_dir) / 'masks'

        subscenes = sorted(self.img_dir.rglob('*.npy'))
        assert len(subscenes) == 513

        self.ids = sorted(map(lambda x: x.with_suffix('').name, subscenes))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        item = self.ids[ix]

        img_file = self.img_dir / (item + '.npy')
        mask_file = self.mask_dir / (item + '.npy')

        data = np.load(img_file)
        mask = np.load(mask_file)
        assert data.shape == (1022, 1022, 13), data.shape
        assert mask.shape == (1022, 1022, 3), mask.shape
        assert mask.dtype == bool

        # RGB+Nir
        data = data[..., [3,2,1,8]]

        # NHWC - NCHW
        data = np.transpose(data, [2,0,1])

        # CLEAR, CLOUD, CLOUD_SHADOW -> CLEAR + CLOUD_SHODOW, CLOUD
        classes = (np.argmax(mask, axis=-1) == 1).astype(np.int64)

        return torch.from_numpy(data).float(), \
            torch.from_numpy(classes).unsqueeze(0).float(), ix

    def plot(self, ix, axes=None):

        if axes is None:
            f,(ax, ax2, ax3) = plt.subplots(1,3, figsize=[15,5], dpi=200)
        else:
            ax, ax2, ax3 = axes

        data, classes = self[ix]

        ax.imshow(data[:,:,[0,1,2]],aspect='equal')
        ax3.imshow(data[...,3], aspect='equal')
        h=ax2.imshow(classes, aspect='equal', vmin=0, vmax=2)

