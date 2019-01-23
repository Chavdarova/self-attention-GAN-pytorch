# Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/datasets.py

import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train',
                 attributes=[], missing_ind=False):
        self.transform = transforms_

        self.selected_attrs = attributes
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]
        self.label_path = "%s/list_attr_celeba.txt" % root
        self.missing_ind = missing_ind
        self.annotations = self.get_annotations()
        self.keys = list(self.annotations.keys())
        
    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, 'r')]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == '1'))
            if self.missing_ind:
                # Basically add a label saying this is the
                # 'everything else' class.
                if 1 not in labels:
                    labels.append(1)
                else:
                    labels.append(0)
            annotations[filename] = labels
        return annotations

    def sample_label(self, bs):
        labels = []
        for i in range(bs):
            rnd_key = np.random.choice(self.keys)
            this_label = self.annotations[rnd_key]
            labels.append(this_label)
        labels = np.asarray(labels)
        return labels

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('/')[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))

        if len(self.selected_attrs) == 0:
            return img
        else:
            return img, label

    def __len__(self):
        return len(self.files)
