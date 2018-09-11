import glob
import os

import numpy as np
from PIL import Image
from torch.utils import data


def load_image(f, mode='RGB'):
    with Image.open(f) as image:
        return image.convert(mode)


def load_label(f):
    with open(f, 'r') as fp:
        s = fp.read().strip()
        return np.fromstring(s, dtype=int, sep=' ')[1:]


class CatDataset(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.samples = None
        self.transform = transform

        self._prepare_samples()

    def _prepare_samples(self):
        self.samples = []
        label_paths = glob.glob(os.path.join(self.root, '*/*.cat'))
        for label_path in label_paths:
            image_path = label_path.rstrip('.cat')
            if os.path.exists(image_path):
                label = load_label(label_path)
                sample = image_path, label
                self.samples.append(sample)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)
