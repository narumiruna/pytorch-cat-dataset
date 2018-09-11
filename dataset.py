import glob
import os

from torch.utils import data
from torchvision.datasets.folder import pil_loader


class CatDataset(data.Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.paths = glob.glob(os.path.join(self.root, '*/*aligned.jpg'))
        self.transform = transform

    def __getitem__(self, index: int):
        image = pil_loader(self.paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.paths)
