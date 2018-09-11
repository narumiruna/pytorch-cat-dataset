import glob
import os
from math import atan2, degrees, sqrt

from PIL import Image
from torch.utils import data


def load_image(f, mode='RGB'):
    with Image.open(f) as image:
        return image.convert(mode)


def load_label(f):
    with open(f, 'r') as fp:
        points = list(map(int, fp.read().strip().split(' ')))
        return dict(
            left_eye=(points[1], points[2]),
            right_eye=(points[3], points[4]),
            mouth=(points[5], points[6]),
            left_ear_1=(points[7], points[8]),
            left_ear_2=(points[9], points[10]),
            left_ear_3=(points[11], points[12]),
            right_ear_1=(points[13], points[14]),
            right_ear_2=(points[15], points[16]),
            right_ear_3=(points[17], points[18]))


def distance(x, y):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def align(image, label):

    left_eye = label['left_eye']
    right_eye = label['right_eye']
    mouth = label['mouth']

    # compute degrees and rotate
    x = right_eye[0] - left_eye[0]
    y = right_eye[1] - left_eye[1]

    theta = degrees(atan2(y, x))
    image = image.rotate(theta, center=mouth)

    # crop
    w = int(distance(left_eye, right_eye))
    r = 1.4
    u = int(w * r)

    image = image.crop(
        box=(mouth[0] - w, mouth[1] - u, mouth[0] + w, mouth[1] + 2 * w - u))

    return image


class CatDataset(data.Dataset):

    def __init__(self, root, transform=None, align=True):
        self.root = root
        self.samples = None
        self.transform = transform
        self.align = align

        self._prepare_samples()

    def _prepare_samples(self):
        self.samples = []
        label_paths = glob.glob(os.path.join(self.root, '*/*.cat'))
        for label_path in label_paths:
            image_path = label_path.rstrip('.cat')
            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))

    def __getitem__(self, index: int):
        image_path, label_path = self.samples[index]
        image = load_image(image_path)
        label = load_label(label_path)

        if self.align:
            image = align(image, label)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.samples)
