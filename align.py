import argparse
import glob
import math
import os

from PIL import Image
from torchvision.datasets.folder import pil_loader


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


def norm(*points, p=2):
    return math.sqrt(sum([(x - y)**p for x, y in zip(*points)]))


def centroid(*points):
    return [sum(numbers) / len(points) for numbers in zip(*points)]


def align(image: Image, label: dict):

    left_eye = label['left_eye']
    right_eye = label['right_eye']
    mouth = label['mouth']

    c = centroid(left_eye, right_eye, mouth)

    # rotate image
    x = right_eye[0] - left_eye[0]
    y = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(y, x))

    image = image.rotate(angle, c=c)

    # crop
    s = int(norm(left_eye, right_eye))

    image = image.crop(box=(c[0] - s, c[1] - s, c[0] + s, c[1] + s))

    return image


def align_images(root):
    label_paths = glob.glob(os.path.join(root, '*/*.cat'))
    for i, label_path in enumerate(label_paths):
        image_path = label_path.rstrip('.cat')

        if not os.path.exists(image_path):
            continue

        image = pil_loader(image_path)
        label = load_label(label_path)
        aligned_image = align(image, label)
        aligned_image_path = image_path.replace('.jpg', '_aligned.jpg')
        aligned_image.save(aligned_image_path)

        if (i + 1) % 1000 == 0:
            print('{}/{}'.format(i + 1, len(label_paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='data')
    args = parser.parse_args()
    print(args)

    align_images(args.root)


if __name__ == '__main__':
    main()
