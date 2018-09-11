import argparse
import glob
import math
import os

from PIL import Image

import utils


def norm(*points, p=2):
    return math.sqrt(sum([(x - y)**p for x, y in zip(*points)]))


def centroid(*points):
    return [sum(numbers) / len(points) for numbers in zip(*points)]


def align(image: Image, label: dict):

    left_eye = label['left_eye']
    right_eye = label['right_eye']
    mouth = label['mouth']

    # compute degrees and rotate
    x = right_eye[0] - left_eye[0]
    y = right_eye[1] - left_eye[1]

    theta = math.degrees(math.atan2(y, x))
    center = centroid(left_eye, right_eye, mouth)

    image = image.rotate(theta, center=center)

    # crop
    w = int(norm(left_eye, right_eye))

    image = image.crop(
        box=(center[0] - w, center[1] - w, center[0] + w, center[1] + w))

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='data')
    args = parser.parse_args()
    print(args)

    label_paths = glob.glob(os.path.join(args.root, '*/*.cat'))
    for i, label_path in enumerate(label_paths):
        image_path = label_path.rstrip('.cat')

        if not os.path.exists(image_path):
            continue

        image = utils.load_image(image_path)
        label = utils.load_label(label_path)
        aligned_image = align(image, label)

        aligned_image_path = image_path.replace('.jpg', '_aligned.jpg')

        aligned_image.save(aligned_image_path)

        if (i + 1) % 1000 == 0:
            print('{}/{}'.format(i + 1, len(label_paths)))


if __name__ == '__main__':
    main()
