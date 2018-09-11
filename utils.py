from PIL import Image


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
