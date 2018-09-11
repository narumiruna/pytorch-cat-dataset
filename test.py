from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image

from dataset import CatDataset


def main():
    image_size = 128
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    cat_loader = data.DataLoader(
        CatDataset('data', transform=transform),
        batch_size=batch_size,
        shuffle=True)

    save_image(next(iter(cat_loader)), 'test.jpg')


if __name__ == '__main__':
    main()
