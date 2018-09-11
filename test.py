from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image

from dataset import CatDataset


def main():
    transform = transforms.Compose(
        [transforms.Resize(128), transforms.ToTensor()])

    cat_loader = data.DataLoader(
        CatDataset('data', transform=transform), batch_size=64, shuffle=True)

    save_image(next(iter(cat_loader)), 'test.jpg')


if __name__ == '__main__':
    main()
