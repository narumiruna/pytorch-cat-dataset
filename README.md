# pytorch-cat-dataset

https://www.kaggle.com/crawford/cat-dataset

Download cats.zip

```
$ unzip cats.zip -d data
$ python3 align.py
```

```python
from torch.utils import data
from torchvision import transforms

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

    for x in cat_loader:
        ...
```