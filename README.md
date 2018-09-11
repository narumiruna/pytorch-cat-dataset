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
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    cat_loader = data.DataLoader(
        CatDataset('data', transform=transform),
        batch_size=64,
        shuffle=True)

    for x in cat_loader:
        ...
```