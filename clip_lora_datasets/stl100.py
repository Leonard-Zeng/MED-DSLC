from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms

class CIFAR100Dataset(Dataset):
    """
    Wrapper over torchvision CIFAR100 with:
      - .class_names for 100 fine labels
      - .superclass_names for 20 coarse labels
    Optionally returns class names per sample.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        to_96_like_stl: bool = False,
        transform=None,
        target_transform=None,
        download: bool = True,
        return_class_name: bool = False,
        return_coarse: bool = False,
    ):
        if transform is None:
            t = [transforms.ToTensor()]
            if to_96_like_stl:
                t.insert(0, transforms.Resize((96, 96)))
            transform = transforms.Compose(t)

        self._ds = CIFAR100(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # fine label names
        self.classnames = self._ds.classes  # 100 fine classes

        # self.classnames = meta["fine_label_names"] 
        self.return_class_name = return_class_name

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, fine_label = self._ds[idx]

        if self.return_class_name:
            fine_name = self.classnames[fine_label]
            return img, fine_label, fine_name

        return img, fine_label

if __name__ == '__main__':
    train100 = CIFAR100Dataset(
        root="../data",
        train=True,
        to_96_like_stl=True,   # resize to 96x96
        return_class_name=True,
        return_coarse=True
    )

    print("Fine classes (100):", train100.classnames[:10])

    img, fine_label, fine_name = train100[0]
    print(f"img shape={img.shape}, fine={fine_label} ({fine_name})")
