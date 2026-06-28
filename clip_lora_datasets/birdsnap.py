from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Any
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset
from torchvision import transforms

@dataclass
class BirdsnapData(Dataset):

    split: str = "train"                       
    cache_dir: Optional[str] = None            
    transform: Optional[Callable] = None      
    return_class_name: bool = False            
    _ds: HFDataset = field(init=False, repr=False)
    class_names: list[str] = field(init=False)

    def __post_init__(self):
        # Load the HF dataset
        self._ds = load_dataset(
            "sasha/birdsnap", 
            split=self.split,
            cache_dir=self.cache_dir
        )

        # Some versions define label as ClassLabel, others just str
        label_feature = self._ds.features["label"]
        if hasattr(label_feature, "names"):
            self.class_names = label_feature.names
        else:
            self.class_names = sorted(set(self._ds["label"]))

        # Default transform (ToTensor only) if none is passed
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Optional[str]]:
        sample = self._ds[idx]
        img, label = sample["image"], sample["label"]

        # Apply transform to image
        if self.transform:
            img = self.transform(img)

        if self.return_class_name:
            return img, label, self.class_names[label]
        return img, label

if __name__ == "__main__":
    train_ds = BirdsnapData(
        split="train",
        cache_dir="/data/hf_cache/birdsnap",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        return_class_name=True,
    )

    print("Num classes:", len(train_ds.class_names))
    print("First 10 classes:", train_ds.class_names[:10])

    img, label, name = train_ds[0]
    print(f"Image shape: {tuple(img.shape)}, label={label}, name={name}")

