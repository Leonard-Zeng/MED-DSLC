from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .cub200 import CUB200
from .resisc45 import RESISC45
from .aibd_cars import AIBDCars
from .imagenet import ImageNet


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "cub200": CUB200,
                "resisc45": RESISC45,
                "aibd_cars": AIBDCars,
                "imagenet": ImageNet,
                }


def build_dataset(dataset, root_path, shots, preprocess, subsample="all"):
    if 'imagenet' in dataset:
        # subsample = "all" if dataset != 'imagenet' else subsample
        return dataset_list['imagenet'](
            root=root_path, num_shots=shots, preprocess=preprocess, subsample=subsample, dataset_dir=dataset
        )
    else:
        return dataset_list[dataset](root_path, shots, subsample=subsample)