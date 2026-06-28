"""Shared configuration for the MED-LCDS training entrypoint."""

from pathlib import Path

from clip_lora_datasets import (
    Caltech101,
    EuroSAT,
    StanfordCars,
    Food101,
    OxfordPets,
    OxfordFlowers,
    DescribableTextures,
    SUN397,
    UCF101,
    FGVCAircraft,
    CUB200,
    RESISC45,
)

# Fixed domain order (matches evaluation)
DOMAIN_ORDER = [
    "caltech101",
    "eurosat",
    "stanford_cars",
    "food101",
    "oxford_pets",
    "oxford_flowers",
    "dtd",
    "ucf101",
    "fgvc",
]

NEW_DATASET_ORDER = [
    "cub200",
    "resisc45",
]

DATASET_MAP = {
    "caltech101": Caltech101,
    "eurosat": EuroSAT,
    "stanford_cars": StanfordCars,
    "food101": Food101,
    "oxford_pets": OxfordPets,
    "oxford_flowers": OxfordFlowers,
    "dtd": DescribableTextures,
    "sun397": SUN397,
    "ucf101": UCF101,
    "fgvc": FGVCAircraft,
    "cub200": CUB200,
    "resisc45": RESISC45,
}

COMBINED_LORA_METHODS = [
    "single_lora",
    "single_lora_from_pretrained",
]

MED_METHODS = [
    "MED",
    "MED_LCDS",
    "mole",
]

PHATGOOSE_METHODS = [
    "phatgoose",
]

METHOD_CHOICES = (
    COMBINED_LORA_METHODS
    + MED_METHODS
    + PHATGOOSE_METHODS
)

DEFAULT_DESCRIPTIONS_DIR = str(Path(__file__).resolve().parents[1] / "meta" / "descriptions")
