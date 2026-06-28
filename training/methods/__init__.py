from training.config import (
    COMBINED_LORA_METHODS,
    MED_METHODS,
    PHATGOOSE_METHODS,
)
from . import (
    mole,
    phatgoose,
    single_lora,
)

def get_trainer(method: str):
    if method in COMBINED_LORA_METHODS:
        return single_lora.train
    if method in MED_METHODS:
        return mole.train
    if method in PHATGOOSE_METHODS:
        return phatgoose.train
    raise ValueError(f"Unknown method: {method}")

