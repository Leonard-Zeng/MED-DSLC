import argparse
from typing import Optional, Union

import torch


DEVICE_CHOICES = ("cpu", "cuda", "mps")


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    device_name = str(device or get_default_device()).lower()
    if device_name not in DEVICE_CHOICES:
        raise ValueError(
            f"Unsupported device '{device_name}'. Choose one of: {', '.join(DEVICE_CHOICES)}."
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected with --device cuda, but CUDA is not available.")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was selected with --device mps, but MPS is not available.")
    return torch.device(device_name)


def device_arg(default: Optional[str] = None) -> dict:
    return {
        "type": argparse_device,
        "default": default or get_default_device(),
        "choices": DEVICE_CHOICES,
        "help": "Device to use: cpu, cuda, or mps.",
    }


def argparse_device(value: str) -> str:
    try:
        return str(resolve_device(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
