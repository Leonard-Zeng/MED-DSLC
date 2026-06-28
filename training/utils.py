import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from device_utils import resolve_device


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    return resolve_device(device_str)


def load_meta_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Meta config must be a JSON object: {config_path}")
    return payload


def domains_from_meta_config(meta_config: dict, default_domain_order: List[str]) -> List[str]:
    raw_domains = meta_config.get("domains")
    if raw_domains is None:
        return list(default_domain_order)
    if not isinstance(raw_domains, list) or not all(isinstance(x, str) for x in raw_domains):
        raise ValueError("meta config field 'domains' must be a list of domain names.")
    if len(raw_domains) == 0:
        raise ValueError("meta config field 'domains' must not be empty.")
    return list(raw_domains)


def expert_weights_from_meta_config(meta_config: dict) -> Dict[str, str]:
    raw = meta_config.get("expert_weights", {})
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(domain): os.path.abspath(str(path)) for domain, path in raw.items()}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict) or "domain" not in item or "path" not in item:
                raise ValueError(
                    "meta config list field 'expert_weights' must contain objects with 'domain' and 'path'."
                )
            out[str(item["domain"])] = os.path.abspath(str(item["path"]))
        return out
    raise ValueError("meta config field 'expert_weights' must be an object or a list.")


def discover_model_paths(base_dir: str, shots: int, seed: int, domain_order: List[str]) -> Dict[str, str]:
    model_name2path = {}
    for expert_dir in domain_order:
        expert_path = os.path.join(base_dir, expert_dir)
        if not os.path.isdir(expert_path):
            continue
        weights_path = os.path.join(expert_path, f"{shots}shots", f"seed{seed}", "lora_weights.pt")
        if os.path.exists(weights_path):
            model_name2path[expert_dir] = weights_path
    return model_name2path


def resolve_expert_model_paths(
    base_dir: str,
    shots: int,
    seed: int,
    domain_order: List[str],
    explicit_paths: Dict[str, str] | None = None,
) -> Dict[str, str]:
    explicit_paths = explicit_paths or {}
    discovered = discover_model_paths(base_dir, shots, seed, domain_order)
    out = {}
    for domain in domain_order:
        path = explicit_paths.get(domain) or discovered.get(domain)
        if path:
            out[domain] = os.path.abspath(path)
    return out


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Compute Focal Loss for classification.
    
    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the true class.
    
    Args:
        logits: Raw logits from model, shape (N, C) where N is batch size and C is number of classes
        targets: Ground truth class indices, shape (N,)
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0). Higher gamma down-weights easy examples more.
    
    Returns:
        Scalar tensor containing the focal loss value.
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Probability of true class
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()
