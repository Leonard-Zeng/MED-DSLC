from typing import List, Tuple

import torch
import open_clip

from lora_mixture.utils import apply_lora_mixture as apply_lora_mixture_std
from lora_med.utils import apply_lora_mixture as apply_lora_mixture_mole


def _load_metadata(load_paths: List[str]):
    metas = []
    for p in load_paths:
        metas.append(torch.load(p)["metadata"])
    return metas


def _build_r_alpha(metas: List[dict], alpha_in: List[float] | None):
    r_real = [m["r"] for m in metas]
    alpha_real = alpha_in if alpha_in is not None else [m["alpha"] for m in metas]
    return r_real, alpha_real


def _zero_out_expert(layer, expert_idx: int, params: List[str]):
    for param in params:
        if param == "o":
            if hasattr(layer, "proj"):
                if hasattr(layer.proj, f"w_lora_A_{expert_idx}"):
                    getattr(layer.proj, f"w_lora_A_{expert_idx}").data.zero_()
                if hasattr(layer.proj, f"w_lora_B_{expert_idx}"):
                    getattr(layer.proj, f"w_lora_B_{expert_idx}").data.zero_()
        else:
            proj = getattr(layer, f"{param}_proj", None)
            if proj is not None:
                if hasattr(proj, f"w_lora_A_{expert_idx}"):
                    getattr(proj, f"w_lora_A_{expert_idx}").data.zero_()
                if hasattr(proj, f"w_lora_B_{expert_idx}"):
                    getattr(proj, f"w_lora_B_{expert_idx}").data.zero_()


def load_lora_mixtures_with_zero(
    clip_model: open_clip.CLIP,
    backbone: str,
    load_paths: List[str],
    dropout_rate: float = 0.0,
    include_base_clip: bool = True,
) -> Tuple[list, int]:
    """Load standard LoRA mixtures, optionally appending an explicit zero/base expert.

    When include_base_clip=True, the base/zero expert is appended as the last index.
    When include_base_clip=False, experts are indexed 0..len(load_paths)-1.
    """
    metas = _load_metadata(load_paths)
    encoder = metas[0]["encoder"]
    params = metas[0]["params"]
    position = metas[0]["position"]
    r_real, alpha_real = _build_r_alpha(metas, None)

    if include_base_clip:
        num_experts = 1 + len(load_paths)  # real + zero
        r_full = r_real + [r_real[0]]
        alpha_full = alpha_real + [0.0]
        expert_start = 0
        base_index = num_experts - 1
    else:
        num_experts = len(load_paths)
        r_full = r_real
        alpha_full = alpha_real
        expert_start = 0
        base_index = None

    list_lora = apply_lora_mixture_std(
        clip_model,
        backbone=backbone,
        encoder=encoder,
        position=position,
        params=params,
        r=r_full,
        alpha=alpha_full,
        dropout_rate=dropout_rate,
    )

    # zero-out base expert when included
    if include_base_clip:
        for layer in list_lora:
            _zero_out_expert(layer, base_index, params)

    # load real experts into indices [expert_start..]
    for expert_offset, load_path in enumerate(load_paths, start=expert_start):
        weights = torch.load(load_path)["weights"]
        for i, layer in enumerate(list_lora):
            layer_weights = weights[f"layer_{i}"]
            for param in params:
                if param == "o":
                    if "proj" in layer_weights:
                        # Saved weights use w_lora_A/w_lora_B (no expert index)
                        getattr(layer.proj, f"w_lora_A_{expert_offset}").data.copy_(
                            layer_weights["proj"]["w_lora_A"]
                        )
                        getattr(layer.proj, f"w_lora_B_{expert_offset}").data.copy_(
                            layer_weights["proj"]["w_lora_B"]
                        )
                else:
                    key = f"{param}_proj"
                    if key in layer_weights:
                        proj = getattr(layer, f"{param}_proj")
                        # Saved weights use w_lora_A/w_lora_B (no expert index)
                        getattr(proj, f"w_lora_A_{expert_offset}").data.copy_(
                            layer_weights[key]["w_lora_A"]
                        )
                        getattr(proj, f"w_lora_B_{expert_offset}").data.copy_(
                            layer_weights[key]["w_lora_B"]
                        )
    return list_lora, num_experts


def load_mole_mixtures_with_zero(
    clip_model: open_clip.CLIP,
    backbone: str,
    load_paths: List[str],
    dropout_rate: float = 0.0,
    include_base_clip: bool = True,
    gate_type: str = "mole",
    top_k_expert: int = 0,
    simplify_mole: bool = False,
) -> Tuple[list, int]:
    """Load MED mixtures, optionally appending an explicit zero/base expert.

    When include_base_clip=True, the base/zero expert is appended as the last index.
    When include_base_clip=False, experts are indexed 0..len(load_paths)-1.
    
    Args:
        gate_type: "mole" for softmax expert gating, "sigmoid" for per-rank sigmoid gating, or "phatgoose".
    """
    metas = _load_metadata(load_paths)
    encoder = metas[0]["encoder"]
    params = metas[0]["params"]
    position = metas[0]["position"]
    r_real, alpha_real = _build_r_alpha(metas, None)

    if include_base_clip:
        num_experts = 1 + len(load_paths)
        r_full = r_real + [r_real[0]]
        alpha_full = alpha_real + [0.0]
        expert_start = 0
        base_index = num_experts - 1
    else:
        num_experts = len(load_paths)
        r_full = r_real
        alpha_full = alpha_real
        expert_start = 0
        base_index = None

    list_lora = apply_lora_mixture_mole(
        clip_model,
        backbone=backbone,
        encoder=encoder,
        position=position,
        params=params,
        r=r_full,
        alpha=alpha_full,
        dropout_rate=dropout_rate,
        gate_type=gate_type,
        top_k_expert=top_k_expert,
        simplify_mole=simplify_mole,
    )
    if include_base_clip:
        for layer in list_lora:
            _zero_out_expert(layer, base_index, params)

    for expert_offset, load_path in enumerate(load_paths, start=expert_start):
        weights = torch.load(load_path)["weights"]
        for i, layer in enumerate(list_lora):
            layer_weights = weights[f"layer_{i}"]
            for param in params:
                if param == "o":
                    if "proj" in layer_weights and hasattr(layer, "proj"):
                        # Saved weights use w_lora_A/w_lora_B (no expert index)
                        getattr(layer.proj, f"w_lora_A_{expert_offset}").data.copy_(
                            layer_weights["proj"]["w_lora_A"]
                        )
                        getattr(layer.proj, f"w_lora_B_{expert_offset}").data.copy_(
                            layer_weights["proj"]["w_lora_B"]
                        )
                else:
                    key = f"{param}_proj"
                    if key in layer_weights and hasattr(layer, f"{param}_proj"):
                        proj = getattr(layer, f"{param}_proj")
                        # Saved weights use w_lora_A/w_lora_B (no expert index)
                        getattr(proj, f"w_lora_A_{expert_offset}").data.copy_(
                            layer_weights[key]["w_lora_A"]
                        )
                        getattr(proj, f"w_lora_B_{expert_offset}").data.copy_(
                            layer_weights[key]["w_lora_B"]
                        )
    return list_lora, num_experts
