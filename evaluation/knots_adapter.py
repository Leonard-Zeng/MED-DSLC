from __future__ import annotations

import copy
import importlib.util
import inspect
import os
import sys
import tempfile
import types
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import loralib


@dataclass
class _KnotsRuntime:
    svd_merger_cls: Any
    get_merge_handler: Any


_KNOTS_RUNTIME: _KnotsRuntime | None = None


class _StateDictBackedModel:
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


class ClipStateDictParamHandler(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def get_ft_parameters(self) -> Dict[str, torch.Tensor]:
        return OrderedDict(
            sorted(
                (k, v.detach().cpu())
                for k, v in self.model.state_dict().items()
                if torch.is_floating_point(v)
            )
        )


def _load_module_from_path(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_utils_shim(merging_functions_module, masking_ops_module):
    shim = types.ModuleType("_knots_utils_shim")

    def get_merging_fn(name: str):
        vector_fns = {
            k.replace("_merging", ""): v
            for (k, v) in inspect.getmembers(merging_functions_module, inspect.isfunction)
            if k.endswith("_merging")
        }
        if name not in vector_fns:
            raise KeyError(f"Unknown KnOTS merging function: {name}")
        return vector_fns[name]

    def get_mask_fn(name: str):
        masking_fns = {
            k.replace("_masking", ""): v
            for (k, v) in inspect.getmembers(masking_ops_module, inspect.isfunction)
            if k.endswith("_masking")
        }
        if name not in masking_fns:
            raise KeyError(f"Unknown KnOTS masking function: {name}")
        return masking_fns[name]

    shim.get_merging_fn = get_merging_fn
    shim.get_mask_fn = get_mask_fn
    return shim


def _load_knots_runtime() -> _KnotsRuntime:
    global _KNOTS_RUNTIME
    if _KNOTS_RUNTIME is not None:
        return _KNOTS_RUNTIME

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    knots_root = os.path.join(repo_root, "KnOTS")
    task_merger_path = os.path.join(knots_root, "task_merger.py")
    merging_functions_path = os.path.join(knots_root, "merging_functions.py")
    masking_ops_path = os.path.join(knots_root, "masking_ops.py")

    if not os.path.exists(task_merger_path):
        raise FileNotFoundError(f"KnOTS task_merger.py not found: {task_merger_path}")

    merging_functions_module = _load_module_from_path("_knots_merging_functions", merging_functions_path)
    masking_ops_module = _load_module_from_path("_knots_masking_ops", masking_ops_path)
    utils_shim = _build_utils_shim(merging_functions_module, masking_ops_module)

    old_utils = sys.modules.get("utils")
    old_masking_ops = sys.modules.get("masking_ops")
    sys.modules["utils"] = utils_shim
    sys.modules["masking_ops"] = masking_ops_module
    try:
        task_merger_module = _load_module_from_path("_knots_task_merger", task_merger_path)
    finally:
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
        if old_masking_ops is None:
            sys.modules.pop("masking_ops", None)
        else:
            sys.modules["masking_ops"] = old_masking_ops

    # Compatibility shim for upstream KnOTS:
    # task_merger.SVDMerger.merge() may call VectorOps.forward without merge_config.
    vector_ops_cls = getattr(task_merger_module, "VectorOps", None)
    if vector_ops_cls is not None:
        forward_fn = getattr(vector_ops_cls, "forward", None)
        if forward_fn is not None:
            sig = inspect.signature(forward_fn)
            merge_cfg_param = sig.parameters.get("merge_config")
            if merge_cfg_param is not None and merge_cfg_param.default is inspect._empty:
                original_forward = forward_fn

                def _forward_compat(self, directions, merging_fn, merge_config=None):
                    # merge_config is not consumed inside upstream VectorOps.forward.
                    return original_forward(self, directions, merging_fn, merge_config or {})

                vector_ops_cls.forward = _forward_compat

    _KNOTS_RUNTIME = _KnotsRuntime(
        svd_merger_cls=task_merger_module.SVDMerger,
        get_merge_handler=task_merger_module.get_merge_handler,
    )
    return _KNOTS_RUNTIME


def _build_expert_models_from_lora(
    base_model: nn.Module,
    load_paths: List[str],
    backbone: str,
) -> List[Any]:
    expert_models: List[Any] = []
    for load_path in load_paths:
        expert_clip, list_lora_layers = loralib.load_lora_from(backbone, load_path)
        expert_state_dict = loralib.save_lora_as_clip(
            expert_clip, list_lora_layers, return_statedict_only=True
        )
        # Use state_dict-backed lightweight holders instead of full model copies.
        expert_models.append(_StateDictBackedModel(expert_state_dict))
        del expert_clip
        del list_lora_layers
        del expert_state_dict
    return expert_models


def _normalize_scaling_coeffs(scaling_coeffs: Any, num_models: int):
    if scaling_coeffs is None:
        return None
    if isinstance(scaling_coeffs, (float, int)):
        return float(scaling_coeffs)
    if isinstance(scaling_coeffs, list):
        if len(scaling_coeffs) == 1:
            return float(scaling_coeffs[0])
        if len(scaling_coeffs) != num_models:
            raise ValueError(
                f"Expected {num_models} scaling coefficients, but got {len(scaling_coeffs)}."
            )
        return [float(v) for v in scaling_coeffs]
    raise TypeError(f"Unsupported knots scaling_coeffs type: {type(scaling_coeffs)}")


@torch.no_grad()
def build_knots_merged_model(
    base_model: nn.Module,
    load_paths: List[str],
    merge_config: Dict[str, Any],
    backbone: str,
    device: torch.device,
    include_base_clip: bool = True,
) -> nn.Module:
    runtime = _load_knots_runtime()
    svd_merger_cls = runtime.svd_merger_cls

    expert_models = _build_expert_models_from_lora(base_model, load_paths, backbone)
    if include_base_clip:
        expert_models.append(_StateDictBackedModel(base_model.state_dict()))

    if len(expert_models) == 0:
        raise ValueError("No expert models were built for KnOTS merging.")

    merge_cfg = dict(merge_config)
    scaling_coeffs = merge_cfg.pop("scaling_coeffs", None)

    tmp_ingredients = tempfile.NamedTemporaryFile(prefix="knots_ingredients_", suffix=".pt", delete=False)
    tmp_ingredients_path = tmp_ingredients.name
    tmp_ingredients.close()
    merge_cfg["ingredients_path"] = tmp_ingredients_path

    merger = svd_merger_cls(
        finetuned_models=expert_models,
        pretrained_model=base_model.cpu(),
        param_handler=ClipStateDictParamHandler,
        device=device,
        merge_config=merge_cfg,
    )
    normalized_scaling = _normalize_scaling_coeffs(scaling_coeffs, num_models=len(expert_models))
    if normalized_scaling is not None:
        merger.set_scaling_coeffs(normalized_scaling)

    try:
        merger.transform(merge_cfg)
        # Free transform-time caches before merge to reduce peak host RAM.
        if hasattr(merger, "ingredients"):
            merger.ingredients = None
        if hasattr(merger, "ftms_params"):
            merger.ftms_params = []
        if hasattr(merger, "finetuned_models"):
            merger.finetuned_models = []
        merged_model = merger.merge(merge_cfg)
    finally:
        if os.path.exists(tmp_ingredients_path):
            try:
                os.remove(tmp_ingredients_path)
            except OSError:
                pass

    merged_model = merged_model.to(device)
    merged_model.eval()
    return merged_model

