import os
import sys
import copy
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import open_clip
import matplotlib.pyplot as plt
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import loralib
from lora_med.utils import load_gate_nets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clip_lora_datasets.utils as ds_utils
from clip_lora_datasets.utils import build_domain_template, create_domain_prompt
from training.lora_loading import (
    load_lora_mixtures_with_zero,
    load_mole_mixtures_with_zero,
)
from training.utils import (
    domains_from_meta_config,
    expert_weights_from_meta_config,
    load_meta_config,
    resolve_expert_model_paths,
)
from lora_med.logit_scalar import infer_clip_feature_dim, load_logit_scalar
from device_utils import device_arg, resolve_device
from clip_lora_datasets import (
    Caltech101,
    EuroSAT,
    StanfordCars,
    Food101,
    OxfordPets,
    OxfordFlowers,
    ImageNet,
    DescribableTextures,
    SUN397,
    UCF101,
    FGVCAircraft,
    CUB200,
    RESISC45,
    AIBDCars,
)

try:
    from . import (
        evaluate_base,
        evaluate_lora_mean,
        evaluate_expert_lora,
        evaluate_mole,
    )
    from .benchmark_builder import BenchmarkBuildResult, build_benchmark_dataset
    from .eval_utils import (
        build_semantic_directional_lookup,
        build_semantic_equivalence_lookup,
    )
    from .overlap import OVERLAP_MERGE_MODES, OVERLAP_MERGE_NONE, write_overlap_resolution
    from .knots_adapter import build_knots_merged_model
except ImportError:
    from evaluation import (
        evaluate_base,
        evaluate_lora_mean,
        evaluate_expert_lora,
        evaluate_mole,
    )
    from evaluation.benchmark_builder import BenchmarkBuildResult, build_benchmark_dataset
    from evaluation.eval_utils import (
        build_semantic_directional_lookup,
        build_semantic_equivalence_lookup,
    )
    from evaluation.overlap import OVERLAP_MERGE_MODES, OVERLAP_MERGE_NONE, write_overlap_resolution
    from evaluation.knots_adapter import build_knots_merged_model


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

NEW_EVAL_DATASET_ORDER = [
    "cub200",
    "resisc45",
]

AIBD_STANFORD_SOURCE_DOMAIN = "stanford_cars"
AIBD_STANFORD_INFERENCE_DOMAIN = "aibd_cars"

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
    "aibd_cars": AIBDCars,
    "imagenet": ImageNet,
}


def build_active_domain_order(
    include_new_ds: bool,
    use_aibd_cars: bool = True,
) -> List[str]:
    domain_order = list(DOMAIN_ORDER)
    if include_new_ds:
        if use_aibd_cars:
            domain_order = [
                AIBD_STANFORD_INFERENCE_DOMAIN if domain == AIBD_STANFORD_SOURCE_DOMAIN else domain
                for domain in domain_order
            ]
        for domain in NEW_EVAL_DATASET_ORDER:
            if domain not in domain_order:
                domain_order.append(domain)

    return domain_order


def build_expert_domain_order(include_new_ds: bool) -> List[str]:
    """Build the training/expert domain order used for loading LoRA checkpoints."""
    domain_order = list(DOMAIN_ORDER)
    if include_new_ds:
        for domain in NEW_EVAL_DATASET_ORDER:
            if domain not in domain_order:
                domain_order.append(domain)

    return domain_order


class GateDomainCSVLogger:
    def __init__(self, method: str, domain_name_by_index: Dict[int, str] | None = None):
        self.method = method
        self.domain_name_by_index = domain_name_by_index or {}
        self.hooks = []
        self._current_domains: List[str] | None = None
        self._sum: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self._count: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._layout_by_source: Dict[str, bool] | None = None

    def start_batch(self, domains=None, current_domain: str | None = None, batch_size: int | None = None):
        self._current_domains = self._resolve_batch_domains(
            domains=domains,
            current_domain=current_domain,
            batch_size=batch_size,
        )

    def end_batch(self):
        self._current_domains = None

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_mixture_layers(self, mixture_layers):
        source_layer_idx: Dict[str, int] = defaultdict(int)
        for layer in mixture_layers:
            gate_net = getattr(layer, "gate_net", None)
            if gate_net is None:
                continue
            layer_type = type(layer).__name__.lower()
            source = "gate_image" if "image" in layer_type else "gate_text"
            source_layer_idx[source] += 1
            layer_name = f"layer_{source_layer_idx[source]}"
            self.hooks.append(
                gate_net.register_forward_hook(self._make_gate_hook(source=source, layer_name=layer_name, joint=False))
            )

    def register_joint_gates(self, joint_gates):
        for idx, gate in enumerate(joint_gates):
            layer_name = f"joint_layer_{idx + 1}"
            self.hooks.append(
                gate.register_forward_hook(self._make_gate_hook(source="gate_image", layer_name=layer_name, joint=True))
            )

    def log_domain_logits(self, logits: torch.Tensor, domains, source: str, layer: str):
        if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
            return
        domain_list = self._resolve_batch_domains(domains=domains, current_domain=None, batch_size=logits.shape[0])
        if domain_list is None:
            return
        # Domain heads output logits; store probabilities as activations.
        values = torch.softmax(logits.detach().float(), dim=-1).cpu()
        self._accumulate(source=source, layer=layer, domain_list=domain_list, values=values)

    def finalize_to_csv(self, csv_path: str):
        self._layout_by_source = self._infer_layout_by_source()
        rows = []
        max_dim = 0
        for key, summed in self._sum.items():
            source, layer, domain = key
            count = self._count[key]
            if count <= 0:
                continue
            mean_activations = self._postprocess_values(
                values=summed / float(count),
                source=source,
            ).tolist()
            max_dim = max(max_dim, len(mean_activations))
            row = {
                "method": self.method,
                "source": source,
                "layer": layer,
                "domain": domain,
                "count": count,
            }
            for i, val in enumerate(mean_activations):
                row[f"expert_{i}"] = float(val)
            rows.append(row)

        if len(rows) == 0:
            return

        columns = ["method", "source", "layer", "domain", "count"] + [f"expert_{i}" for i in range(max_dim)]
        out_df = pd.DataFrame(rows)
        out_df = out_df.reindex(columns=columns)
        out_df.to_csv(csv_path, index=False)
        print(f"Gate/domain activations saved to {csv_path}")

    def finalize_heatmaps(self, output_dir: str):
        if len(self._sum) == 0:
            return
        self._layout_by_source = self._infer_layout_by_source()
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        by_source: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for key, summed in self._sum.items():
            source, layer, domain = key
            count = self._count.get(key, 0)
            if count <= 0:
                continue
            mean_values = self._postprocess_values(
                values=(summed / float(count)).detach().float(),
                source=source,
            ).cpu().numpy()
            by_source.setdefault(source, {}).setdefault(layer, {})[domain] = mean_values

        for source, layer_to_domain in by_source.items():
            if len(layer_to_domain) == 0:
                continue
            ordered_domains = self._ordered_domains(layer_to_domain)
            if len(ordered_domains) == 0:
                continue
            num_experts = self._infer_num_experts(layer_to_domain)
            if num_experts <= 0:
                continue
            expert_labels = self._infer_expert_labels(num_experts)
            plot_indices = [
                idx for idx, label in enumerate(expert_labels)
                if str(label) != "base_clip"
            ]
            if len(plot_indices) == 0:
                continue
            plot_labels = [expert_labels[idx] for idx in plot_indices]

            layer_names = sorted(layer_to_domain.keys(), key=self._layer_sort_key)
            layer_mats = []
            for layer_name in layer_names:
                mat = np.full((len(ordered_domains), num_experts), np.nan, dtype=np.float32)
                domain_to_values = layer_to_domain[layer_name]
                for row_idx, domain_name in enumerate(ordered_domains):
                    values = domain_to_values.get(domain_name)
                    if values is None:
                        continue
                    width = min(num_experts, int(values.shape[0]))
                    mat[row_idx, :width] = values[:width]
                plot_mat = mat[:, plot_indices]
                layer_mats.append(plot_mat)
                layer_filename = (
                    f"{self._safe_name(self.method)}_{self._safe_name(source)}_{self._safe_name(layer_name)}.png"
                )
                self._save_heatmap(
                    matrix=plot_mat,
                    row_labels=ordered_domains,
                    col_labels=plot_labels,
                    title=f"{self._display_method_name()} | Layer {self._display_layer_name(layer_name)}",
                    output_path=os.path.join(heatmap_dir, layer_filename),
                )

            if len(layer_mats) == 0:
                continue
            stack = np.stack(layer_mats, axis=0)
            avg_mat = np.nanmean(stack, axis=0)
            avg_filename = (
                f"{self._safe_name(self.method)}_{self._safe_name(source)}_avg_across_layers.png"
            )
            self._save_heatmap(
                matrix=avg_mat,
                row_labels=ordered_domains,
                col_labels=plot_labels,
                title=f"{self._display_method_name()} | Avg across layers",
                output_path=os.path.join(heatmap_dir, avg_filename),
            )
        print(f"Gate heatmaps saved under {heatmap_dir}")

    def _display_method_name(self) -> str:
        aliases = {
            "MED_standard": "MED",
            "MED_LCDS_domain-wise": "MED-LCDS",
            "mole_standard": "mole",
        }
        return aliases.get(self.method, self.method)

    def _display_layer_name(self, layer_name: str) -> str:
        if "_" in layer_name:
            suffix = layer_name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return suffix
        return layer_name

    def _ordered_domains(self, layer_to_domain: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
        observed = set()
        for domain_map in layer_to_domain.values():
            observed.update(domain_map.keys())
        if len(observed) == 0:
            return []
        ordered = []
        if self.domain_name_by_index:
            for idx in sorted(self.domain_name_by_index.keys()):
                domain = str(self.domain_name_by_index[idx])
                if domain in observed:
                    ordered.append(domain)
        for domain in sorted(observed):
            if domain not in ordered:
                ordered.append(domain)
        return ordered

    def _infer_num_experts(self, layer_to_domain: Dict[str, Dict[str, np.ndarray]]) -> int:
        max_dim = 0
        for domain_map in layer_to_domain.values():
            for values in domain_map.values():
                max_dim = max(max_dim, int(values.shape[0]))
        return max_dim

    def _infer_expert_labels(self, num_experts: int) -> List[str]:
        mapped_domains = []
        if self.domain_name_by_index:
            mapped_domains = [str(self.domain_name_by_index[idx]) for idx in sorted(self.domain_name_by_index.keys())]
        if len(mapped_domains) == num_experts:
            return mapped_domains
        if len(mapped_domains) + 1 == num_experts:
            return mapped_domains + ["base_clip"]
        return [f"expert_{i}" for i in range(num_experts)]

    def _postprocess_values(self, values: torch.Tensor, source: str) -> torch.Tensor:
        out = values.detach().float()
        out = self._ensure_probability_vector(out, source=source)
        out = self._maybe_move_base_to_last(out, source=source)
        return out

    def _ensure_probability_vector(self, values: torch.Tensor, source: str) -> torch.Tensor:
        # Defensive normalization: some older call paths logged raw domain-head logits.
        if values.ndim != 1 or not source.startswith("logit_scalar"):
            return values
        if self._looks_like_probability_vector(values):
            return values
        return torch.softmax(values, dim=-1)

    def _looks_like_probability_vector(self, values: torch.Tensor) -> bool:
        if values.ndim != 1 or values.numel() == 0:
            return False
        if bool(torch.any(values < -1e-6)) or bool(torch.any(values > 1.0 + 1e-6)):
            return False
        total = float(values.sum().item())
        return abs(total - 1.0) <= 1e-3

    def _maybe_move_base_to_last(self, values: torch.Tensor, source: str) -> torch.Tensor:
        # Most current loaders append base CLIP as the last expert; preserve that layout unless observed data strongly matches the legacy "base-first" ordering.
        if values.ndim != 1:
            return values
        rotate_first_to_last = False
        if self._layout_by_source is not None:
            rotate_first_to_last = bool(self._layout_by_source.get(source, False))
        if not rotate_first_to_last:
            return values
        if int(values.shape[0]) < 2:
            return values
        return torch.cat([values[1:], values[:1]], dim=0)

    def _infer_layout_by_source(self) -> Dict[str, bool]:
        """
        Infer whether each source uses legacy expert ordering (base at index 0).
        Returns: source -> rotate_first_to_last
        """
        out: Dict[str, bool] = {}
        if not self.domain_name_by_index or len(self._sum) == 0:
            return out
        mapped_domains = [str(self.domain_name_by_index[idx]) for idx in sorted(self.domain_name_by_index.keys())]
        num_domains = len(mapped_domains)
        if num_domains == 0:
            return out
        domain_to_pos = {d: i for i, d in enumerate(mapped_domains)}

        score_last: Dict[str, float] = defaultdict(float)
        score_first: Dict[str, float] = defaultdict(float)
        used: Dict[str, int] = defaultdict(int)

        for key, summed in self._sum.items():
            source, _layer, domain = key
            count = self._count.get(key, 0)
            if count <= 0:
                continue
            pos = domain_to_pos.get(str(domain))
            if pos is None:
                continue
            vec = (summed / float(count)).detach().float()
            if vec.ndim != 1:
                continue
            num_experts = int(vec.shape[0])
            if num_experts != num_domains + 1:
                continue
            # Base-last layout: domain i maps to expert i.
            score_last[source] += float(vec[pos].item())
            # Base-first legacy layout: domain i maps to expert (i+1).
            score_first[source] += float(vec[pos + 1].item())
            used[source] += 1

        for source, n in used.items():
            if n <= 0:
                continue
            out[source] = score_first[source] > score_last[source]
        return out

    def _layer_sort_key(self, layer_name: str):
        if "_" in layer_name:
            suffix = layer_name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return (0, int(suffix), layer_name)
        return (1, 0, layer_name)

    def _safe_name(self, value: str) -> str:
        out = str(value)
        keep = []
        for ch in out:
            if ch.isalnum() or ch in ("-", "_", "."):
                keep.append(ch)
            else:
                keep.append("_")
        safe = "".join(keep).strip("_")
        return safe or "value"

    def _save_heatmap(
        self,
        matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
        output_path: str,
    ):
        fig_w = max(8.0, 0.7 * max(1, len(col_labels)))
        fig_h = max(6.0, 0.5 * max(1, len(row_labels)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Expert / domain j")
        ax.set_ylabel("Inference domain i")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean gate activation")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def _make_gate_hook(self, source: str, layer_name: str, joint: bool):
        def hook_fn(module, inputs, output):
            if self._current_domains is None:
                return
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            activations = self._reduce_gate_values(output[0], source=source, joint=joint)
            if activations is None or activations.shape[0] != len(self._current_domains):
                return
            self._accumulate(
                source=source,
                layer=layer_name,
                domain_list=self._current_domains,
                values=activations.detach().float().cpu(),
            )
        return hook_fn

    def _reduce_gate_values(self, values: torch.Tensor, source: str, joint: bool):
        if values is None:
            return None
        if values.ndim == 2:
            return values
        if values.ndim != 3:
            return None
        # Token-wise gate values: (B, L, E) -> (B, E)
        # Maple-MED2 joint values: (B_img, B_txt+1, E), image token is last.
        if joint and source == "gate_image" and values.shape[1] > 1:
            return values[:, -1, :]
        return values.mean(dim=1)

    def _resolve_batch_domains(self, domains, current_domain: str | None, batch_size: int | None):
        if domains is not None:
            if isinstance(domains, torch.Tensor):
                domains = domains.detach().cpu().tolist()
            elif isinstance(domains, np.ndarray):
                domains = domains.tolist()
            return [self._map_domain_value(d) for d in domains]
        if current_domain is not None and batch_size is not None:
            return [str(current_domain)] * int(batch_size)
        return None

    def _map_domain_value(self, value):
        if isinstance(value, str):
            return value
        try:
            ivalue = int(value)
        except Exception:
            return str(value)
        if ivalue in self.domain_name_by_index:
            return self.domain_name_by_index[ivalue]
        if (ivalue - 1) in self.domain_name_by_index:
            return self.domain_name_by_index[ivalue - 1]
        return str(ivalue)

    def _accumulate(self, source: str, layer: str, domain_list: List[str], values: torch.Tensor):
        for idx, domain in enumerate(domain_list):
            key = (source, layer, domain)
            row = values[idx]
            if key not in self._sum:
                self._sum[key] = torch.zeros_like(row)
            self._sum[key] += row
            self._count[key] += 1


DEFAULT_SEMANTIC_EQUIVALENCE_GROUPS = [
    ("pasture", "Pasture Land"),
    ("highway", "Highway or Road"),
    ("Basketball", "basketball_court", "outdoor basketball_court"),
    ("Bowling", "bowling_alley"),
    ("cliff", "Cliff_Diving"),
]
DEFAULT_SEMANTIC_EQUIVALENCE_EXCLUDED = {
    "lobster",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "strawberry",
    "strawberry_shortcake",
    "car_side",
    "carside",
}
DEFAULT_SEMANTIC_DIRECTIONAL_RULES = [
    # GT pizza_tossing predicted as pizza is acceptable (one-way only).
    ("Pizza_Tossing", "pizza"),
]


def discover_model_paths(base_dir, shots, seed, domain_order=None):
    """Discover LoRA weight paths for each expert domain."""
    domain_order = domain_order or DOMAIN_ORDER
    model_name2path = {}
    for expert_dir in domain_order:
        expert_path = os.path.join(base_dir, expert_dir)
        if not os.path.isdir(expert_path):
            continue
        weights_path = os.path.join(expert_path, f"{shots}shots", f"seed{seed}", "lora_weights.pt")
        if os.path.exists(weights_path):
            model_name2path[expert_dir] = weights_path
        else:
            print(f"Warning: weights not found for {weights_path}")
    return model_name2path


def discover_stage2_model_paths(meta_weight_path, shots, seed, domain_order=None):
    """Discover stage-2 LoRA experts saved under <meta_weight_path>/stage2_lora."""
    stage2_root = os.path.join(os.path.abspath(meta_weight_path), "stage2_lora")
    model_paths = discover_model_paths(stage2_root, shots, seed, domain_order=domain_order)
    if len(model_paths) == 0:
        raise ValueError(
            f"--mole_stage2 is enabled but no stage-2 LoRA weights were found under {stage2_root} "
            f"for shots={shots}, seed={seed}."
        )
    print(f"[INFO] Using stage-2 LoRA experts from {stage2_root}")
    return model_paths


def _make_benchmark_view(
    classnames: List[str],
    data_source,
    preprocess,
    return_domain: bool,
) -> SimpleNamespace:
    """Wrap raw Datum lists with DatasetWrapper for evaluation."""
    wrapper = ds_utils.DatasetWrapper(
        data_source=data_source,
        transform=preprocess,
        return_domain=return_domain,
        is_train=False,
        classnames=classnames,
    )
    return SimpleNamespace(classnames=classnames, test=wrapper)


def _build_classname_to_domain_map(benchmark_result) -> Dict[str, str]:
    """Build a classname -> domain mapping from benchmark metadata."""
    mapping: Dict[str, str] = {}
    per_domain_classnames = getattr(benchmark_result, "per_domain_classnames", None)
    if per_domain_classnames:
        for domain, classnames in per_domain_classnames.items():
            for classname in classnames:
                mapping[classname] = domain
        return mapping
    per_domain_datasets = getattr(benchmark_result, "per_domain_datasets", None)
    if per_domain_datasets:
        for domain, dataset in per_domain_datasets.items():
            for classname in dataset.classnames:
                mapping[classname] = domain
    return mapping


def _normalize_per_domain_accuracy(
    per_domain_acc: Dict,
    domain_map: Dict[int, str],
) -> Dict[str, float]:
    """Convert per-domain accuracy keys to readable domain names."""
    if not per_domain_acc:
        return {}
    normalized = {}
    for key, value in per_domain_acc.items():
        if isinstance(key, str):
            normalized[key] = value
        else:
            domain_name = domain_map.get(key)
            if domain_name is None:
                domain_name = str(key)
            normalized[domain_name] = value
    return normalized


def _parse_layer_list(raw_value, default):
    """Parse comma-separated layer indices into a list of ints."""
    if raw_value is None:
        return default
    if isinstance(raw_value, (list, tuple)):
        return [int(v) for v in raw_value]
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if cleaned == "":
            return default
        return [int(v) for v in cleaned.split(",") if v.strip() != ""]
    try:
        return [int(raw_value)]
    except Exception:
        return default


def _build_knots_merge_config(args):
    """Build KnOTS-TIES or KnOTS-DARE-TIES merge config."""
    merge_config = {
        "rep_type": "svd-vector",
        "merge_method": "ties",
        "merging_type": "mean",
        "topK": 100.0,
        "concat_across_output": True,
        "merge_other_params": True,
        "dare": bool(getattr(args, "knots_dare", False)),
        "scaling_coeffs": 1.0,
    }
    if merge_config["dare"]:
        merge_config["dare_pruning_coeffs"] = float(getattr(args, "knots_dare_pruning_coeffs", 0.5))
        merge_config["dare_seed"] = int(getattr(args, "knots_dare_seed", 0))
    return merge_config


def _resolve_checkpoint_path(
    meta_weight_path: str,
    preferred_name: str,
    latest_name: str | None = None,
    fallback_name: str | None = None,
) -> str:
    """Resolve checkpoint path with latest -> best -> legacy fallback order."""
    candidates = []
    if latest_name:
        candidates.append(os.path.join(meta_weight_path, latest_name))
    candidates.append(os.path.join(meta_weight_path, preferred_name))
    if fallback_name:
        candidates.append(os.path.join(meta_weight_path, fallback_name))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Expected checkpoint not found. Tried: {candidates}")


def _write_prompts_txt(prompt_output_path: str, prompt_rows: List[Tuple[str, str, str]]) -> None:
    """Write prompts used for classification to a text file."""
    with open(prompt_output_path, "w", encoding="utf-8") as f:
        f.write("domain\tclassname\tprompt\n")
        for domain, classname, prompt in prompt_rows:
            f.write(f"{domain}\t{classname}\t{prompt}\n")


def _uses_topk_subdir(meta_type: str) -> bool:
    """Selected MED-LCDS modes do not use top-k checkpoint subdirectories."""
    return False


def _resolve_meta_weight_path_with_topk(args) -> str:
    """Resolve effective meta weight path."""
    return os.path.abspath(args.meta_weight_path)


def _requires_meta_weight_path(meta_type: str) -> bool:
    return meta_type in (
        "single_lora",
        "single_lora_from_pretrained",
        "MED",
        "MED_LCDS",
        "mole",
        "phatgoose",
    )


def load_meta_weights(meta_type, meta_weight_path, num_experts, device):
    """Load meta network weights based on selected MED-LCDS meta_type."""
    if meta_type in (
        "base",
        "base_clip",
        "expert_lora",
        "lora_mean",
        "knots_svd",
        "single_lora",
        "single_lora_from_pretrained",
    ):
        return None, None

    meta_weight_path = os.path.abspath(meta_weight_path)
    if not os.path.isdir(meta_weight_path):
        raise ValueError(f"meta_weight_path must be a directory: {meta_weight_path}")

    if meta_type in ("MED", "MED_LCDS", "mole", "phatgoose"):
        weight_file = _resolve_checkpoint_path(
            meta_weight_path=meta_weight_path,
            preferred_name="meta_net.pt",
            latest_name="meta_net_latest.pt",
            fallback_name=f"{meta_type}.pt",
        )
        return weight_file, None

    raise ValueError(f"Unknown meta_type: {meta_type}")


def _infer_expert_count_from_weight_file(meta_type: str, weight_file: str) -> int | None:
    """Best-effort: infer expert count from selected gate checkpoints."""
    if meta_type not in ("MED", "MED_LCDS", "mole", "phatgoose"):
        return None
    try:
        sd = torch.load(weight_file, map_location="cpu")
    except Exception:
        return None
    if not isinstance(sd, dict) or len(sd) == 0:
        return None
    first_gate = next(iter(sd.values()))
    if isinstance(first_gate, dict) and "linear.weight" in first_gate:
        return int(first_gate["linear.weight"].shape[0])
    if isinstance(first_gate, dict) and "linear.linear.weight" in first_gate:
        return int(first_gate["linear.linear.weight"].shape[0])
    return None

def _create_eval_runner(
    meta_type: str,
    args,
    clip_model,
    tokenizer,
    preprocess_val,
    device: torch.device,
    domain_name_by_index: Dict[int, str] | None = None,
    domain_order: List[str] | None = None,
) -> Callable[[SimpleNamespace], Tuple[float, Dict]]:
    """Build an evaluator for the selected MED-LCDS modes."""
    domain_order = domain_order or DOMAIN_ORDER

    def discover_experts():
        model_paths = resolve_expert_model_paths(
            args.model_root,
            args.model_shots,
            args.seed,
            domain_order=domain_order,
            explicit_paths=getattr(args, "expert_weight_paths", None),
        )
        load_paths = list(model_paths.values())
        if len(load_paths) == 0:
            raise ValueError(f"No LoRA weights found under {args.model_root}")
        return model_paths, load_paths

    def make_gate_logger() -> GateDomainCSVLogger:
        logit_scalar_mode = getattr(args, "logit_scalar", "standard")
        return GateDomainCSVLogger(
            method=f"{meta_type}_{logit_scalar_mode}",
            domain_name_by_index=domain_name_by_index,
        )

    if meta_type == "base_clip":
        def runner(benchmark_dataset, current_domain=None):
            return evaluate_base(
                clip_model,
                tokenizer,
                benchmark_dataset,
                preprocess_val,
                args.batch_size,
                args.num_workers,
                device,
            )

        return runner

    if meta_type in ("single_lora", "single_lora_from_pretrained"):
        weight_file = os.path.abspath(args.meta_weight_path)
        if not os.path.isfile(weight_file):
            raise ValueError(
                f"For {meta_type}, --meta_weight_path must be a checkpoint file path: {weight_file}"
            )
        merged_lora_clip, list_lora_layers = loralib.load_lora_from(args.backbone, weight_file)
        merged_state_dict = loralib.save_lora_as_clip(
            merged_lora_clip,
            list_lora_layers,
            return_statedict_only=True,
        )
        model = copy.deepcopy(clip_model)
        model.load_state_dict(merged_state_dict)
        model = model.to(device)
        model.eval()

        def runner(benchmark_dataset, current_domain=None):
            return evaluate_base(
                model, tokenizer, benchmark_dataset, preprocess_val,
                args.batch_size, args.num_workers, device
            )

        return runner

    if meta_type == "knots_svd":
        _, load_paths = discover_experts()
        merge_config = _build_knots_merge_config(args)
        print(f"[INFO] Running KnOTS merge with config: {merge_config}")
        merged_model = build_knots_merged_model(
            base_model=clip_model,
            load_paths=load_paths,
            merge_config=merge_config,
            backbone=args.backbone,
            device=device,
            include_base_clip=args.include_base_clip,
        )

        def runner(benchmark_dataset, current_domain=None):
            return evaluate_base(
                merged_model,
                tokenizer,
                benchmark_dataset,
                preprocess_val,
                args.batch_size,
                args.num_workers,
                device,
            )

        return runner

    if meta_type == "lora_mean":
        _, load_paths = discover_experts()
        model = copy.deepcopy(clip_model)
        mixtures, num_experts = load_lora_mixtures_with_zero(
            clip_model=model,
            backbone=args.backbone,
            load_paths=load_paths,
            dropout_rate=0.0,
            include_base_clip=args.include_base_clip,
        )
        model = model.to(device)
        model.eval()

        def runner(benchmark_dataset, current_domain=None):
            return evaluate_lora_mean(
                model, mixtures, num_experts, tokenizer, benchmark_dataset, preprocess_val,
                args.batch_size, args.num_workers, device
            )

        return runner

    if meta_type == "expert_lora":
        model_paths, _ = discover_experts()
        model_by_domain_name = {}
        for domain in domain_order:
            if domain not in model_paths:
                continue
            expert_clip, list_lora_layers = loralib.load_lora_from(args.backbone, model_paths[domain])
            expert_clip_sd = loralib.save_lora_as_clip(
                expert_clip, list_lora_layers, return_statedict_only=True
            )
            expert_clip = copy.deepcopy(clip_model)
            expert_clip.load_state_dict(expert_clip_sd)
            expert_clip = expert_clip.to(device)
            expert_clip.eval()
            model_by_domain_name[domain] = expert_clip
        dname_by_idx = domain_name_by_index if domain_name_by_index is not None else {}

        def runner(benchmark_dataset, current_domain=None):
            return evaluate_expert_lora(
                model_by_domain_name,
                dname_by_idx,
                tokenizer,
                benchmark_dataset,
                preprocess_val,
                args.batch_size,
                args.num_workers,
                device,
                current_domain=current_domain,
            )

        return runner

    if meta_type in ("MED", "MED_LCDS", "mole", "phatgoose"):
        _, load_paths = discover_experts()
        model = copy.deepcopy(clip_model)
        gate_type = "phatgoose" if meta_type == "phatgoose" else "mole"
        include_base_clip = False if meta_type == "phatgoose" else args.include_base_clip
        list_lora_mixtures, num_experts = load_mole_mixtures_with_zero(
            clip_model=model,
            backbone=args.backbone,
            load_paths=load_paths,
            dropout_rate=0.0,
            include_base_clip=include_base_clip,
            gate_type=gate_type,
            simplify_mole=getattr(args, "simplify_mole", False),
        )
        model = model.to(device)
        model.eval()

        weight_file, _ = load_meta_weights(meta_type, args.meta_weight_path, num_experts, device)
        expected = _infer_expert_count_from_weight_file(meta_type, weight_file)
        if expected is not None and expected != num_experts:
            suggestion = "--no-include_base_clip" if args.include_base_clip else "--include_base_clip"
            raise ValueError(
                f"{meta_type} gate weights expect num_expert={expected}, but mixtures were built with "
                f"num_expert={num_experts}. Try running with {suggestion}."
            )

        class _LoadArgs:
            pass

        largs = _LoadArgs()
        largs.params = ["q", "k", "v", "o"]
        largs.gate_filename = weight_file.replace(".pt", "")
        load_gate_nets(largs, list_lora_mixtures)
        feature_dim = infer_clip_feature_dim(model)
        logit_scalar_module = load_logit_scalar(
            meta_weight_path=args.meta_weight_path,
            mode=getattr(args, "logit_scalar", "standard"),
            feature_dim=feature_dim,
            num_domains=len(domain_order),
            device=device,
        )
        gate_logger = make_gate_logger()
        gate_logger.register_mixture_layers(list_lora_mixtures)

        def runner(benchmark_dataset, current_domain=None):
            return evaluate_mole(
                model, tokenizer, benchmark_dataset, preprocess_val,
                args.batch_size, args.num_workers, device,
                logit_scalar_module=logit_scalar_module,
                logit_scalar_mode=getattr(args, "logit_scalar", "standard"),
                gate_logger=gate_logger,
                current_domain=current_domain,
            )

        runner.gate_logger = gate_logger
        return runner

    raise ValueError(f"Unknown meta_type: {meta_type}")

def main():
    parser = argparse.ArgumentParser(description="Unified meta benchmark evaluation")
    
    # Required args
    parser.add_argument("--subsample", type=str, required=True, choices=["all", "base", "new"],
                        help="Subsample mode for datasets")
    parser.add_argument("--firstk_cnt", type=int, default=5,
                        help="Number of classes per domain to include in benchmark")
    parser.add_argument("--benchmark_mode", type=str, default="cross_domain",
                        choices=["firstk", "randomk", "in_domain", "cross_domain", "target_domain_randomk"],
                        help="Benchmark construction strategy")
    parser.add_argument("--meta_type", type=str, required=True,
                        choices=[
                            "base_clip",
                            "single_lora",
                            "single_lora_from_pretrained",
                            "lora_mean",
                            "knots_svd",
                            "expert_lora",
                            "MED",
                            "MED_LCDS",
                            "mole",
                            "phatgoose",
                        ],
                        help="Meta type to evaluate")
    parser.add_argument("--meta_weight_path", type=str, default=None,
                        help="Directory containing meta network weights, or a single checkpoint file for single_lora modes.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for randomk benchmark mode")
    parser.add_argument(
        "--class_filter_path",
        type=str,
        default=None,
        help="Optional JSON class filter with {'domains': {domain: [classnames...]}}.",
    )
    parser.add_argument(
        "--meta_config_path",
        type=str,
        default=None,
        help="Optional JSON config with ordered domains, exact expert LoRA paths, and optional meta_weight_path.",
    )
    
    # Additional args with defaults
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--model_root", type=str, default="./output-rank2/base_split/vitb16",
                        help="Root directory containing expert LoRA weights")
    parser.add_argument("--shots", type=int, default=16,
                        help="Number of shots (benchmark/training setting).")
    parser.add_argument(
        "--model_shots",
        type=int,
        default=None,
        help="Shots value used only to locate pretrained LoRA expert paths (defaults to --shots).",
    )
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed (for finding expert paths)")
    parser.add_argument("--backbone", type=str, default="ViT-B/16",
                        help="CLIP backbone model")
    parser.add_argument(
        "--include_base_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include base CLIP as the zero expert when constructing mixtures/MoE.",
    )
    parser.add_argument(
        "--logit_scalar",
        type=str,
        default="standard",
        choices=["standard", "domain-wise", "input-wise", "affine-wise", "img-wise"],
        help="Logit scaling mode for MED-family methods.",
    )
    parser.add_argument(
        "--simplify_mole",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use summary-token gating for MED-family methods during evaluation (image CLS, text EOS).",
    )
    parser.add_argument(
        "--mole_stage2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For MED-family evaluation, load LoRA experts from stage-2 outputs under --meta_weight_path/stage2_lora.",
    )
    parser.add_argument(
        "--knots_dare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DARE pruning in KnOTS.",
    )
    parser.add_argument("--knots_dare_pruning_coeffs", type=float, default=0.5,
                        help="DARE pruning coefficient for KnOTS.")
    parser.add_argument("--knots_dare_seed", type=int, default=0,
                        help="DARE seed for KnOTS.")
    parser.add_argument("--device", **device_arg())
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save CSV results (default: current directory)")
    parser.add_argument(
        "--dump_wrong_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dump wrongly classified images with GT and top-k overlays.",
    )
    parser.add_argument(
        "--dump_wrong_images_topk",
        type=int,
        default=5,
        help="Top-k predictions to render on wrongly classified image overlays.",
    )
    parser.add_argument(
        "--enable_semantic_equivalence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally treat selected semantically-equivalent classnames as "
            "correct during evaluation (default profile includes pasture/Pasture Land)."
        ),
    )
    parser.add_argument(
        "--include_overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable overlap-aware cross-domain benchmark construction and scoring. "
            "Direct equivalents are canonicalized, semantic equivalents receive "
            "partial credit, and generic-to-fine generic images are removed while "
            "their text labels remain as distractors."
        ),
    )
    parser.add_argument(
        "--overlap_merge_mode",
        type=str,
        default=OVERLAP_MERGE_NONE,
        choices=sorted(OVERLAP_MERGE_MODES),
        help=(
            "Prediction-time merge for --include_overlap. "
            "'sum_generic_to_fine' adds each generic overlap label probability "
            "to all related fine labels before argmax, e.g. Caltech car_side "
            "probability is added to every Stanford Cars label."
        ),
    )
    parser.add_argument(
        "--overlap_groups_path",
        type=str,
        default=None,
        help="Optional path to overlap group metadata JSON (defaults to evaluation/overlap_groups.json).",
    )
    parser.add_argument(
        "--dump_eval_details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dump per-sample prediction/logit/routing details to CSV when supported.",
    )
    parser.add_argument(
        "--include_new_ds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include opt-in new benchmark datasets. CUB200 and RESISC45 are appended; "
            "AIBD Cars replaces Stanford Cars as the inference dataset when --use_aibd_cars is set."
        ),
    )
    parser.add_argument(
        "--use_aibd_cars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --include_new_ds is set, evaluate on AIBD Cars instead of Stanford Cars. "
            "Disable to keep Stanford Cars and skip AIBD Cars."
        ),
    )
    args = parser.parse_args()
    meta_config = load_meta_config(args.meta_config_path)
    args.domain_order = domains_from_meta_config(meta_config, DOMAIN_ORDER)
    args.expert_weight_paths = expert_weights_from_meta_config(meta_config)
    if args.meta_weight_path is None:
        config_meta_path = meta_config.get("meta_weight_path") or meta_config.get("meta_network")
        if config_meta_path:
            args.meta_weight_path = str(config_meta_path)
    if args.meta_type == "MED_LCDS":
        args.logit_scalar = "domain-wise"
    elif args.meta_type in ("MED", "mole"):
        args.logit_scalar = "standard"
    if _requires_meta_weight_path(args.meta_type) and args.meta_weight_path is None:
        parser.error("--meta_weight_path is required")
    if args.model_shots is None:
        args.model_shots = args.shots
    if args.meta_weight_path is not None:
        args.meta_weight_path = _resolve_meta_weight_path_with_topk(args)
    if _uses_topk_subdir(args.meta_type):
        print(f"Using top-k meta weight path: {args.meta_weight_path}")
    
    device = resolve_device(args.device)
    args.device = str(device)
    if meta_config:
        active_domain_order = list(args.domain_order)
        active_expert_domain_order = list(args.domain_order)
    else:
        active_domain_order = build_active_domain_order(
            include_new_ds=bool(args.include_new_ds),
            use_aibd_cars=bool(args.use_aibd_cars),
        )
        active_expert_domain_order = build_expert_domain_order(
            include_new_ds=bool(args.include_new_ds),
        )
    
    print(
        f"Building {args.benchmark_mode} benchmark: "
        f"subsample={args.subsample}, firstk_cnt={args.firstk_cnt}"
    )
    if bool(args.include_new_ds):
        if bool(args.use_aibd_cars):
            print(
                "New dataset inclusion is enabled: using AIBD Cars in place of Stanford Cars, "
                "and adding CUB200 and RESISC45."
            )
        else:
            print(
                "New dataset inclusion is enabled: keeping Stanford Cars, "
                "adding CUB200 and RESISC45 (AIBD Cars skipped)."
            )
        print(f"Expert domain order uses Stanford Cars for the car expert: {active_expert_domain_order}")
    if args.overlap_merge_mode != OVERLAP_MERGE_NONE and not bool(args.include_overlap):
        raise ValueError("--overlap_merge_mode requires --include_overlap.")
    benchmark_result = build_benchmark_dataset(
        benchmark_mode=args.benchmark_mode,
        domain_order=active_domain_order,
        dataset_map=DATASET_MAP,
        data_root=args.data_root,
        subsample=args.subsample,
        class_count=args.firstk_cnt,
        random_seed=args.random_seed,
        enable_semantic_equivalence=bool(args.enable_semantic_equivalence),
        class_filter_path=args.class_filter_path,
        include_overlap=bool(args.include_overlap),
        overlap_groups_path=args.overlap_groups_path,
    )
    if benchmark_result.combined_dataset is not None:
        print(f"Benchmark dataset created with {len(benchmark_result.combined_dataset.classnames)} classes")
    else:
        print(f"Constructed in-domain benchmark across {len(benchmark_result.per_domain_datasets)} domains")
    if benchmark_result.classnames_log_path:
        print(f"Random benchmark classnames saved to {benchmark_result.classnames_log_path}")

    # Load base CLIP model
    print("Loading base CLIP model...")
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B/16", pretrained="openai", force_quick_gelu=True, device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    eval_runner = _create_eval_runner(
        args.meta_type,
        args,
        clip_model,
        tokenizer,
        preprocess_val,
        device,
        domain_name_by_index=benchmark_result.domain_name_by_index,
        domain_order=active_expert_domain_order,
    )
    gate_logger = getattr(eval_runner, "gate_logger", None)

    prompt_filename_parts = [
        "prompts",
        args.benchmark_mode,
        args.subsample,
        f"k{args.firstk_cnt}",
    ]
    if args.benchmark_mode == "randomk":
        prompt_filename_parts.append(f"seed{args.random_seed}")
    prompt_filename = "_".join(prompt_filename_parts) + ".txt"
    os.makedirs(args.output_dir, exist_ok=True)
    if bool(args.include_overlap) and getattr(benchmark_result, "overlap_resolved", None) is not None:
        overlap_resolution_path = write_overlap_resolution(
            args.output_dir,
            benchmark_result.overlap_resolved,
        )
        print(f"Overlap resolution saved to {overlap_resolution_path}")
    prompt_output_path = os.path.join(args.output_dir, prompt_filename)
    wrong_dir = os.path.join(
        args.output_dir,
        f"wrongly_classified_{args.subsample}",
        args.meta_type,
    )
    if args.dump_wrong_images:
        os.makedirs(wrong_dir, exist_ok=True)
        print(f"Wrongly classified images will be saved to {wrong_dir}")

    if benchmark_result.combined_dataset is not None:
        raw_classnames = benchmark_result.combined_dataset.classnames
        benchmark_view = _make_benchmark_view(
            raw_classnames,
            benchmark_result.combined_dataset.test,
            preprocess_val,
            return_domain=True,
        )
        # Apply per-domain prompt templates to text inputs
        selected_domain_targets = benchmark_result.selected_domain_targets
        domain_name_by_index = benchmark_result.domain_name_by_index
        if selected_domain_targets:
            domain_by_class_index = [
                domain_name_by_index.get(domain_idx)
                for domain_idx in selected_domain_targets
            ]
        else:
            domain_by_classname = _build_classname_to_domain_map(benchmark_result)
            domain_by_class_index = [
                domain_by_classname.get(cname) for cname in raw_classnames
            ]
        benchmark_view.raw_classnames = list(raw_classnames)
        benchmark_view.class_domains = list(domain_by_class_index)
        benchmark_view.domain_name_by_index = dict(benchmark_result.domain_name_by_index or {})
        benchmark_view.current_domain = None
        benchmark_view.dump_wrong_images = bool(args.dump_wrong_images)
        benchmark_view.dump_wrong_images_dir = wrong_dir
        benchmark_view.dump_wrong_images_topk = max(1, int(args.dump_wrong_images_topk))
        benchmark_view.overlap_resolved = getattr(benchmark_result, "overlap_resolved", None)
        benchmark_view.overlap_merge_mode = args.overlap_merge_mode
        benchmark_view.dump_eval_details = bool(args.dump_eval_details) or bool(args.include_overlap)
        benchmark_view.eval_method = args.meta_type
        benchmark_view.per_sample_output_path = (
            os.path.join(args.output_dir, f"{args.meta_type}_per_sample.csv")
            if benchmark_view.dump_eval_details
            else None
        )
        benchmark_view.semantic_equivalence_lookup = {}
        benchmark_view.semantic_directional_lookup = {}
        if bool(args.enable_semantic_equivalence) and not bool(args.include_overlap):
            benchmark_view.semantic_equivalence_lookup = build_semantic_equivalence_lookup(
                groups=DEFAULT_SEMANTIC_EQUIVALENCE_GROUPS,
                excluded_classnames=DEFAULT_SEMANTIC_EQUIVALENCE_EXCLUDED,
            )
            benchmark_view.semantic_directional_lookup = build_semantic_directional_lookup(
                rules=DEFAULT_SEMANTIC_DIRECTIONAL_RULES,
                excluded_classnames=DEFAULT_SEMANTIC_EQUIVALENCE_EXCLUDED,
            )
        prompts = [
            create_domain_prompt(cname, domain_by_class_index[i])
            for i, cname in enumerate(raw_classnames)
        ]
        benchmark_view.classnames = prompts
        prompt_rows = [
            (domain_by_class_index[i] or "unknown", cname, prompt)
            for i, (cname, prompt) in enumerate(zip(raw_classnames, prompts))
        ]
        _write_prompts_txt(prompt_output_path, prompt_rows)
        print(f"Prompt list saved to {prompt_output_path}")

        # DEBUG: Print template, full prompt, and classname for each prompt
        classnames = raw_classnames
        
        print(f"\n[DEBUG] Total prompts/classnames: {len(classnames)}")
        print(f"[DEBUG] Showing first 10 examples:")
        print(f"{'Classname':<30} {'Domain':<20} {'Template':<50} {'Full Prompt (if applied)':<50}")
        print("-" * 150)
        
        for i in range(min(10, len(classnames))):
            classname = classnames[i]
            domain = domain_by_class_index[i] or "unknown"
            
            # Build template string (matching create_domain_prompt logic)
            template_str = build_domain_template(domain)
            
            # Get full prompt (applied during inference)
            full_prompt = prompts[i]
            
            print(f"{classname:<30} {domain:<20} {template_str:<50} {full_prompt:<50}")
        
        if len(classnames) > 10:
            print(f"\n[DEBUG] ... and {len(classnames) - 10} more classnames")
            print(f"[DEBUG] Last 5 examples:")
            for i in range(max(10, len(classnames) - 5), len(classnames)):
                classname = classnames[i]
                domain = domain_by_class_index[i] or "unknown"
                full_prompt = prompts[i]
                template_str = build_domain_template(domain)
                print(f"{classname:<30} {domain:<20} {template_str:<50} {full_prompt:<50}")
        
        overall_acc, per_domain_acc = eval_runner(benchmark_view)
        overlap_metrics = getattr(benchmark_view, "overlap_metrics", {})
        per_domain_acc = _normalize_per_domain_accuracy(
            per_domain_acc, benchmark_result.domain_name_by_index
        )
    else:
        per_domain_acc = {}
        total_correct = 0.0
        total_samples = 0
        prompt_rows = []
        for domain, dataset in benchmark_result.per_domain_datasets.items():
            if dataset.test is None or len(dataset.test) == 0:
                continue
            raw_classnames = dataset.classnames
            benchmark_view = _make_benchmark_view(
                raw_classnames,
                dataset.test,
                preprocess_val,
                return_domain=False,
            )
            # Apply per-domain prompt templates to text inputs
            prompts = [create_domain_prompt(cname, domain) for cname in raw_classnames]
            benchmark_view.raw_classnames = list(raw_classnames)
            benchmark_view.class_domains = [domain for _ in raw_classnames]
            benchmark_view.domain_name_by_index = dict(benchmark_result.domain_name_by_index or {})
            benchmark_view.current_domain = domain
            benchmark_view.dump_wrong_images = bool(args.dump_wrong_images)
            benchmark_view.dump_wrong_images_dir = wrong_dir
            benchmark_view.dump_wrong_images_topk = max(1, int(args.dump_wrong_images_topk))
            benchmark_view.overlap_resolved = None
            benchmark_view.overlap_merge_mode = OVERLAP_MERGE_NONE
            benchmark_view.dump_eval_details = bool(args.dump_eval_details)
            benchmark_view.eval_method = args.meta_type
            benchmark_view.per_sample_output_path = None
            benchmark_view.semantic_equivalence_lookup = {}
            benchmark_view.semantic_directional_lookup = {}
            if bool(args.enable_semantic_equivalence) and not bool(args.include_overlap):
                benchmark_view.semantic_equivalence_lookup = build_semantic_equivalence_lookup(
                    groups=DEFAULT_SEMANTIC_EQUIVALENCE_GROUPS,
                    excluded_classnames=DEFAULT_SEMANTIC_EQUIVALENCE_EXCLUDED,
                )
                benchmark_view.semantic_directional_lookup = build_semantic_directional_lookup(
                    rules=DEFAULT_SEMANTIC_DIRECTIONAL_RULES,
                    excluded_classnames=DEFAULT_SEMANTIC_EQUIVALENCE_EXCLUDED,
                )
            benchmark_view.classnames = prompts
            prompt_rows.extend(
                (domain, cname, prompt)
                for cname, prompt in zip(raw_classnames, prompts)
            )
            # DEBUG: Print template, full prompt, and classname for this domain
            classnames = raw_classnames
            print(f"\n[DEBUG] Domain '{domain}': {len(classnames)} prompts")
            
            # Build template string (matching create_domain_prompt logic)
            template_str = build_domain_template(domain)
            
            print(f"{'Classname':<30} {'Template':<50} {'Full Prompt (if applied)':<50}")
            print("-" * 130)
            
            num_examples = min(10, len(classnames))
            for i in range(num_examples):
                classname = classnames[i]
                full_prompt = prompts[i]
                print(f"{classname:<30} {template_str:<50} {full_prompt:<50}")
            
            if len(classnames) > num_examples:
                print(f"\n... and {len(classnames) - num_examples} more classnames")
            
            domain_acc, _ = eval_runner(benchmark_view, current_domain=domain)
            per_domain_acc[domain] = domain_acc
            domain_samples = len(dataset.test)
            total_correct += domain_acc * domain_samples
            total_samples += domain_samples
        overlap_metrics = {}
        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
        if prompt_rows:
            _write_prompts_txt(prompt_output_path, prompt_rows)
            print(f"Prompt list saved to {prompt_output_path}")
    
    # Print results
    print(f"\nOverall accuracy: {overall_acc:.4f}")
    for domain_name in active_domain_order:
        if domain_name in per_domain_acc:
            print(f"{domain_name}: {per_domain_acc[domain_name]:.4f}")
    extra_domains = [d for d in per_domain_acc.keys() if d not in active_domain_order]
    for domain_name in extra_domains:
        print(f"{domain_name}: {per_domain_acc[domain_name]:.4f}")
    
    # Save to CSV
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    csv_filename = os.path.join(args.output_dir, f"{args.meta_type}.csv")
    
    # Prepare data row
    row_data = {
        'meta_type': args.meta_type,
        'benchmark_mode': args.benchmark_mode,
        'subsample': args.subsample,
        'firstk_cnt': args.firstk_cnt,
        'include_new_ds': bool(args.include_new_ds),
        'use_aibd_cars': bool(args.use_aibd_cars),
        'include_overlap': bool(args.include_overlap),
        'overlap_merge_mode': args.overlap_merge_mode,
        'mean_accuracy': overall_acc,
    }
    if overlap_metrics:
        row_data.update(overlap_metrics)
    if args.benchmark_mode == "randomk":
        row_data['random_seed'] = args.random_seed
        if benchmark_result.classnames_log_path:
            row_data['classnames_log_path'] = benchmark_result.classnames_log_path
    
    # Calculate mean of per-domain accuracies
    domain_columns = list(active_domain_order)
    for domain in per_domain_acc.keys():
        if domain not in domain_columns:
            domain_columns.append(domain)
    domain_accuracies = [per_domain_acc.get(domain, np.nan) for domain in domain_columns]
    row_data['mean_domain_accuracy'] = (
        np.nanmean(domain_accuracies) if len(domain_accuracies) > 0 else np.nan
    )
    
    # Add per-domain accuracy columns
    for domain in domain_columns:
        row_data[domain] = per_domain_acc.get(domain, np.nan)
    
    df_new = pd.DataFrame([row_data])
    
    # Append to CSV or create new file
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    if gate_logger is not None:
        gate_csv = os.path.join(args.output_dir, f"{args.meta_type}_gate.csv")
        gate_logger.finalize_to_csv(gate_csv)
        gate_logger.finalize_heatmaps(args.output_dir)
        gate_logger.close()


if __name__ == "__main__":
    main()
