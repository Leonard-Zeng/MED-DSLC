import json
import os
import random
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import clip_lora_datasets.utils as ds_utils
from clip_lora_datasets.class_filter import (
    apply_class_filter_to_domain_datasets,
    load_class_filter,
)
try:
    from .overlap import (
        build_overlap_benchmark,
        load_overlap_groups,
    )
except ImportError:
    from evaluation.overlap import (
        build_overlap_benchmark,
        load_overlap_groups,
    )

CALTECH101_CROSS_DOMAIN_EXCLUDE = {
    "water_lilly",
    "lotus",
    "pagoda",
    "pizza",
    "sunflower",
    "airplane",
    "barrel",
    "ice_cream",
    "car_side",
    "carside",
    "lobster",
    "helicopter",
    "soccer_ball"
}

# When semantic equivalence is enabled, exclude these Caltech101 classes (and their images)
# so that "car_side" / "carside" never appear in the benchmark.
CALTECH101_CARSIDE_EXCLUDE = {"car_side", "carside"}

SEMANTIC_EQUIVALENCE_EXCLUDE_BY_DOMAIN = {
    "caltech101": set(CALTECH101_CROSS_DOMAIN_EXCLUDE) | CALTECH101_CARSIDE_EXCLUDE,
}


@dataclass
class BenchmarkBuildResult:
    mode: str
    combined_dataset: Optional[ds_utils.DatasetBase] = None
    per_domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]" = field(default_factory=OrderedDict)
    per_domain_classnames: Dict[str, List[str]] = field(default_factory=dict)
    aggregated_classnames: List[str] = field(default_factory=list)
    selected_domain_targets: Optional[List[int]] = None
    classnames_log_path: Optional[str] = None
    domain_name_by_index: Dict[int, str] = field(default_factory=dict)
    overlap_groups: list = field(default_factory=list)
    overlap_resolved: object = None


def _load_domain_datasets(
    domain_order: List[str],
    dataset_map: Dict[str, Type[ds_utils.DatasetBase]],
    data_root: str,
    subsample: str,
) -> "OrderedDict[str, ds_utils.DatasetBase]":
    """Instantiate datasets for each domain and keep them ordered."""
    datasets = OrderedDict()
    for domain in domain_order:
        DatasetCls = dataset_map.get(domain)
        if DatasetCls is None:
            continue
        datasets[domain] = DatasetCls(root=data_root, num_shots=-1, subsample=subsample)
    if len(datasets) == 0:
        raise ValueError("No valid datasets found for the provided domain order.")
    return datasets


def _create_combined_dataset(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
    selected_classnames: List,
) -> Tuple[ds_utils.DatasetBase, List[int]]:
    """Use CombinedDataset helper to build a filtered benchmark dataset."""
    combine_ds = ds_utils.CombinedDataset(list(domain_datasets.values()))
    final_ds, selected_domain_targets = combine_ds.create_cross_labels_from_selected_classnames(
        selected_classnames
    )
    return final_ds, selected_domain_targets


def _prepare_domain_index_map(domains: "OrderedDict[str, ds_utils.DatasetBase]") -> Dict[int, str]:
    return {idx: domain for idx, domain in enumerate(domains.keys())}


def _filter_dataset_excluded_classnames(
    dataset: ds_utils.DatasetBase,
    excluded_classnames: set[str],
) -> ds_utils.DatasetBase:
    selected_classnames = [
        cname for cname in list(dataset.classnames)
        if cname not in excluded_classnames
    ]
    if len(selected_classnames) == len(list(dataset.classnames)):
        return dataset

    relabeler = {cname: idx for idx, cname in enumerate(selected_classnames)}

    def _filter_split(split):
        if split is None:
            return None
        filtered = []
        for item in split:
            if item.classname not in relabeler:
                continue
            filtered.append(
                ds_utils.Datum(
                    impath=item.impath,
                    label=relabeler[item.classname],
                    domain=item.domain,
                    classname=item.classname,
                    dataset_name=getattr(item, "dataset_name", ""),
                    real_classname=getattr(item, "real_classname", ""),
                    real_dataset_name=getattr(item, "real_dataset_name", ""),
                    bboxes=getattr(item, "bboxes", None),
                    scores=getattr(item, "scores", None),
                )
            )
        return filtered

    train_x = _filter_split(dataset.train_x)
    train_u = _filter_split(dataset.train_u)
    val = _filter_split(dataset.val)
    test = _filter_split(dataset.test)

    # Preserve original dataset metadata (e.g., dataset_dir) expected by CombinedDataset.
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset._train_x = train_x
    filtered_dataset._train_u = train_u
    filtered_dataset._val = val
    filtered_dataset._test = test
    filtered_dataset._num_classes = len(selected_classnames)
    filtered_dataset._lab2cname = {idx: cname for idx, cname in enumerate(selected_classnames)}
    filtered_dataset._classnames = list(selected_classnames)
    return filtered_dataset


def _apply_semantic_equivalence_exclusions(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
) -> "OrderedDict[str, ds_utils.DatasetBase]":
    filtered_datasets = OrderedDict()
    for domain, dataset in domain_datasets.items():
        excluded = SEMANTIC_EQUIVALENCE_EXCLUDE_BY_DOMAIN.get(domain, set())
        if excluded:
            filtered_datasets[domain] = _filter_dataset_excluded_classnames(dataset, excluded)
        else:
            filtered_datasets[domain] = dataset
    return filtered_datasets


def _assert_no_carside_when_semantic_equivalence(
    result: BenchmarkBuildResult,
    enable_semantic_equivalence: bool,
) -> None:
    """When semantic equivalence is enabled, ensure no car_side/carside from Caltech appears."""
    if not enable_semantic_equivalence:
        return
    for domain, dataset in result.per_domain_datasets.items():
        if domain != "caltech101":
            continue
        for c in CALTECH101_CARSIDE_EXCLUDE:
            if c in dataset.classnames:
                raise ValueError(
                    f"When --enable_semantic_equivalence is set, Caltech101 must not contain "
                    f"class '{c}'; found in dataset classnames."
                )
        if dataset.test:
            for d in dataset.test:
                if getattr(d, "classname", None) in CALTECH101_CARSIDE_EXCLUDE:
                    raise ValueError(
                        f"When --enable_semantic_equivalence is set, no Caltech101 test image "
                        f"may have classname '{d.classname}'."
                    )
    if result.combined_dataset is not None:
        for cname in result.combined_dataset.classnames:
            if cname in CALTECH101_CARSIDE_EXCLUDE:
                raise ValueError(
                    f"When --enable_semantic_equivalence is set, combined benchmark must not "
                    f"contain classname '{cname}'."
                )
        if result.combined_dataset.test:
            for d in result.combined_dataset.test:
                domain = getattr(d, "domain", None)
                if domain == "caltech101" and getattr(d, "classname", None) in CALTECH101_CARSIDE_EXCLUDE:
                    raise ValueError(
                        f"When --enable_semantic_equivalence is set, no Caltech101 test image "
                        f"may appear in combined dataset (found classname '{d.classname}')."
                    )


def build_firstk_benchmark(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
    class_count: int,
) -> BenchmarkBuildResult:
    if class_count is None or class_count <= 0:
        raise ValueError("--firstk_cnt must be a positive integer for firstk benchmark mode.")

    per_domain_classnames = {}
    selected_classnames = []
    for domain, dataset in domain_datasets.items():
        num_to_take = min(class_count, len(dataset.classnames))
        selected = dataset.classnames[:num_to_take]
        per_domain_classnames[domain] = selected
        selected_classnames.extend((domain, cname) for cname in selected)

    combined_dataset, selected_targets = _create_combined_dataset(domain_datasets, selected_classnames)
    return BenchmarkBuildResult(
        mode="firstk",
        combined_dataset=combined_dataset,
        per_domain_datasets=domain_datasets,
        per_domain_classnames=per_domain_classnames,
        aggregated_classnames=list(combined_dataset.classnames),
        selected_domain_targets=selected_targets,
    )


def build_randomk_benchmark(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
    class_count: int,
    random_seed: int,
) -> BenchmarkBuildResult:
    if class_count is None or class_count <= 0:
        raise ValueError("--firstk_cnt must be a positive integer for randomk benchmark mode.")

    rng = random.Random(random_seed)
    per_domain_classnames = {}
    selected_classnames = []

    for domain, dataset in domain_datasets.items():
        available = len(dataset.classnames)
        num_to_take = min(class_count, available)
        if num_to_take >= available:
            selected = list(dataset.classnames)
        else:
            sampled_indices = rng.sample(range(available), k=num_to_take)
            sampled_indices.sort()
            selected = [dataset.classnames[idx] for idx in sampled_indices]
        per_domain_classnames[domain] = selected
        selected_classnames.extend((domain, cname) for cname in selected)

    combined_dataset, selected_targets = _create_combined_dataset(domain_datasets, selected_classnames)

    log_filename = f"benchmark_classnames_randomk_seed{random_seed}_k{class_count}.json"
    log_path = os.path.join(os.path.dirname(__file__), log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": random_seed,
                "class_count": class_count,
                "domains": per_domain_classnames,
            },
            f,
            indent=2,
        )

    return BenchmarkBuildResult(
        mode="randomk",
        combined_dataset=combined_dataset,
        per_domain_datasets=domain_datasets,
        per_domain_classnames=per_domain_classnames,
        aggregated_classnames=list(combined_dataset.classnames),
        selected_domain_targets=selected_targets,
        classnames_log_path=log_path,
    )


def build_in_domain_benchmark(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
) -> BenchmarkBuildResult:
    return BenchmarkBuildResult(
        mode="in_domain",
        per_domain_datasets=domain_datasets,
        per_domain_classnames={domain: dataset.classnames for domain, dataset in domain_datasets.items()},
        aggregated_classnames=[],
    )


def build_cross_domain_benchmark(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
    enable_semantic_equivalence: bool = False,
    include_overlap: bool = False,
    overlap_groups_path: Optional[str] = None,
) -> BenchmarkBuildResult:
    if include_overlap:
        groups = load_overlap_groups(overlap_groups_path)
        combined_dataset, selected_targets, classnames, resolved_groups, resolved = build_overlap_benchmark(
            domain_datasets=domain_datasets,
            configured_groups=groups,
        )
        return BenchmarkBuildResult(
            mode="cross_domain",
            combined_dataset=combined_dataset,
            per_domain_datasets=domain_datasets,
            per_domain_classnames={
                domain: list(dataset.classnames)
                for domain, dataset in domain_datasets.items()
            },
            aggregated_classnames=list(classnames),
            selected_domain_targets=selected_targets,
            overlap_groups=list(resolved_groups),
            overlap_resolved=resolved,
        )

    per_domain_classnames = {}
    selected_classnames = []
    for domain, dataset in domain_datasets.items():
        classnames = list(dataset.classnames)
        per_domain_classnames[domain] = classnames
        selected_classnames.extend((domain, cname) for cname in classnames)

    combined_dataset, selected_targets = _create_combined_dataset(
        domain_datasets, selected_classnames
    )
    return BenchmarkBuildResult(
        mode="cross_domain",
        combined_dataset=combined_dataset,
        per_domain_datasets=domain_datasets,
        per_domain_classnames=per_domain_classnames,
        aggregated_classnames=list(combined_dataset.classnames),
        selected_domain_targets=selected_targets,
    )


def build_benchmark_dataset(
    benchmark_mode: str,
    domain_order: List[str],
    dataset_map: Dict[str, Type[ds_utils.DatasetBase]],
    data_root: str,
    subsample: str,
    class_count: Optional[int],
    random_seed: Optional[int] = None,
    enable_semantic_equivalence: bool = False,
    class_filter_path: Optional[str] = None,
    include_overlap: bool = False,
    overlap_groups_path: Optional[str] = None,
) -> BenchmarkBuildResult:
    """Dispatcher for building benchmark datasets across all supported modes."""
    domain_datasets = _load_domain_datasets(domain_order, dataset_map, data_root, subsample)
    if enable_semantic_equivalence and not include_overlap:
        domain_datasets = _apply_semantic_equivalence_exclusions(domain_datasets)
    class_filter = load_class_filter(class_filter_path)
    if class_filter:
        domain_datasets = apply_class_filter_to_domain_datasets(
            domain_datasets,
            class_filter,
            strict=True,
        )

    if benchmark_mode == "firstk":
        result = build_firstk_benchmark(domain_datasets, class_count)
    elif benchmark_mode == "randomk":
        if random_seed is None:
            raise ValueError("--random_seed must be provided for randomk benchmark mode.")
        result = build_randomk_benchmark(domain_datasets, class_count, random_seed)
    elif benchmark_mode == "in_domain":
        result = build_in_domain_benchmark(domain_datasets)
    elif benchmark_mode in ("cross_domain", "target_domain_randomk"):
        result = build_cross_domain_benchmark(
            domain_datasets,
            enable_semantic_equivalence=enable_semantic_equivalence,
            include_overlap=include_overlap,
            overlap_groups_path=overlap_groups_path,
        )
        result.mode = benchmark_mode
    else:
        raise ValueError(f"Unknown benchmark_mode: {benchmark_mode}")

    result.domain_name_by_index = _prepare_domain_index_map(domain_datasets)
    _assert_no_carside_when_semantic_equivalence(result, enable_semantic_equivalence and not include_overlap)
    return result
