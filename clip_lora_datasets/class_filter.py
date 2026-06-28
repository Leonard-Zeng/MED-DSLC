"""Utilities for filtering dataset classes by domain."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from typing import Dict, Iterable, Mapping

from .utils import Datum, DatasetBase


def load_class_filter(path: str | None) -> Dict[str, list[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    domains = payload.get("domains", payload)
    if not isinstance(domains, dict):
        raise ValueError(f"class filter must contain a domain->classnames mapping: {path}")
    out: Dict[str, list[str]] = {}
    for domain, classnames in domains.items():
        if classnames is None:
            continue
        if not isinstance(classnames, list) or not all(isinstance(c, str) for c in classnames):
            raise ValueError(f"class filter for domain {domain!r} must be a list of strings")
        out[str(domain)] = list(classnames)
    return out


def filter_dataset_to_classnames(
    dataset: DatasetBase,
    selected_classnames: Iterable[str],
    *,
    domain: str | None = None,
    strict: bool = True,
) -> DatasetBase:
    selected = list(selected_classnames)
    if not selected:
        raise ValueError(f"class filter for {domain or 'dataset'} is empty")

    available = list(dataset.classnames)
    missing = [c for c in selected if c not in available]
    if missing and strict:
        raise ValueError(
            f"class filter for {domain or 'dataset'} contains unavailable classes: {missing[:10]}"
        )
    selected = [c for c in selected if c in available]
    label_by_class = {c: idx for idx, c in enumerate(selected)}

    def _filter_split(split):
        if split is None:
            return None
        filtered = []
        for item in split:
            if item.classname not in label_by_class:
                continue
            filtered.append(
                Datum(
                    impath=item.impath,
                    label=label_by_class[item.classname],
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

    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset._train_x = _filter_split(dataset.train_x)
    filtered_dataset._train_u = _filter_split(dataset.train_u)
    filtered_dataset._val = _filter_split(dataset.val)
    filtered_dataset._test = _filter_split(dataset.test)
    filtered_dataset._num_classes = len(selected)
    filtered_dataset._lab2cname = {idx: cname for idx, cname in enumerate(selected)}
    filtered_dataset._classnames = selected
    return filtered_dataset


def apply_class_filter_to_domain_datasets(
    domain_datasets: Mapping[str, DatasetBase],
    class_filter: Mapping[str, list[str]] | None,
    *,
    strict: bool = True,
) -> "OrderedDict[str, DatasetBase]":
    filtered = OrderedDict()
    class_filter = class_filter or {}
    for domain, dataset in domain_datasets.items():
        selected = class_filter.get(domain)
        if selected is None:
            filtered[domain] = dataset
        else:
            filtered[domain] = filter_dataset_to_classnames(
                dataset,
                selected,
                domain=domain,
                strict=strict,
            )
    return filtered
