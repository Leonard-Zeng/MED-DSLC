"""Shared evaluation helpers."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set

import torch


def normalize_classname(name: str) -> str:
    """Normalize class strings for robust semantic-equivalence checks."""
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_semantic_equivalence_lookup(
    groups: Iterable[Sequence[str]] | None,
    excluded_classnames: Iterable[str] | None = None,
) -> Dict[str, str]:
    """Build normalized classname -> group-id lookup from equivalence groups."""
    excluded: Set[str] = set()
    if excluded_classnames:
        excluded = {normalize_classname(name) for name in excluded_classnames}

    lookup: Dict[str, str] = {}
    if not groups:
        return lookup

    for group in groups:
        normalized_group: List[str] = []
        for name in group:
            normalized_name = normalize_classname(name)
            if not normalized_name or normalized_name in excluded:
                continue
            normalized_group.append(normalized_name)
        normalized_group = sorted(set(normalized_group))
        if len(normalized_group) < 2:
            continue
        group_id = "|".join(normalized_group)
        for normalized_name in normalized_group:
            lookup[normalized_name] = group_id
    return lookup


def build_semantic_directional_lookup(
    rules: Iterable[Sequence[str]] | None,
    excluded_classnames: Iterable[str] | None = None,
) -> Dict[str, Set[str]]:
    """Build directional GT->allowed-pred lookup from (gt, pred1, pred2, ...) tuples."""
    excluded: Set[str] = set()
    if excluded_classnames:
        excluded = {normalize_classname(name) for name in excluded_classnames}

    lookup: Dict[str, Set[str]] = {}
    if not rules:
        return lookup

    for rule in rules:
        if not rule or len(rule) < 2:
            continue
        gt_name = normalize_classname(rule[0])
        if not gt_name or gt_name in excluded:
            continue
        allowed = {
            normalize_classname(name)
            for name in rule[1:]
            if normalize_classname(name) and normalize_classname(name) not in excluded
        }
        if not allowed:
            continue
        if gt_name not in lookup:
            lookup[gt_name] = set()
        lookup[gt_name].update(allowed)
    return lookup


def are_class_indices_semantically_equivalent(
    pred_idx: int,
    label_idx: int,
    classnames: Sequence[str] | None,
    semantic_lookup: Dict[str, str] | None,
) -> bool:
    """Check if predicted/GT class indices belong to the same semantic group."""
    if pred_idx == label_idx:
        return True
    if not classnames or not semantic_lookup:
        return False
    if pred_idx < 0 or label_idx < 0:
        return False
    if pred_idx >= len(classnames) or label_idx >= len(classnames):
        return False
    pred_name = normalize_classname(classnames[pred_idx])
    label_name = normalize_classname(classnames[label_idx])
    return (
        pred_name in semantic_lookup
        and label_name in semantic_lookup
        and semantic_lookup[pred_name] == semantic_lookup[label_name]
    )


def is_directionally_semantic_match(
    pred_idx: int,
    label_idx: int,
    classnames: Sequence[str] | None,
    directional_lookup: Dict[str, Set[str]] | None,
) -> bool:
    """Check if pred is allowed for GT according to directional semantic rules."""
    if not classnames or not directional_lookup:
        return False
    if pred_idx < 0 or label_idx < 0:
        return False
    if pred_idx >= len(classnames) or label_idx >= len(classnames):
        return False
    pred_name = normalize_classname(classnames[pred_idx])
    label_name = normalize_classname(classnames[label_idx])
    return pred_name in directional_lookup.get(label_name, set())


def compute_correct_mask(
    preds: torch.Tensor,
    labels: torch.Tensor,
    benchmark_dataset,
) -> torch.Tensor:
    """Return per-sample correctness mask with optional semantic equivalence."""
    mask = preds.eq(labels)
    semantic_lookup = getattr(benchmark_dataset, "semantic_equivalence_lookup", None)
    directional_lookup = getattr(benchmark_dataset, "semantic_directional_lookup", None)
    if not semantic_lookup and not directional_lookup:
        return mask

    raw_classnames = list(getattr(benchmark_dataset, "raw_classnames", []) or [])
    if not raw_classnames:
        raw_classnames = list(getattr(benchmark_dataset, "classnames", []) or [])
    if not raw_classnames:
        return mask

    mask_cpu = mask.detach().cpu().clone()
    preds_cpu = preds.detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    for idx, is_correct in enumerate(mask_cpu.tolist()):
        if is_correct:
            continue
        if are_class_indices_semantically_equivalent(
            pred_idx=int(preds_cpu[idx]),
            label_idx=int(labels_cpu[idx]),
            classnames=raw_classnames,
            semantic_lookup=semantic_lookup,
        ) or is_directionally_semantic_match(
            pred_idx=int(preds_cpu[idx]),
            label_idx=int(labels_cpu[idx]),
            classnames=raw_classnames,
            directional_lookup=directional_lookup,
        ):
            mask_cpu[idx] = True
    return mask_cpu.to(mask.device)
