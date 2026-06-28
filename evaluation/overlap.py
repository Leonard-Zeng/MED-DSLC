from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import clip_lora_datasets.utils as ds_utils
from .eval_utils import normalize_classname


OVERLAP_DIRECT = "direct_equivalent"
OVERLAP_SEMANTIC = "semantic_equivalent"
OVERLAP_GENERIC_TO_FINE = "generic_to_fine"
OVERLAP_TYPES = {OVERLAP_DIRECT, OVERLAP_SEMANTIC, OVERLAP_GENERIC_TO_FINE}
OVERLAP_MERGE_NONE = "none"
OVERLAP_MERGE_SUM_GENERIC_TO_FINE = "sum_generic_to_fine"
OVERLAP_MERGE_MODES = {OVERLAP_MERGE_NONE, OVERLAP_MERGE_SUM_GENERIC_TO_FINE}


@dataclass(frozen=True)
class ClassRef:
    domain: str
    classname: str

    @classmethod
    def from_payload(cls, payload) -> "ClassRef":
        if isinstance(payload, Mapping):
            return cls(domain=str(payload["domain"]), classname=str(payload["classname"]))
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            return cls(domain=str(payload[0]), classname=str(payload[1]))
        raise ValueError(f"Invalid class reference: {payload!r}")

    def key(self) -> Tuple[str, str]:
        return (self.domain, self.classname)

    def normalized_classname(self) -> str:
        return normalize_classname(self.classname)

    def to_dict(self) -> dict:
        return {"domain": self.domain, "classname": self.classname}


@dataclass
class OverlapGroup:
    group_id: str
    type: str
    members: List[ClassRef] = field(default_factory=list)
    generic_members: List[ClassRef] = field(default_factory=list)
    fine_members: List[ClassRef] = field(default_factory=list)
    canonical: ClassRef | None = None
    partial_credit: float = 0.5
    directional: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping) -> "OverlapGroup":
        group_type = str(payload["type"])
        if group_type not in OVERLAP_TYPES:
            raise ValueError(f"Unsupported overlap group type: {group_type}")

        canonical_payload = payload.get("canonical")
        if canonical_payload is None and "canonical_domain" in payload and "canonical_classname" in payload:
            canonical_payload = {
                "domain": payload["canonical_domain"],
                "classname": payload["canonical_classname"],
            }

        return cls(
            group_id=str(payload["group_id"]),
            type=group_type,
            members=[ClassRef.from_payload(x) for x in payload.get("members", [])],
            generic_members=[ClassRef.from_payload(x) for x in payload.get("generic_members", [])],
            fine_members=[ClassRef.from_payload(x) for x in payload.get("fine_members", [])],
            canonical=ClassRef.from_payload(canonical_payload) if canonical_payload else None,
            partial_credit=float(payload.get("partial_credit", 0.5)),
            directional=bool(payload.get("directional", False)),
        )

    def to_dict(self) -> dict:
        out = {
            "group_id": self.group_id,
            "type": self.type,
            "partial_credit": self.partial_credit,
        }
        if self.canonical is not None:
            out["canonical"] = self.canonical.to_dict()
        if self.members:
            out["members"] = [x.to_dict() for x in self.members]
        if self.generic_members:
            out["generic_members"] = [x.to_dict() for x in self.generic_members]
        if self.fine_members:
            out["fine_members"] = [x.to_dict() for x in self.fine_members]
        if self.directional:
            out["directional"] = True
        return out


@dataclass
class OverlapResolved:
    groups: List[OverlapGroup]
    label_to_group: Dict[int, str] = field(default_factory=dict)
    label_to_overlap_type: Dict[int, str] = field(default_factory=dict)
    semantic_members_by_label: Dict[int, set[int]] = field(default_factory=dict)
    generic_labels_by_fine_label: Dict[int, set[int]] = field(default_factory=dict)
    direct_member_indices_by_label: Dict[int, List[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "label_to_group": {str(k): v for k, v in self.label_to_group.items()},
            "label_to_overlap_type": {str(k): v for k, v in self.label_to_overlap_type.items()},
            "semantic_members_by_label": {
                str(k): sorted(v) for k, v in self.semantic_members_by_label.items()
            },
            "generic_labels_by_fine_label": {
                str(k): sorted(v) for k, v in self.generic_labels_by_fine_label.items()
            },
            "direct_member_indices_by_label": {
                str(k): list(v) for k, v in self.direct_member_indices_by_label.items()
            },
        }


def default_overlap_groups_path() -> str:
    return os.path.join(os.path.dirname(__file__), "overlap_groups.json")


def load_overlap_groups(path: str | None = None) -> List[OverlapGroup]:
    path = path or default_overlap_groups_path()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    groups_payload = payload.get("groups", payload)
    if not isinstance(groups_payload, list):
        raise ValueError(f"Overlap groups must be a list or {{'groups': [...]}}: {path}")
    return [OverlapGroup.from_payload(x) for x in groups_payload]


def write_overlap_resolution(output_dir: str, resolved: OverlapResolved) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "overlap_resolved.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resolved.to_dict(), f, indent=2)
    return path


def _available_class_refs(domain_datasets: Mapping[str, ds_utils.DatasetBase]) -> List[ClassRef]:
    refs = []
    for domain, dataset in domain_datasets.items():
        for classname in dataset.classnames:
            refs.append(ClassRef(domain=str(domain), classname=str(classname)))
    return refs


def _class_ref_exists(ref: ClassRef, domain_datasets: Mapping[str, ds_utils.DatasetBase]) -> bool:
    dataset = domain_datasets.get(ref.domain)
    return dataset is not None and ref.classname in set(dataset.classnames)


def _domain_priority(domain: str) -> int:
    # Prefer more task-specific domains for canonical labels.
    priorities = {
        "stanford_cars": 0,
        "aibd_cars": 0,
        "fgvc": 0,
        "food101": 1,
        "oxford_flowers": 1,
        "ucf101": 1,
        "caltech101": 5,
    }
    return priorities.get(domain, 3)


def _choose_canonical(refs: Sequence[ClassRef], requested: ClassRef | None = None) -> ClassRef:
    if requested is not None:
        for ref in refs:
            if ref.key() == requested.key():
                return ref
    return sorted(refs, key=lambda r: (_domain_priority(r.domain), r.domain, r.classname))[0]


def _auto_direct_groups(domain_datasets: Mapping[str, ds_utils.DatasetBase]) -> List[OverlapGroup]:
    by_norm: Dict[str, List[ClassRef]] = defaultdict(list)
    for ref in _available_class_refs(domain_datasets):
        by_norm[ref.normalized_classname()].append(ref)

    groups = []
    for norm_name, refs in sorted(by_norm.items()):
        domains = {ref.domain for ref in refs}
        if len(refs) < 2 or len(domains) < 2:
            continue
        canonical = _choose_canonical(refs)
        groups.append(
            OverlapGroup(
                group_id=f"direct_auto_{norm_name.replace(' ', '_')}",
                type=OVERLAP_DIRECT,
                members=list(refs),
                canonical=canonical,
                partial_credit=1.0,
            )
        )
    return groups


def _merge_direct_groups(
    configured: Sequence[OverlapGroup],
    auto_groups: Sequence[OverlapGroup],
    domain_datasets: Mapping[str, ds_utils.DatasetBase],
) -> List[OverlapGroup]:
    explicit_keys = set()
    out = []
    for group in configured:
        if group.type == OVERLAP_DIRECT:
            members = [
                ref for ref in group.members
                if _class_ref_exists(ref, domain_datasets)
            ]
            if len(members) < 2:
                continue
            canonical = _choose_canonical(members, group.canonical)
            out.append(
                OverlapGroup(
                    group_id=group.group_id,
                    type=group.type,
                    members=members,
                    canonical=canonical,
                    partial_credit=1.0,
                )
            )
            explicit_keys.update(ref.key() for ref in members)
        else:
            out.append(group)

    for group in auto_groups:
        if any(ref.key() in explicit_keys for ref in group.members):
            continue
        out.append(group)
    return out


def _dataset_domain(dataset: ds_utils.DatasetBase) -> str:
    dataset_dir = getattr(dataset, "dataset_dir", "")
    basename = os.path.basename(str(dataset_dir))
    return ds_utils.dir_name2domain_name.get(basename, basename)


def _relabel_item(item, label: int, classname: str | None = None) -> ds_utils.Datum:
    return ds_utils.Datum(
        impath=item.impath,
        label=int(label),
        domain=item.domain,
        classname=classname if classname is not None else item.classname,
        dataset_name=getattr(item, "dataset_name", ""),
        real_classname=getattr(item, "real_classname", "") or item.classname,
        real_dataset_name=getattr(item, "real_dataset_name", "") or getattr(item, "dataset_name", ""),
        bboxes=getattr(item, "bboxes", None),
        scores=getattr(item, "scores", None),
    )


def _item_key(item) -> Tuple[str, str]:
    domain = getattr(item, "dataset_name", "") or getattr(item, "real_dataset_name", "")
    if domain == "other":
        domain = getattr(item, "real_dataset_name", "") or domain
    return (str(domain), str(item.classname))


def _copy_dataset_with_splits(
    template_dataset: ds_utils.DatasetBase,
    train_x,
    train_u,
    val,
    test,
    classnames: Sequence[str],
) -> ds_utils.DatasetBase:
    dataset = ds_utils.DatasetBase(
        train_x=train_x,
        train_u=train_u,
        val=val,
        test=test,
        remap_labels=False,
    )
    dataset._num_classes = len(classnames)
    dataset._lab2cname = {idx: cname for idx, cname in enumerate(classnames)}
    dataset._classnames = list(classnames)
    return dataset


def build_overlap_benchmark(
    domain_datasets: "OrderedDict[str, ds_utils.DatasetBase]",
    configured_groups: Sequence[OverlapGroup],
) -> tuple[ds_utils.DatasetBase, List[int], List[str], List[OverlapGroup], OverlapResolved]:
    """Build an overlap-aware combined benchmark dataset.

    Direct equivalents are canonicalized at construction time. Generic-to-fine
    generic images are removed from test/val, while their text labels remain as
    candidate prompts.
    """
    combined = ds_utils.CombinedDataset(list(domain_datasets.values()))
    domain_index_by_name = {domain: idx for idx, domain in enumerate(domain_datasets.keys())}

    groups = _merge_direct_groups(
        configured=configured_groups,
        auto_groups=_auto_direct_groups(domain_datasets),
        domain_datasets=domain_datasets,
    )

    direct_map: Dict[Tuple[str, str], ClassRef] = {}
    direct_members_by_canonical: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    semantic_groups = []
    generic_to_fine_groups = []
    generic_keys = set()
    for group in groups:
        if group.type == OVERLAP_DIRECT:
            if group.canonical is None:
                continue
            for ref in group.members:
                direct_map[ref.key()] = group.canonical
                direct_members_by_canonical[group.canonical.key()].append(ref.key())
        elif group.type == OVERLAP_SEMANTIC:
            semantic_groups.append(group)
        elif group.type == OVERLAP_GENERIC_TO_FINE:
            generic_to_fine_groups.append(group)
            generic_keys.update(ref.key() for ref in group.generic_members)

    selected_keys = []
    seen = set()
    per_domain_classnames: Dict[str, List[str]] = OrderedDict((d, []) for d in domain_datasets.keys())
    for domain, dataset in domain_datasets.items():
        for classname in dataset.classnames:
            key = (domain, classname)
            canonical_ref = direct_map.get(key, ClassRef(domain=domain, classname=classname))
            canonical_key = canonical_ref.key()
            if canonical_key not in seen:
                seen.add(canonical_key)
                selected_keys.append(canonical_key)
                per_domain_classnames.setdefault(canonical_ref.domain, []).append(canonical_ref.classname)

    key_to_label = {key: idx for idx, key in enumerate(selected_keys)}
    label_classnames = [classname for _domain, classname in selected_keys]
    selected_domain_targets = [
        domain_index_by_name.get(domain, 0) for domain, _classname in selected_keys
    ]

    def _filter_split(split, *, drop_generic: bool) -> list[ds_utils.Datum] | None:
        if split is None:
            return None
        out = []
        for item in split:
            source_key = _item_key(item)
            if drop_generic and source_key in generic_keys:
                continue
            canonical_ref = direct_map.get(source_key, ClassRef(*source_key))
            canonical_key = canonical_ref.key()
            if canonical_key not in key_to_label:
                continue
            out.append(
                _relabel_item(
                    item,
                    label=key_to_label[canonical_key],
                    classname=canonical_ref.classname,
                )
            )
        return out

    train_x = _filter_split(combined.train_x, drop_generic=False)
    train_u = _filter_split(combined.train_u, drop_generic=False)
    val = _filter_split(combined.val, drop_generic=True)
    test = _filter_split(combined.test, drop_generic=True)
    overlap_dataset = _copy_dataset_with_splits(
        template_dataset=combined,
        train_x=train_x,
        train_u=train_u,
        val=val,
        test=test,
        classnames=label_classnames,
    )

    resolved = OverlapResolved(groups=groups)
    for key, label in key_to_label.items():
        member_keys = direct_members_by_canonical.get(key, [])
        if member_keys:
            resolved.direct_member_indices_by_label[label] = [
                key_to_label[direct_map.get(member_key, ClassRef(*member_key)).key()]
                for member_key in member_keys
                if direct_map.get(member_key, ClassRef(*member_key)).key() in key_to_label
            ]

    key_to_group = {}
    key_to_type = {}
    for group in groups:
        refs = list(group.members) + list(group.generic_members) + list(group.fine_members)
        for ref in refs:
            canonical_key = direct_map.get(ref.key(), ref).key()
            if canonical_key in key_to_label:
                label = key_to_label[canonical_key]
                resolved.label_to_group[label] = group.group_id
                resolved.label_to_overlap_type[label] = group.type
            key_to_group[ref.key()] = group.group_id
            key_to_type[ref.key()] = group.type

    for group in semantic_groups:
        member_labels = []
        for ref in group.members:
            key = direct_map.get(ref.key(), ref).key()
            if key in key_to_label:
                member_labels.append(key_to_label[key])
        if group.directional and member_labels:
            # Convention: first member is GT/source, remaining members are allowed predictions.
            resolved.semantic_members_by_label[member_labels[0]] = set(member_labels[1:])
        else:
            for label in member_labels:
                resolved.semantic_members_by_label[label] = set(member_labels)

    for group in generic_to_fine_groups:
        generic_labels = set()
        for ref in group.generic_members:
            key = direct_map.get(ref.key(), ref).key()
            if key in key_to_label:
                generic_labels.add(key_to_label[key])
        fine_labels = set()
        if group.fine_members:
            for ref in group.fine_members:
                key = direct_map.get(ref.key(), ref).key()
                if key in key_to_label:
                    fine_labels.add(key_to_label[key])
        else:
            # Empty fine_members means all non-generic labels in this group's target domains.
            target_domains = {
                ref.domain for ref in group.members
                if ref.domain not in {g.domain for g in group.generic_members}
            }
            for key, label in key_to_label.items():
                if key[0] in target_domains and key not in generic_keys:
                    fine_labels.add(label)
        for label in fine_labels:
            resolved.generic_labels_by_fine_label[label] = set(generic_labels)
            if label not in resolved.label_to_group:
                resolved.label_to_group[label] = group.group_id
                resolved.label_to_overlap_type[label] = group.type
        for label in generic_labels:
            resolved.label_to_group[label] = group.group_id
            resolved.label_to_overlap_type[label] = group.type

    return overlap_dataset, selected_domain_targets, label_classnames, groups, resolved


def _domain_name(domain_idx, domain_name_by_index: Mapping[int, str] | None, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback)
    if domain_name_by_index is None:
        return str(domain_idx)
    try:
        idx = int(domain_idx)
    except Exception:
        return str(domain_idx)
    return str(domain_name_by_index.get(idx, domain_name_by_index.get(idx - 1, idx)))


def _safe_class(raw_classnames: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(raw_classnames):
        return str(raw_classnames[idx])
    return f"class_{idx}"


def _to_numpy_domains(domains):
    if domains is None:
        return None
    if isinstance(domains, torch.Tensor):
        return domains.detach().cpu().numpy()
    return np.asarray(domains)


def compute_overlap_scores(
    preds: torch.Tensor,
    labels: torch.Tensor,
    benchmark_dataset,
) -> torch.Tensor:
    resolved: OverlapResolved | None = getattr(benchmark_dataset, "overlap_resolved", None)
    if resolved is None:
        return preds.eq(labels).float()

    preds_cpu = preds.detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    scores = []
    for pred, label in zip(preds_cpu, labels_cpu):
        pred = int(pred)
        label = int(label)
        if pred == label:
            scores.append(1.0)
            continue
        semantic_members = resolved.semantic_members_by_label.get(label, set())
        if pred in semantic_members:
            scores.append(0.5)
            continue
        generic_labels = resolved.generic_labels_by_fine_label.get(label, set())
        if pred in generic_labels:
            scores.append(0.5)
            continue
        scores.append(0.0)
    return torch.tensor(scores, device=labels.device, dtype=torch.float32)


def apply_overlap_prediction_merge(
    scores: torch.Tensor,
    benchmark_dataset,
    *,
    scores_are_probabilities: bool = False,
) -> torch.Tensor:
    """Optionally merge generic overlap probability mass into related fine labels."""
    merge_mode = getattr(benchmark_dataset, "overlap_merge_mode", OVERLAP_MERGE_NONE)
    if merge_mode in (None, "", False):
        merge_mode = OVERLAP_MERGE_NONE
    if merge_mode == OVERLAP_MERGE_NONE:
        return scores
    if merge_mode not in OVERLAP_MERGE_MODES:
        raise ValueError(
            f"Unsupported overlap merge mode: {merge_mode!r}. "
            f"Expected one of {sorted(OVERLAP_MERGE_MODES)}."
        )

    resolved: OverlapResolved | None = getattr(benchmark_dataset, "overlap_resolved", None)
    if resolved is None or not resolved.generic_labels_by_fine_label:
        return scores

    if merge_mode == OVERLAP_MERGE_SUM_GENERIC_TO_FINE:
        probs = scores if scores_are_probabilities else torch.softmax(scores, dim=-1)
        merged = probs.clone()
        num_labels = probs.shape[-1]
        generic_labels_to_suppress = set()
        for fine_label, generic_labels in resolved.generic_labels_by_fine_label.items():
            if fine_label < 0 or fine_label >= num_labels:
                continue
            valid_generic = [
                int(label) for label in generic_labels
                if 0 <= int(label) < num_labels
            ]
            if not valid_generic:
                continue
            generic_labels_to_suppress.update(valid_generic)
            generic_mass = probs[:, valid_generic].sum(dim=-1)
            merged[:, int(fine_label)] = merged[:, int(fine_label)] + generic_mass
        if generic_labels_to_suppress:
            merged[:, sorted(generic_labels_to_suppress)] = 0.0
        return merged

    return scores


class PerSampleOverlapLogger:
    """Append per-sample prediction, score, and routing rows to CSV."""

    def __init__(self, output_path: str, benchmark_dataset, method: str):
        self.output_path = output_path
        self.benchmark_dataset = benchmark_dataset
        self.method = method
        self._file = None
        self._writer = None
        self.rows_written = 0

    def __enter__(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "method",
            "sample_index",
            "image_path",
            "source_domain",
            "gt_domain",
            "gt_class",
            "pred_domain",
            "pred_class",
            "strict_correct",
            "overlap_score",
            "group_id",
            "overlap_type",
            "generic_fallback",
            "gt_logit",
            "pred_logit",
            "generic_logit",
            "fine_logit",
            "image_gate_top_expert",
            "image_gate_top_value",
            "text_gate_top_expert",
            "text_gate_top_value",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def log_batch(
        self,
        *,
        logits: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        domains=None,
        batch_start_idx: int,
        image_gate_mean_logits: torch.Tensor | None = None,
        text_gate_mean_logits: torch.Tensor | None = None,
        gate_expert_names: Sequence[str] | None = None,
    ):
        if self._writer is None:
            return

        raw_classnames = list(getattr(self.benchmark_dataset, "raw_classnames", []) or [])
        class_domains = list(getattr(self.benchmark_dataset, "class_domains", []) or [])
        domain_name_by_index = getattr(self.benchmark_dataset, "domain_name_by_index", {}) or {}
        current_domain = getattr(self.benchmark_dataset, "current_domain", None)
        resolved: OverlapResolved | None = getattr(self.benchmark_dataset, "overlap_resolved", None)
        data_source = getattr(getattr(self.benchmark_dataset, "test", None), "data_source", None)

        logits_cpu = logits.detach().cpu()
        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()
        scores_cpu = compute_overlap_scores(preds, labels, self.benchmark_dataset).detach().cpu()
        domains_np = _to_numpy_domains(domains)
        image_gate_cpu = image_gate_mean_logits.detach().cpu() if image_gate_mean_logits is not None else None
        text_gate_cpu = text_gate_mean_logits.detach().cpu() if text_gate_mean_logits is not None else None
        expert_names = list(gate_expert_names) if gate_expert_names is not None else []

        for i in range(logits_cpu.shape[0]):
            sample_idx = batch_start_idx + i
            pred_idx = int(preds_cpu[i].item())
            label_idx = int(labels_cpu[i].item())
            pred_domain = class_domains[pred_idx] if 0 <= pred_idx < len(class_domains) else "unknown"
            gt_domain = class_domains[label_idx] if 0 <= label_idx < len(class_domains) else None
            if gt_domain is None:
                sample_domain = domains_np[i] if domains_np is not None and i < len(domains_np) else None
                gt_domain = _domain_name(sample_domain, domain_name_by_index, current_domain)
            item = data_source[sample_idx] if data_source is not None and sample_idx < len(data_source) else None
            source_domain = getattr(item, "real_dataset_name", "") or getattr(item, "dataset_name", "") or gt_domain
            image_path = getattr(item, "impath", "") if item is not None else ""

            group_id = ""
            overlap_type = ""
            generic_fallback = False
            if resolved is not None:
                group_id = resolved.label_to_group.get(label_idx, resolved.label_to_group.get(pred_idx, ""))
                overlap_type = resolved.label_to_overlap_type.get(
                    label_idx,
                    resolved.label_to_overlap_type.get(pred_idx, ""),
                )
                generic_fallback = pred_idx in resolved.generic_labels_by_fine_label.get(label_idx, set())

            generic_logit = ""
            if resolved is not None:
                generic_labels = sorted(resolved.generic_labels_by_fine_label.get(label_idx, set()))
                if generic_labels:
                    generic_logit = float(logits_cpu[i, generic_labels].max().item())

            image_gate_top_expert = ""
            image_gate_top_value = ""
            if image_gate_cpu is not None and i < image_gate_cpu.shape[0]:
                idx = int(image_gate_cpu[i].argmax().item())
                image_gate_top_expert = expert_names[idx] if idx < len(expert_names) else f"expert_{idx}"
                image_gate_top_value = float(image_gate_cpu[i, idx].item())

            text_gate_top_expert = ""
            text_gate_top_value = ""
            if text_gate_cpu is not None and label_idx < text_gate_cpu.shape[0]:
                idx = int(text_gate_cpu[label_idx].argmax().item())
                text_gate_top_expert = expert_names[idx] if idx < len(expert_names) else f"expert_{idx}"
                text_gate_top_value = float(text_gate_cpu[label_idx, idx].item())

            self._writer.writerow(
                {
                    "method": self.method,
                    "sample_index": sample_idx,
                    "image_path": image_path,
                    "source_domain": source_domain,
                    "gt_domain": gt_domain,
                    "gt_class": _safe_class(raw_classnames, label_idx),
                    "pred_domain": pred_domain,
                    "pred_class": _safe_class(raw_classnames, pred_idx),
                    "strict_correct": int(pred_idx == label_idx),
                    "overlap_score": float(scores_cpu[i].item()),
                    "group_id": group_id,
                    "overlap_type": overlap_type,
                    "generic_fallback": int(generic_fallback),
                    "gt_logit": float(logits_cpu[i, label_idx].item()),
                    "pred_logit": float(logits_cpu[i, pred_idx].item()),
                    "generic_logit": generic_logit,
                    "fine_logit": float(logits_cpu[i, label_idx].item()),
                    "image_gate_top_expert": image_gate_top_expert,
                    "image_gate_top_value": image_gate_top_value,
                    "text_gate_top_expert": text_gate_top_expert,
                    "text_gate_top_value": text_gate_top_value,
                }
            )
            self.rows_written += 1
