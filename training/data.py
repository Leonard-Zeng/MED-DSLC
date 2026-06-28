import json
import os
import random
import itertools
from typing import Dict, List, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

import clip_lora_datasets.utils as ds_utils
from clip_lora_datasets.class_filter import (
    apply_class_filter_to_domain_datasets,
    load_class_filter,
)
from clip_lora_datasets.utils import DatasetWrapper

from .config import DOMAIN_ORDER, DATASET_MAP


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_domain_datasets(
    data_root: str,
    shots: int,
    subsample: str,
    domains: List[str] = None,
    class_filter_path: str = None,
) -> Dict[str, ds_utils.DatasetBase]:
    """Instantiate per-domain datasets with train/val/test splits."""
    domains = domains or DOMAIN_ORDER
    datasets = {}
    for name in domains:
        cls = DATASET_MAP.get(name)
        if cls is None:
            continue
        ds_train_only = cls(data_root, num_shots=shots, subsample=subsample)
        ds_full = cls(data_root, num_shots=-1, subsample=subsample)
        ds_train_only._val = ds_full.val
        ds_train_only._test = ds_full.test
        datasets[name] = ds_train_only
    if len(datasets) == 0:
        raise ValueError("No datasets constructed; check domain list and paths.")
    class_filter = load_class_filter(class_filter_path)
    if class_filter:
        datasets = apply_class_filter_to_domain_datasets(datasets, class_filter, strict=True)
    return datasets


class PromptedDataset(IterableDataset):
    """Iterates a base DatasetWrapper and injects tokenized prompts."""

    def __init__(
        self,
        base_dataset: DatasetWrapper,
        tokenizer,
        domain_idx: int,
        prompt_source: str,
        description_bank: Dict[str, List[str]],
        template: str,
        batch_size: int = None,
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.domain_idx = domain_idx
        self.prompt_source = prompt_source
        self.description_bank = description_bank
        self.template = template
        self.classnames = base_dataset.classnames
        self.domain_idx = domain_idx
        self.batch_size = batch_size
        # prompts used for random_class_tokens (sample uniformly across classes)
        self.prompts = self._build_prompt_list()
        # Track chosen description per class for current batch (reset periodically)
        self._current_batch_descriptions: Dict[str, str] = {}
        self._item_count = 0

    def _build_prompt_list(self) -> List[str]:
        prompts = []
        for cname in self.classnames:
            if self.prompt_source == "descriptions" and cname in self.description_bank:
                prompts.extend(self.description_bank[cname])
            elif self.prompt_source == "combined":
                # Include both template and descriptions
                prompts.append(self.template.format(cname))
                if cname in self.description_bank:
                    prompts.extend(self.description_bank[cname])
            else:
                prompts.append(self.template.format(cname))
        # if descriptions list empty for a class, fall back to template
        if not prompts:
            prompts = [self.template.format(c) for c in self.classnames]
        return prompts

    def _sample_prompt(self, classname: str) -> str:
        # Check if we've already chosen a description for this class in current batch
        if classname in self._current_batch_descriptions:
            return self._current_batch_descriptions[classname]
        
        # Choose a new description for this class
        if self.prompt_source == "descriptions":
            candidates = self.description_bank.get(classname)
            if candidates:
                chosen = random.choice(candidates)
            else:
                chosen = self.template.format(classname)
        elif self.prompt_source == "combined":
            # Combine template and descriptions, randomly choose from both
            candidates = [self.template.format(classname)]
            if classname in self.description_bank:
                candidates.extend(self.description_bank[classname])
            chosen = random.choice(candidates)
        else:
            chosen = self.template.format(classname)
        
        # Store for reuse within current batch
        self._current_batch_descriptions[classname] = chosen
        return chosen

    def get_current_batch_descriptions(self) -> Dict[str, str]:
        """Return the current batch descriptions dictionary for consistency with random_k sampling."""
        return self._current_batch_descriptions.copy()

    def __len__(self):
        return len(self.base_dataset)

    def __iter__(self):
        # Reset batch descriptions at the start of each iteration
        self._current_batch_descriptions = {}
        self._item_count = 0
        for img, label in self.base_dataset:
            # Reset descriptions at the start of each new batch to allow variation across batches
            if self.batch_size and self._item_count > 0 and self._item_count % self.batch_size == 0:
                self._current_batch_descriptions = {}
            classname = self.classnames[label]
            prompt = self._sample_prompt(classname)
            self._item_count += 1
            yield img, self.tokenizer(prompt).squeeze(), self.domain_idx


class InterleavedBatchBufferDataset(IterableDataset):
    """Interleave prompted per-domain datasets with optional round-robin balancing."""

    def __init__(self, datasets, buffer_size=4, seed=1, balanced=False):
        self.datasets = list(datasets)
        self.num_datasets = len(self.datasets)
        self.buffer_size = buffer_size
        self.balanced = balanced
        self.rng = random.Random(seed) if seed is not None else random
        self._build_prompt_bank()
        if self.balanced:
            self.dataset_iterators = [itertools.cycle(iter(ds)) for ds in self.datasets]
            self.max_samples_per_epoch = 1000 // max(1, self.num_datasets)

    def _build_prompt_bank(self):
        self.all_prompts = []
        self.domain_indices = []
        self.select_weights = []
        self.prompt_to_classname = []
        self.domain_to_prompt_ids = {}
        self.domain_to_prompt_weights = {}
        for ds in self.datasets:
            classnames = getattr(ds, "classnames", None)
            prompts = getattr(ds, "prompts", [])
            description_bank = getattr(ds, "description_bank", {})
            template = getattr(ds, "template", "a photo of a {}.")
            if not prompts:
                continue
            per_prompt_weight = 1 / len(prompts)
            for prompt in prompts:
                classname = None
                if classnames is not None:
                    for cname in classnames:
                        if prompt == template.format(cname):
                            classname = cname
                            break
                        if cname in description_bank and prompt in description_bank[cname]:
                            classname = cname
                            break
                prompt_id = len(self.all_prompts)
                self.prompt_to_classname.append(classname)
                self.all_prompts.append(prompt)
                self.domain_indices.append(ds.domain_idx)
                self.select_weights.append(per_prompt_weight)
                self.domain_to_prompt_ids.setdefault(ds.domain_idx, []).append(prompt_id)
                self.domain_to_prompt_weights.setdefault(ds.domain_idx, []).append(per_prompt_weight)

    def _sample_prompt_id_for_domain(self, domain_id, batch_descriptions):
        prompt_ids = self.domain_to_prompt_ids.get(domain_id)
        if not prompt_ids:
            return self.rng.choices(range(len(self.all_prompts)), weights=self.select_weights, k=1)[0]
        weights = self.domain_to_prompt_weights.get(domain_id)
        for _ in range(20):
            candidate_id = self.rng.choices(prompt_ids, weights=weights, k=1)[0]
            candidate_classname = self.prompt_to_classname[candidate_id]
            if candidate_classname and candidate_classname in batch_descriptions:
                chosen_desc = batch_descriptions[candidate_classname]
                for idx in prompt_ids:
                    if self.all_prompts[idx] == chosen_desc and self.prompt_to_classname[idx] == candidate_classname:
                        return idx
                continue
            return candidate_id
        return self.rng.choice(prompt_ids)

    def random_class_tokens(self, tokenizer, k=10):
        batch_descriptions = {}
        for ds in self.datasets:
            if hasattr(ds, "get_current_batch_descriptions"):
                for classname, desc in ds.get_current_batch_descriptions().items():
                    batch_descriptions.setdefault(classname, desc)
        selected_prompts = []
        selected_domains = []
        domain_ids = list(self.domain_to_prompt_ids.keys())
        if domain_ids:
            self.rng.shuffle(domain_ids)
            per_domain = k // len(domain_ids)
            remainder = k % len(domain_ids)
            for idx, domain_id in enumerate(domain_ids):
                take = per_domain + (1 if idx < remainder else 0)
                for _ in range(take):
                    prompt_id = self._sample_prompt_id_for_domain(domain_id, batch_descriptions)
                    selected_prompts.append(self.all_prompts[prompt_id])
                    selected_domains.append(self.domain_indices[prompt_id])
        while len(selected_prompts) < k:
            prompt_id = self.rng.choices(range(len(self.all_prompts)), weights=self.select_weights, k=1)[0]
            selected_prompts.append(self.all_prompts[prompt_id])
            selected_domains.append(self.domain_indices[prompt_id])
        return tokenizer(selected_prompts).squeeze(), selected_domains

    def _balanced_iter(self):
        domain_idx = 0
        samples_yielded = 0
        while samples_yielded < self.max_samples_per_epoch * self.num_datasets:
            yield next(self.dataset_iterators[domain_idx])
            samples_yielded += 1
            domain_idx = (domain_idx + 1) % self.num_datasets

    def __iter__(self):
        if self.balanced:
            yield from self._balanced_iter()
            return
        iterators = [iter(ds) for ds in self.datasets]
        buffers = [[] for _ in range(self.num_datasets)]
        active = [True] * self.num_datasets
        for i, iterator in enumerate(iterators):
            while len(buffers[i]) < self.buffer_size:
                try:
                    buffers[i].append(next(iterator))
                except StopIteration:
                    active[i] = False
                    break
        while any(buffers) or any(active):
            valid_domains = [i for i in range(self.num_datasets) if buffers[i]]
            if not valid_domains:
                break
            domain_idx = self.rng.choice(valid_domains)
            buffer_idx = self.rng.randint(0, len(buffers[domain_idx]) - 1)
            yield buffers[domain_idx].pop(buffer_idx)
            if active[domain_idx] and len(buffers[domain_idx]) < self.buffer_size:
                try:
                    buffers[domain_idx].append(next(iterators[domain_idx]))
                except StopIteration:
                    active[domain_idx] = False


def _load_description_bank(descriptions_dir: str, domain: str) -> Dict[str, List[str]]:
    path = os.path.join(descriptions_dir, f"{domain}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # json structure: {classname: [descriptions...]}
    return {k: [d for d in v if isinstance(d, str)] for k, v in data.items()}


def build_training_loader(
    tokenizer,
    preprocess_train,
    data_root: str,
    subsample: str,
    shots: int,
    dataset_mode: str,
    prompt_source: str,
    descriptions_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    class_filter_path: str = None,
    domain_order: List[str] = None,
) -> Tuple[DataLoader, Dict[str, int], List[str]]:
    """Create the interleaved training dataloader."""
    set_random_seed(seed)
    datasources = build_domain_datasets(
        data_root=data_root,
        shots=shots,
        subsample=subsample,
        domains=domain_order,
        class_filter_path=class_filter_path,
    )
    domain_to_idx = {name: i + 1 for i, name in enumerate(datasources.keys())}  # +1 reserve 0 for zero-expert

    per_domain_iterables = []
    for domain_name, ds in datasources.items():
        fallback_template = ds_utils.build_domain_template(domain_name)
        template = getattr(ds, "template", fallback_template)
        desc_bank = _load_description_bank(descriptions_dir, domain_name) if prompt_source in ["descriptions", "combined"] else {}
        wrapper = DatasetWrapper(
            ds._train_x,
            transform=preprocess_train,
            tokenizer=None,
            classnames=ds.classnames,
            templates=template,
        )
        iterable = PromptedDataset(
            wrapper,
            tokenizer=tokenizer,
            domain_idx=domain_to_idx[domain_name],
            prompt_source=prompt_source,
            description_bank=desc_bank,
            template=template if isinstance(template, str) else fallback_template,
            batch_size=batch_size,
        )
        per_domain_iterables.append(iterable)

    combined = InterleavedBatchBufferDataset(
        per_domain_iterables,
        balanced=(dataset_mode == "balanced"),
        seed=seed,
    )

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return loader, domain_to_idx, list(datasources.keys())


def build_in_domain_training_loaders(
    tokenizer,
    preprocess_train,
    data_root: str,
    subsample: str,
    shots: int,
    prompt_source: str,
    descriptions_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    class_filter_path: str = None,
    domain_order: List[str] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """Create one dataloader per domain for in-domain training."""
    set_random_seed(seed)
    datasources = build_domain_datasets(
        data_root=data_root,
        shots=shots,
        subsample=subsample,
        domains=domain_order,
        class_filter_path=class_filter_path,
    )
    domain_to_idx = {name: i + 1 for i, name in enumerate(datasources.keys())}
    loaders: Dict[str, DataLoader] = {}

    for domain_name, ds in datasources.items():
        fallback_template = ds_utils.build_domain_template(domain_name)
        template = getattr(ds, "template", fallback_template)
        desc_bank = (
            _load_description_bank(descriptions_dir, domain_name)
            if prompt_source in ["descriptions", "combined"]
            else {}
        )
        wrapper = DatasetWrapper(
            ds._train_x,
            transform=preprocess_train,
            tokenizer=None,
            classnames=ds.classnames,
            templates=template,
        )
        iterable = PromptedDataset(
            wrapper,
            tokenizer=tokenizer,
            domain_idx=domain_to_idx[domain_name],
            prompt_source=prompt_source,
            description_bank=desc_bank,
            template=template if isinstance(template, str) else fallback_template,
            batch_size=batch_size,
        )
        loaders[domain_name] = DataLoader(
            iterable,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    return loaders, domain_to_idx, list(datasources.keys())
