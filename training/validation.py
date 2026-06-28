import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_lora_datasets.utils import CombinedDataset, DatasetWrapper, create_domain_prompt
from training.config import DATASET_MAP, DOMAIN_ORDER


def _build_classname_to_domain_map(domain2classes):
    mapping = {}
    for domain, classnames in domain2classes.items():
        for classname in classnames:
            mapping[classname] = domain
    return mapping


def _datum_key(datum):
    return (datum.impath, datum.label, datum.classname)


def build_cross_domain_val_data(
    preprocess_val,
    data_root,
    subsample,
    batch_size,
    num_workers,
    val_shots,
    seed,
    shots=-1,
    domain_order=None,
):
    datasets = []
    rng = random.Random(seed)
    for domain in (domain_order or DOMAIN_ORDER):
        cls = DATASET_MAP.get(domain)
        if cls is None:
            continue
        ds = cls(root=data_root, num_shots=-1, subsample=subsample)

        # Default: validation split (not test split).
        val_source = list(ds.val) if ds.val else []

        # If training uses a subset (few-shot), use the remaining train_x as validation.
        if shots is not None and shots > 0:
            ds_train_subset = cls(root=data_root, num_shots=shots, subsample=subsample)
            train_full = list(ds.train_x) if ds.train_x else []
            train_subset = list(ds_train_subset.train_x) if ds_train_subset.train_x else []
            if len(train_subset) < len(train_full):
                used_keys = {_datum_key(x) for x in train_subset}
                remainder = [x for x in train_full if _datum_key(x) not in used_keys]
                if len(remainder) > 0:
                    val_source = remainder

        if val_shots is not None and val_shots > 0 and val_source:
            if len(val_source) > val_shots:
                val_source = rng.sample(val_source, k=val_shots)

        ds._val = val_source
        datasets.append(ds)

    combined = CombinedDataset(datasets)
    raw_classnames = combined.classnames
    domain_by_classname = _build_classname_to_domain_map(combined.domain2classes)
    domain_by_class_index = [domain_by_classname.get(cname) for cname in raw_classnames]
    prompts = [
        create_domain_prompt(cname, domain_by_class_index[i])
        for i, cname in enumerate(raw_classnames)
    ]

    val_data_source = combined.val if combined.val else combined.test
    val_wrapper = DatasetWrapper(
        data_source=val_data_source,
        transform=preprocess_val,
        return_domain=False,
        is_train=False,
        classnames=raw_classnames,
    )
    val_loader = DataLoader(
        val_wrapper,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return val_loader, prompts


def _resolve_logit_scale(model):
    if hasattr(model, "logit_scale"):
        return model.logit_scale.exp()
    if hasattr(model, "clip_model") and hasattr(model.clip_model, "logit_scale"):
        return model.clip_model.logit_scale.exp()
    raise AttributeError("Model must expose `logit_scale` or `clip_model.logit_scale`.")


@torch.no_grad()
def run_cross_domain_val(
    model,
    tokenizer,
    val_loader,
    prompts,
    device,
    before_encode_text=None,
    after_encode_text=None,
):
    model.eval()
    text_tokens = tokenizer(prompts).to(device)
    text_features = []
    for i in range(0, text_tokens.shape[0], val_loader.batch_size):
        batch_tokens = text_tokens[i : i + val_loader.batch_size]
        if before_encode_text is not None:
            before_encode_text(batch_tokens)
        batch_txt_feats = model.encode_text(batch_tokens)
        if after_encode_text is not None:
            after_encode_text()
        text_features.append(batch_txt_feats)
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = _resolve_logit_scale(model)

    total = 0
    total_loss = 0.0
    correct = 0
    pbar = tqdm(val_loader, desc="Val cross-domain")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features.T)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        batch_total = len(labels)
        total_loss += loss.item() * batch_total
        correct += (preds == labels).sum().item()
        total += batch_total
        running_acc = correct / total if total > 0 else 0.0
        pbar.set_description(f"Val cross-domain (acc {running_acc:.4f})")

    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def run_cross_domain_val_joint(model, tokenizer, val_loader, prompts, device):
    model.eval()
    text_tokens = tokenizer(prompts).to(device)
    total = 0
    total_loss = 0.0
    correct = 0
    pbar = tqdm(val_loader, desc="Val cross-domain")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images, text_tokens)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        batch_total = len(labels)
        total_loss += loss.item() * batch_total
        correct += (preds == labels).sum().item()
        total += batch_total
        running_acc = correct / total if total > 0 else 0.0
        pbar.set_description(f"Val cross-domain (acc {running_acc:.4f})")

    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc
