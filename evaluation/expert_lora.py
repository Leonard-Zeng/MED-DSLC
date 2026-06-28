import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .eval_utils import compute_correct_mask
from .overlap import apply_overlap_prediction_merge
from .wrongly_classified import save_wrong_predictions


@torch.no_grad()
def evaluate_expert_lora(
    model_by_domain_name,
    domain_name_by_index,
    tokenizer,
    benchmark_dataset,
    preprocess,
    batch_size,
    num_workers,
    device,
    current_domain=None,
):
    for model in model_by_domain_name.values():
        model.eval()
        model.to(device)

    prompts = benchmark_dataset.classnames
    text_tokens = tokenizer(prompts).to(device)

    if current_domain is not None:
        # Single-domain (in_domain) mode: one expert for the whole dataset
        model = model_by_domain_name.get(current_domain)
        if model is None:
            raise ValueError(f"Missing expert model for domain '{current_domain}'.")
        text_features = []
        for i in range(0, text_tokens.shape[0], batch_size):
            batch_tokens = text_tokens[i:i + batch_size]
            batch_txt_feats = model.encode_text(batch_tokens)
            text_features.append(batch_txt_feats)
        text_features = torch.cat(text_features, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()

        dataloader = DataLoader(
            benchmark_dataset.test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        correct = 0
        total = 0
        dataset_offset = 0
        pbar = tqdm(dataloader, desc="Evaluating expert LoRA")
        for batch in pbar:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.T)
            prediction_scores = apply_overlap_prediction_merge(logits, benchmark_dataset)
            preds = prediction_scores.argmax(dim=-1)
            batch_correct_mask = compute_correct_mask(preds, labels, benchmark_dataset)
            correct += batch_correct_mask.sum().item()
            total += len(labels)
            save_wrong_predictions(
                benchmark_dataset=benchmark_dataset,
                logits=prediction_scores,
                preds=preds,
                labels=labels,
                domains=None,
                batch_start_idx=dataset_offset,
            )
            dataset_offset += len(labels)
            running_acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating expert LoRA (acc {running_acc:.4f})")
        overall_accuracy = correct / total if total > 0 else 0.0
        return overall_accuracy, {current_domain: overall_accuracy}

    # Combined benchmark: require return_domain=True (3-tuple batches)
    if not domain_name_by_index:
        raise ValueError("domain_name_by_index is required for expert_lora (combined benchmark).")

    text_features_by_domain = {}
    for domain_idx, domain_name in domain_name_by_index.items():
        model = model_by_domain_name.get(domain_name)
        if model is None:
            raise ValueError(f"Missing expert model for domain '{domain_name}'.")
        text_features = []
        for i in range(0, text_tokens.shape[0], batch_size):
            batch_tokens = text_tokens[i:i + batch_size]
            batch_txt_feats = model.encode_text(batch_tokens)
            text_features.append(batch_txt_feats)
        text_features = torch.cat(text_features, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_by_domain[domain_idx] = text_features

    dataloader = DataLoader(
        benchmark_dataset.test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    correct = 0
    total = 0
    per_domain_correct = {}
    per_domain_total = {}
    dataset_offset = 0

    pbar = tqdm(dataloader, desc="Evaluating expert LoRA")
    for batch in pbar:
        if len(batch) == 3:
            images, labels, domains = batch
        else:
            raise ValueError("expert_lora (combined) requires datasets that return (img, label, domain).")
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        domains_cpu = domains.cpu().numpy()

        unique_domains = torch.unique(domains)
        image_features = None

        for domain_id in unique_domains:
            domain_idx = int(domain_id.item())
            domain_name = domain_name_by_index.get(domain_idx)
            if domain_name is None:
                raise ValueError(f"Unknown domain index {domain_idx} in batch.")
            model = model_by_domain_name.get(domain_name)
            if model is None:
                raise ValueError(f"Missing expert model for domain '{domain_name}'.")
            mask = domains == domain_id
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            domain_images = images[idxs]
            domain_features = model.encode_image(domain_images)
            domain_features = domain_features / domain_features.norm(dim=-1, keepdim=True)
            if image_features is None:
                image_features = torch.empty(
                    (images.shape[0], domain_features.shape[-1]),
                    device=domain_features.device,
                )
            image_features[idxs] = domain_features

        logits = torch.empty((images.shape[0], len(prompts)), device=device)
        for domain_id in unique_domains:
            domain_idx = int(domain_id.item())
            domain_name = domain_name_by_index.get(domain_idx)
            if domain_name is None:
                raise ValueError(f"Unknown domain index {domain_idx} in batch.")
            model = model_by_domain_name.get(domain_name)
            if model is None:
                raise ValueError(f"Missing expert model for domain '{domain_name}'.")
            mask = domains == domain_id
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            domain_text_features = text_features_by_domain.get(domain_idx)
            if domain_text_features is None:
                raise ValueError(f"Missing text features for domain index {domain_idx}.")
            logit_scale = model.logit_scale.exp()
            logits[idxs] = logit_scale * (image_features[idxs] @ domain_text_features.T)

        prediction_scores = apply_overlap_prediction_merge(logits, benchmark_dataset)
        preds = prediction_scores.argmax(dim=-1)
        batch_correct_mask = compute_correct_mask(preds, labels, benchmark_dataset)
        batch_correct = batch_correct_mask.sum().item()
        batch_total = len(labels)
        save_wrong_predictions(
            benchmark_dataset=benchmark_dataset,
            logits=prediction_scores,
            preds=preds,
            labels=labels,
            domains=domains_cpu,
            batch_start_idx=dataset_offset,
        )
        dataset_offset += batch_total

        correct += batch_correct
        total += batch_total
        running_acc = correct / total if total > 0 else 0.0
        pbar.set_description(f"Evaluating expert LoRA (acc {running_acc:.4f})")

        for img_idx in range(batch_total):
            domain = domains_cpu[img_idx]
            if domain not in per_domain_correct:
                per_domain_correct[domain] = 0
                per_domain_total[domain] = 0
            if bool(batch_correct_mask[img_idx].item()):
                per_domain_correct[domain] += 1
            per_domain_total[domain] += 1

    overall_accuracy = correct / total if total > 0 else 0.0
    per_domain_accuracy = {
        domain: per_domain_correct[domain] / per_domain_total[domain]
        for domain in per_domain_correct
        if per_domain_total[domain] > 0
    }

    return overall_accuracy, per_domain_accuracy
