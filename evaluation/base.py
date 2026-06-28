"""Base CLIP evaluation mode."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .eval_utils import compute_correct_mask
from .overlap import (
    PerSampleOverlapLogger,
    apply_overlap_prediction_merge,
    compute_overlap_scores,
)
from .wrongly_classified import save_wrong_predictions


@torch.no_grad()
def evaluate_base(model, tokenizer, benchmark_dataset, preprocess, batch_size, num_workers, device):
    """Evaluate using base CLIP model only."""
    model.eval()
    
    # Build prompts from benchmark classnames
    prompts = benchmark_dataset.classnames
    text_tokens = tokenizer(prompts).to(device)
    
    # Encode all text prompts
    text_features = []
    for i in range(0, text_tokens.shape[0], batch_size):
        batch_tokens = text_tokens[i:i + batch_size]
        batch_txt_feats = model.encode_text(batch_tokens)
        text_features.append(batch_txt_feats)
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Create dataloader with return_domain=True
    dataloader = DataLoader(
        benchmark_dataset.test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Evaluate
    correct = 0
    total = 0
    overlap_score_sum = 0.0
    generic_fallback_count = 0
    per_domain_correct = {}
    per_domain_total = {}
    dataset_offset = 0
    resolved_overlap = getattr(benchmark_dataset, "overlap_resolved", None)
    detail_path = getattr(benchmark_dataset, "per_sample_output_path", None)
    detail_logger = (
        PerSampleOverlapLogger(
            output_path=detail_path,
            benchmark_dataset=benchmark_dataset,
            method=getattr(benchmark_dataset, "eval_method", "base"),
        )
        if detail_path
        else None
    )
    
    logit_scale = model.logit_scale.exp()
    
    context = detail_logger if detail_logger is not None else _NullContext()
    with context as logger:
        pbar = tqdm(dataloader, desc="Evaluating base CLIP")
        for batch in pbar:
            if len(batch) == 3:
                images, labels, domains = batch
                domains = domains.cpu().numpy()
            else:
                images, labels = batch
                domains = None
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Encode images
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logits = logit_scale * (image_features @ text_features.T)
            
            # Compute accuracy
            prediction_scores = apply_overlap_prediction_merge(logits, benchmark_dataset)
            preds = prediction_scores.argmax(dim=-1)
            batch_correct_mask = compute_correct_mask(preds, labels, benchmark_dataset)
            batch_correct = batch_correct_mask.sum().item()
            batch_total = len(labels)
            if resolved_overlap is not None:
                overlap_scores = compute_overlap_scores(preds, labels, benchmark_dataset)
                overlap_score_sum += float(overlap_scores.sum().item())
                for pred_idx, label_idx in zip(preds.detach().cpu().tolist(), labels.detach().cpu().tolist()):
                    if int(pred_idx) in resolved_overlap.generic_labels_by_fine_label.get(int(label_idx), set()):
                        generic_fallback_count += 1
            if logger is not None:
                logger.log_batch(
                    logits=prediction_scores,
                    preds=preds,
                    labels=labels,
                    domains=domains,
                    batch_start_idx=dataset_offset,
                )
            save_wrong_predictions(
                benchmark_dataset=benchmark_dataset,
                logits=prediction_scores,
                preds=preds,
                labels=labels,
                domains=domains,
                batch_start_idx=dataset_offset,
            )
            dataset_offset += batch_total
            
            correct += batch_correct
            total += batch_total
            running_acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating base CLIP (acc {running_acc:.4f})")
            
            # Per-domain tracking
            if domains is not None:
                for img_idx in range(batch_total):
                    domain = domains[img_idx]
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
    if resolved_overlap is not None:
        benchmark_dataset.overlap_metrics = {
            "strict_accuracy": overall_accuracy,
            "overlap_partial_accuracy": overlap_score_sum / total if total > 0 else 0.0,
            "generic_fallback_rate": generic_fallback_count / total if total > 0 else 0.0,
        }
    
    return overall_accuracy, per_domain_accuracy


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

