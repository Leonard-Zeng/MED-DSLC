import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lora_mixture import update_alpha_lora_mixtures
from .eval_utils import compute_correct_mask
from .overlap import apply_overlap_prediction_merge
from .wrongly_classified import save_wrong_predictions


@torch.no_grad()
def evaluate_lora_mean(model, mixtures, num_experts, tokenizer, benchmark_dataset, preprocess, batch_size, num_workers, device):
    """Evaluate using LoRA mean (uniform mixture of all experts)."""
    model.eval()
    
    # Set uniform alphas
    uniform_alphas = [1.0 / num_experts] * num_experts
    # uniform_alphas = [1] * num_experts
    update_alpha_lora_mixtures(mixtures, uniform_alphas)
    
    # Build prompts from benchmark classnames
    prompts = benchmark_dataset.classnames
    text_tokens = tokenizer(prompts).to(device)
    
    # Encode all text prompts with LoRA mixture
    text_features = []
    for i in range(0, text_tokens.shape[0], batch_size):
        batch_tokens = text_tokens[i:i + batch_size]
        batch_txt_feats = model.encode_text(batch_tokens)
        text_features.append(batch_txt_feats)
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Create dataloader
    dataloader = DataLoader(
        benchmark_dataset.test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Evaluate
    correct = 0
    total = 0
    per_domain_correct = {}
    per_domain_total = {}
    dataset_offset = 0
    
    logit_scale = model.logit_scale.exp()
    
    pbar = tqdm(dataloader, desc="Evaluating LoRA mean")
    for batch in pbar:
        if len(batch) == 3:
            images, labels, domains = batch
            domains = domains.cpu().numpy()
        else:
            images, labels = batch
            domains = None
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images with LoRA mixture
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
        pbar.set_description(f"Evaluating LoRA mean (acc {running_acc:.4f})")
        
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
    
    return overall_accuracy, per_domain_accuracy

