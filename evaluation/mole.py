import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lora_med.gate_hook import GateValueHook
from .eval_utils import compute_correct_mask
from .overlap import (
    PerSampleOverlapLogger,
    apply_overlap_prediction_merge,
    compute_overlap_scores,
)
from .wrongly_classified import save_wrong_predictions


def _reduce_gate_logits(logits: torch.Tensor | None) -> torch.Tensor | None:
    if logits is None:
        return None
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits.mean(dim=1)
    return None


def _collect_mean_gate_logits(hook_manager: GateValueHook, prefix: str) -> torch.Tensor | None:
    per_layer = []
    for key in hook_manager.gate_values.keys():
        if prefix not in key:
            continue
        values = hook_manager.gate_values.get(key, [])
        if len(values) == 0:
            continue
        output = values[-1]
        if not (isinstance(output, tuple) and len(output) >= 2):
            continue
        logits = _reduce_gate_logits(output[1])
        if logits is None:
            continue
        per_layer.append(logits)
    if len(per_layer) == 0:
        return None
    return torch.stack(per_layer, dim=0).mean(dim=0)


def _infer_gate_expert_names(num_experts: int, benchmark_dataset) -> list[str]:
    mapping = getattr(benchmark_dataset, "domain_name_by_index", {}) or {}
    ordered = [mapping[idx] for idx in sorted(mapping.keys())]
    if len(ordered) == num_experts:
        return [str(x) for x in ordered]
    if len(ordered) + 1 == num_experts:
        return ["base_clip"] + [str(x) for x in ordered]
    return [f"expert_{i}" for i in range(num_experts)]


def _build_domain_to_idx(benchmark_dataset) -> dict:
    """Build domain->index mapping from benchmark metadata."""
    idx_to_domain = getattr(benchmark_dataset, "domain_name_by_index", {}) or {}
    mapping = {}
    for idx, domain_name in idx_to_domain.items():
        if domain_name is None:
            continue
        mapping[str(domain_name)] = int(idx)
    return mapping


def _num_scalar_domains(logit_scalar_module) -> int | None:
    if logit_scalar_module is None:
        return None
    num_domains = getattr(logit_scalar_module, "num_domains", None)
    if num_domains is not None:
        return int(num_domains)
    head = getattr(logit_scalar_module, "img_domain_head", None)
    if head is not None and hasattr(head, "out_features"):
        return int(head.out_features)
    return None


@torch.no_grad()
def evaluate_mole(
    model,
    tokenizer,
    benchmark_dataset,
    preprocess,
    batch_size,
    num_workers,
    device,
    logit_scalar_module=None,
    logit_scalar_mode: str = "standard",
    gate_logger=None,
    current_domain=None,
):
    model.eval()
    
    # Build prompts from benchmark classnames
    prompts = benchmark_dataset.classnames
    text_tokens = tokenizer(prompts).to(device)
    
    # Encode all text prompts
    collect_gate_for_wrong = bool(getattr(benchmark_dataset, "dump_wrong_images", False))
    collect_gate_for_details = bool(getattr(benchmark_dataset, "dump_eval_details", False))
    collect_gate_for_wrong = collect_gate_for_wrong or collect_gate_for_details
    hook_manager = None
    text_gate_mean_logits = None
    gate_expert_names = None
    if collect_gate_for_wrong:
        gate_modules = [m for m in model.modules() if hasattr(m, "gate_net")]
        if len(gate_modules) > 0:
            hook_manager = GateValueHook(expert_names=[])
            hook_manager.register_hooks(gate_modules, training=False)

    text_features = []
    text_gate_chunks = []
    for i in range(0, text_tokens.shape[0], batch_size):
        batch_tokens = text_tokens[i:i + batch_size]
        batch_txt_feats = model.encode_text(batch_tokens)
        text_features.append(batch_txt_feats)
        if hook_manager is not None:
            batch_text_gate = _collect_mean_gate_logits(hook_manager, prefix="text")
            if batch_text_gate is not None:
                text_gate_chunks.append(batch_text_gate.detach().cpu())
            hook_manager.clear_step()
    text_features = torch.cat(text_features, dim=0)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    if len(text_gate_chunks) > 0:
        text_gate_mean_logits = torch.cat(text_gate_chunks, dim=0)
        if text_gate_mean_logits.shape[0] == text_features.shape[0]:
            gate_expert_names = _infer_gate_expert_names(
                num_experts=text_gate_mean_logits.shape[1],
                benchmark_dataset=benchmark_dataset,
            )

    text_domain = None
    class_domains = getattr(benchmark_dataset, "class_domains", None)
    domain_to_idx = _build_domain_to_idx(benchmark_dataset)
    scalar_num_domains = _num_scalar_domains(logit_scalar_module)
    if logit_scalar_mode == "domain-wise" or logit_scalar_mode == "affine-wise" :
        if class_domains is not None and len(class_domains) == text_features.shape[0]:
            mapped = []
            for d in class_domains:
                if isinstance(d, str):
                    mapped.append(domain_to_idx.get(d, -1))
                else:
                    try:
                        mapped.append(int(d))
                    except Exception:
                        mapped.append(-1)
            if scalar_num_domains is not None:
                invalid = [v for v in mapped if v < 0 or v >= scalar_num_domains]
                if len(invalid) > 0:
                    raise ValueError(
                        "Invalid text_domain targets for domain-wise logit scalar. "
                        f"Found invalid values {sorted(set(invalid))} with num_domains={scalar_num_domains}. "
                        f"domain_name_by_index={getattr(benchmark_dataset, 'domain_name_by_index', {})}, "
                        f"sample class_domains={list(class_domains)[:10]}"
                    )
            text_domain = torch.tensor(mapped, device=device, dtype=torch.long)
    
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
    overlap_score_sum = 0.0
    generic_fallback_count = 0
    per_domain_correct = {}
    per_domain_total = {}
    dataset_offset = 0
    logged_text_scalar = False
    resolved_overlap = getattr(benchmark_dataset, "overlap_resolved", None)
    detail_path = getattr(benchmark_dataset, "per_sample_output_path", None)
    detail_logger = (
        PerSampleOverlapLogger(
            output_path=detail_path,
            benchmark_dataset=benchmark_dataset,
            method=getattr(benchmark_dataset, "eval_method", "mole"),
        )
        if detail_path
        else None
    )
    
    logit_scale = model.logit_scale.exp()
    
    context = detail_logger if detail_logger is not None else _NullContext()
    with context as logger:
        pbar = tqdm(dataloader, desc="Evaluating MED")
        for batch in pbar:
            if len(batch) == 3:
                images, labels, domains = batch
                domains = domains.cpu().numpy()
            else:
                images, labels = batch
                domains = None
            
            images = images.to(device)
            labels = labels.to(device)
            if gate_logger is not None:
                gate_logger.start_batch(domains=domains, current_domain=current_domain, batch_size=images.shape[0])
            
            # Encode images (gating happens automatically in MED layers)
            image_features = model.encode_image(images)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            image_gate_mean_logits = None
            if hook_manager is not None:
                image_gate_mean_logits = _collect_mean_gate_logits(hook_manager, prefix="image")
            
            # Compute logits
            if logit_scalar_module is None or logit_scalar_mode == "standard":
                logits = logit_scale * (image_features_norm @ text_features_norm.T)
            else:
                image_domain = None
                if logit_scalar_mode == "domain-wise" and domains is not None:
                    image_domain = torch.as_tensor(domains, device=device, dtype=torch.long)
                    if image_domain.numel() > 0 and int(image_domain.min().item()) >= 1:
                        image_domain = image_domain - 1
                if (
                    gate_logger is not None
                    and logit_scalar_mode == "domain-wise"
                    and logit_scalar_module is not None
                ):
                    img_domain_logits = logit_scalar_module.img_domain_head(image_features)
                    gate_logger.log_domain_logits(
                        logits=img_domain_logits,
                        domains=domains if domains is not None else [current_domain] * image_features.shape[0],
                        source="logit_scalar_img",
                        layer="img_domain_head",
                    )
                    if not logged_text_scalar and class_domains is not None and len(class_domains) == text_features.shape[0]:
                        txt_domain_logits = logit_scalar_module.txt_domain_head(text_features)
                        gate_logger.log_domain_logits(
                            logits=txt_domain_logits,
                            domains=class_domains,
                            source="logit_scalar_txt",
                            layer="txt_domain_head",
                        )
                        logged_text_scalar = True
                logits, _ = logit_scalar_module(
                    image_features=image_features,
                    text_features=text_features,
                    image_features_norm=image_features_norm,
                    text_features_norm=text_features_norm,
                    image_domain=image_domain,
                    text_domain=text_domain,
                )
            if gate_logger is not None:
                gate_logger.end_batch()
            
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
                    image_gate_mean_logits=image_gate_mean_logits,
                    text_gate_mean_logits=text_gate_mean_logits,
                    gate_expert_names=gate_expert_names,
                )
            save_wrong_predictions(
                benchmark_dataset=benchmark_dataset,
                logits=prediction_scores,
                preds=preds,
                labels=labels,
                domains=domains,
                batch_start_idx=dataset_offset,
                image_gate_mean_logits=image_gate_mean_logits,
                text_gate_mean_logits=text_gate_mean_logits,
                gate_expert_names=gate_expert_names,
            )
            dataset_offset += batch_total
            if hook_manager is not None:
                hook_manager.clear_step()
            
            correct += batch_correct
            total += batch_total
            running_acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating MED (acc {running_acc:.4f})")
            
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
    if hook_manager is not None:
        hook_manager.remove_hooks()
        hook_manager.clear()
    
    return overall_accuracy, per_domain_accuracy


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

