import os
import re
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .eval_utils import (
    are_class_indices_semantically_equivalent,
    is_directionally_semantic_match,
)


def _to_list(domains) -> List[int] | None:
    if domains is None:
        return None
    if isinstance(domains, torch.Tensor):
        return [int(x) for x in domains.detach().cpu().tolist()]
    if isinstance(domains, np.ndarray):
        return [int(x) for x in domains.tolist()]
    if isinstance(domains, Iterable):
        return [int(x) for x in domains]
    return None


def _safe_get(seq: Sequence | None, idx: int, default):
    if seq is None:
        return default
    if 0 <= idx < len(seq):
        return seq[idx]
    return default


def _sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned[:80] if cleaned else "unknown"


def _resize_if_too_small(image: Image.Image, min_side: int = 320) -> Image.Image:
    width, height = image.size
    if min(width, height) >= min_side:
        return image
    scale = float(min_side) / float(min(width, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)


def _draw_text_block(
    overlay_draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
    corner: str,
):
    if not lines:
        return
    width, _ = image_size
    line_spacing = 4
    pad = 6

    line_boxes = [overlay_draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [(box[3] - box[1]) for box in line_boxes]
    line_widths = [(box[2] - box[0]) for box in line_boxes]
    block_w = max(line_widths) + (2 * pad)
    block_h = sum(line_heights) + line_spacing * (len(lines) - 1) + (2 * pad)

    if corner == "top_left":
        x0 = 8
    elif corner == "top_right":
        x0 = max(8, width - block_w - 8)
    elif corner == "bottom_left":
        x0 = 8
    else:
        x0 = 8
    if corner == "bottom_left":
        y0 = max(8, image_size[1] - block_h - 8)
    else:
        y0 = 8
    x1 = x0 + block_w
    y1 = y0 + block_h

    overlay_draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 180))
    y = y0 + pad
    for line, line_h in zip(lines, line_heights):
        overlay_draw.text((x0 + pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h + line_spacing


def _wrap_text_to_width(
    text: str,
    font: ImageFont.ImageFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
) -> List[str]:
    """Greedy word-wrap for overlay text blocks."""
    words = text.split()
    if not words:
        return [text]

    wrapped: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
        if (box[2] - box[0]) <= max_width:
            current = candidate
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def _resolve_domain_name(
    domain_index: int | None,
    domain_name_by_index: dict | None,
    fallback: str | None = None,
) -> str:
    if fallback:
        return fallback
    if domain_index is None:
        return "unknown"
    if domain_name_by_index and domain_index in domain_name_by_index:
        return str(domain_name_by_index[domain_index])
    return str(domain_index)


def _same_domain(lhs: str, rhs: str) -> bool:
    return lhs.strip().lower() == rhs.strip().lower()


def save_wrong_predictions(
    benchmark_dataset,
    logits: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    domains=None,
    batch_start_idx: int = 0,
    image_gate_mean_logits: torch.Tensor | None = None,
    text_gate_mean_logits: torch.Tensor | None = None,
    gate_expert_names: Sequence[str] | None = None,
):
    """Save wrongly classified samples with GT/pred/top-k logits overlays."""
    if not getattr(benchmark_dataset, "dump_wrong_images", False):
        return

    output_dir = getattr(benchmark_dataset, "dump_wrong_images_dir", None)
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    dataset_wrapper = getattr(benchmark_dataset, "test", None)
    data_source = getattr(dataset_wrapper, "data_source", None)
    if data_source is None:
        return

    raw_classnames = list(getattr(benchmark_dataset, "raw_classnames", []))
    class_domains = list(getattr(benchmark_dataset, "class_domains", []))
    domain_name_by_index = getattr(benchmark_dataset, "domain_name_by_index", None)
    current_domain = getattr(benchmark_dataset, "current_domain", None)
    topk = max(1, int(getattr(benchmark_dataset, "dump_wrong_images_topk", 5)))
    txt_log_path = os.path.join(output_dir, "wrong_predictions.txt")

    logits_cpu = logits.detach().cpu()
    preds_cpu = preds.detach().cpu()
    labels_cpu = labels.detach().cpu()
    domains_list = _to_list(domains)
    k = min(topk, logits_cpu.shape[-1])
    font = ImageFont.load_default()
    image_gate_cpu = image_gate_mean_logits.detach().cpu() if image_gate_mean_logits is not None else None
    text_gate_cpu = text_gate_mean_logits.detach().cpu() if text_gate_mean_logits is not None else None
    expert_names = list(gate_expert_names) if gate_expert_names is not None else None
    expert_lookup = {}
    if expert_names:
        for idx, name in enumerate(expert_names):
            expert_lookup[str(name).strip().lower()] = idx

    for i in range(logits_cpu.shape[0]):
        pred_idx = int(preds_cpu[i].item())
        label_idx = int(labels_cpu[i].item())
        semantic_lookup = getattr(benchmark_dataset, "semantic_equivalence_lookup", None)
        directional_lookup = getattr(benchmark_dataset, "semantic_directional_lookup", None)
        if are_class_indices_semantically_equivalent(
            pred_idx=pred_idx,
            label_idx=label_idx,
            classnames=raw_classnames,
            semantic_lookup=semantic_lookup,
        ) or is_directionally_semantic_match(
            pred_idx=pred_idx,
            label_idx=label_idx,
            classnames=raw_classnames,
            directional_lookup=directional_lookup,
        ):
            continue

        data_idx = batch_start_idx + i
        if data_idx >= len(data_source):
            continue
        item = data_source[data_idx]
        impath = getattr(item, "impath", None)
        if not impath or not os.path.exists(impath):
            continue

        gt_name = _safe_get(raw_classnames, label_idx, f"class_{label_idx}")
        pred_name = _safe_get(raw_classnames, pred_idx, f"class_{pred_idx}")

        sample_domain_idx = None
        if domains_list is not None and i < len(domains_list):
            sample_domain_idx = domains_list[i]

        gt_domain = _resolve_domain_name(
            domain_index=sample_domain_idx,
            domain_name_by_index=domain_name_by_index,
            fallback=current_domain,
        )
        pred_domain = _safe_get(class_domains, pred_idx, None) or "unknown"
        gt_class_domain = _safe_get(class_domains, label_idx, None)
        if gt_class_domain:
            gt_domain = gt_class_domain
        if _same_domain(pred_domain, gt_domain):
            # Keep only OOD wrong predictions:
            # predicted class domain must differ from GT domain.
            continue

        gt_logit = float(logits_cpu[i, label_idx].item())
        pred_logit = float(logits_cpu[i, pred_idx].item())

        top_vals, top_indices = torch.topk(logits_cpu[i], k=k)
        right_lines = []
        top_entries = []
        for rank in range(k):
            cls_idx = int(top_indices[rank].item())
            cls_name = _safe_get(raw_classnames, cls_idx, f"class_{cls_idx}")
            cls_domain = _safe_get(class_domains, cls_idx, None) or "unknown"
            cls_logit = float(top_vals[rank].item())
            right_lines.append(f"({cls_domain}, {cls_name}) -> {cls_logit:.4f}")
            top_entries.append((cls_domain, cls_name, cls_logit))

        left_lines = [
            f"({gt_domain}, {gt_name}) -> {gt_logit:.4f}",
            f"({pred_domain}, {pred_name}) -> {pred_logit:.4f}",
        ]
        gate_lines = []
        gt_domain_key = str(gt_domain).strip().lower()
        gt_domain_expert_idx = expert_lookup.get(gt_domain_key, None)

        if image_gate_cpu is not None and i < image_gate_cpu.shape[0]:
            img_gate_vec = image_gate_cpu[i]
            pred_idx_gate = int(img_gate_vec.argmax().item())
            pred_name_gate = (
                expert_names[pred_idx_gate]
                if expert_names and pred_idx_gate < len(expert_names)
                else f"expert_{pred_idx_gate}"
            )
            if gt_domain_expert_idx is not None and gt_domain_expert_idx < img_gate_vec.shape[0]:
                gt_logit_gate = float(img_gate_vec[gt_domain_expert_idx].item())
                gate_lines.append(
                    f"img_gate meanL pred={pred_name_gate} gt_logit={gt_logit_gate:.4f}"
                )
            else:
                gate_lines.append(
                    f"img_gate meanL pred={pred_name_gate} gt_logit=n/a"
                )

        if (
            text_gate_cpu is not None
            and label_idx < text_gate_cpu.shape[0]
        ):
            txt_gate_vec = text_gate_cpu[label_idx]
            pred_idx_gate = int(txt_gate_vec.argmax().item())
            pred_name_gate = (
                expert_names[pred_idx_gate]
                if expert_names and pred_idx_gate < len(expert_names)
                else f"expert_{pred_idx_gate}"
            )
            if gt_domain_expert_idx is not None and gt_domain_expert_idx < txt_gate_vec.shape[0]:
                gt_logit_gate = float(txt_gate_vec[gt_domain_expert_idx].item())
                gate_lines.append(
                    f"txt_gate meanL pred={pred_name_gate} gt_logit={gt_logit_gate:.4f}"
                )
            else:
                gate_lines.append(
                    f"txt_gate meanL pred={pred_name_gate} gt_logit=n/a"
                )
        if gate_lines:
            left_lines.extend(gate_lines)

        try:
            image = Image.open(impath).convert("RGB")
        except OSError:
            continue
        image = _resize_if_too_small(image, min_side=320)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        _draw_text_block(
            overlay_draw=overlay_draw,
            lines=left_lines,
            font=font,
            image_size=image.size,
            corner="top_left",
        )
        _draw_text_block(
            overlay_draw=overlay_draw,
            lines=right_lines,
            font=font,
            image_size=image.size,
            corner="top_right",
        )

        # Add original source image path at the bottom for traceability.
        path_prefix = "path:"
        max_text_w = max(80, image.size[0] - 32)
        path_lines = _wrap_text_to_width(
            f"{path_prefix} {impath}",
            font=font,
            draw=overlay_draw,
            max_width=max_text_w,
        )
        _draw_text_block(
            overlay_draw=overlay_draw,
            lines=path_lines,
            font=font,
            image_size=image.size,
            corner="bottom_left",
        )
        rendered = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        top1_domain, top1_name, top1_logit = top_entries[0]
        filename = (
            f"{data_idx:07d}__gt_{_sanitize_name(gt_name)}"
            f"__pred_{_sanitize_name(pred_name)}"
            f"__top1_{_sanitize_name(top1_domain)}_{_sanitize_name(top1_name)}_{top1_logit:.4f}.jpg"
        )
        out_path = os.path.join(output_dir, filename)
        rendered.save(out_path, quality=95)

        pred_text = " | ".join(
            [f"({d}, {c}) -> {v:.4f}" for d, c, v in top_entries]
        )
        gt_text = f"({gt_domain}, {gt_name}) -> {gt_logit:.4f}"
        gate_text = ""
        if gate_lines:
            gate_text = f"; gate: {' | '.join(gate_lines)}"
        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{filename}\tpredicted: {pred_text}; GT: {gt_text}{gate_text}\n"
            )
