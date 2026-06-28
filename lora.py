import torch
import torch.nn.functional as F
import clip

import tqdm
import time
from typing import Dict, List, Tuple
from contextlib import nullcontext

import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import importlib.util
utils_spec = importlib.util.spec_from_file_location("main_utils", os.path.join(parent_dir, "utils.py"))
main_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(main_utils)

clip_classifier = main_utils.clip_classifier
cls_acc = main_utils.cls_acc
correct_flg = main_utils.correct_flg
pre_load_features = main_utils.pre_load_features

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from device_utils import resolve_device


def _device(args):
    return resolve_device(getattr(args, "device", None))


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(device):
    return torch.cuda.amp.GradScaler(enabled=torch.device(device).type == "cuda")


def _maybe_half(tensor, device):
    if torch.device(device).type == "cuda":
        return tensor.half()
    return tensor

def evaluate_lora_on_combined_dataset(args, clip_model, loader, dataset, logit_scale=100.0, has_ood=False):
    device = _device(args)
    assert loader.dataset.return_domain
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with _autocast(device):
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    domain_names = dataset.domains
    num_classes = dataset.num_classes
    ds2acc = {domain_name: 0 for domain_name in domain_names} 
    ds2count = {domain_name: 0 for domain_name in domain_names} 

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target, domain) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            images, target = images.to(device), target.to(device)

            if has_ood:
                target = (target + (num_classes + 1)) % (num_classes + 1)
                domain = (domain + (len(domain_names) + 1)) % (len(domain_names) + 1)

            with _autocast(device):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()

            correct_flags = correct_flg(cosine_similarity, target)  # (k, B)
            # print(target)
            # print(correct_flags.shape)
            # print(correct_flags)
            # print(target)
            # print(cosine_similarity.argmax(-1))

            batch_size = target.size(0)
            tot_samples += batch_size

            # accumulate overall correct count
            acc += float(correct_flags.sum())

            # accumulate per-domain correct & counts
            for idx in range(batch_size):
                dname = domain_names[domain[idx].item()]
                ds2acc[dname]  += correct_flags[:, idx].sum()
                ds2count[dname] += 1

    overall_acc = acc / tot_samples
    per_domain_acc = {
        d: (ds2acc[d] / ds2count[d] if ds2count[d] > 0 else 0.0)
        for d in domain_names
    }
    return overall_acc, per_domain_acc


def evaluate_lora(args, clip_model, loader, dataset, logit_scale=100.0):
    device = _device(args)
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template if isinstance(dataset.template, str) else dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with _autocast(device):
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            images, target = images.to(device), target.to(device)
            with _autocast(device):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    device = _device(args)
    if torch.device(device).type == "mps":
        # for MPS, we need to have consistent type, so we all narrow down to float32 whenever in mps mode
        clip_model.float()
    
    VALIDATION = False

    # print(f"length of test dataloader {len(test_loader)}")
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model, device=device)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader, device=device)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader, device=device)
    
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    # zs_acc = evaluate_lora(args, clip_model, test_loader, dataset)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device)
    
    if args.eval_only:
        args.dataset = "imagenet" if "imagenet" in args.dataset else args.dataset
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset, logit_scale)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return zs_acc, acc_test

    mark_only_lora_as_trainable(clip_model)
    # Calculate total iterations for scheduler: epochs * batches per epoch
    total_iters = args.epochs * len(train_loader)
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = _grad_scaler(device)
    
    for epoch in range(args.epochs):
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = _maybe_half(textual_features.t(), device)
        for i, (images, target) in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            
            template = dataset.template
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.to(device), target.to(device)
            if args.encoder == 'text' or args.encoder == 'both':
                with _autocast(device):
                    texts = clip.tokenize(texts).to(device)
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with _autocast(device):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with _autocast(device):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}')

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset, logit_scale)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
        if args.save_path != None:
            save_lora(args, list_lora_layers)
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset, logit_scale)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return zs_acc, acc_test


class LoRASignalTracker:
    """Tracks LoRA delta signal from all active LinearLoRA modules."""

    def __init__(self, list_lora_layers, eps: float = 1e-8):
        self.eps = eps
        self.hooks = []
        self.delta_signals = []
        self.norm_signals = []
        self._register(list_lora_layers)

    def _register(self, list_lora_layers):
        for layer in list_lora_layers:
            for proj_name in ["q_proj", "k_proj", "v_proj", "proj"]:
                proj = getattr(layer, proj_name, None)
                if isinstance(proj, lora_layers.LinearLoRA):
                    self.hooks.append(proj.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, inputs, output):
        x = inputs[0]
        base = F.linear(x, module.weight, module.bias)
        delta = output - base

        mean_delta_sq = delta.float().pow(2).mean()
        delta_mag = torch.sqrt(mean_delta_sq + self.eps)
        base_mag = torch.sqrt(base.float().pow(2).mean() + self.eps)
        norm_mag = delta_mag / (base_mag + self.eps)

        self.delta_signals.append(delta_mag)
        self.norm_signals.append(norm_mag)

    def reset(self):
        self.delta_signals.clear()
        self.norm_signals.clear()

    def get_signals(self, device):
        if len(self.delta_signals) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        raw = torch.stack(self.delta_signals).mean()
        norm = torch.stack(self.norm_signals).mean()
        return raw, norm

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def _build_class_text_tokens(dataset, device=None):
    template = dataset.template if isinstance(dataset.template, str) else dataset.template[0]
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    return clip.tokenize(texts).to(resolve_device(device))


def _extract_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


@torch.no_grad()
def evaluate_ood_signal(
    args,
    clip_model,
    loader,
    class_text_tokens,
    signal_tracker: LoRASignalTracker,
):
    device = _device(args)
    clip_model.eval()
    total_raw = 0.0
    total_norm = 0.0
    total_samples = 0

    for batch in tqdm.tqdm(loader, total=len(loader)):
        images = _extract_images(batch).to(device)
        batch_size = images.shape[0]

        signal_tracker.reset()
        if args.encoder in ['text', 'both']:
            with _autocast(device):
                _ = clip_model.encode_text(class_text_tokens)
        if args.encoder in ['vision', 'both']:
            with _autocast(device):
                _ = clip_model.encode_image(images)
        else:
            with torch.no_grad():
                with _autocast(device):
                    _ = clip_model.encode_image(images)
        raw_signal, norm_signal = signal_tracker.get_signals(images.device)

        total_raw += raw_signal.item() * batch_size
        total_norm += norm_signal.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_raw / total_samples, total_norm / total_samples


@torch.no_grad()
def evaluate_lora_with_signal(args, clip_model, loader, dataset, logit_scale, signal_tracker: LoRASignalTracker):
    device = _device(args)
    clip_model.eval()
    class_text_tokens = _build_class_text_tokens(dataset, device=device)

    total_acc = 0.0
    total_raw = 0.0
    total_norm = 0.0
    total_samples = 0

    for images, target in tqdm.tqdm(loader, total=len(loader)):
        images, target = images.to(device), target.to(device)
        batch_size = images.shape[0]

        signal_tracker.reset()
        if args.encoder in ['text', 'both']:
            with _autocast(device):
                text_features = clip_model.encode_text(class_text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            with _autocast(device):
                text_features = clip_model.encode_text(class_text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if args.encoder in ['vision', 'both']:
            with _autocast(device):
                image_features = clip_model.encode_image(images)
        else:
            with torch.no_grad():
                with _autocast(device):
                    image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        cosine_similarity = logit_scale * image_features @ text_features.t()
        acc_batch = cls_acc(cosine_similarity, target)
        raw_signal, norm_signal = signal_tracker.get_signals(images.device)

        total_acc += acc_batch * batch_size
        total_raw += raw_signal.item() * batch_size
        total_norm += norm_signal.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    return total_acc / total_samples, total_raw / total_samples, total_norm / total_samples


def _save_lora_with_name(args, list_lora_layers, filename):
    old_name = args.filename
    args.filename = filename
    save_lora(args, list_lora_layers)
    args.filename = old_name


def run_lora_ood(
    args,
    clip_model,
    logit_scale,
    dataset,
    train_loader,
    val_loader,
    test_loader,
    ood_train_loaders: Dict[str, torch.utils.data.DataLoader],
    ood_val_loaders: Dict[str, torch.utils.data.DataLoader],
):
    device = _device(args)
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model, device=device)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader, device=device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device)

    if args.eval_only:
        args.dataset = "imagenet" if "imagenet" in args.dataset else args.dataset
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset, logit_scale)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return zs_acc, acc_test

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = _grad_scaler(device)
    signal_tracker = LoRASignalTracker(list_lora_layers)

    ood_names = list(ood_train_loaders.keys())
    ood_iters = {name: iter(loader) for name, loader in ood_train_loaders.items()}
    id_text_tokens = _build_class_text_tokens(dataset, device=device)

    best_score = float("-inf")
    best_metrics = {}
    val_interval = 5

    for epoch in range(args.epochs):
        clip_model.train()
        running_total = 0.0
        running_id_ce = 0.0
        running_ood_penalty = 0.0
        running_acc = 0.0
        running_id_signal = 0.0
        running_ood_signal = 0.0
        total_samples = 0
        num_batches = 0

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        for step, (images, target) in pbar:
            images, target = images.to(device), target.to(device)
            batch_size = images.shape[0]

            selected_ood = ood_names[step % len(ood_names)]
            ood_batch, ood_iters[selected_ood] = _next_batch(ood_iters[selected_ood], ood_train_loaders[selected_ood])
            ood_images = _extract_images(ood_batch).to(device)

            signal_tracker.reset()
            if args.encoder in ['text', 'both']:
                with _autocast(device):
                    text_features = clip_model.encode_text(id_text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                text_features = _maybe_half(textual_features.t(), device)

            if args.encoder in ['vision', 'both']:
                with _autocast(device):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with _autocast(device):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()
            id_ce = F.cross_entropy(cosine_similarity, target)
            id_signal_raw, id_signal_norm = signal_tracker.get_signals(images.device)

            signal_tracker.reset()
            if args.encoder in ['text', 'both']:
                with _autocast(device):
                    _ = clip_model.encode_text(id_text_tokens)
            if args.encoder in ['vision', 'both']:
                with _autocast(device):
                    _ = clip_model.encode_image(ood_images)
            else:
                with torch.no_grad():
                    with _autocast(device):
                        _ = clip_model.encode_image(ood_images)
            ood_signal_raw, ood_signal_norm = signal_tracker.get_signals(ood_images.device)

            if args.ood_term == 'normalized':
                ood_penalty = ood_signal_norm.pow(2)
                id_signal_for_log = id_signal_norm
                ood_signal_for_log = ood_signal_norm
            else:
                ood_penalty = ood_signal_raw.pow(2)
                id_signal_for_log = id_signal_raw
                ood_signal_for_log = ood_signal_raw

            loss = id_ce + args.ood_lambda * ood_penalty

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc_batch = cls_acc(cosine_similarity, target)
            running_total += loss.item() * batch_size
            running_id_ce += id_ce.item() * batch_size
            running_ood_penalty += ood_penalty.item() * batch_size
            running_acc += acc_batch * batch_size
            running_id_signal += id_signal_for_log.item() * batch_size
            running_ood_signal += ood_signal_for_log.item() * batch_size
            total_samples += batch_size
            num_batches += 1

            pbar.set_description(
                f"Epoch {epoch+1}/{args.epochs} "
                f"loss {running_total/max(total_samples,1):.4f} "
                f"id_ce {running_id_ce/max(total_samples,1):.4f} "
                f"ood {running_ood_penalty/max(total_samples,1):.4f} "
                f"acc {running_acc/max(total_samples,1):.4f} "
                f"id_sig {running_id_signal/max(total_samples,1):.4f} "
                f"ood_sig {running_ood_signal/max(total_samples,1):.4f}"
            )

        train_id_acc = running_acc / max(total_samples, 1)
        train_id_signal = running_id_signal / max(total_samples, 1)
        train_ood_signal = running_ood_signal / max(total_samples, 1)

        should_validate = ((epoch + 1) % val_interval == 0)
        if should_validate:
            val_id_acc, val_id_raw_signal, val_id_norm_signal = evaluate_lora_with_signal(
                args, clip_model, val_loader, dataset, logit_scale, signal_tracker
            )
            val_id_signal = val_id_norm_signal if args.ood_term == 'normalized' else val_id_raw_signal

            per_ood_val = {}
            ood_signals = []
            for ood_name, ood_val_loader in ood_val_loaders.items():
                ood_raw_signal, ood_norm_signal = evaluate_ood_signal(
                    args=args,
                    clip_model=clip_model,
                    loader=ood_val_loader,
                    class_text_tokens=id_text_tokens,
                    signal_tracker=signal_tracker,
                )
                ood_sig = ood_norm_signal if args.ood_term == 'normalized' else ood_raw_signal
                per_ood_val[ood_name] = {
                    "raw_signal": ood_raw_signal,
                    "norm_signal": ood_norm_signal,
                    "signal": ood_sig,
                }
                ood_signals.append(ood_sig)

            mean_ood_signal = sum(ood_signals) / max(len(ood_signals), 1)
            signal_gap = val_id_signal - mean_ood_signal

            if args.best_criterion == 'id_acc':
                score = val_id_acc
            elif args.best_criterion == 'signal_gap':
                score = signal_gap
            else:
                score = args.best_weight_id * val_id_acc + args.best_weight_gap * signal_gap

            print(
                f"[Epoch {epoch+1}] "
                f"train_id_acc={train_id_acc:.4f} train_id_signal={train_id_signal:.6f} train_ood_signal={train_ood_signal:.6f} "
                f"val_id_acc={val_id_acc:.4f} val_id_signal={val_id_signal:.6f} mean_ood_signal={mean_ood_signal:.6f} "
                f"signal_gap={signal_gap:.6f} score={score:.6f}"
            )
            for ood_name, metrics in per_ood_val.items():
                print(
                    f"  [OOD:{ood_name}] signal={metrics['signal']:.6f} "
                    f"raw_signal={metrics['raw_signal']:.6f} norm_signal={metrics['norm_signal']:.6f}"
                )

            if score > best_score:
                best_score = score
                best_metrics = {
                    "epoch": epoch + 1,
                    "val_id_acc": val_id_acc,
                    "signal_gap": signal_gap,
                    "mean_ood_signal": mean_ood_signal,
                    "score": score,
                }
                if args.save_path is not None:
                    _save_lora_with_name(args, list_lora_layers, "lora_weights_best")
        else:
            print(
                f"[Epoch {epoch+1}] "
                f"train_id_acc={train_id_acc:.4f} train_id_signal={train_id_signal:.6f} "
                f"train_ood_signal={train_ood_signal:.6f} (validation skipped; interval={val_interval})"
            )

    signal_tracker.close()

    acc_test = evaluate_lora(args, clip_model, test_loader, dataset, logit_scale)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    if len(best_metrics) > 0:
        print(f"Best checkpoint metrics: {best_metrics}")

    if args.save_path is not None:
        _save_lora_with_name(args, list_lora_layers, "lora_weights_last")
    return zs_acc, acc_test
