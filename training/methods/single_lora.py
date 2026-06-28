import os
from types import SimpleNamespace

import open_clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loralib.utils import apply_lora, get_lora_parameters, mark_only_lora_as_trainable
from training.config import DOMAIN_ORDER
from training.data import build_training_loader
from training.utils import focal_loss, get_device, resolve_expert_model_paths, set_random_seed
from training.validation import build_cross_domain_val_data, run_cross_domain_val


def _save_single_lora(weight_path, list_lora_layers, metadata):
    weights = {}
    params = metadata["params"]
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if "q" in params:
            layer_weights["q_proj"] = {
                "w_lora_A": layer.q_proj.w_lora_A.data.detach().cpu(),
                "w_lora_B": layer.q_proj.w_lora_B.data.detach().cpu(),
            }
        if "k" in params:
            layer_weights["k_proj"] = {
                "w_lora_A": layer.k_proj.w_lora_A.data.detach().cpu(),
                "w_lora_B": layer.k_proj.w_lora_B.data.detach().cpu(),
            }
        if "v" in params:
            layer_weights["v_proj"] = {
                "w_lora_A": layer.v_proj.w_lora_A.data.detach().cpu(),
                "w_lora_B": layer.v_proj.w_lora_B.data.detach().cpu(),
            }
        if "o" in params:
            layer_weights["proj"] = {
                "w_lora_A": layer.proj.w_lora_A.data.detach().cpu(),
                "w_lora_B": layer.proj.w_lora_B.data.detach().cpu(),
            }
        weights[f"layer_{i}"] = layer_weights

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save({"weights": weights, "metadata": metadata}, weight_path)


def _init_lora_from_pretrained_mean(list_lora_layers, checkpoint_paths):
    loaded = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "weights" in ckpt:
            loaded.append(ckpt["weights"])

    if not loaded:
        raise ValueError("No valid pretrained LoRA checkpoints found for initialization.")

    proj_to_attr = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "proj": "proj",
    }
    for layer_idx, layer in enumerate(list_lora_layers):
        layer_key = f"layer_{layer_idx}"
        for proj_name, attr_name in proj_to_attr.items():
            module = getattr(layer, attr_name, None)
            if module is None or not hasattr(module, "w_lora_A") or not hasattr(module, "w_lora_B"):
                continue

            tgt_a = module.w_lora_A.data
            tgt_b = module.w_lora_B.data
            sum_a = torch.zeros_like(tgt_a)
            sum_b = torch.zeros_like(tgt_b)
            cnt_a = torch.zeros_like(tgt_a)
            cnt_b = torch.zeros_like(tgt_b)

            for src_weights in loaded:
                layer_weights = src_weights.get(layer_key, {})
                if proj_name not in layer_weights:
                    continue
                src_a = layer_weights[proj_name]["w_lora_A"].to(device=tgt_a.device, dtype=tgt_a.dtype)
                src_b = layer_weights[proj_name]["w_lora_B"].to(device=tgt_b.device, dtype=tgt_b.dtype)
                r = min(src_a.shape[0], tgt_a.shape[0], src_b.shape[1], tgt_b.shape[1])
                if r <= 0:
                    continue
                sum_a[:r, :] += src_a[:r, :]
                sum_b[:, :r] += src_b[:, :r]
                cnt_a[:r, :] += 1
                cnt_b[:, :r] += 1

            valid_a = cnt_a > 0
            valid_b = cnt_b > 0
            if torch.any(valid_a):
                tgt_a[valid_a] = (sum_a[valid_a] / cnt_a[valid_a])
            if torch.any(valid_b):
                tgt_b[valid_b] = (sum_b[valid_b] / cnt_b[valid_b])


def train(args, weights_dir):
    device = get_device(args.device)
    set_random_seed(args.seed)

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="openai",
        force_quick_gelu=True,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # single_lora is trained as one shared LoRA over all training domains.
    lora_encoder = "both"
    lora_position = "all"
    lora_params = ["q", "k", "v"]
    lora_alpha = 1
    lora_args = SimpleNamespace(
        encoder=lora_encoder,
        position=lora_position,
        params=lora_params,
        r=int(args.single_lora_rank),
        alpha=lora_alpha,
        dropout_rate=args.dropout_rate,
        backbone=args.backbone,
    )
    list_lora_layers = apply_lora(lora_args, clip_model)
    if getattr(args, "single_lora_init_from_pretrained", False):
        domain_order = list(getattr(args, "domain_order", DOMAIN_ORDER))
        model_paths = resolve_expert_model_paths(
            args.model_dir,
            args.model_shots,
            args.seed,
            domain_order,
            explicit_paths=getattr(args, "expert_weight_paths", None),
        )
        load_paths = list(model_paths.values())
        if len(load_paths) == 0:
            raise ValueError(
                "single_lora_from_pretrained requested but no pretrained LoRA checkpoints were found "
                f"under {args.model_dir} for shots={args.shots}, seed={args.seed}."
            )
        _init_lora_from_pretrained_mean(list_lora_layers, load_paths)
        print(f"Initialized single LoRA from mean of {len(load_paths)} pretrained LoRA checkpoints.")

    clip_model = clip_model.to(device)
    clip_model.train()
    mark_only_lora_as_trainable(clip_model)

    train_loader, _, _ = build_training_loader(
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        data_root=args.data_root,
        subsample=args.subsample,
        shots=args.shots,
        dataset_mode=args.dataset_mode,
        prompt_source=args.prompt_source,
        descriptions_dir=args.descriptions_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        class_filter_path=getattr(args, "class_filter_path", None),
        domain_order=getattr(args, "domain_order", None),
    )
    val_loader = None
    val_prompts = None
    if getattr(args, "val_cross_domain", True):
        val_batch_size = getattr(args, "val_batch_size", None) or args.batch_size
        val_shots = getattr(args, "val_shots", 32)
        val_loader, val_prompts = build_cross_domain_val_data(
            preprocess_val=preprocess_val,
            data_root=args.data_root,
            subsample=args.subsample,
            batch_size=val_batch_size,
            num_workers=args.num_workers,
            val_shots=val_shots,
            seed=args.seed,
            shots=args.shots,
            domain_order=getattr(args, "domain_order", None),
        )

    optimizer = torch.optim.AdamW(
        get_lora_parameters(clip_model),
        lr=1e-3,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.8,
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, texts, domains in pbar:
            del domains  # Not needed for plain combined LoRA training.
            images = images.to(device)
            texts = texts.to(device)

            rdn_texts, _ = train_loader.dataset.random_class_tokens(tokenizer, k=args.random_k)
            rdn_texts = rdn_texts.to(device)
            all_texts = torch.cat((texts, rdn_texts), dim=0)
            unique_texts, inverse = torch.unique(all_texts, dim=0, return_inverse=True)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = clip_model.encode_text(unique_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = clip_model.logit_scale.exp() * (image_features @ text_features.T)
            targets = inverse[: images.shape[0]].to(device)

            if getattr(args, "use_focal_loss", False):
                loss = focal_loss(
                    logits,
                    targets,
                    alpha=getattr(args, "focal_alpha", 0.25),
                    gamma=getattr(args, "focal_gamma", 2.0),
                )
            else:
                loss = F.cross_entropy(logits, targets)

            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1
            running_loss += loss.item()
            running_acc += acc.item()
            pbar.set_description(
                f"Epoch {epoch} loss {running_loss/num_batches:.4f} "
                f"acc {running_acc/num_batches:.4f}"
            )
        if val_loader is not None:
            val_loss, val_acc = run_cross_domain_val(
                model=clip_model,
                tokenizer=tokenizer,
                val_loader=val_loader,
                prompts=val_prompts,
                device=device,
            )
            print(f"Epoch {epoch} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
        scheduler.step()
        clip_model.train()

    checkpoint_path = os.path.join(weights_dir, "single_lora.pt")
    metadata = {
        "r": int(args.single_lora_rank),
        "alpha": lora_alpha,
        "encoder": lora_encoder,
        "params": lora_params,
        "position": lora_position,
    }
    _save_single_lora(checkpoint_path, list_lora_layers, metadata)
    print(f"Saved single LoRA checkpoint to {checkpoint_path}")
