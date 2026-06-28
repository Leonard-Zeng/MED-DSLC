import os

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from lora_med.utils import clear_text_input_ids, save_gate_nets, set_text_input_ids
from clip_lora_datasets import build_dataset
from clip_lora_datasets.utils import build_data_loader
from training.config import DOMAIN_ORDER
from training.lora_loading import load_mole_mixtures_with_zero
from training.utils import focal_loss, get_device, resolve_expert_model_paths, set_random_seed
from training.validation import build_cross_domain_val_data, run_cross_domain_val


def _collect_phatgoose_u_params(mixtures):
    params = []
    for layer in mixtures:
        gate_net = getattr(layer, "gate_net", None)
        if gate_net is None:
            continue
        linear = getattr(gate_net, "linear", None)
        if linear is None or not hasattr(linear, "weight"):
            continue
        linear.weight.requires_grad = True
        params.append(linear.weight)
    return params


def _set_active_expert(mixtures, expert_idx: int | None):
    for layer in mixtures:
        gate_net = getattr(layer, "gate_net", None)
        if gate_net is None:
            continue
        if hasattr(gate_net, "active_expert_idx"):
            gate_net.active_expert_idx = expert_idx


def _build_domain_train_loaders_main_style(args, preprocess_train):
    """Build per-domain train loaders with the same pattern used in main.py."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    domain_datasets = {}
    domain_loaders = {}
    active_domains = list(getattr(args, "domain_order", DOMAIN_ORDER))
    for domain_name in active_domains:
        dataset = build_dataset(
            domain_name,
            args.data_root,
            args.shots,
            preprocess_train,
            subsample=args.subsample,
        )
        loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=args.batch_size,
            tfm=train_transform,
            is_train=True,
            shuffle=True,
            num_workers=8,
        )
        domain_datasets[domain_name] = dataset
        domain_loaders[domain_name] = loader
    return domain_datasets, domain_loaders


def train(args, weights_dir):
    """Train phatgoose with in-domain data and u-only optimization."""
    device = get_device(args.device)
    set_random_seed(args.seed)
    if not hasattr(args, "params"):
        args.params = ["q", "k", "v", "o"]

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="openai",
        force_quick_gelu=True,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    domain_order = list(getattr(args, "domain_order", DOMAIN_ORDER))
    model_paths = resolve_expert_model_paths(
        args.model_dir,
        args.model_shots,
        args.seed,
        domain_order,
        explicit_paths=getattr(args, "expert_weight_paths", None),
    )
    if len(model_paths) == 0:
        raise ValueError(f"No LoRA weights found under {args.model_dir}")
    load_paths = list(model_paths.values())

    mixtures, num_experts = load_mole_mixtures_with_zero(
        clip_model,
        backbone=args.backbone,
        load_paths=load_paths,
        dropout_rate=args.dropout_rate,
        include_base_clip=False,
        gate_type="phatgoose",
        top_k_expert=0,
        simplify_mole=getattr(args, "simplify_mole", False),
    )

    clip_model = clip_model.to(device)
    clip_model._list_lora_mixtures = mixtures
    clip_model.train()

    # Freeze everything first, then unfreeze only gate u parameters.
    for param in clip_model.parameters():
        param.requires_grad = False
    train_params = _collect_phatgoose_u_params(mixtures)
    if len(train_params) == 0:
        raise ValueError("No phatgoose gate parameters found for optimization.")
    optimizer = torch.optim.Adam(train_params, lr=args.lr)

    domain_datasets, domain_loaders = _build_domain_train_loaders_main_style(args, preprocess_train)

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
            domain_order=domain_order,
        )

    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    expert_name_to_idx = {name: i for i, name in enumerate(model_paths.keys())}
    print(f"Training phatgoose with {num_experts} experts (no base expert).")
    print("Only gate vectors u are trainable; CLIP and LoRA are frozen.")

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        active_epoch_domains = [d for d in domain_order if d in domain_loaders]
        for domain_name in active_epoch_domains:
            expert_idx = expert_name_to_idx.get(domain_name, None)
            if expert_idx is None:
                continue
            _set_active_expert(mixtures, expert_idx)
            loader = domain_loaders[domain_name]
            dataset = domain_datasets[domain_name]
            template = dataset.template if isinstance(dataset.template, str) else dataset.template[0]
            prompts = [template.format(cname.replace("_", " ")) for cname in dataset.classnames]
            text_tokens = tokenizer(prompts).to(device)
            pbar = tqdm(loader, desc=f"Epoch {epoch} [{domain_name}]")
            for batch in pbar:
                images, targets = batch[0], batch[1]
                images = images.to(device)
                targets = targets.to(device)
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                set_text_input_ids(mixtures, text_tokens)
                text_features = clip_model.encode_text(text_tokens)
                clear_text_input_ids(mixtures)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = clip_model.logit_scale.exp() * (image_features @ text_features.T)

                if getattr(args, "use_focal_loss", False):
                    loss = focal_loss(
                        logits,
                        targets,
                        alpha=getattr(args, "focal_alpha", 0.25),
                        gamma=getattr(args, "focal_gamma", 2.0),
                    )
                else:
                    loss = F.cross_entropy(logits, targets)

                pred = logits.argmax(dim=-1)
                acc = (pred == targets).float().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_batches += 1
                running_loss += loss.item()
                running_acc += acc.item()
                pbar.set_description(
                    f"Epoch {epoch} [{domain_name}] loss {running_loss/max(num_batches,1):.4f} "
                    f"acc {running_acc/max(num_batches,1):.4f}"
                )
        _set_active_expert(mixtures, None)

        os.makedirs(weights_dir, exist_ok=True)

        class _SaveArgs:
            pass

        sargs_latest = _SaveArgs()
        sargs_latest.params = ["q", "k", "v", "o"]
        sargs_latest.gate_filename = os.path.join(weights_dir, "meta_net_latest")
        save_gate_nets(sargs_latest, mixtures)

        epoch_train_loss = running_loss / max(num_batches, 1)
        save_best = False
        best_reason = None
        if val_loader is not None:
            val_loss, val_acc = run_cross_domain_val(
                model=clip_model,
                tokenizer=tokenizer,
                val_loader=val_loader,
                prompts=val_prompts,
                device=device,
                before_encode_text=lambda text_tokens: set_text_input_ids(mixtures, text_tokens),
                after_encode_text=lambda: clear_text_input_ids(mixtures),
            )
            print(f"Epoch {epoch} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
            if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = float(val_acc)
                best_val_loss = float(val_loss)
                save_best = True
                best_reason = f"val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}"
        elif epoch_train_loss < best_train_loss:
            best_train_loss = float(epoch_train_loss)
            save_best = True
            best_reason = f"train_loss={best_train_loss:.4f}"

        if save_best:
            bargs = _SaveArgs()
            bargs.params = ["q", "k", "v", "o"]
            bargs.gate_filename = os.path.join(weights_dir, "meta_net")
            save_gate_nets(bargs, mixtures)
            print(f"Updated best checkpoint ({best_reason})")

        clip_model.train()

    print(f"Saved phatgoose gates to {os.path.join(weights_dir, 'meta_net.pt')}")
