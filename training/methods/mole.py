import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip

from lora_med.utils import (
    get_gate_parameters,
    get_lora_parameters_from_mixtures,
    freeze_gate_networks,
    load_gate_nets,
    save_gate_nets,
    save_stage2_lora_experts,
    set_text_input_ids,
    clear_text_input_ids,
)
from training.config import DOMAIN_ORDER
from training.data import build_training_loader
from training.lora_loading import load_mole_mixtures_with_zero
from training.utils import set_random_seed, get_device, resolve_expert_model_paths, focal_loss
from training.validation import build_cross_domain_val_data, run_cross_domain_val
from lora_med.gate_hook import GateValueHook
from lora_med.logit_scalar import MEDLogitScalar, infer_clip_feature_dim, save_logit_scalar


def _set_requires_grad(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def train(args, weights_dir):
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
    expert_names = list(model_paths.keys())
    if "domain_ce" in getattr(args, "mole_mode", "") and expert_names != domain_order:
        missing = [domain for domain in domain_order if domain not in model_paths]
        unexpected = [domain for domain in expert_names if domain not in domain_order]
        details = []
        if missing:
            details.append(f"missing experts: {missing}")
        if unexpected:
            details.append(f"unexpected experts: {unexpected}")
        detail_msg = f" ({'; '.join(details)})" if details else ""
        raise ValueError(
            "MED domain_ce gate supervision requires one loaded expert LoRA checkpoint "
            "for each configured training domain. "
            f"Loaded experts: {expert_names}; training domains: {domain_order}{detail_msg}. "
            "Train the missing expert LoRAs or use an unsupervised gate-balance mode for subset smoke tests."
        )
    load_paths = list(model_paths.values())
    metadata_template = torch.load(load_paths[0], map_location="cpu").get("metadata", {})
    gate_type = "mole"
    mixtures, num_experts = load_mole_mixtures_with_zero(
        clip_model,
        backbone=args.backbone,
        load_paths=load_paths,
        dropout_rate=args.dropout_rate,
        gate_type=gate_type,
        simplify_mole=getattr(args, "simplify_mole", False),
    )

    clip_model = clip_model.to(device)
    _set_requires_grad(clip_model.parameters(), False)
    clip_model.train()
    clip_model._list_lora_mixtures = mixtures

    train_loader, _, domain_list = build_training_loader(
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
        domain_order=domain_order,
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
            domain_order=domain_order,
        )

    hook_manager = GateValueHook(expert_names=domain_list)
    hook_manager.register_hooks(mixtures, training=True)
    logit_scalar_mode = getattr(args, "logit_scalar", "standard")
    feature_dim = infer_clip_feature_dim(clip_model)
    logit_scalar_module = None
    if logit_scalar_mode in ("domain-wise", "input-wise", "affine-wise", "img-wise"):
        logit_scalar_module = MEDLogitScalar(
            mode=logit_scalar_mode,
            feature_dim=feature_dim,
            num_domains=len(domain_list),
        ).to(device)

    stage2_mode = bool(getattr(args, "mole_stage2", False))
    if stage2_mode:
        gate_dir = getattr(args, "gate_dir", None)
        if gate_dir:
            gate_path = os.path.join(os.path.abspath(gate_dir), "meta_net.pt")
            if os.path.isfile(gate_path):
                class _LoadArgs:
                    pass
                largs = _LoadArgs()
                largs.gate_filename = gate_path.replace(".pt", "")
                load_gate_nets(largs, mixtures)
                print(f"Loaded MED gating networks from {gate_path}")
            else:
                raise FileNotFoundError(f"Gate weights not found: {gate_path} (--gate_dir={gate_dir})")
        freeze_gate_networks(mixtures)
        train_params = get_lora_parameters_from_mixtures(mixtures, params=args.params)
        if len(train_params) == 0:
            raise ValueError("No LoRA parameters found for stage-2 MED training.")
        _set_requires_grad(train_params, True)
        if logit_scalar_module is not None:
            train_params = list(train_params) + list(logit_scalar_module.parameters())
        optimizer = torch.optim.Adam(train_params, lr=args.lr)
        print("MED stage-2 enabled: gating networks frozen, training LoRA parameters.")
    else:
        gate_params = get_gate_parameters(args, mixtures)
        _set_requires_grad(gate_params, True)
        if logit_scalar_module is not None:
            gate_params = list(gate_params) + list(logit_scalar_module.parameters())
        optimizer = torch.optim.Adam(gate_params, lr=args.lr)

    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(args.epochs):
        running_loss = running_clip = running_balance = running_acc = running_dom_acc = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, texts, domains in pbar:
            images = images.to(device)
            texts = texts.to(device)
            domains = domains.to(device)

            rdn_texts, rdn_domains = train_loader.dataset.random_class_tokens(tokenizer, k=args.random_k)
            rdn_texts = rdn_texts.to(device)
            rdn_domains = torch.tensor(rdn_domains, device=device)

            all_texts = torch.cat((texts, rdn_texts), dim=0)
            txt_domains = torch.cat((domains, rdn_domains), dim=0)

            unique_texts, inverse = torch.unique(all_texts, dim=0, return_inverse=True)
            unique_txt_domain = torch.full((unique_texts.size(0),), -1, device=device, dtype=torch.long)
            for batch_idx, unique_idx in enumerate(inverse):
                unique_txt_domain[unique_idx] = txt_domains[batch_idx]

            image_features = clip_model.encode_image(images)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            set_text_input_ids(mixtures, unique_texts)
            text_features = clip_model.encode_text(unique_texts)
            clear_text_input_ids(mixtures)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

            target = inverse[: images.shape[0]].to(device)
            img_domain_for_hook = domains - 1
            txt_domain_for_hook = unique_txt_domain.clone()
            valid_mask = txt_domain_for_hook >= 0
            txt_domain_for_hook[valid_mask] = txt_domain_for_hook[valid_mask] - 1

            if logit_scalar_mode == "standard":
                logits = clip_model.logit_scale.exp() * (image_features_norm @ text_features_norm.T)
                logit_aux_loss = logits.new_tensor(0.0)
            else:
                logits, logit_aux_loss = logit_scalar_module(
                    image_features=image_features,
                    text_features=text_features,
                    image_features_norm=image_features_norm,
                    text_features_norm=text_features_norm,
                    image_domain=img_domain_for_hook if logit_scalar_mode == "domain-wise" else None,
                    text_domain=txt_domain_for_hook if logit_scalar_mode == "domain-wise" else None,
                )
            if getattr(args, "use_focal_loss", False):
                clip_loss = focal_loss(logits, target, alpha=getattr(args, "focal_alpha", 0.25), gamma=getattr(args, "focal_gamma", 2.0))
            else:
                clip_loss = F.cross_entropy(logits, target)
            
            # Image classification accuracy (CLIP matching)
            clip_pred = logits.argmax(dim=-1)
            clip_acc = (clip_pred == target).float().mean()

            # Convert domain indices from 1-10 (with zero expert) to 0-9 (for hook which expects real expert indices)
            # The hook expects indices matching the expert_names list (real experts only, no zero expert)
            balance_mode = args.mole_mode
            img_domain_for_loss = img_domain_for_hook
            txt_domain_for_loss = txt_domain_for_hook

            balanced_loss, msg = hook_manager.get_balance_loss(
                img_domain=img_domain_for_loss,
                txt_domain=txt_domain_for_loss,
                balance_reg=1,
                domain_reg=0.1,
                mode=balance_mode,
            )
            loss = clip_loss + args.alpha * balanced_loss + logit_aux_loss

            # Domain/expert prediction accuracy from gate logits
            # Gate logits are stored as (gate, logits) tuples in gate_values
            img_gate_logits_list = []
            for key in hook_manager.gate_values.keys():
                if "image" in key:
                    # Get the latest gate values for this layer
                    gate, logits = hook_manager.gate_values[key][-1]
                    # Handle both (B, N) and (B, L, N) shapes for token-wise gates
                    if len(logits.shape) == 3:
                        # Token-wise: (B, L, N) -> average over sequence -> (B, N)
                        logits = logits.mean(dim=1)
                    img_gate_logits_list.append(logits)  # [B, K]
            
            if len(img_gate_logits_list) > 0:
                # Average logits across layers, then take argmax
                avg_logits = torch.stack(img_gate_logits_list, dim=0).mean(dim=0)  # [B, K]
                pred_experts = avg_logits.argmax(dim=-1)  # [B] - indices 0-9 (real experts)
                # Convert back to domain indices (0-9 -> 1-10) for comparison
                pred_domains = pred_experts + 1
                dom_acc = (pred_domains == domains).float().mean()
            else:
                dom_acc = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1
            running_loss += loss.item()
            running_clip += clip_loss.item()
            running_balance += balanced_loss.item()
            running_acc += clip_acc.item()
            running_dom_acc += dom_acc.item()
            pbar.set_description(
                f"Epoch {epoch} loss {running_loss/num_batches:.4f} "
                f"clip {running_clip/num_batches:.4f} bal {running_balance/num_batches:.4f} "
                f"acc {running_acc/num_batches:.4f} dom_acc {running_dom_acc/num_batches:.4f}"
            )
            hook_manager.clear_step()

        # Save latest checkpoint every epoch.
        os.makedirs(weights_dir, exist_ok=True)
        class _SaveArgs:
            pass
        sargs_latest = _SaveArgs()
        sargs_latest.params = ["q", "k", "v", "o"]
        sargs_latest.gate_filename = os.path.join(weights_dir, "meta_net_latest")
        save_gate_nets(sargs_latest, mixtures)
        if logit_scalar_module is not None:
            save_logit_scalar(weights_dir, logit_scalar_module, filename="logit_scalar_latest.pt")

        epoch_train_loss = running_loss / max(num_batches, 1)
        save_best = False
        best_reason = None
        if stage2_mode:
            stage2_root = os.path.join(weights_dir, "stage2_lora")
            base_idx = (num_experts - 1) if getattr(args, "include_base_clip", True) else None
            save_stage2_lora_experts(
                stage2_root=stage2_root,
                list_lora_mixtures=mixtures,
                expert_names=expert_names,
                params=args.params,
                base_expert_idx=base_idx,
                shots=args.model_shots,
                seed=args.seed,
                metadata_template=metadata_template,
            )
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
            if logit_scalar_module is not None:
                save_logit_scalar(weights_dir, logit_scalar_module, filename="logit_scalar.pt")
            print(f"Updated best checkpoint ({best_reason})")
        clip_model.train()

    print(f"Saved MED gates to {os.path.join(weights_dir, 'meta_net.pt')}")
    if stage2_mode:
        print(f"Saved stage-2 LoRA experts to {os.path.join(weights_dir, 'stage2_lora')}")
