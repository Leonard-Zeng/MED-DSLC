import os, itertools
from importlib.metadata import metadata

import open_clip
import torch
import torch.nn as nn
# from tqdm.auto import tqdm
import torch.nn.functional as F

from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt

from .text_layers import TextMEDLayer, TextPlainMultiheadAttentionMED
from .image_layers import ImageMEDLayer, ImagePlainMultiheadAttentionMED

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, TextMEDLayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def get_lora_parameters_from_mixtures(list_lora_mixtures, params=("q", "k", "v", "o")):
    """Collect LoRA parameters from MED mixture layers."""
    collected = []
    for layer in list_lora_mixtures:
        for param_name in params:
            proj = layer.proj if param_name == "o" else getattr(layer, f"{param_name}_proj", None)
            if proj is None:
                continue
            for name, param in proj.named_parameters():
                if "lora_" in name:
                    collected.append(param)
    return collected


def freeze_gate_networks(list_lora_mixtures):
    """Freeze per-layer gate networks attached to mixture layers."""
    for layer in list_lora_mixtures:
        gate = getattr(layer, "gate_net", None)
        if gate is None:
            continue
        for param in gate.parameters():
            param.requires_grad = False


def save_stage2_lora_experts(
    stage2_root: str,
    list_lora_mixtures,
    expert_names: List[str],
    params: List[str],
    base_expert_idx: int | None = None,
    shots: int | None = None,
    seed: int | None = None,
    metadata_template: dict | None = None,
):
    """Save stage-2 MED LoRA experts in per-domain lora_weights.pt format.

    The output layout intentionally differs from pretrained source paths:
    stage2_root/<domain>/<shots>shots/seed<seed>/lora_weights.pt
    """
    if len(list_lora_mixtures) == 0:
        raise ValueError("No mixture layers to save.")
    first_layer = list_lora_mixtures[0]
    total_experts = int(first_layer.num_experts)

    trainable_indices = [idx for idx in range(total_experts) if idx != base_expert_idx]
    if len(trainable_indices) < len(expert_names):
        raise ValueError(
            f"Not enough expert slots ({len(trainable_indices)}) for expert names ({len(expert_names)})."
        )
    if len(trainable_indices) != len(expert_names):
        # Keep deterministic mapping by truncating extra indices when present.
        trainable_indices = trainable_indices[: len(expert_names)]

    os.makedirs(stage2_root, exist_ok=True)
    for expert_name, expert_idx in zip(expert_names, trainable_indices):
        weights = {}
        for layer_idx, layer in enumerate(list_lora_mixtures):
            layer_weights = {}
            for param in params:
                if param == "o":
                    proj = getattr(layer, "proj", None)
                    if proj is None:
                        continue
                    key_a = f"w_lora_A_{expert_idx}"
                    key_b = f"w_lora_B_{expert_idx}"
                    if hasattr(proj, key_a) and hasattr(proj, key_b):
                        layer_weights["proj"] = {
                            "w_lora_A": getattr(proj, key_a).data.detach().cpu(),
                            "w_lora_B": getattr(proj, key_b).data.detach().cpu(),
                        }
                else:
                    proj = getattr(layer, f"{param}_proj", None)
                    if proj is None:
                        continue
                    key_a = f"w_lora_A_{expert_idx}"
                    key_b = f"w_lora_B_{expert_idx}"
                    if hasattr(proj, key_a) and hasattr(proj, key_b):
                        layer_weights[f"{param}_proj"] = {
                            "w_lora_A": getattr(proj, key_a).data.detach().cpu(),
                            "w_lora_B": getattr(proj, key_b).data.detach().cpu(),
                        }
            weights[f"layer_{layer_idx}"] = layer_weights

        rank = int(first_layer.r[expert_idx]) if hasattr(first_layer, "r") else int(first_layer.r[0])
        alpha = float(first_layer.lora_alpha[expert_idx]) if hasattr(first_layer, "lora_alpha") else 1.0
        metadata = {
            "r": rank,
            "alpha": alpha,
            "encoder": (metadata_template or {}).get("encoder", "both"),
            "params": (metadata_template or {}).get("params", list(params)),
            "position": (metadata_template or {}).get("position", "all"),
        }
        save_dir = os.path.join(stage2_root, expert_name)
        if shots is not None:
            save_dir = os.path.join(save_dir, f"{int(shots)}shots")
        if seed is not None:
            save_dir = os.path.join(save_dir, f"seed{int(seed)}")
        save_path = os.path.join(save_dir, "lora_weights.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"weights": weights, "metadata": metadata}, save_path)

    print(f"Saved stage-2 LoRA experts to {stage2_root}")


def apply_lora_mixture(
        clip_model, backbone:str, encoder:str, position:str, params:list[str], r:list[int], alpha:list[float], dropout_rate:float, gate_type:str = "mole", top_k_expert: int = 0, simplify_mole: bool = False):
    """
    :param encoder: "text", "vision" or "both"
    :param position: choices=['bottom','mid','up','half-up','half-bottom','all','top3'], i.e. 'up'
    :param params: i.e. ["q", "v"]
    :param r: i.e. [2, 4, 8]
    :param alpha: i.e. [0.1, 0.2, 0.7]
    :param gate_type: "mole" for softmax expert gating, "sigmoid" for per-rank sigmoid gating, or "phatgoose"
    """
    list_lora_layers = []
    assert len(r) == len(alpha)
    num_experts = len(r)
    if encoder == 'text' or encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = TextPlainMultiheadAttentionMED(
                            existing_mha=submodule,
                            num_experts=num_experts,
                            enable_lora=params,
                            r=r,
                            lora_alpha=alpha,
                            dropout_rate=dropout_rate,
                            backbone=backbone,
                            gate_type=gate_type,
                            top_k_expert=top_k_expert,
                            simplify_mole=simplify_mole,
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if encoder == 'vision' or encoder == 'both':
        indices = INDEX_POSITIONS_VISION[backbone][position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = ImagePlainMultiheadAttentionMED(
                            existing_mha=submodule,
                            num_experts=num_experts,
                            enable_lora=params,
                            r=r,
                            lora_alpha=alpha,
                            dropout_rate=dropout_rate,
                            backbone=backbone,
                            gate_type=gate_type,
                            top_k_expert=top_k_expert,
                            simplify_mole=simplify_mole,
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora_mixture(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        for expert_idx in range(layer.num_experts):
            for param in args.params:
                if param == 'o':
                    layer_weights[f'proj'] = {
                        f'w_lora_A_{expert_idx}': eval(f"layer.proj.w_lora_A_{expert_idx}.data"),
                        f'w_lora_B_{expert_idx}': eval(f"layer.proj.w_lora_B_{expert_idx}.data"),
                    }
                else:
                    layer_weights[f'{param}_proj'] = {
                        f'w_lora_A_{expert_idx}': eval(f"layer.{param}_proj.w_lora_A_{expert_idx}.data"),
                        f'w_lora_B_{expert_idx}': eval(f"layer.{param}_proj.w_lora_B_{expert_idx}.data"),
                    }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'num_experts': len(args.r),
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()

    save_path = f'{args.filename}'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora_mixture(args, list_lora_layers):
    # to manage names like ViT-B/16
    load_path = f'{args.filename}'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    # if metadata['alpha'] != args.alpha:
    #     raise ValueError(
    #         f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    num_experts = metadata['num_experts']
    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        for expert_idx in range(num_experts):
            for param in args.params:
                if param == 'o':
                    if f'proj' in layer_weights:
                        eval(f"layer.proj.w_lora_A_{expert_idx}").data.copy_(
                            layer_weights[f'proj'][f'w_lora_A_{expert_idx}'])
                        eval(f"layer.proj.w_lora_B_{expert_idx}").data.copy_(
                            layer_weights[f'proj'][f'w_lora_B_{expert_idx}'])
                else:
                    if f'{param}_proj' in layer_weights:
                        eval(f"layer.{param}_proj.w_lora_A_{expert_idx}").data.copy_(
                            layer_weights[f'{param}_proj'][f'w_lora_A_{expert_idx}'])
                        eval(f"layer.{param}_proj.w_lora_B_{expert_idx}").data.copy_(
                            layer_weights[f'{param}_proj'][f'w_lora_B_{expert_idx}'])

    print(f'LoRA weights loaded from {load_path}')

'''load_paths are weights saved by loralib'''
def load_loras_as_mixtures(
        clip_model, backbone:str, load_paths: list[str],
        alpha_in: list[float] = None, dropout_rate=0, shared_expert_rank = 0
):
    r = []
    alpha = [] if alpha_in is None else alpha_in
    dropout_rate = dropout_rate
    encoder = None
    params = None
    position = None

    num_experts = len(load_paths)
    assert (len(alpha) > 0 and len(alpha) == num_experts) or (alpha == 0)

    for i in range(num_experts):
        load_path = load_paths[i]
        loaded_data = torch.load(load_path)
        metadata = loaded_data['metadata']
        r.append(metadata['r'])
        if alpha_in is None:
            alpha.append(metadata['alpha'])
        if encoder is None:
            encoder = metadata['encoder']
        else:
            assert encoder == metadata['encoder']
        if params is None:
            params = metadata['params']
        else:
            assert sorted(params) == sorted(metadata['params'])
        if position is None:
            position = metadata['position']
        else:
            assert position == metadata['position']

    if shared_expert_rank > 0:
        r.append(shared_expert_rank)
        alpha = [1/len(r)]*len(r)

    list_lora_mixtures = apply_lora_mixture(clip_model, backbone, encoder, position, params, r, alpha, dropout_rate)

    # loading the lora from load_paths
    for expert_idx in range(num_experts):
        load_path = load_paths[expert_idx]
        loaded_data = torch.load(load_path)
        weights = loaded_data['weights']
        for i, layer in enumerate(list_lora_mixtures):
            layer_weights = weights[f'layer_{i}']
            for param in params:
                if param == 'o':
                    if f'proj' in layer_weights:
                        eval(f"layer.proj.w_lora_A_{expert_idx}").data.copy_(
                            layer_weights[f'proj'][f'w_lora_A'])
                        eval(f"layer.proj.w_lora_B_{expert_idx}").data.copy_(
                            layer_weights[f'proj'][f'w_lora_B'])
                else:
                    if f'{param}_proj' in layer_weights:
                        eval(f"layer.{param}_proj.w_lora_A_{expert_idx}").data.copy_(
                            layer_weights[f'{param}_proj'][f'w_lora_A'])
                        eval(f"layer.{param}_proj.w_lora_B_{expert_idx}").data.copy_(
                            layer_weights[f'{param}_proj'][f'w_lora_B'])
        print(f'LoRA weights loaded from {load_path} as expert {expert_idx} with alpha = {alpha[expert_idx]}')
    print(f'LoRA weights are loaded as total {len(load_paths)} experts')

    shared_expert = []
    if shared_expert_rank > 0:
        for i, layer in enumerate(list_lora_mixtures):
            for param in params:
                if param == 'o':
                    if hasattr(layer, 'proj.w_lora_A'):
                        shared_expert.append(eval(f"layer.proj.w_lora_A_{num_experts}"))
                        shared_expert.append(eval(f"layer.proj.w_lora_B_{num_experts}"))
                else:
                    if hasattr(layer, f'{param}_proj.w_lora_A'):
                        shared_expert.append(eval(f"layer.{param}_proj.w_lora_A_{num_experts}"))
                        shared_expert.append(eval(f"layer.{param}_proj.w_lora_B_{num_experts}"))
    # list_lora_mixtures, return the number of experts, last lora_experts
    return list_lora_mixtures, len(load_paths), shared_expert

def set_text_prompts(
        list_lora_mixtures,
        text_feats,
):
    for layer in list_lora_mixtures:
        if isinstance(layer, ImagePlainMultiheadAttentionMED) or isinstance(layer, ImageMEDLayer):
            layer.set_prompts_feats(text_feats)


def set_text_input_ids(list_lora_mixtures, input_ids: torch.Tensor):
    """Attach current text token ids for EOS-aware gating."""
    for layer in list_lora_mixtures:
        if isinstance(layer, TextPlainMultiheadAttentionMED):
            layer.input_ids_override = input_ids


def clear_text_input_ids(list_lora_mixtures):
    """Clear transient text token ids after text encoding."""
    for layer in list_lora_mixtures:
        if isinstance(layer, TextPlainMultiheadAttentionMED):
            layer.input_ids_override = None

def save_gate_nets(
        args,
        list_lora_mixtures,
):
    gate_state_dict = {}
    for i, layer in enumerate(list_lora_mixtures):
        # for param in args.params:
        #     if param == 'o':
        #         if hasattr(layer.proj, 'gate_net'):
        #             gate_state_dict[f"layer_{i}_{param}_proj"] = eval(f"layer.proj.gate_net").state_dict()
        #     else:
        #         if hasattr(eval(f"layer.{param}_proj"), 'gate_net'):
        #             gate_state_dict[f"layer_{i}_{param}_proj"] = eval(f"layer.{param}_proj.gate_net").state_dict()
        if hasattr(layer, 'gate_net'):
            gate_state_dict[f"layer_{i}"] = eval(f"layer.gate_net").state_dict()

    # to manage names like ViT-B/16
    save_path = f'{args.gate_filename}.pt'
    torch.save(gate_state_dict, save_path)
    print(f'LoRA weights saved to {save_path}')

def load_gate_nets(
        args,
        list_lora_mixtures,
):
    gate_state_dict = torch.load(f"{args.gate_filename}.pt")
    for i, layer in enumerate(list_lora_mixtures):
        if hasattr(layer, 'gate_net'):
            gate_net = eval(f"layer.gate_net")
            saved_state = gate_state_dict[f"layer_{i}"]
            # Filter state_dict to only include keys that exist in the model
            # This handles cases where saved weights have different activation (tau vs no tau)
            model_state = gate_net.state_dict()
            filtered_state = {k: v for k, v in saved_state.items() if k in model_state}
            gate_net.load_state_dict(filtered_state, strict=False)
            # for param in args.params:
            # if param == 'o':
            #     if hasattr(layer.proj, 'gate_net'):
            #         eval(f"layer.proj.gate_net").load_state_dict(gate_state_dict[f"layer_{i}_{param}_proj"])
            # else:
            #     if hasattr(eval(f"layer.{param}_proj"), 'gate_net'):
            #         eval(f"layer.{param}_proj.gate_net").load_state_dict(gate_state_dict[f"layer_{i}_{param}_proj"])


def get_gate_parameters(args, list_lora_mixtures):
    params = []
    for i, layer in enumerate(list_lora_mixtures):
        # for param in args.params:
        #     if param == 'o':
        #         if hasattr(eval("layer.proj"), 'gate_net'):
        #             params += list(eval(f"layer.proj.gate_net").parameters())
        #             # print(eval(f"layer.proj.gate_net").parameters())
        #     else:
        #         if hasattr(layer, param + '_proj') and hasattr(eval(f"layer.{param}_proj"), "gate_net"):
        #             params += list(eval(f"layer.{param}_proj.gate_net").parameters())
        #             # print(eval(f"layer.{param}_proj.gate_net").parameters())
        if hasattr(layer, 'gate_net'):
            params += list(layer.gate_net.parameters())
    return params
