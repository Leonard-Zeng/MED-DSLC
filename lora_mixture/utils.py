import os, itertools
from importlib.metadata import metadata

import open_clip
import torch
import torch.nn as nn
# from tqdm.auto import tqdm
import torch.nn.functional as F

from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt

from .layers import LoRAMixtureLayer, PlainMultiheadAttentionLoRAMixture

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
            if isinstance(m, LoRAMixtureLayer) and \
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


def apply_lora_mixture(
        clip_model, backbone:str, encoder:str, position:str, params:list[str], r:list[int], alpha:list[float], dropout_rate:float):
    """
    :param encoder: "text", "vision" or "both"
    :param position: choices=['bottom','mid','up','half-up','half-bottom','all','top3'], i.e. 'up'
    :param params: i.e. ["q", "v"]
    :param r: i.e. [2, 4, 8]
    :param alpha: i.e. [0.1, 0.2, 0.7]
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
                        new_multi_head_lora = PlainMultiheadAttentionLoRAMixture(
                            existing_mha=submodule,
                            num_experts=num_experts,
                            enable_lora=params,
                            r=r,
                            lora_alpha=alpha,
                            dropout_rate=dropout_rate
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
                        new_multi_head_lora = PlainMultiheadAttentionLoRAMixture(
                            existing_mha=submodule,
                            num_experts=num_experts,
                            enable_lora=params,
                            r=r,
                            lora_alpha=alpha,
                            dropout_rate=dropout_rate
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
        clip_model, backbone:str, load_paths: list[str], alpha_in: list[float] = None, dropout_rate=0, include_baseclip=False
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

    if include_baseclip:
        num_experts = num_experts + 1
        r = [1] + r
        alpha = [1/num_experts] * num_experts

    list_lora_mixtures = apply_lora_mixture(clip_model, backbone, encoder, position, params, r, alpha, dropout_rate)

    # loading the lora from load_paths
    for expert_idx in range(num_experts):
        # The expert_idxfirst one need to be base clip
        if include_baseclip and expert_idx == 0:
            continue

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
    return list_lora_mixtures, len(load_paths) # return the number of experts

def update_alpha_lora_mixtures(
        list_lora_mixtures,
        alphas: list[float]
):
    for layer in list_lora_mixtures:
        for expert_idx, alpha in enumerate(alphas):
            layer.update_alpha(alpha, expert_idx)

def set_alpha_batch_lora_mixtures(
        list_lora_mixtures,
        alpha_batch,
):
    for layer in list_lora_mixtures:
        layer.set_alpha_batch(alpha_batch)

def unset_alpha_batch_lora_mixtures(
        list_lora_mixtures
):
    for layer in list_lora_mixtures:
        layer.unset_alpha_batch()
        

def apply_svd_lora_mixtures(
    list_lora_mixtures,
    train_mode = False,
    svd_kind = "softplus",
    svd_beta = 2.0,
):
    print("Applying SVD to each expert")
    for layer in list_lora_mixtures:
        layer.apply_svd(
            train_mode=train_mode,
            svd_transform = svd_kind,
            svd_beta = svd_beta
        )
    print("SVD has applied")


from typing import List

def save_lora_mixture_separate(
    list_lora_layers: list,
    expert_names: List[str],       
    filename: str,                 
    params: List[str],             
    r: List[int],                                 
    alpha: List[float],             
    encoder: str,                   
    position: str,                  
    keep_on_gpu: bool = False       
) -> None:
    num_experts = len(expert_names)
    assert len(r)     == num_experts, "r must have one entry per expert"
    assert len(alpha) == num_experts, "alpha must have one entry per expert"

    base_dir  = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)


    for expert_idx, expert_name in enumerate(expert_names):
        weights = {}
        for layer_idx, layer in enumerate(list_lora_layers):
            layer.svd_update_BA()
            layer_dict = {}
            for p in params:
                if p == "o":
                    proj_mod = layer.proj
                    key = "proj"
                else:
                    proj_mod = getattr(layer, f"{p}_proj")
                    key = f"{p}_proj"

                layer_dict[key] = {
                    "w_lora_A": getattr(proj_mod, f"w_lora_A_{expert_idx}").detach().cpu()
                                 if not keep_on_gpu else getattr(proj_mod, f"w_lora_A_{expert_idx}").detach(),
                    "w_lora_B": getattr(proj_mod, f"w_lora_B_{expert_idx}").detach().cpu()
                                 if not keep_on_gpu else getattr(proj_mod, f"w_lora_B_{expert_idx}").detach(),
                }

            weights[f"layer_{layer_idx}"] = layer_dict

        metadata = {
            "expert_name": expert_name,
            "expert_idx":  expert_idx,
            "r":           r[expert_idx],
            "alpha":       alpha[expert_idx],
            "encoder":     encoder,
            "params":      params,
            "position":    position,
        }

        save_data = {"weights": weights, "metadata": metadata}
        save_path = os.path.join(base_dir, f"{base_name}_{expert_name}.pt")
        torch.save(save_data, save_path)
        print(f"[LoRA-Mixture] expert {expert_idx:02d} (“{expert_name}”) saved -> {save_path}")

def margin_penality_loss(list_lora_layers, margin=0.05, lambda_margin=1e-2):
    loss = 0
    for layer in list_lora_layers:
        for name, p in layer.named_parameters():
            # print(name)
            if "lora_S_" in name:
                # sigma = _sigma_transform(p, kind=layer._svd_kind, beta=layer._svd_beta)
                sigma = p
                loss += lambda_margin * F.relu(margin - sigma).sum()
    return loss

def _sigma_transform(raw_s: torch.Tensor,
                     kind: str = "softplus",
                     beta: float = 2.0) -> torch.Tensor:
    if kind == "relu":
        return F.relu(raw_s)
    elif kind == "square":
        return raw_s.pow(2)
    elif kind == "exp":
        return torch.exp(raw_s)
    elif kind == "softplus":
        return F.softplus(raw_s, beta=beta)
    elif kind == "abs":
        return raw_s.abs()
    elif kind == "none":
        return raw_s
    else:
        raise ValueError(f"Unknown SVD transform {kind}")

def plot_singular_distributions(
    list_lora_layers      : List,              
    params                : List[str],          # e.g. ["q","v"] or ["proj"]
    output_dir            : str,
    expert_names          : List[str],         
    transform             : bool  = False,   
    svd_kind              : str   = "softplus",
    svd_beta              : float = 2.0,
    figsize_per_layer     : Tuple[int,int] = (6, 2),
    epoch                 : int = -1
):
    os.makedirs(output_dir, exist_ok=True)
    num_experts = len(expert_names)
    assert num_experts == list_lora_layers[0].num_experts, \
        "expert_names length must equal mixture’s num_experts"

    n_layers = len(list_lora_layers)
    for expert_idx, expert_name in enumerate(expert_names):
        for param in params:
            # ––––– build the figure –––––
            fig, axes = plt.subplots(
                n_layers, 1,
                figsize=(figsize_per_layer[0],
                         figsize_per_layer[1] * n_layers),
                sharex=True
            )
            if n_layers == 1: axes = [axes]

            for layer_idx, (layer, ax) in enumerate(zip(list_lora_layers, axes)):
                # pick the projection module
                if param in ("o", "proj"):
                    sub_mod = layer.proj
                else:
                    sub_mod = getattr(layer, f"{param}_proj")

                S_param = getattr(sub_mod, f"w_lora_S_{expert_idx}", None)
                if S_param is None:
                    raise ValueError(
                        f"Layer {layer_idx} / expert {expert_idx} missing "
                        f"w_lora_S - did you call apply_svd(train_mode=…)?"
                    )

                S = S_param.detach()
                if transform:
                    S = _sigma_transform(S, kind=svd_kind, beta=svd_beta)
                s_norm = (S / S.sum()).cpu().numpy()

                ax.bar(range(len(s_norm)), s_norm, width=0.8)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel(f"L{layer_idx}", rotation=0, labelpad=20, fontsize=7)

                if layer_idx == 0:
                    title = (
                        f"{expert_name} • {param} – "
                        + ("σ(S)" if transform else "raw S")
                    )
                    ax.set_title(title, fontsize=10, pad=12)

            axes[-1].set_xlabel("rank index", fontsize=8)
            plt.tight_layout()

            if epoch < 0:
                stem        = f"{expert_name}_{param}"
                candidate   = os.path.join(output_dir, f"{stem}.png")
                counter     = itertools.count(1)
                while os.path.exists(candidate):
                    candidate = os.path.join(output_dir, f"{stem}_{next(counter)}.png")
            else:
                stem        = f"{expert_name}_{param}"
                candidate   = os.path.join(output_dir, f"{stem}_epoch_{epoch}.png")

            fig.savefig(candidate, dpi=150)
            plt.close(fig)          # free memory
            print(f"[plot] saved → {candidate}")
