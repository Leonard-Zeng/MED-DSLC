import os

import torch
import torch.nn as nn

from typing import Dict

import clip
from .layers import LoRALayer, PlainMultiheadAttentionLoRA

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
            if isinstance(m, LoRALayer) and \
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


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def save_lora_as_clip(clip_model, list_lora_layers, save_path=None, return_statedict_only=False):
    if not return_statedict_only:
        assert save_path is not None

    list_mha_layers = []
    for lora_layer in list_lora_layers:
        
        # rebuild a MultiheadAttention layer
        embed_dim = lora_layer.embed_dim
        kdim = lora_layer.kdim
        vdim = lora_layer.vdim
        _qkv_same_embed_dim = lora_layer._qkv_same_embed_dim
        num_heads = lora_layer.num_heads
        batch_first = lora_layer.batch_first
        head_dim = lora_layer.head_dim
        
        mha = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = 0,
            bias = lora_layer.q_proj.bias is not None,
            kdim = kdim,
            vdim = vdim,
            batch_first=batch_first,
        )

        q_proj = lora_layer.q_proj
        if hasattr(q_proj, "lora_train"):
            q_proj.lora_train(False) # merge the weights into model
        mha.in_proj_weight[:embed_dim, :].data.copy_(q_proj.weight.data)
        if hasattr(q_proj, "bias"):
            mha.in_proj_bias[:embed_dim].data.copy_(q_proj.bias.data)

        k_proj = lora_layer.k_proj
        if hasattr(k_proj, "lora_train"):
            k_proj.lora_train(False) # merge the weights into model
        mha.in_proj_weight[embed_dim:embed_dim*2, :].data.copy_(k_proj.weight.data)
        if hasattr(k_proj, "bias"):
            mha.in_proj_bias[embed_dim:embed_dim*2].data.copy_(k_proj.bias.data)

        v_proj = lora_layer.v_proj
        if hasattr(v_proj, "lora_train"):
            v_proj.lora_train(False) # merge the weights into model
        mha.in_proj_weight[embed_dim*2:, :].data.copy_(v_proj.weight.data)
        if hasattr(v_proj, "bias"):
            mha.in_proj_bias[embed_dim*2:].data.copy_(v_proj.bias.data)

        out_proj = lora_layer.proj
        if hasattr(out_proj, "lora_train"):
            out_proj.lora_train(False) # merge the weights into model
        mha.out_proj.weight.data.copy_(out_proj.weight.data)
        if hasattr(out_proj, "bias"):
            mha.out_proj.bias.data.copy_(out_proj.bias.data)
        
        list_mha_layers.append(mha)
    idx = 0
    text_encoder = clip_model.transformer
    for i, block in enumerate(text_encoder.resblocks):
        for name, submodule in block.named_children():
            if isinstance(submodule, PlainMultiheadAttentionLoRA):
                setattr(block, name, list_mha_layers[idx])
                idx += 1
    vision_encoder = clip_model.visual.transformer
    for i, block in enumerate(vision_encoder.resblocks):
        for name, submodule in block.named_children():
            if isinstance(submodule, PlainMultiheadAttentionLoRA):
                setattr(block, name, list_mha_layers[idx])
                idx+=1
    assert idx == len(list_mha_layers), "Not every LoRA layer was replaced"

    if return_statedict_only:
        return clip_model.state_dict()
    else:
        torch.save(clip_model.state_dict(), save_path)


def apply_lora_with_teacher(args, clip_model, teacher_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer

        text_encoder_teacher = teacher_model.transformer
        teacher_resblocks_modules = [dict(block.named_children()) for block in text_encoder_teacher.resblocks]

        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )

                        teacher_multi_head_lora = PlainMultiheadAttentionLoRA(
                            teacher_resblocks_modules[i][name],
                            enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        new_multi_head_lora.init_BA_with_teacher(teacher_multi_head_lora)

                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer

        vision_encoder_teacher = teacher_model.visual.transformer
        teacher_resblocks_modules = [dict(block.named_children()) for block in vision_encoder_teacher.resblocks]

        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )

                        teacher_multi_head_lora = PlainMultiheadAttentionLoRA(
                            teacher_resblocks_modules[i][name],
                            enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        new_multi_head_lora.init_BA_with_teacher(teacher_multi_head_lora)

                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
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
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        if metadata['r'] < args.r:
            print(f"loading {metadata['r']} into the first {args.r}")
        else:
            raise ValueError(
                f"r mismatch: expected less than {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    r = metadata['r']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data[:r, :].copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data[:, :r].copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data[:r, :].copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data[:, :r].copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data[:r, :].copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data[:, :r].copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data[:r, :].copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data[:, :r].copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')

def apply_lora_(clip_model, backbone, encoder, position, params, r, alpha, dropout_rate=0.25):
    list_lora_layers = []
    if encoder == 'text' or encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=params, r=r, lora_alpha=alpha, dropout_rate=dropout_rate)
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
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=params, r=r, lora_alpha=alpha, dropout_rate=dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def load_lora_from(backbone, load_path):

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    clip_model, _ = clip.load(backbone)

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    # print(metadata)
    r = metadata['r']
    alpha = metadata['alpha']
    encoder = metadata['encoder']
    params = metadata['params']
    position = metadata['position']

    list_lora_layers = apply_lora_(clip_model, backbone, encoder, position, params, r, alpha)

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
    return clip_model, list_lora_layers

