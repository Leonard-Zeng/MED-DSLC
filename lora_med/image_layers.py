#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
from typing import Optional, List


from .gating_network import (
    CrossAttnGating,
    MEDGatingNetwork,
    SigmoidRankGatingNetwork,
    PhatGooseGatingNetwork,
)
from .constants import openclip_backbones

apply_bayesian_noise_to_lora = None

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class ImageMEDLayer:
    def __init__(
        self,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        fan_in_fan_out: bool = False,
        dropout_rate:float = 0,
        backbone='ViT-L/14',
    ):
        self.num_experts = num_experts
        assert len(r) == len(lora_alpha)
        assert len(r) == num_experts
        self.r = copy.deepcopy(r)
        self.lora_alpha = copy.deepcopy(lora_alpha) # we must do deep copy here
        self.dropout_rate = dropout_rate

        self.scalings = []
        for i in range(num_experts):
            assert self.r[i] > 0
            self.scalings.append(1/math.sqrt(self.r[i]))

        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

        self.prompt_feats = None

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            for i in range(self.num_experts):
                self.register_parameter(f'{lora_name}_lora_A_{i}',
                    nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r[i], eval(f'self.{param_name}').size()[1])))
                    )
                self.register_parameter(f'{lora_name}_lora_B_{i}',
                    nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r[i])))
                    )
                
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                for i in range(self.num_experts):
                    nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A_{i}'), a=math.sqrt(5))
                    nn.init.zeros_(eval(f'self.{lora_name}_lora_B_{i}'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    # def mole_train(self, mode:bool):
    #     self.gate_net.train(mode)

    def mole_out(self, x: torch.Tensor):
        out_dict = {}
        # weight, weight_lora_B_i, weight_lora_A_i
        for param_name, lora_name in self.params_with_lora.items():
            out = []
            for i in range(self.num_experts):
                B = getattr(self, f"{lora_name}_lora_B_{i}")
                A = getattr(self, f"{lora_name}_lora_A_{i}")
                if hasattr(self, "_apply_bayesian_noise"):
                    A, B = self._apply_bayesian_noise(A=A, B=B, expert_idx=i)
                z = x@(B@A).T * self.scalings[i] # (B, L, d)
                out.append(z)
            out_dict[param_name] = torch.stack(out, dim=1)  # (B, N, L, d)
        return out_dict

    def set_prompt_feats(self, prompt_feats: torch.Tensor):
        self.prompt_feats = prompt_feats

class ImageLinearMED(nn.Linear, ImageMEDLayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        backbone='ViT-L/14',
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        ImageMEDLayer.__init__(
            self, num_experts=num_experts, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out, backbone=backbone)

        # Set sequence_length from backbone constants
        sequence_length = openclip_backbones[backbone]["vision"]["sequence_length"]
        self.sequence_length = sequence_length
        self.total_rank = sum(r)

        self.load_state_dict(existing_linear.state_dict())

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        # if r > 0:
        self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.bayes_enabled = False
        self.bayes_axis = "d"
        self.bayes_eval_sampling = "mean"
        self.bayes_net_type = "linear"
        self.bayes_hidden_dim = 128
        self.bayes_F_A = None
        self.bayes_F_B = None

    def train(self, mode: bool = True):
        super().train(mode)

        
    def forward(self, x: torch.Tensor, alpha, **kwargs):
        # alpha: (B, N) for expert-gating, or (B, T) for rank-gating where T = sum(r)
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)  #(L, B, d) or (L*B, d)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # detect rank-gating vs expert-gating based on alpha's second dimension
        if alpha.shape[1] == self.total_rank:
            return self._forward_rank_gated(x, alpha, original_output)
        else:
            return self._forward_expert_gated(x, alpha, original_output)

    def _forward_expert_gated(self, x: torch.Tensor, alpha: torch.Tensor, original_output: torch.Tensor):
        """Standard expert-level gating with alpha: (B, N)"""
        delta_x = self.mole_out(x)["weight"]  # Shape depends on input format
        if len(alpha.shape) == 3:
            # Token-wise gating: alpha is (B, L, N)
            if len(delta_x.shape) == 3:
                # (L*B, N, d) -> (B, L, N, d)
                L_B, N, d = delta_x.shape
                B = alpha.shape[0]
                L = alpha.shape[1]
                delta_x = delta_x.view(L, B, N, d).permute(1, 0, 2, 3)
            else:
                if delta_x.shape[0] == self.sequence_length:
                    # (L, N, B, d) -> (B, L, N, d)
                    delta_x = delta_x.permute(2, 0, 1, 3)
                else:
                    # (B, N, L, d) -> (B, L, N, d)
                    delta_x = delta_x.permute(0, 2, 1, 3)
            delta_x = (alpha.unsqueeze(-1) * delta_x).sum(dim=2)  # (B, L, d)
            if len(original_output.shape) == 3:
                if original_output.shape[0] == self.sequence_length:
                    delta_x = delta_x.permute(1, 0, 2).contiguous()
            else:
                delta_x = delta_x.reshape(-1, delta_x.size(-1))
            return original_output + delta_x

        if len(delta_x.shape) == 3:
            L_B, N, d = delta_x.shape
            delta_x = delta_x.view(self.sequence_length, L_B//self.sequence_length, N, d)
            delta_x = torch.permute(delta_x, (0, 2, 1, 3))  # (L, N, B, d)
            # alpha: (B, N) -> (1, N, B, 1)
            alpha = torch.permute(alpha, (1, 0)).contiguous()
            alpha = alpha.view(1, alpha.size(0), alpha.size(1), 1)
            delta_x = alpha * delta_x  # (L, N, B, d)
            delta_x = torch.sum(delta_x, dim=1)  # (L, B, d)
        else:
            if delta_x.shape[0] == self.sequence_length:
                # (L, N, B, d) format - original assumption
                alpha = torch.permute(alpha, (1, 0)).contiguous()  # (N, B)
                alpha = alpha.view(1, alpha.size(0), alpha.size(1), 1)  # (1, N, B, 1)
                delta_x = alpha * delta_x  # (L, N, B, d)
                delta_x = torch.sum(delta_x, dim=1)  # (L, B, d)
            else:
                # (B, N, L, d) format - when input was (B, L, d)
                alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)  # (B, N, 1, 1)
                delta_x = alpha * delta_x  # (B, N, L, d)
                delta_x = torch.sum(delta_x, dim=1)  # (B, L, d)

        delta_x = delta_x.view(*original_output.shape)
        return original_output + delta_x

    def _forward_rank_gated(self, x: torch.Tensor, alpha: torch.Tensor, original_output: torch.Tensor):
        input_3d = len(x.shape) == 3
        if input_3d:
            if x.shape[0] == self.sequence_length:
                # (L, B, d) format
                L, B, d_in = x.shape
                x_flat = x.reshape(L * B, d_in)  # (L*B, d_in)
            else:
                # (B, L, d) format
                B, L, d_in = x.shape
                x_flat = x.reshape(B * L, d_in)  # (B*L, d_in)
        else:
            L_B, d_in = x.shape
            B = L_B // self.sequence_length
            L = self.sequence_length
            x_flat = x  # already (L*B, d_in)

        # alpha: (B, T) -> expand for sequence: (B*L, T) by repeating for each token
        # each sample's alpha applies to all its tokens
        alpha_expanded = alpha.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)  # (B*L, T)

        delta = torch.zeros(B * L, self.out_features, device=x.device, dtype=x.dtype)
        rank_offset = 0
        for i in range(self.num_experts):
            r_i = self.r[i]
            A_i = getattr(self, f"w_lora_A_{i}")  # (r_i, d_in)
            B_i = getattr(self, f"w_lora_B_{i}")  # (d_out, r_i)
            A_i, B_i = self._apply_bayesian_noise(A=A_i, B=B_i, expert_idx=i)
            
            # x @ A_i.T -> (B*L, r_i)
            u = x_flat @ A_i.T
            
            # get alpha slice for this expert's ranks
            alpha_slice = alpha_expanded[:, rank_offset:rank_offset + r_i]  # (B*L, r_i)
            u = u * alpha_slice
            
            # u @ B_i.T -> (B*L, d_out), scaled
            delta = delta + (u @ B_i.T) * self.scalings[i]
            
            rank_offset += r_i

        delta = delta.view(*original_output.shape)
        return original_output + delta

    def _should_sample_bayesian(self) -> bool:
        if self.training:
            return True
        return self.bayes_eval_sampling == "mc"

    def _apply_bayesian_noise(self, A: torch.Tensor, B: torch.Tensor, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.bayes_enabled:
            return A, B
        # Keep explicit base/zero experts deterministic.
        if self.lora_alpha[expert_idx] == 0:
            return A, B
        if self.bayes_axis == "r":
            if len(self.r) == 0 or any(v != self.r[0] for v in self.r):
                raise ValueError(
                    "Bayesian noise axis='r' requires identical rank across experts "
                    f"for each projection module. Got ranks={self.r}"
                )
        if self.bayes_F_A is None or self.bayes_F_B is None:
            return A, B
        if apply_bayesian_noise_to_lora is None:
            raise ImportError("Bayesian MED support is not included in this MED-DSLC import.")
        return apply_bayesian_noise_to_lora(
            A=A,
            B=B,
            f_a=self.bayes_F_A,
            f_b=self.bayes_F_B,
            axis=self.bayes_axis,
            sample=self._should_sample_bayesian(),
        )


class ImagePlainMultiheadAttentionMED(nn.Module, ImageMEDLayer):
    def __init__(
            self,
            num_experts: int,
            r: list[int],
            lora_alpha: list[float],
            existing_mha: nn.MultiheadAttention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            dropout_rate:float = 0.,
            backbone='ViT-L/14',
            gate_type = "mole",
            top_k_expert: int = 0,
            simplify_mole: bool = False,
            **kwargs
        ):
        super().__init__()

        self.dropout = 0 # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = True  # Match open_clip's batch_first setting
        self.head_dim = existing_mha.head_dim
        #self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=existing_mha.in_proj_bias is not None)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Initialize parameters
        with torch.no_grad():
            
            # Extract the existing weights and biases
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            # Initialize q_proj
            self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

            # Initialize k_proj
            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

            # Initialize v_proj
            self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

            # Initialize proj
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        # Initialize ImageMEDLayer attributes
        ImageMEDLayer.__init__(self, num_experts=num_experts, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate, backbone=backbone)
        in_dim = openclip_backbones[backbone]["vision"]["hidden_size"]
        out_dim = openclip_backbones[backbone]["text"]["hidden_size"]
        sequence_length = openclip_backbones[backbone]["vision"]["sequence_length"]
        self.sequence_length = sequence_length
        # Per-layer expert selector: gate depends on layer input, not prompts
        self.gate_type = gate_type
        if gate_type == 'mole':
            self.gate_net = MEDGatingNetwork(
                num_experts, sequence_length, in_dim, simplify_mole=simplify_mole
            )
        elif gate_type == "phatgoose":
            self.gate_net = PhatGooseGatingNetwork(
                num_experts,
                sequence_length,
                in_dim,
                top_k_expert=top_k_expert,
                simplify_mole=simplify_mole,
            )
        elif gate_type == 'maple':
            txt_dim = openclip_backbones[backbone]["text"]["hidden_size"]
            self.gate_net = CrossAttnGating(sequence_length, in_dim, txt_dim, num_experts)
        elif gate_type == 'sigmoid':
            total_rank = sum(r)
            self.gate_net = SigmoidRankGatingNetwork(
                total_rank, sequence_length, in_dim, simplify_mole=simplify_mole
            )
        elif gate_type == 'fixed':
            self.register_buffer("fixed_alpha", torch.full((num_experts,), 1.0))
        else:
            raise NotImplementedError("This gate type {} is not implemented".format(gate_type))
        
        # Init qkv as a new lora linear layer
        self.enable_lora = enable_lora
        for item in self.enable_lora:
            if item == 'q':
                self.q_proj = ImageLinearMED(self.q_proj,
                                              num_experts=num_experts,
                                              r=r,
                                              lora_alpha=lora_alpha,
                                              fan_in_fan_out=False,
                                              dropout_rate = dropout_rate,
                                              backbone = backbone)
            elif item == 'k':
                self.k_proj = ImageLinearMED(self.k_proj,
                                              num_experts=num_experts,
                                              r=r,
                                              lora_alpha=lora_alpha,
                                              fan_in_fan_out=False,
                                              dropout_rate = dropout_rate,
                                              backbone = backbone)
            elif item == 'v':
                self.v_proj = ImageLinearMED(self.v_proj,
                                              num_experts=num_experts,
                                              r=r,
                                              lora_alpha=lora_alpha,
                                              fan_in_fan_out=False,
                                              dropout_rate = dropout_rate,
                                              backbone=backbone)
            elif item == 'o':
                self.proj = ImageLinearMED(self.proj,
                                            num_experts=num_experts,
                                            r=r,
                                            lora_alpha=lora_alpha,
                                            fan_in_fan_out=False,
                                            dropout_rate = dropout_rate,
                                            backbone=backbone)
        
    def forward_module(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """

        # Check for alpha_override FIRST (used by external joint-gating experiments)
        # This avoids calling gate_net when alpha is already computed by joint gating
        if hasattr(self, 'alpha_override') and self.alpha_override is not None:
            alpha = self.alpha_override  # (B, E) - override from joint gating
        elif self.gate_type == "fixed":
            alpha = self.fixed_alpha.unsqueeze(0).expand(bsz, -1)
        elif type(self.gate_net) is ImageLinearMED:
            alpha = self.gate_net(query)[0]
        elif type(self.gate_net) is CrossAttnGating:
            assert self.prompt_feats is not None, "prompt_feats should be set"
            alpha = self.gate_net(query, self.prompt_feats)[0]
        elif (
            isinstance(self.gate_net, MEDGatingNetwork)
            or isinstance(self.gate_net, PhatGooseGatingNetwork)
        ):
            alpha = self.gate_net(query)[0]
        elif isinstance(self.gate_net, SigmoidRankGatingNetwork):
            alpha = self.gate_net(query)[0]
        elif getattr(self.gate_net, "uses_prompt_feats", False):
            # Support for self-attention gates (e.g., SelfAttnTokenExpertGating)
            # Note: For joint gating, alpha should come from alpha_override, not gate_net
            assert self.prompt_feats is not None, "prompt_feats should be set"
            alpha = self.gate_net(query, self.prompt_feats)[0]
        else:
            raise ValueError(f"Unknown gate_net type: {type(self.gate_net)}")
        # print(f"images: {[ [ f'{v:.4f}' for v in vector] for vector in alpha]}")
        # print(f"images: {alpha.argmax(-1)}")
        if type(self.q_proj) is ImageLinearMED:
            q = self.q_proj(query, alpha)
        else:
            q = self.q_proj(query)

        if type(self.k_proj) is ImageLinearMED:
            k = self.k_proj(key, alpha)
        else:
            k = self.k_proj(key)

        if type(self.v_proj) is ImageLinearMED:
            v = self.v_proj(value, alpha)
        else:
            v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        if type(self.proj) is ImageLinearMED:
            attn_output = self.proj(attn_output, alpha)
        else:
            attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        self.mole_train(mode)
        #self.lora_train(mode)  

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):

        return self.forward_module(query, key, value, **kwargs)

    def set_prompts_feats(self, prompt_feats: torch.Tensor):
        # Set on main layer for forward_module check
        self.prompt_feats = prompt_feats
        # Also set on individual projection layers
        for item in self.enable_lora:
            if item == 'o':
                eval(f"self.proj").set_prompt_feats(prompt_feats)
            else:
                eval(f"self.{item}_proj").set_prompt_feats(prompt_feats)
            # print(f"self.{item}_proj is set")

    def mole_train(self, mode:bool):
        if hasattr(self, "gate_net"):
            self.gate_net.train(mode)

