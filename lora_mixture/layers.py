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

def _sigma_inverse(s: torch.Tensor, kind: str = "softplus", beta: float = 2.0):
    if kind == "relu":
        return s # singular values will always be non-negative
    elif kind == "square":
        return torch.sqrt(torch.clamp_min(s, 0.0))
    elif kind == "exp":
        return torch.log(torch.clamp_min(s, 1e-12))
    elif kind == "softplus":
        return 1 / beta * torch.log( torch.exp(beta*s) - 1 )
    elif kind == "abs":
        return s # singular values will always be non-negative
    elif kind == "none":
        return s
    else:
        raise ValueError(f"Unknown SVD transform {kind!r}")

class LoRAMixtureLayer():
    def __init__(
        self,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        fan_in_fan_out: bool = False,
        dropout_rate:float = 0,
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
            self.scalings.append(self.lora_alpha[i]/math.sqrt(self.r[i]))

        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

        # alpha_batch for parallelism
        self.alpha_batch = None # (B, num_expert)

        # SVD
        self.svd = False
        self.svd_train = False
        self.delta_r = None
        self.pytest = False # this flag is only set for doing pytest

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

    def merge_BA(self, param_name: str, expert_idx: int):
        lora_name = self.params_with_lora[param_name]
        if self.svd:
            # print("hello there!")
            singular_values = eval(f'self.{lora_name}_lora_S_{expert_idx}') # (r, )
            # sigma = nn.ReLU()(singular_values.unsqueeze(0)) # (1, r)
            sigma = _sigma_transform(
                singular_values.unsqueeze(0),
                kind=self._svd_kind,
                beta=self._svd_beta
            )
            # scalar * (d_out, r) * (1, r) @ (r, d_in)
            return self.lora_alpha[expert_idx] * self.transpose(
                (
                    eval(f'self.{lora_name}_lora_U_{expert_idx}') *
                    sigma @
                    eval(f'self.{lora_name}_lora_Vh_{expert_idx}')
                ).view(eval(f'self.{param_name}').shape) 
            ) / math.sqrt(self.r[expert_idx])
        else:
            return self.lora_alpha[expert_idx] * self.transpose(
                (
                    eval(f'self.{lora_name}_lora_B_{expert_idx}') @
                    eval(f'self.{lora_name}_lora_A_{expert_idx}')
                ).view(eval(f'self.{param_name}').shape)
            ) / math.sqrt(self.r[expert_idx])

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')

            p_new = p.detach() # detach() is very important here
            for i in range(self.num_experts):
                p_new += self.merge_BA(param_name, i)
            
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            # print(f"original:{eval(f'self.{param_name}').data})")
            for i in range(self.num_experts):
                # print(f"to merge {i}: {self.merge_BA(param_name, i)}")
                eval(f'self.{param_name}').data += self.merge_BA(param_name, i)
                # print(f"after {i}:{eval(f'self.{param_name}').data})")
    
    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            for i in range(self.num_experts):
                eval(f'self.{param_name}').data -= self.merge_BA(param_name, i)

    def add_expert_lora_data(self, i):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name, i)
    
    def sub_expert_lora_data(self, i):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name, i)

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged:
            # Make sure that the weights are not merged
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True

    def update_alpha(self, alpha: float, expert_idx: int):
        r"""NOT differentiable"""
        # print(self.weight)
        # print(self.lora_alpha)
        if self.merged:
            self.sub_expert_lora_data(expert_idx)
        self.lora_alpha[expert_idx] = alpha
        self.scalings[expert_idx] = self.lora_alpha[expert_idx] / math.sqrt(self.r[expert_idx])
        if self.merged:
            self.add_expert_lora_data(expert_idx)
        # print(f"after {expert_idx}: {self.weight}")

    def set_alpha_batch(self, alpha_batch: torch.Tensor):
        r"""NOT differentiable"""
        assert alpha_batch.shape[1] == self.num_experts
        self.alpha_batch = alpha_batch
        if self.merged:
            self.sub_lora_data()
        self.merged = False # when alpha_batch is not None, it can never be merged

    def unset_alpha_batch(self):
        r"""NOT differentiable"""
        self.alpha_batch = None

    def apply_svd(
        self, train_mode = False, svd_transform: str = "softplus", svd_beta: float = 2.0, delta_r=None, pytest=False
    ):
        r"""NOT differentiable"""
        if self.merged:
            self.sub_lora_data()

        self.svd_train = train_mode
        self._svd_kind = svd_transform
        self._svd_beta = svd_beta
        self.pytest = pytest # this flag is only set for doing pytest

        if delta_r is not None:
            self.delta_r = delta_r

        for param_name, lora_name in self.params_with_lora.items():
            for expert_idx in range(self.num_experts):
                delta_param = self.transpose(
                    (
                        eval(f'self.{lora_name}_lora_B_{expert_idx}') @
                        eval(f'self.{lora_name}_lora_A_{expert_idx}')
                    ).view(eval(f'self.{param_name}').shape)
                )

                # print(eval(f'self.{lora_name}_lora_B_{expert_idx}').shape, eval(f'self.{lora_name}_lora_A_{expert_idx}').shape)
                # r = torch.linalg.matrix_rank(delta_param)
                r = self.r[expert_idx]
                U, S, Vh = torch.linalg.svd(delta_param.data, full_matrices=False)
                U = U[:, :r]
                Vh = Vh[:r, :]
                S = S[:r]

                inv_S = _sigma_inverse(S, kind=svd_transform, beta=svd_beta)
                # print(U.shape, S.shape, Vh.shape)

                self.register_buffer(
                    f'{lora_name}_lora_U_{expert_idx}',
                    U
                )
                self.register_parameter(
                    f'{lora_name}_lora_S_{expert_idx}',
                    nn.Parameter(inv_S)
                )
                self.register_buffer(
                    f'{lora_name}_lora_Vh_{expert_idx}',
                    Vh
                )

                if self.delta_r is not None:
                    self.register_parameter(f'{lora_name}_lora_deltaA_{expert_idx}',
                        nn.Parameter(eval(f'self.{param_name}').new_zeros((self.delta_r, eval(f'self.{param_name}').size()[1])))
                        )
                    self.register_parameter(f'{lora_name}_lora_deltaB_{expert_idx}',
                        nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.delta_r)))
                        )

                if self.svd_train:
                    # eval(f'self.{lora_name}_lora_U_{expert_idx}').requires_grad = False
                    # eval(f'self.{lora_name}_lora_Vh_{expert_idx}').requires_grad = False
                    eval(f'self.{lora_name}_lora_B_{expert_idx}').requires_grad = False
                    eval(f'self.{lora_name}_lora_A_{expert_idx}').requires_grad = False
                    eval(f'self.{lora_name}_lora_S_{expert_idx}').requires_grad = True

        self.svd = True
        if self.merged:
            self.add_lora_data()

    def revert_svd(self, train_mode=False):
        r"""NOT differentiable"""
        if self.merged:
            self.sub_lora_data()

        self.svd = False
        self.svd_train = train_mode
        for param_name, lora_name in self.params_with_lora.items():
            for expert_idx in range(self.num_experts):
                singular_values = eval(f'self.{lora_name}_lora_S_{expert_idx}') # (r, )
                sigma = _sigma_transform(
                    singular_values.unsqueeze(0),
                    kind=self._svd_kind,
                    beta=self._svd_beta
                )
                if self._svd_kind == "none":
                    r = self.r[expert_idx]
                    deltaW = eval(f'self.{lora_name}_lora_U_{expert_idx}') * torch.sqrt(sigma).reshape(1, -1) @ eval(f'self.{lora_name}_lora_Vh_{expert_idx}')
                    Q, R = torch.linalg.qr(deltaW, mode='reduced')
                    Q = Q[:, :r]
                    R = R[:r, :]
                    getattr(self, f'{lora_name}_lora_B_{expert_idx}').data.copy_(Q)
                    getattr(self, f'{lora_name}_lora_A_{expert_idx}').data.copy_(R)
                else:
                    eval(f'self.{lora_name}_lora_B_{expert_idx}').data = (
                        eval(f'self.{lora_name}_lora_U_{expert_idx}') * 
                        torch.sqrt(sigma).reshape(1, -1)
                    )

                    eval(f'self.{lora_name}_lora_A_{expert_idx}').data = (
                        torch.sqrt(sigma).reshape(-1, 1) * 
                        eval(f'self.{lora_name}_lora_Vh_{expert_idx}')
                    )
                if not self.svd_train:
                    eval(f'self.{lora_name}_lora_B_{expert_idx}').requires_grad = True
                    eval(f'self.{lora_name}_lora_A_{expert_idx}').requires_grad = True
                    eval(f'self.{lora_name}_lora_S_{expert_idx}').requires_grad = False

        if self.merged:
            self.add_lora_data()

    def svd_update_BA(self):
        origin_svd = self.svd
        if self.svd:
            self.revert_svd(True)
            self.svd = origin_svd


class Embedding(nn.Embedding, LoRAMixtureLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRAMixtureLayer.__init__(self, num_experts=num_experts, r=r, lora_alpha=lora_alpha)

        self.params_with_lora = {'weight': 'w'}
        self.register_lora_param()
        nn.Embedding.reset_parameters(self)
        self.init_lora_param()

    def init_lora_param(self):
        if hasattr(self, 'w_lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.w_lora_A)
            nn.init.normal_(self.w_lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_train(mode)
        
    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Embedding.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Embedding.forward(self, x, **kwargs)

class LinearLoRA(nn.Linear, LoRAMixtureLayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict())
        LoRAMixtureLayer.__init__(
            self, num_experts=num_experts, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

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

    def train(self, mode: bool = True):
        super().train(mode)     
        self.lora_train(mode)

        
    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, d_in) or (seq_len, B, d_in)
        # print(x.shape)
        # if self.alpha_batch is not None:
        #     print(self.alpha_batch)
        if (self.alpha_batch is not None) and (not self.svd_train):
            if self.svd:
                original_output = nn.Linear.forward(self, x)

                XVs = []
                Ss = []
                Uhs = []
                for expert_idx in range(self.num_experts):
                    lora_name = self.params_with_lora['weight']
                    U = eval(f'self.{lora_name}_lora_U_{expert_idx}') # (d_out, r)
                    Vh = eval(f'self.{lora_name}_lora_Vh_{expert_idx}') # (r, d_in)
                    S = eval(f'self.{lora_name}_lora_S_{expert_idx}') # (r, )
                    S = _sigma_transform(S, kind=self._svd_kind, beta=self._svd_beta)

                    # print(U.shape, Vh.shape, S.shape)
                    XV = x @ Vh.T # (B, r) or (seq_len, B, r)
                    XVs.append(XV)

                    Ss.append(S.unsqueeze(0)) # (1, r)
                    Uhs.append(U.T) # (r, d_out)

                XVs = torch.stack(XVs) # (n, seq_len, B, r), or (n, B, r)
                Ss = torch.vstack(Ss) # (n, r)
                Uhs = torch.stack(Uhs) # (n, r, d_out)
                # print(Ss.shape)
                # print(Uhs.shape)

                alpha_batch = self.alpha_batch.clone() # (B, n)
                # self.r a list with shape= (num_experts, )
                alpha_batch /= torch.sqrt(torch.tensor(self.r, dtype=self.alpha_batch.dtype, device=self.alpha_batch.device)).reshape(1, -1)
                # print(f"alpha_batch: {alpha_batch}")
                alpha_batch = alpha_batch.unsqueeze(-1) # (B, n, 1)

                B = alpha_batch.shape[0]
                r = Ss.shape[-1]
                
                # alpha_batch: (B, n, r)
                if self.pytest:
                    alpha_batch = torch.cat([alpha_batch]*r, dim=-1)
                else:
                    alpha_batch = torch.cat((alpha_batch, torch.zeros(size=(B, self.num_experts, r-1)).to(alpha_batch.device)), dim=-1)
                    # alpha_batch = torch.cat((alpha_batch, torch.ones(size=(B, self.num_experts, r-1)).to(alpha_batch.device)), dim=-1)
                # print(alpha_batch)
                # print(alpha_batch.shape)

                sigma = alpha_batch * Ss.unsqueeze(0) # (B, n, r)
                sigma = torch.diag_embed(sigma) # (B, n, r, r)
                sigma = torch.permute(sigma, (1, 0, 2, 3)) # (n, B, r, r)
                Uhs = Uhs.unsqueeze(1) # (n, 1, r, d_out)
                SigmaUh = torch.bmm(sigma.reshape(-1, r, r), Uhs.repeat(1, B, 1, 1).reshape(-1, r, Uhs.shape[-1])) # (n * B, r, d_out)

                if len(x.shape) == 2:
                    # XVs = (n, B, r)
                    XVs = XVs.unsqueeze(-2).reshape(-1, 1, r) # (n * B, 1, r)
                    lora_adjustment = torch.bmm(XVs, SigmaUh) # (n * B, 1, d_out)
                    lora_adjustment = lora_adjustment.reshape(self.num_experts, B, lora_adjustment.shape[-1])
                    lora_adjustment = torch.sum(lora_adjustment, dim=0) #(B, d_out)
                    result = original_output + lora_adjustment
                    return result
                else:
                    # XVs = (n, seq_len, B, r)
                    # print(XVs.shape)
                    XVs = XVs.permute(0, 2, 1, 3) # (n, B, seq_len, r)
                    seq_len = XVs.shape[-2]
                    XVs = XVs.reshape(-1, seq_len, r) # (n * B, seq_len, r)
                    # print(XVs.shape)
                    # print(SigmaUh.shape)
                    lora_adjustment = torch.bmm(XVs, SigmaUh)  # (n * B, seq_len, d_out)
                    lora_adjustment = lora_adjustment.reshape(self.num_experts, B, seq_len, lora_adjustment.shape[-1]) # (n, B, seq_len, d_out)
                    lora_adjustment = torch.sum(lora_adjustment, dim=0)  # (B, seq_len, d_out)
                    lora_adjustment = lora_adjustment.permute(1, 0, 2) # (seq_len, B, d_out)
                    result = original_output + lora_adjustment
                    return result
            else:
                # print(x.shape)
                # print(self.alpha_batch.shape)
                # Compute the original linear transformation
                original_output = nn.Linear.forward(self, x)
                # print(original_output.shape)

                # alpha_batch (B, num_experts)
                # x (B, D_x)
                # BAs (3, D_x, D_z)
                # BAx (num_experts, B, D_z)
                BAs = []
                for expert_idx in range(self.num_experts):
                    lora_name = self.params_with_lora['weight']
                    BA = self.transpose(
                    (
                        eval(f'self.{lora_name}_lora_B_{expert_idx}') @
                        eval(f'self.{lora_name}_lora_A_{expert_idx}')
                    ).view(eval(f'self.weight').shape))
                    BAs.append(torch.matmul(x, BA.transpose(0, 1)))
                BAs = torch.stack(BAs)

                alpha_batch = self.alpha_batch.clone() # (B, n)
                # self.r a list with shape= (num_experts, )
                alpha_batch /= torch.sqrt(torch.tensor(self.r, dtype=self.alpha_batch.dtype, device=self.alpha_batch.device)).reshape(1, -1)

                if len(x.shape) == 2:
                    alpha_batch = alpha_batch.transpose(0, 1).unsqueeze(-1)
                    lora_adjustment = torch.sum(alpha_batch * BAs, dim=0)
                    result = original_output + lora_adjustment
                    return result
                else:
                    alpha_batch = alpha_batch.transpose(0, 1).unsqueeze(1).unsqueeze(-1)
                    # print(alpha_batch.shape)
                    lora_adjustment = torch.sum(alpha_batch * BAs, dim=0)
                    result = original_output + lora_adjustment
                    return result
        if (not self.training) or (self.dropout is None): # do as before
            if not self.merged:
                self.merge_lora_param()
                result = nn.Linear.forward(self, x, **kwargs)
                self.sub_lora_data()
                return result
            else:
                return nn.Linear.forward(self, x, **kwargs)
            
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)

        if self.training and self.dropout.p > 0:
            x = self.dropout(x)
        
        if not self.merged:
            # lora_adjustment = torch.matmul(x,self.merge_BA('weight').transpose(0, 1)) * self.scaling
            lora_adjustment = torch.zeros_like(original_output)
            for i in range(self.num_experts):
                lora_adjustment += torch.matmul(x, self.merge_BA('weight', i).transpose(0, 1))
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result

class Conv1d(nn.Conv1d, LoRAMixtureLayer):
    # LoRA implemented in a Conv1d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRAMixtureLayer.__init__(self, num_experts=num_experts, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        # if r > 0:
        for i, r in enumerate(self.r):
            self.register_parameter(
                f'w_lora_A_{i}',
                nn.Parameter(
                    self.weight.new_zeros(
                        (r*kernel_size, in_channels*kernel_size)
                    )
                )
            )
            self.register_parameter(
                f'w_lora_B_{i}',
                    nn.Parameter(
                        self.weight.new_zeros(
                            (out_channels//self.groups*kernel_size, r*kernel_size)
                        )
                    )
            )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        nn.Conv1d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv1d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if not self.merged:
            self.merge_lora_param()
            result = nn.Conv1d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv1d.forward(self, x, **kwargs)

class Conv2d(nn.Conv2d, LoRAMixtureLayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRAMixtureLayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        for i, r in enumerate(self.r):
            self.register_parameter(
                f'w_lora_A_{i}',
                nn.Parameter(
                    self.weight.new_zeros(
                        (r*kernel_size, in_channels*kernel_size)
                    )
                )
            )
            self.register_parameter(
                f'w_lora_B_{i}',
                    nn.Parameter(
                        self.weight.new_zeros(
                            (out_channels//self.groups*kernel_size, r*kernel_size)
                        )
                    )
            )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)

class Conv3d(nn.Conv3d, LoRAMixtureLayer):
    # LoRA implemented in a Conv3d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        num_experts: int,
        r: list[int],
        lora_alpha: list[float],
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRAMixtureLayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        for i, r in enumerate(self.r):
            self.register_parameter(
                f'w_lora_A_{i}',
                nn.Parameter(
                    self.weight.new_zeros(
                        (r*kernel_size, in_channels*kernel_size)
                    )
                )
            )
            self.register_parameter(
                f'w_lora_B_{i}',
                    nn.Parameter(
                        self.weight.new_zeros(
                            (out_channels//self.groups*kernel_size, r*kernel_size)
                        )
                    )
            )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        nn.Conv3d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv3d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv3d.forward(self, x, **kwargs)


class PlainMultiheadAttentionLoRAMixture(nn.Module):
    def __init__(
            self,
            num_experts: int,
            r: list[int],
            lora_alpha: list[float],
            existing_mha: nn.MultiheadAttention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            dropout_rate:float = 0.,
            **kwargs
        ):
        super().__init__()
        
        self.dropout = 0 # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
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

        LoRAMixtureLayer.__init__(self, num_experts=num_experts, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
        
        # Init qkv as a new lora linear layer
        self.enable_lora = enable_lora
        for item in self.enable_lora:
            if item == 'q':
                self.q_proj = LinearLoRA(self.q_proj,
                                         num_experts=num_experts,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'k':
                self.k_proj = LinearLoRA(self.k_proj,
                                         num_experts=num_experts,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'v':
                self.v_proj = LinearLoRA(self.v_proj,
                                         num_experts=num_experts,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'o':
                self.proj = LinearLoRA(self.proj,
                                         num_experts=num_experts,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
        
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
        
        q = self.q_proj(query)
        k = self.k_proj(key)
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
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        #self.lora_train(mode)  

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):

        return self.forward_module(query, key, value, **kwargs)

    def update_alpha(self, alpha: float, expert_idx: int):
        for item in self.enable_lora:
            eval(f"self.{item}_proj").update_alpha(alpha, expert_idx)

    def apply_svd(
        self, train_mode=False, svd_transform: str = "softplus", svd_beta: float = 2.0
    ):
        self._svd_kind = svd_transform
        self._svd_beta = svd_beta
        for item in self.enable_lora:
            eval(f"self.{item}_proj").apply_svd(train_mode, svd_transform, svd_beta)

    def svd_update_BA(self):
        for item in self.enable_lora:
            eval(f"self.{item}_proj").svd_update_BA()

    def revert_svd(self):
        for item in self.enable_lora:
            eval(f"self.{item}_proj").revert_svd()

    def set_alpha_batch(self, alpha_batch: torch.Tensor):
        for item in self.enable_lora:
            eval(f"self.{item}_proj").set_alpha_batch(alpha_batch)
            # print(f"self.{item}_proj is set")

    def unset_alpha_batch(self):
        for item in self.enable_lora:
            eval(f"self.{item}_proj").unset_alpha_batch()

class MergedLinear(nn.Linear, LoRAMixtureLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,                # ← now used below
        r: list[int],
        lora_alpha: list[float],
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRAMixtureLayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert out_features % num_experts == 0, \
            'out_features must be divisible by num_experts'
        assert len(enable_lora) == num_experts, \
            'enable_lora length must equal num_experts'

        self.enable_lora = enable_lora
        self.num_experts = num_experts

        # register A/B for only the enabled experts...
        if any(self.enable_lora):
            for i, rank in enumerate(self.r):
                self.register_parameter(
                    f"w_lora_A_{i}",
                    nn.Parameter(self.weight.new_zeros((rank * sum(self.enable_lora), in_features)))
                )
                self.register_parameter(
                    f"w_lora_B_{i}",
                    nn.Parameter(self.weight.new_zeros(
                        ( (out_features // num_experts) * sum(self.enable_lora), rank )
                    ))
                )

            block_size = out_features // num_experts
            mask = torch.zeros(out_features, dtype=torch.bool, device=self.weight.device)
            for expert_idx, enabled in enumerate(self.enable_lora):
                if enabled:
                    start = expert_idx * block_size
                    mask[start : start + block_size] = True
            self.lora_ind = mask

        # freeze and init
        self.weight.requires_grad = False
        nn.Linear.reset_parameters(self)
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        # create full-size delta and scatter in only the active‑expert rows
        result = x.new_zeros(self.weight.size())
        result[self.lora_ind] = x
        return result

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        delta = F.conv1d(
            getattr(self, f"{lora_name}_lora_A").unsqueeze(0),
            getattr(self, f"{lora_name}_lora_B").unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return self.transpose(self.zero_pad(delta))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.r and not self.merged:
            self.merge_lora_param()
            out = super().forward(x, **kwargs)
            self.sub_lora_data()
            return out
        else:
            return super().forward(x, **kwargs)
        
