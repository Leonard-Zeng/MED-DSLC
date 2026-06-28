import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MEDGatingNetwork(nn.Module):
    def __init__(self, num_expert, sequence_length, hidden_dim, simplify_mole: bool = False):
        super(MEDGatingNetwork, self).__init__()
        # Per-layer expert selector: gate depends only on the current layer input (L, B, d)
        in_dim = hidden_dim if simplify_mole else sequence_length * hidden_dim
        self.linear = nn.Linear(in_dim, num_expert, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.tau = nn.Parameter(torch.ones(1))
        self.sequence_length = sequence_length
        self.simplify_mole = simplify_mole

    def _to_bld(self, x):
        # x: (L, B, d), (B, L, d), or (L*B, d)
        if len(x.shape) == 3:
            if x.shape[0] == self.sequence_length:
                # (L, B, d) -> (B, L, d)
                x = x.permute(1, 0, 2).contiguous()
            # else: already (B, L, d), no permutation needed
            else:
                x = x.contiguous()
        else:
            # (L*B, d) -> (B, L, d)
            L_B, d = x.shape
            x = x.view(self.sequence_length, L_B//self.sequence_length, d)
            x = x.permute(1, 0, 2).contiguous()
        return x

    def _select_summary_token(self, x, input_ids=None):
        # x: (B, L, d)
        if input_ids is None:
            # image path: CLS token
            return x[:, 0, :]
        if input_ids.dim() != 2 or input_ids.shape[0] != x.shape[0]:
            return x[:, 0, :]
        eos_positions = input_ids.argmax(dim=-1).to(x.device)
        eos_positions = eos_positions.clamp(min=0, max=x.shape[1] - 1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return x[batch_idx, eos_positions]

    def forward(self, x, input_ids=None):
        x = self._to_bld(x)
        if self.simplify_mole:
            # Simplified path: gate on summary token only, shape (B, d).
            x = self._select_summary_token(x, input_ids=input_ids)
        else:
            # Normalize token features per token dim
            # x = self.norm(x)
            # Flatten (B, L, d) -> (B, L*d)
            x = x.reshape(x.size(0), -1)
        # Project to expert logits and softmax temperature
        x = self.linear(x)
        tau = F.softplus(self.tau) + 1e-6
        gate = F.softmax(x/tau, dim=-1)
        return gate, x


class TokenWiseMEDGatingNetwork(nn.Module):
    def __init__(
        self,
        num_expert,
        sequence_length,
        hidden_dim,
        activation="sigmoid",
        top_k_expert: int = 0,
        simplify_mole: bool = False,
    ):
        super(TokenWiseMEDGatingNetwork, self).__init__()
        # Per-token expert selector: gate depends on each token's hidden state
        self.linear = nn.Linear(hidden_dim, num_expert, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.sequence_length = sequence_length
        self.activation = activation
        self.top_k_expert = int(top_k_expert) if top_k_expert is not None else 0
        self.simplify_mole = simplify_mole
        if activation == "softmax":
            self.tau = nn.Parameter(torch.ones(1))

    def _to_bld(self, x):
        # x: (L, B, d), (B, L, d), or (L*B, d)
        if len(x.shape) == 3:
            if x.shape[0] == self.sequence_length:
                # (L, B, d) -> (B, L, d)
                x = x.permute(1, 0, 2).contiguous()
            else:
                x = x.contiguous()
        else:
            # (L*B, d) -> (B, L, d)
            L_B, d = x.shape
            x = x.view(self.sequence_length, L_B // self.sequence_length, d)
            x = x.permute(1, 0, 2).contiguous()
        return x

    def _select_summary_token(self, x, input_ids=None):
        # x: (B, L, d)
        if input_ids is None:
            return x[:, 0, :]
        if input_ids.dim() != 2 or input_ids.shape[0] != x.shape[0]:
            return x[:, 0, :]
        eos_positions = input_ids.argmax(dim=-1).to(x.device)
        eos_positions = eos_positions.clamp(min=0, max=x.shape[1] - 1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return x[batch_idx, eos_positions]

    def forward(self, x, input_ids=None):
        x = self._to_bld(x)
        token_len = x.shape[1]
        if self.simplify_mole:
            summary = self._select_summary_token(x, input_ids=input_ids)
            x = self.linear(summary).unsqueeze(1).expand(-1, token_len, -1)
        else:
            # Project each token to expert logits
            x = self.linear(x)  # (B, L, N)
        # Project each token to expert logits
        # Apply activation: sigmoid for independent probabilities, softmax for probability distribution
        if self.activation == "sigmoid":
            gate = torch.sigmoid(x)
            if self.top_k_expert > 0:
                if self.top_k_expert < gate.shape[-1]:
                    topk_vals, topk_idx = torch.topk(gate, k=self.top_k_expert, dim=-1)
                    sparse_gate = torch.zeros_like(gate).scatter_(-1, topk_idx, topk_vals)
                else:
                    sparse_gate = gate
                gate = sparse_gate / sparse_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:  # softmax
            tau = F.softplus(self.tau) + 1e-6
            gate = F.softmax(x / tau, dim=-1)
        return gate, x


class PhatGooseGatingNetwork(nn.Module):
    def __init__(
        self,
        num_expert,
        sequence_length,
        hidden_dim,
        top_k_expert: int = 0,
        simplify_mole: bool = False,
    ):
        super(PhatGooseGatingNetwork, self).__init__()
        # Per-token expert selector with no bias:
        # logits = x @ U where U in R^{d x N}
        self.linear = nn.Linear(hidden_dim, num_expert, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.norm = nn.LayerNorm(hidden_dim)
        self.sequence_length = sequence_length
        self.top_k_expert = int(top_k_expert) if top_k_expert is not None else 0
        self.simplify_mole = simplify_mole
        # When set, train/evaluate using only one expert column u_j.
        self.active_expert_idx: int | None = None

    def _to_bld(self, x):
        # x: (L, B, d), (B, L, d), or (L*B, d)
        if len(x.shape) == 3:
            if x.shape[0] == self.sequence_length:
                # (L, B, d) -> (B, L, d)
                x = x.permute(1, 0, 2).contiguous()
            else:
                x = x.contiguous()
        else:
            # (L*B, d) -> (B, L, d)
            L_B, d = x.shape
            x = x.view(self.sequence_length, L_B // self.sequence_length, d)
            x = x.permute(1, 0, 2).contiguous()
        return x

    def _select_summary_token(self, x, input_ids=None):
        # x: (B, L, d)
        if input_ids is None:
            return x[:, 0, :]
        if input_ids.dim() != 2 or input_ids.shape[0] != x.shape[0]:
            return x[:, 0, :]
        eos_positions = input_ids.argmax(dim=-1).to(x.device)
        eos_positions = eos_positions.clamp(min=0, max=x.shape[1] - 1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return x[batch_idx, eos_positions]

    def forward(self, x, input_ids=None):
        x = self._to_bld(x)
        token_len = x.shape[1]
        if self.simplify_mole:
            summary = self._select_summary_token(x, input_ids=input_ids)
            logits = self.linear(summary).unsqueeze(1).expand(-1, token_len, -1)
        else:
            logits = self.linear(x)  # (B, L, N)
        if self.active_expert_idx is not None:
            # Keep exactly one expert active during in-domain phatgoose training.
            idx = int(self.active_expert_idx)
            masked = torch.full_like(logits, -30.0)
            masked[..., idx] = logits[..., idx]
            logits = masked
        gate = torch.sigmoid(logits)
        if self.top_k_expert > 0:
            if self.top_k_expert < gate.shape[-1]:
                topk_vals, topk_idx = torch.topk(gate, k=self.top_k_expert, dim=-1)
                sparse_gate = torch.zeros_like(gate).scatter_(-1, topk_idx, topk_vals)
            else:
                sparse_gate = gate
            gate = sparse_gate / sparse_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return gate, logits


class SigmoidRankGatingNetwork(nn.Module):
    def __init__(self, total_rank: int, sequence_length: int, hidden_dim: int, simplify_mole: bool = False):
        super(SigmoidRankGatingNetwork, self).__init__()
        # Per-layer rank selector: gate depends only on the current layer input (L, B, d)
        in_dim = hidden_dim if simplify_mole else sequence_length * hidden_dim
        self.linear = nn.Linear(in_dim, total_rank, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.sequence_length = sequence_length
        self.total_rank = total_rank
        self.simplify_mole = simplify_mole

    def _to_bld(self, x):
        # x: (L, B, d), (B, L, d), or (L*B, d)
        if len(x.shape) == 3:
            # Determine format based on shape - sequence_length is known
            if x.shape[0] == self.sequence_length:
                # (L, B, d) -> (B, L, d)
                x = x.permute(1, 0, 2).contiguous()
            # else: already (B, L, d), no permutation needed
            else:
                x = x.contiguous()
        else:
            # (L*B, d) -> (B, L, d)
            L_B, d = x.shape
            x = x.view(self.sequence_length, L_B // self.sequence_length, d)
            x = x.permute(1, 0, 2).contiguous()
        return x

    def _select_summary_token(self, x, input_ids=None):
        if input_ids is None:
            return x[:, 0, :]
        if input_ids.dim() != 2 or input_ids.shape[0] != x.shape[0]:
            return x[:, 0, :]
        eos_positions = input_ids.argmax(dim=-1).to(x.device)
        eos_positions = eos_positions.clamp(min=0, max=x.shape[1] - 1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return x[batch_idx, eos_positions]

    def forward(self, x, input_ids=None):
        x = self._to_bld(x)
        if self.simplify_mole:
            # Simplified path: gate on summary token only, shape (B, d).
            x = self._select_summary_token(x, input_ids=input_ids)
        else:
            # Flatten (B, L, d) -> (B, L*d)
            x = x.reshape(x.size(0), -1)
        # Project to rank logits
        logits = self.linear(x)  # (B, T)
        # Apply sigmoid for independent per-rank gating
        alpha = torch.sigmoid(logits)  # (B, T)
        return alpha, logits


class CrossAttnGating(nn.Module):
    def __init__(self, sequence_length, img_dim, txt_dim, num_experts, num_heads=8):
        super(CrossAttnGating, self).__init__()

        self.txt_proj = nn.Sequential(*[
            nn.Linear(txt_dim, 4*txt_dim),
            nn.ReLU(),
            nn.Linear(4*txt_dim, img_dim)
        ])

        self.hidden_dim = txt_dim
        self.sequence_length = sequence_length

        self.mha = nn.MultiheadAttention(img_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(*[nn.Linear(img_dim, 4*img_dim),nn.ReLU(),nn.Linear(4*img_dim, img_dim)])
        self.q_norm = nn.LayerNorm(img_dim)
        self.v_norm = nn.LayerNorm(img_dim)
        self.k_norm = nn.LayerNorm(img_dim)
        self.mlp_norm = nn.LayerNorm(img_dim)

        self.tau = nn.Parameter(torch.ones(1))
        self.linear = nn.Linear(img_dim, num_experts)

    def forward(self, x:torch.Tensor, txt_feats: torch.Tensor):
        # x: (L, B, d) or (L*B, d)
        if len(x.shape) == 3:
            # (L, B, d) -> (B, L, d)
            x = torch.permute(x, (1, 0, 2))
        else:
            # (L*B, d) -> (B, L, d)
            L_B, d = x.shape
            x = x.view(self.sequence_length, L_B // self.sequence_length, d)
            x = torch.permute(x, (1, 0, 2))

        x = x[:, 0, :].unsqueeze(1) #(B, 1, d)
        B = x.shape[0]  # image batch size
        # ensure txt_feats is 2D (M, d) before projection
        if len(txt_feats.shape) != 2:
            raise ValueError(f"txt_feats must be 2D (M, d), got shape {txt_feats.shape}")
        txt_feats_proj = self.txt_proj(txt_feats)  # (M, img_dim)
        txt_feats = txt_feats_proj.unsqueeze(0).expand(B, -1, -1)  # (B, M, img_dim)
        attn_out, _ = self.mha(self.q_norm(x), self.k_norm(txt_feats), self.v_norm(txt_feats))
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        x = x.squeeze()
        x = self.linear(x)
        tau = F.softplus(self.tau) + 1e-6
        gate = F.softmax(x/tau, dim=-1)
        return gate, x

