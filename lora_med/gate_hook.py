import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from lora_med.image_layers import ImagePlainMultiheadAttentionMED
from lora_med.text_layers import TextPlainMultiheadAttentionMED
from lora_med.constants import openclip_backbones

def entropy(q, eps=1e-8):
    # expect q in (B, K)
    # print(q.shape)
    q = q.clamp_min(eps)
    return torch.sum(- q * q.log(), dim=1)

def get_probability(elems, num_experts):
    prob = torch.zeros(num_experts)
    for elem in elems:
        prob[elem] += 1
    return prob / prob.sum()

def _margin_loss(g):
    top2 = torch.topk(g, k=2, dim=-1).values  # (B,2)
    margin = (top2[:, 0] - top2[:, 1])
    return (1.0 - margin).clamp_min(0).mean()

def _balance_kl(g):
    q = g.mean(dim=0)                      # (E,)
    q = q / (q.sum() + 1e-8)
    E = q.numel()
    return (q * torch.log(q.clamp_min(1e-8) * E)).sum()


_KNOWN_SEQ_LENS = {
    openclip_backbones[name][enc]["sequence_length"]
    for name in openclip_backbones
    for enc in ("vision", "text")
}

class GateValueHook:
    def __init__(self, expert_names: list[str]):
        self.expert_names = expert_names
        self.gate_values = {}
        self.layer_tensors = {}
        self.hooks = []

    def _create_hook_fn(self, module_name: str, training: bool):
        def hook_fn(module, input, output):
            if module_name not in self.gate_values:
                self.gate_values[module_name] = []
            # if training:
            self.gate_values[module_name].append(output)
            # else:
            #     self.gate_values[module_name].append(output.detach().cpu().clone())
        return hook_fn
    
    def _create_layer_tensor_hook_fn(self, module_name: str, training: bool):
        # currently ignore training to save VRAM
        def hook_fn(module, input, output):
            if module_name not in self.layer_tensors:
                self.layer_tensors[module_name] = []
            self.layer_tensors[module_name].append(input[0].detach().cpu().clone())
        return hook_fn

    def register_hooks(self, mole_mixtures: list[nn.Module], training: bool = False):
        for idx, module in enumerate(mole_mixtures):
            if hasattr(module, "gate_net"):
                if type(module) is ImagePlainMultiheadAttentionMED:
                    prefix = "image"
                else:
                    prefix = "text"
                self.hooks.append(
                    module.gate_net.register_forward_hook(self._create_layer_tensor_hook_fn(f"{prefix}_layer_{idx}", training)))
                self.hooks.append(
                    module.gate_net.register_forward_hook(self._create_hook_fn(f"{prefix}_layer_{idx}", training)))

    # Mixture of LoRA formula
    # def get_balance_loss(self):
    #     q = torch.concatenate([torch.stack(self.gate_values[key], dim=0) for key in self.gate_values.keys()], dim=0)
    #     q = torch.mean(q, dim=0)
    #     loss = - (1.0 / q.shape[-1]) * torch.log(q.clamp_min(1e-8)).sum()
    #     return loss

    # def get_balance_loss(self, img_domain=None, txt_domain=None, balance_reg=0.1, domain_reg=1,eps=1e-8):
    #     # Collect current-batch gate probs per layer
    #     img_per_layer = []
    #     txt_per_layer = []
    #     img_domain_ce = []
    #     txt_domain_ce = []
    #     for key in self.gate_values.keys():
    #         # stack over (possibly multiple calls in the same forward; keep the latest one)
    #         x = self.gate_values[key][-1]  # ([B, K], [B, K]) -> (gate, logits)
    #         if key.__contains__('image'):
    #             img_per_layer.append(x[0])  # [B, K]
    #             if img_domain is not None:
    #                 img_domain_ce.append(F.cross_entropy(x[1], img_domain))
    #         else:
    #             txt_per_layer.append(x[0])  # [B, K]
    #             if txt_domain is not None:
    #                 txt_domain_ce.append(F.cross_entropy(x[1], txt_domain))
    #
    #
    #     img_q = torch.stack(img_per_layer, dim=1).mean(dim=1).clamp_min(eps)  # [B, #Layers, K] -> [B, K]
    #     txt_q = torch.stack(txt_per_layer, dim=1).mean(dim=1).clamp_min(eps)  # [B, #Layers, K] -> [B, K]
    #     ce_like = -torch.log(img_q).sum(dim=-1).mean() - torch.log(txt_q).sum(dim=-1).mean()
    #     loss = balance_reg*ce_like
    #
    #     if img_domain is not None:
    #         img_domain_ce = torch.stack(img_domain_ce).mean()
    #         loss += img_domain_ce * domain_reg
    #     if txt_domain is not None:
    #         txt_domain_ce = torch.stack(txt_domain_ce).mean()
    #         loss += txt_domain_ce * domain_reg
    #
    #     return loss

    def get_balance_loss(
        self, img_domain=None, txt_domain=None, 
        balance_reg=0.1, domain_reg=0.1,eps=1e-8,
        mode="domain_ce"
    ):
        # Collect current-batch gate probs per layer
        # use this if we have no idea what the target distribution
        img_per_layer = []
        txt_per_layer = []

        # use this if we know the distribution from img_domain or txt_domain
        img_match_loss = []
        txt_match_loss = []
        img_domain_ce = []
        txt_domain_ce = []
        img_target, txt_target = None, None
        if img_domain is not None:
            img_domain_prob = get_probability(img_domain, len(self.expert_names)).to(img_domain.device)
            img_target = entropy(img_domain_prob.unsqueeze(0))
        if txt_domain is not None:
            txt_domain_prob = get_probability(txt_domain, len(self.expert_names)).to(txt_domain.device)
            txt_target = entropy(txt_domain_prob.unsqueeze(0))

        for key in self.gate_values.keys():
            # stack over (possibly multiple calls in the same forward; keep the latest one)
            x = self.gate_values[key][-1]  # ([B, K], [B, K]) or ([B, T+1, E], [B, T+1, E]) for joint gates
            
            if len(x[0].shape) == 3:
                if x[0].shape[1] in _KNOWN_SEQ_LENS:
                    # Token-wise gates: average across tokens to get (B, E)
                    x = (x[0].mean(dim=1), x[1].mean(dim=1))
                elif key.__contains__('image'):
                    # Joint gate output: extract image token (last token)
                    alpha_img = x[0][:, -1, :]  # (B_img, E) - image token alpha
                    logits_img = x[1][:, -1, :]  # (B_img, E) - image token logits
                    x = (alpha_img, logits_img)
                else:
                    # Text gates from joint gates: use all text tokens (first B_txt tokens)
                    # For now, average over text tokens
                    alpha_txt = x[0][:, :-1, :].mean(dim=1)  # (B_img, E) - average over text tokens
                    logits_txt = x[1][:, :-1, :].mean(dim=1)  # (B_img, E)
                    x = (alpha_txt, logits_txt)
            
            if key.__contains__('image'):
                # if img_domain is not None:
                if "domain_ce" in mode:
                    assert img_domain is not None
                    img_domain_ce.append(F.cross_entropy(x[1], img_domain))
                if "entropy_match" in mode:
                    assert img_domain is not None
                    img_match_loss.append(((entropy(x[0])-img_target)**2).mean())
                if "entropy_balance" in mode:
                    img_per_layer.append(x[0])  # [B, K]
            else:
                if "domain_ce" in mode:
                    assert txt_domain is not None
                    txt_domain_ce.append(F.cross_entropy(x[1], txt_domain))
                if "entropy_match" in mode:
                    assert txt_domain is not None
                    txt_match_loss.append(((entropy(x[0])-txt_target)**2).mean())
                if "entropy_balance" in mode:
                    txt_per_layer.append(x[0])  # [B, K]

        loss = 0
        message = {}
        if len(img_per_layer) > 0:
            img_q = torch.stack(img_per_layer, dim=1).mean(dim=1).clamp_min(eps)  # [B, #Layers, K] -> [B, K]
            img_entropy_balance = -torch.log(img_q).sum(dim=-1).mean() * balance_reg

            loss += img_entropy_balance
            message["img_entropy_balance"] = img_entropy_balance.item()
        if len(txt_per_layer) > 0:
            txt_q = torch.stack(txt_per_layer, dim=1).mean(dim=1).clamp_min(eps)  # [B, #Layers, K] -> [B, K]
            txt_entropy_balance = -torch.log(txt_q).sum(dim=-1).mean() * balance_reg
            
            loss += txt_entropy_balance
            message["txt_entropy_balance"] = txt_entropy_balance.item()

        if len(img_domain_ce) > 0:
            img_domain_ce = torch.stack(img_domain_ce).mean()
            loss += img_domain_ce * domain_reg
            message["img_domain_ce"] = img_domain_ce.item()
        if len(img_match_loss) > 0:
            img_match_loss = torch.stack(img_match_loss).mean()
            loss += img_match_loss * balance_reg
            message["img_entropy_match"] = img_match_loss.item()
        if len(txt_domain_ce) > 0:
            txt_domain_ce = torch.stack(txt_domain_ce).mean()
            loss += txt_domain_ce * domain_reg
            message["txt_domain_ce"] = txt_domain_ce.item()
        if len(txt_match_loss) > 0:
            txt_match_loss = torch.stack(txt_match_loss).mean()
            loss += txt_match_loss * balance_reg
            message["txt_entropy_match"] = txt_match_loss.item()

        return loss, message

    def clear_step(self):
        # call after computing the loss each step
        for k in list(self.gate_values.keys()):
            self.gate_values[k].clear()
        for k in list(self.layer_tensors.keys()):
            self.layer_tensors[k].clear()

    # def get_balance_loss(self,
    #                      w_spike: float = 1.0,
    #                      w_balance: float = 0.05,
    #                      domain: torch.Tensor = None,
    #                      spike: str = "entropy"):
    #     layer_losses = []
    #     for key, gs in self.gate_values.items():
    #         # print("****************************")
    #         # print(len(gs))
    #         # print("****************************")
    #         gs = gs[0]
    #         spike_terms, balance_terms = [], []
    #         domain_ce = []
    #         # --- spikiness ---
    #         if spike == "entropy":
    #             spike_terms.append(_entropy_mean(gs[0]))
    #         elif spike == "margin":
    #             spike_terms.append(_margin_loss(gs[0]))
    #         else:
    #             raise ValueError("spike must be 'entropy' or 'margin'")
    #
    #         if domain is not None:
    #             domain_ce.append(F.cross_entropy(gs[1], domain))
    #
    #         # --- load-balance across batch/examples ---
    #         balance_terms.append(_balance_kl(gs[0]))
    #
    #         # average across multiple routers in the same layer (if any)
    #         layer_spike = torch.mean(torch.stack(spike_terms))
    #         layer_bal = torch.mean(torch.stack(balance_terms))
    #         if domain is not None:
    #             layer_domain = torch.mean(torch.stack(domain_ce))
    #             # layer_losses.append(w_spike * layer_spike + w_balance * layer_bal + layer_domain)
    #             layer_losses.append(layer_domain)
    #         else:
    #             layer_losses.append(w_spike * layer_spike + w_balance * layer_bal)
    #
    #     # average across layers so hyperparameters are depth-invariant
    #     return torch.mean(torch.stack(layer_losses))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.gate_values.clear()
        self.layer_tensors.clear()

    def keys(self) -> list[str]:
        return list(self.gate_values.keys())

    def get_gate_values(self) -> dict[str, list[torch.Tensor]]:
        return self.gate_values

    def get_layer_tensors(self) -> dict[str, list[torch.Tensor]]:
        return self.layer_tensors

    def report_gate_values(self, file_path="gate_report.txt"):
        with open(file_path, "w") as f:
            for k in self.keys():
                f.write(f"At this module {k}\n")
                gate_values = self.get_gate_values()[k]
                gate_values = torch.concatenate(gate_values, dim=0)
                gate_values = gate_values.mean(dim=0)
                message = " ".join(
                    [f"{self.expert_names[i]}: {gate_values[i]:.4f}" for i in range(len(self.expert_names))]
                )
                f.write(message + "\n")

    def report_layer_tensors(
        self, indices: list[int], save_dir: str, 
        text_tokens: torch.Tensor=None, text:str=None,
        H = None, W = None
    ):
        assert (text_tokens is not None and text is not None) or (H is not None and W is not None)
        mode = "text" if text_tokens is not None and text is not None else "image"

        if mode == "text":
            text2heatmap = {}
        
        # indices are the sample we want to check
        indices = torch.tensor(indices)
        layer_tensors = self.get_layer_tensors()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for k in layer_tensors.keys():
            layer_tensors_ = torch.concatenate(layer_tensors[k], dim=0)
            layer_tensors_ = layer_tensors_[indices]
            
            for i in range(layer_tensors_.shape[0]):
                if mode == "text":
                    endIdx = text_tokens[i].argmax()
                    layer_tensors_i = layer_tensors_[i, :endIdx+1]
                    text2heatmap[text[i]] = layer_tensors_i.mean(dim=-1).detach().cpu().numpy()

                elif mode == "image":
                    layer_tensors_i = layer_tensors_[i]
                    layer_tensors_i = layer_tensors_i.mean(dim=-1).detach().cpu().numpy()
                    heatmap = layer_tensors_i.reshape(H, W)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(heatmap, cmap='viridis')
                    plt.colorbar()
                    plt.savefig(os.path.join(save_dir, f"heatmap_sample{i}_{k}.png"))
                    plt.close()
        
        if mode == "text":
            self._plot_text_heatmaps(text2heatmap, save_dir)

    def _plot_text_heatmaps(self, text2heatmap, save_dir, samples_per_image=5):
        """Plot token-level heatmaps for text samples."""
        texts = list(text2heatmap.keys())
        scores = list(text2heatmap.values())
        
        # Group samples into batches
        num_images = (len(texts) + samples_per_image - 1) // samples_per_image
        
        for img_idx in range(num_images):
            start = img_idx * samples_per_image
            end = min(start + samples_per_image, len(texts))
            
            batch_texts = texts[start:end]
            batch_scores = scores[start:end]
            max_length = max(len(s) for s in batch_scores)
            
            # Create figure
            fig_width = max(8, max_length * 0.8)
            fig_height = max(4, len(batch_texts) * 1.2)
            fig, axes = plt.subplots(len(batch_texts), 1, figsize=(fig_width, fig_height))
            if len(batch_texts) == 1:
                axes = [axes]
            
            for i, (text, score) in enumerate(zip(batch_texts, batch_scores)):
                ax = axes[i]
                tokens = text.split()
                score = score[:len(tokens)]
                
                # Normalize scores
                if len(score) > 0:
                    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
                
                # Draw bars
                for j, (token, s) in enumerate(zip(tokens, score)):
                    ax.barh(0.5, 1, height=0.8, left=j, 
                           color=plt.cm.Greys(s), alpha=0.8, 
                           edgecolor='black', linewidth=0.5)
                    ax.text(j + 0.5, 0.9, token, ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
                
                ax.set_xlim(0, len(tokens))
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{text[:50]}{'...' if len(text) > 50 else ''}", fontsize=10)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"text_heatmap_{img_idx}.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            