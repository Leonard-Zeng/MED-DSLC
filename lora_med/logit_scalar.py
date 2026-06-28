import os
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGIT_SCALAR_FILENAME = "logit_scalar.pt"
LOGIT_SCALAR_LATEST_FILENAME = "logit_scalar_latest.pt"


def infer_clip_feature_dim(model: nn.Module, default: int = 512) -> int:
    text_projection = getattr(model, "text_projection", None)
    if isinstance(text_projection, torch.Tensor):
        if text_projection.ndim == 2:
            return int(text_projection.shape[1])
        if text_projection.ndim == 1:
            return int(text_projection.shape[0])
    return int(default)


class MEDLogitScalar(nn.Module):
    def __init__(self, mode: str, feature_dim: int, num_domains: int):
        super().__init__()
        self.mode = mode
        self.feature_dim = int(feature_dim)
        self.num_domains = int(num_domains)

        if self.mode == "domain-wise" or self.mode == "affine-wise":
            self.img_domain_head = nn.Linear(self.feature_dim, self.num_domains)
            self.txt_domain_head = nn.Linear(self.feature_dim, self.num_domains)
            self.img_domain_scalar_raw = nn.Parameter(torch.zeros(self.num_domains))
            self.txt_domain_scalar_raw = nn.Parameter(torch.zeros(self.num_domains))
            if self.mode == "affine-wise":
                self.bias_scalar = nn.Parameter(torch.zeros(self.num_domains))
                self.bias_scalar_text = nn.Parameter(torch.zeros(self.num_domains))
        elif self.mode == "img-wise":
            self.img_domain_head = nn.Linear(self.feature_dim, self.num_domains)
            # self.txt_domain_head = nn.Linear(self.feature_dim, self.num_domains)
            self.img_domain_scalar_raw = nn.Parameter(torch.zeros(self.num_domains))
            # self.txt_domain_scalar_raw = nn.Parameter(torch.zeros(self.num_domains))
        elif self.mode == "input-wise":
            self.img_scalar_head = nn.Linear(self.feature_dim, 1)
            self.txt_scalar_head = nn.Linear(self.feature_dim, 1)
            nn.init.zeros_(self.img_scalar_head.weight)
            nn.init.zeros_(self.img_scalar_head.bias)
            nn.init.zeros_(self.txt_scalar_head.weight)
            nn.init.zeros_(self.txt_scalar_head.bias)
        elif self.mode != "standard":
            raise ValueError(f"Unknown logit scalar mode: {self.mode}")

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_features_norm: torch.Tensor,
        text_features_norm: torch.Tensor,
        image_domain: torch.Tensor | None = None,
        text_domain: torch.Tensor | None = None,
    ):
        base_similarity = image_features_norm @ text_features_norm.T

        if self.mode == "standard":
            zero = base_similarity.new_tensor(0.0)
            return base_similarity, zero

        if self.mode == "domain-wise" or self.mode == "affine-wise" or self.mode == "img-wise":
            img_domain_logits = self.img_domain_head(image_features)
            if self.mode == "domain-wise" or self.mode == "affine-wise":
                txt_domain_logits = self.txt_domain_head(text_features)
                pred_txt_domain = txt_domain_logits.argmax(dim=-1)

            pred_img_domain = img_domain_logits.argmax(dim=-1)
            
            img_scalar = torch.exp(self.img_domain_scalar_raw[pred_img_domain])
            if self.mode == "domain-wise" or self.mode == "affine-wise":
                txt_scalar = torch.exp(self.txt_domain_scalar_raw[pred_txt_domain])
                logits = base_similarity * (img_scalar[:, None] * txt_scalar[None, :])
            else:
                logits = base_similarity * (img_scalar[:, None])
            if self.mode == "affine-wise":
                logits += torch.tanh(self.bias_scalar[pred_img_domain])[:, None] / 2.
                logits += torch.tanh(self.bias_scalar_text[pred_txt_domain])[None, :] / 2.

            aux_loss = logits.new_tensor(0.0)
            if image_domain is not None:
                aux_loss = aux_loss + F.cross_entropy(img_domain_logits, image_domain)
            if text_domain is not None:
                aux_loss = aux_loss + F.cross_entropy(txt_domain_logits, text_domain)
            return logits, aux_loss

        img_scalar = torch.exp(self.img_scalar_head(image_features).squeeze(-1))
        txt_scalar = torch.exp(self.txt_scalar_head(text_features).squeeze(-1))
        logits = base_similarity * (img_scalar[:, None] * txt_scalar[None, :])
        zero = logits.new_tensor(0.0)
        return logits, zero


def save_logit_scalar(
    weights_dir: str,
    module: MEDLogitScalar,
    filename: str = LOGIT_SCALAR_FILENAME,
) -> str:
    save_path = os.path.join(os.path.abspath(weights_dir), filename)
    payload = {
        "mode": module.mode,
        "feature_dim": int(module.feature_dim),
        "num_domains": int(module.num_domains),
        "state_dict": module.state_dict(),
    }
    torch.save(payload, save_path)
    print(f"Saved logit scalar weights to {save_path}")
    return save_path


def load_logit_scalar(
    meta_weight_path: str,
    mode: str,
    feature_dim: int,
    num_domains: int,
    device: torch.device | str,
) -> MEDLogitScalar | None:
    meta_weight_path = os.path.abspath(meta_weight_path)
    latest_file = os.path.join(meta_weight_path, LOGIT_SCALAR_LATEST_FILENAME)
    weight_file = latest_file if os.path.exists(latest_file) else os.path.join(meta_weight_path, LOGIT_SCALAR_FILENAME)

    if mode == "standard":
        best_file = os.path.join(meta_weight_path, LOGIT_SCALAR_FILENAME)
        if os.path.exists(weight_file) or os.path.exists(best_file):
            print(
                f"[WARN] Detected logit scalar weights under {meta_weight_path}, "
                "but --logit_scalar standard was requested. "
                "Ignoring logit scalar weights."
            )
        return None

    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"Expected logit scalar weight file not found for --logit_scalar {mode}: {weight_file}"
        )

    payload = torch.load(weight_file, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        payload_mode = payload.get("mode", None)
        payload_domains = payload.get("num_domains", None)
    elif isinstance(payload, dict):
        state_dict = payload
        payload_mode = None
        payload_domains = None
    else:
        raise ValueError(f"Unexpected logit scalar payload format: {weight_file}")

    if payload_mode is not None and payload_mode != mode:
        raise ValueError(
            f"logit_scalar mode mismatch: checkpoint has {payload_mode}, but requested {mode}"
        )
    if payload_domains is not None and int(payload_domains) != int(num_domains):
        raise ValueError(
            f"logit_scalar num_domains mismatch: checkpoint has {payload_domains}, expected {num_domains}"
        )

    module = MEDLogitScalar(mode=mode, feature_dim=feature_dim, num_domains=num_domains)
    module.load_state_dict(state_dict, strict=True)
    module = module.to(device)
    module.eval()
    return module
