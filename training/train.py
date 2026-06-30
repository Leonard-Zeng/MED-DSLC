import argparse
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.logging_utils import setup_logging, tee_stdout_stderr
from training.config import METHOD_CHOICES, DEFAULT_DESCRIPTIONS_DIR
from training import config
from training.methods import get_trainer
from device_utils import device_arg, resolve_device
from training.utils import (
    domains_from_meta_config,
    expert_weights_from_meta_config,
    load_meta_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entrypoint")
    parser.add_argument("--method", type=str, required=True, choices=METHOD_CHOICES)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./output-rank2/base_split/vitb16")
    parser.add_argument(
        "--meta_config_path",
        type=str,
        default=None,
        help="Optional JSON config with ordered domains and exact expert LoRA paths.",
    )
    parser.add_argument(
        "--output_weights_dir",
        type=str,
        default=None,
        help="Optional exact output directory for this meta checkpoint.",
    )
    parser.add_argument("--subsample", type=str, default="base", choices=["all", "base", "new"])
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument(
        "--include_new_ds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include opt-in new training datasets: CUB200 and RESISC45. AIBD Cars is eval-only.",
    )
    parser.add_argument(
        "--model_shots",
        type=int,
        default=16,
        help="Shots value used only to locate pretrained LoRA weights under --model_dir (defaults to --shots).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--dataset_mode", type=str, default="balanced", choices=["balanced", "combined"])
    parser.add_argument("--prompt_source", type=str, default="default", choices=["default", "descriptions", "combined"])
    parser.add_argument("--descriptions_dir", type=str, default=DEFAULT_DESCRIPTIONS_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--single_lora_rank",
        type=int,
        default=20,
        help="LoRA rank used by single_lora methods.",
    )
    parser.add_argument(
        "--single_lora_init_from_pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize single_lora weights from the mean of discovered expert LoRA checkpoints under --model_dir.",
    )
    parser.add_argument("--random_k", type=int, default=50)
    parser.add_argument(
        "--class_filter_path",
        type=str,
        default=None,
        help="Optional JSON class filter with {'domains': {domain: [classnames...]}}.",
    )
    parser.add_argument("--loss_mode", type=str, default="hybrid", choices=["domain_only", "clip_only", "hybrid"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for domain/balance loss (gating mole methods).")
    parser.add_argument(
        "--use_focal_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Focal Loss instead of vanilla cross entropy for CLIP loss.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for Focal Loss (default: 0.25).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (default: 2.0).",
    )
    parser.add_argument(
        "--mole_mode",
        type=str,
        default="domain_ce",
        choices=[
            "",
            "domain_ce",
            "domain_ce_entropy_match",
            "domain_ce_entropy_balance",
            "entropy_match",
            "entropy_balance",
        ],
    )
    parser.add_argument(
        "--logit_scalar",
        type=str,
        default="standard",
        choices=["standard", "domain-wise", "input-wise", "affine-wise", "img-wise"],
        help="Logit scaling mode for MED-family methods.",
    )
    parser.add_argument(
        "--include_base_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include base CLIP as zero expert / fallback when building mixtures.",
    )
    parser.add_argument(
        "--simplify_mole",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use summary-token gating for MED-family methods (image CLS, text EOS).",
    )
    parser.add_argument(
        "--mole_stage2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stage-2 MED training: freeze gate networks and train LoRA parameters.",
    )
    parser.add_argument(
        "--gate_dir",
        type=str,
        default=None,
        help="Directory containing meta_net.pt to load gating networks for stage-2 MED; used when --mole_stage2 is set (e.g. path from a prior stage-1 run).",
    )
    parser.add_argument(
        "--val_cross_domain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run cross-domain validation loss each epoch.",
    )
    parser.add_argument(
        "--val_shots",
        type=int,
        default=32,
        help="Max samples per domain for cross-domain validation (<=0 for full set).",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=32,
        help="Batch size for cross-domain validation (defaults to --batch_size).",
    )
    parser.add_argument("--device", **device_arg())
    parser.add_argument("--log_dir", type=str, default="training/logs")
    parser.add_argument("--weights_dir", type=str, default="training/weights")
    return parser.parse_args()


def normalize_method_args(args):
    """Map public MED-DSLC method names onto the shared trainer implementations."""
    if args.method == "MED":
        args.mole_mode = "domain_ce"
        args.logit_scalar = "standard"
    elif args.method == "MED_LCDS":
        args.mole_mode = "domain_ce"
        args.logit_scalar = "domain-wise"
    elif args.method == "mole":
        args.mole_mode = "entropy_balance"
        args.logit_scalar = "standard"
    elif args.method == "single_lora":
        args.logit_scalar = "standard"
    elif args.method == "single_lora_from_pretrained":
        args.single_lora_init_from_pretrained = True
        args.logit_scalar = "standard"
    return args


def main():
    args = normalize_method_args(parse_args())
    args.device = str(resolve_device(args.device))
    if bool(getattr(args, "include_new_ds", False)):
        for domain in config.NEW_DATASET_ORDER:
            if domain not in config.DOMAIN_ORDER:
                config.DOMAIN_ORDER.append(domain)
    if args.model_shots is None:
        args.model_shots = args.shots
    meta_config = load_meta_config(args.meta_config_path)
    args.domain_order = domains_from_meta_config(meta_config, config.DOMAIN_ORDER)
    args.expert_weight_paths = expert_weights_from_meta_config(meta_config)
    log_path = os.path.join(args.log_dir, f"{args.method}.log")
    weights_root = os.path.join(args.weights_dir, f"{args.method}_{args.logit_scalar}")
    weights_dir = os.path.abspath(args.output_weights_dir) if args.output_weights_dir else weights_root
    os.makedirs(weights_dir, exist_ok=True)

    with tee_stdout_stderr(log_path):
        setup_logging(log_path)
        trainer = get_trainer(args.method)
        trainer(args, weights_dir)


if __name__ == "__main__":
    main()
