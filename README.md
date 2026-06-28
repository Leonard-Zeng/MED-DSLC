# MED-LCDS

Curated MED-LCDS import from `/home/leo/svcl/clip-lora`.

This repository keeps only the training and evaluation path needed to recreate the selected `final_results` experiments:

- `MED`
- `MED_LCDS`
- `mole`
- `single_lora`
- `single_lora_from_pretrained`
- `phatgoose`
- `base_clip` inference
- `expert_lora` results under `expert-lora/`
- `lora_mean` results under `mean-lora/`
- `knots_svd` results under `knots-tries/` and `knots-dare-tries/`

## Layout

```text
training/              # selected training entrypoint and trainers
evaluation/            # selected benchmark evaluation entrypoint
lora_med/              # renamed MED/MED LoRA mixture layers
lora_mixture/          # mean-LoRA mixture utilities
loralib/               # base LoRA load/save/apply utilities
clip_lora_datasets/    # benchmark dataset definitions
main.py                # per-domain expert LoRA training entrypoint
train_expert_loras.sh  # convenience wrapper for prerequisite expert LoRAs
train_meta.sh          # train trainable meta methods
run_meta_inference.sh  # evaluate meta methods and inference-only baselines
meta/descriptions/     # optional prompt descriptions
KnOTS/                 # upstream KnOTS git submodule
```

Large or generated paths are intentionally ignored by git:

```text
data/
output-rank2/
final_weights/
final_results/
training/weights/
training/logs/
```

## Setup

```bash
uv sync
git submodule update --init --recursive
```

This project pins Python 3.10 in `.python-version` and declares dependencies in `pyproject.toml`. The PyTorch packages are resolved from the CUDA 11.7 PyTorch index through `uv`.

If `uv` is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can still install from the legacy requirements file if needed:

```bash
uv pip install -r requirements.txt
```

Prepare datasets following `DATASETS.md`. The default script paths are:

```bash
DATA_ROOT=./data
MODEL_ROOT=./output-rank2/base_split/vitb16
WEIGHTS_ROOT=final_weights
OUT_ROOT=final_results
```

`MODEL_ROOT` must contain prerequisite per-domain expert LoRA checkpoints:

```text
${MODEL_ROOT}/${domain}/16shots/seed1/lora_weights.pt
```

### 1. Train Expert LoRAs

Train the prerequisite per-domain expert LoRAs with rank 2 into the default layout:

```bash
./train_expert_loras.sh
```

The wrapper saves to `SAVE_ROOT=./output-rank2/base_split`, producing the default `MODEL_ROOT=./output-rank2/base_split/vitb16` used by the meta scripts.

### 2. Train Meta Methods

Train the meta methods that learn checkpoints:

```bash
./train_meta.sh
```

By default this trains one method, `MED`.

Select a different trainable meta method with `META_TYPE`:

```bash
META_TYPE=MED_LCDS ./train_meta.sh
```

Use `META_CONFIG_PATH` to choose the ordered datasets and exact expert LoRA checkpoint paths:

```bash
META_TYPE=MED \
META_CONFIG_PATH=meta_config.example.json \
META_OUTPUT_DIR=final_weights/MED_custom \
./train_meta.sh
```

### 3. Run Meta Inference

Evaluate one meta method or inference-only baseline:

```bash
./run_meta_inference.sh
```

By default this evaluates `MED`.

Select the method with `META_TYPE` and the meta network/checkpoint with `META_WEIGHT_PATH`:

```bash
META_TYPE=MED_LCDS \
META_CONFIG_PATH=meta_config.example.json \
META_WEIGHT_PATH=final_weights/MED_LCDS_domain-wise \
./run_meta_inference.sh
```

`base_clip` is vanilla CLIP inference over the same meta benchmark. It uses the existing `evaluation/base.py` path and does not require `--meta_weight_path`.

The meta config format is:

```json
{
  "domains": ["caltech101", "eurosat"],
  "expert_weights": {
    "caltech101": "/path/to/caltech101/lora_weights.pt",
    "eurosat": "/path/to/eurosat/lora_weights.pt"
  },
  "meta_weight_path": "/path/to/meta_network_dir_or_checkpoint"
}
```

`meta_weight_path` is optional in the JSON; `META_WEIGHT_PATH` overrides it for inference.

Override artifact roots without changing source:

```bash
DATA_ROOT=/home/leo/data MODEL_ROOT=/path/to/output-rank2/base_split/vitb16 ./train_meta.sh
WEIGHTS_ROOT=/path/to/final_weights OUT_ROOT=/path/to/final_results ./run_meta_inference.sh
```

## Acknowledgements

This codebase builds on the base implementation provided by [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA). We are thankful to the CLIP-LoRA authors for making their implementation available.
