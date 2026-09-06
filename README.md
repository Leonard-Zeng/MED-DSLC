# MED-DSLC

**Multi-Expert-Domain Classification via Domain Supervision and Logit Calibration**

Zheng Zeng, Deepak Sridhar, and Nuno Vasconcelos · ECCV 2026

[Paper](https://arxiv.org/abs/2607.10985) · [Project website](https://leonard-zeng.github.io/MED-DSLC/) · [Pretrained weights](https://drive.google.com/file/d/1lgB9s06DUvzqjLSQVekA5Htjm5f6BUdI/view?usp=sharing)

MED-DSLC combines domain expert LoRA adapters with domain-supervised routing and logit calibration for classification across domains. This repository contains the training and evaluation code and dataset setup instructions.

### Main variants

The project uses the name **MED-DSLC**. Earlier paper files and the command-line identifiers below retain the implementation's older naming.

| Variant | `META_TYPE` | Logit scaling | Default checkpoint directory |
| --- | --- | --- | --- |
| Domain supervision | `MED` | Standard CLIP scale | `trained_weights/MED_standard/` |
| Domain supervision + logit calibration | `MED_LCDS` | Learned domain-wise scalar | `trained_weights/MED_LCDS_domain-wise/` |

Training and evaluation also support these methods and baselines:

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
training/              # training entrypoint and trainers
evaluation/            # benchmark evaluation entrypoint
lora_med/              # MED routing and LoRA mixture layers
lora_mixture/          # mean-LoRA mixture utilities
loralib/               # base LoRA load/save/apply utilities
clip_lora_datasets/    # benchmark dataset definitions
main.py                # per-domain expert LoRA training entrypoint
train_expert_loras.sh  # convenience wrapper for prerequisite expert LoRAs
train_meta.sh          # train trainable meta methods
run_meta_inference.sh  # evaluate meta methods and inference-only baselines
meta/descriptions/     # optional prompt descriptions
KnOTS/                 # upstream KnOTS git submodule
website/index.html     # standalone project website, deployed with GitHub Pages
```

Large or generated paths are intentionally ignored by git:

```text
data/
output-rank2/
trained_weights/
results/
training/weights/
training/logs/
```

## Setup

```bash
uv sync
source .venv/bin/activate
git submodule update --init --recursive
```

This project pins Python 3.10 in `.python-version` and declares dependencies in `pyproject.toml`. These setup commands install the development environment into `.venv`.

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
WEIGHTS_ROOT=trained_weights
OUT_ROOT=results
```

`MODEL_ROOT` must contain prerequisite per-domain expert LoRA checkpoints:

```text
${MODEL_ROOT}/${domain}/16shots/seed1/lora_weights.pt
```

### Pretrained weights

Download the [pretrained weights](https://drive.google.com/file/d/1lgB9s06DUvzqjLSQVekA5Htjm5f6BUdI/view?usp=sharing) and place the checkpoints in the default paths listed above.

After environment and dataset setup, evaluate the pretrained models directly:

```bash
META_TYPE=MED_LCDS ./run_meta_inference.sh
META_TYPE=MED ./run_meta_inference.sh
```

Set `DATA_ROOT=/path/to/data` if needed. The base CLIP backbone is downloaded separately by the loader.

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
META_OUTPUT_DIR=trained_weights/MED_custom \
./train_meta.sh
```

### 3. Run Meta Inference

Evaluate one meta method or inference-only baseline:

```bash
./run_meta_inference.sh
```

By default this evaluates `MED` in both `cross_domain` and `in_domain` modes, over the `all`, `base`, and `new` splits, and saves wrongly classified images.

Select the method with `META_TYPE` and the meta network/checkpoint with `META_WEIGHT_PATH`:

```bash
META_TYPE=MED_LCDS \
META_CONFIG_PATH=meta_config.example.json \
META_WEIGHT_PATH=trained_weights/MED_LCDS_domain-wise \
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
  "meta_weight_path": "/path/to/meta_network_dir"
}
```

`meta_weight_path` is optional in the JSON and is used by direct `evaluation/evaluate.py` calls when `--meta_weight_path` is omitted. The shell wrapper supplies a method-specific default, so set `META_WEIGHT_PATH` explicitly when choosing a checkpoint through `run_meta_inference.sh`.

For `MED` and `MED_LCDS`, pass a checkpoint **directory** containing `meta_net.pt` and, for `MED_LCDS`, its matching `logit_scalar.pt` from the same run and saved epoch. The loader prefers `meta_net_latest.pt` and `logit_scalar_latest.pt` when present; to evaluate a specific saved pair, place it in a separate directory without competing latest files. Both variants also require the per-domain expert adapters under `MODEL_ROOT` or the paths in `META_CONFIG_PATH`.

Override artifact roots without changing source:

```bash
DATA_ROOT=/path/to/data MODEL_ROOT=/path/to/output-rank2/base_split/vitb16 ./train_meta.sh
WEIGHTS_ROOT=/path/to/trained_weights OUT_ROOT=/path/to/results ./run_meta_inference.sh
```

## BibTeX

```
@inproceedings{zengeccv26,
      author = {Zeng, Zheng and Sridhar, Deepak and Vasconcelos, Nuno},
      title = {MED-DSLC: Multi-Expert-Domain Classification via Domain Supervision and Logit Calibration},
      booktitle = {European Conference on Computer Vision},
      year = {2026},
  }
```

## Acknowledgements

This codebase builds on the base implementation provided by [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA). We are thankful to the CLIP-LoRA authors for making their implementation available.
