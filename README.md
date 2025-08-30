Of course. Here is the revised and cleaned-up version of your README.

-----

# Superposition Project: SAE Steering on Qwen3

This repository contains experiments on steering the **`Qwen/Qwen3-4B-Instruct-2507`** model using a Sparse Autoencoder (SAE).

The core process involves five steps:

1.  Cache hidden activations from a decoder layer.
2.  Train a small SAE on these activations.
3.  Identify and select candidate features (e.g., a "code-writing" feature).
4.  Inject a scaled feature vector back into the model during a forward pass.
5.  Measure the resulting **interference** with other tasks (math logit margin) and the impact on output formatting (code/LaTeX validity).

All artifacts, including datasets, model weights, and analysis results (`JSON`/`CSV`/plots), are located in the `results/` directory for full transparency.

-----

## TL;DR: Key Findings

  * **Model**: `Qwen/Qwen3-4B-Instruct-2507`
  * **Hook Point**: Decoder layer **L18** (hidden size: **2560**)
  * **SAE**: **4096** features, trained on activations from **112** mixed code and math prompts.
  * **Steering**: We steered the model using a learned **`code_feature`** versus a **`random_control`** vector, sweeping the activation strength (`alpha`) across a symmetric range.
  * **Results**:
      * The `code_feature` only slightly nudges the last-token math logit margin. Surprisingly, the `random_control` vector often had a larger impact.
      * LaTeX formatting is highly robust, with the model consistently producing leading LaTeX (`$$...`) at a rate near **100%** across all alpha values for mixed prompts.

See the plots for more details:

  * `results/plots/interference_all_compare.png`
  * `results/plots/interference_fine_lfirst.png`

-----

## Repository Layout

```
.
├── data/
│   └── prompts.json              # 112 mixed code/math prompts
├── logs/
├── results/
│   ├── activations/              # Cached activations (*.pt)
│   ├── saes/                     # SAE checkpoints (sae_simple.pt, sae_decoder.pt)
│   ├── runs/                     # Optional per-prompt dumps (*.jsonl)
│   ├── plots/                    # Figures (PNGs)
│   └── *.json, *.csv             # Aggregate outputs
├── scripts/
│   ├── 01_activation_caching.py  # Cache L18 activations
│   ├── 02_train_sae.py           # Train SimpleSAE (d_sae=4096)
│   ├── 02b_pick_features.py      # Pick candidate feature indices
│   ├── 03_steering_and_eval.py   # Run steering sweeps and compute metrics
│   └── 03c_dump_lfirst.py        # Optional: Dump per-prompt LaTeX-first data
├── utils/                        # Small validators/helpers
├── README.md
└── requirements.txt
```

-----

## Environment Setup

  * **GPU**: H200 (141 GB) recommended for main runs; A100 is also sufficient.
  * **CUDA/Driver**: CUDA 12.8 / Driver 570.xx or newer.
  * **Python**: 3.11–3.12 (the development VM used 3.12).
  * **Key Libraries**: `torch==2.8.0`, `transformers==4.55.4`, `transformer-lens`, `sae-lens`.

#### Installation

To get started, create a virtual environment and install the dependencies.

```bash
# Set up optional caches for faster Hugging Face downloads
export HF_HOME=/data/hf
export HF_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_CACHE=/data/hf/transformers

# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

-----

## End-to-End Pipeline

> **Note**: All commands should be run from the repository root within the activated virtual environment. The GPU is auto-detected.

### 1\. Cache Activations

```bash
python scripts/01_activation_caching.py
# Output → results/activations/prompt_*.pt
```

### 2\. Train the SAE

```bash
python scripts/02_train_sae.py
# Output → results/saes/sae_simple.pt
```

### 3\. Pick Candidate Features

This script identifies features that activate strongly on code or math prompts.

```bash
python scripts/02b_pick_features.py
# Output → results/feature_candidates.json
```

The output JSON contains lists of feature indices:

```json
{
  "code_candidates": [123, 456, ...],
  "math_candidates": [789, 101, ...]
}
```

### 4\. Run Steering and Evaluation

Perform a basic sweep using the learned `code_feature` and a `random_control` baseline.

```bash
FEATURES=code_feature,random_control \
ALPHAS="-4,-3,-2,-1,0,1,2,3,4" \
MAX_NEW_TOKENS=64 \
python scripts/03_steering_and_eval.py

# Output → results/interference_results.json
```

**(Optional)** To generate a fine-grained sweep and an overlay plot for LaTeX-first rate:

```bash
# Run the dump script
FEATURES=code_feature \
MIXED_LIMIT=24 \
ALPHAS="-4,-3.5,-3,-2.5,-2,-1.5,-1,0,1" \
MAX_NEW_TOKENS=64 \
RUN_DUMP=results/runs/fine_lfirst.jsonl \
python scripts/03c_dump_lfirst.py

# Generate the plot
python scripts/generate_overlay_plot.py # (Assuming the logic is moved to a script)
```

> **Note on Math Feature**: Steering with `FEATURES=math_feature,random_control` is supported but experimental. Ensure `MATH_FEATURE_IDX` is set in your environment or read from `feature_candidates.json`.

-----

## Metrics Explained

  * **Math Logit Margin (Last Token)**: For the final token in the prompt, this is `max(logits[math_tokens]) - max(logits[non_math_tokens])`, averaged across all prompts. It measures how strongly the model prefers a math-related token over any other token.
  * **Code/LaTeX Validity**: Simple string-based validators that check for well-formed fenced code blocks (` ```...``` `) and LaTeX blocks (`$$...$$`).
  * **LaTeX-First Rate**: The proportion of model responses that begin with a LaTeX delimiter (`$$`) immediately at the turn boundary.

-----

## How Steering Works

We register a forward **pre-hook** at decoder **layer L18**. During the forward pass, we add the vector $\\alpha \\cdot v$ to the residual stream, where:

  * $v$ is the decoder-weight column corresponding to a selected SAE feature.
  * $\\alpha$ is the steering coefficient (alpha).

The **random control** baseline uses a random unit vector for $v$ of the same dimensionality to isolate the effect of the specific feature.

-----

## Artifacts

  * **Aggregate Data**
      * `results/interference_results.json`
      * `results/interference_results_all.json`
      * `results/interference_all_compare.csv`
      * `results/interference_results_fine_lfirst.csv`
  * **Per-Prompt Dumps**
      * `results/runs/fine_lfirst.jsonl`
  * **Plots**
      * `results/plots/interference_all_compare.png`
      * `results/plots/interference_fine_lfirst.png`

-----

## Troubleshooting

  * **Error: “No `W_dec` in checkpoint”**
    The training script saves a `state_dict`, but the steering script may expect a raw decoder weight tensor. Run this snippet once to extract it:

    ```python
    import torch

    # Load the SimpleSAE checkpoint
    ck = torch.load("results/saes/sae_simple.pt", map_location="cpu")
    sd = ck.get("state_dict", {})

    # Find the decoder weight tensor
    W = sd.get("W_dec") or sd.get("decoder.weight")
    if W is None:
        # Fallback: find the largest 2D tensor
        W = max((v for v in sd.values() if getattr(v, "ndim", 0) == 2), key=lambda t: t.numel())

    # Ensure correct shape (d_sae, d_in) and dtype
    W = W.float()
    if W.shape == (2560, 4096):
        W = W.T # Transpose if necessary

    # Save the extracted decoder
    torch.save({"W_dec": W, "d_in": 2560, "d_sae": 4096}, "results/saes/sae_decoder.pt")
    ```

  * **Long-Running Jobs on a VM**
    Always use a terminal multiplexer like `tmux` or run your command with `nohup`. This prevents your job from being killed if your SSH connection drops.

    ```bash
    # Start a new tmux session named "mats"
    tmux new -s mats

    # Run your commands...

    # Detach from the session: Ctrl-b, then d
    # Re-attach later: tmux attach -t mats
    ```

-----

## Changelog

  * **v0.1.2**: Added math-feature sweep and comparison plot.
  * **v0.1.1**: Added fine-grained sweep, LaTeX-first dump, and overlay plot.
  * **v0.1.0**: Initial GPU-ready version with SimpleSAE refactor and stable paths.

-----

## License

This project is licensed under the **Apache License 2.0**. See the [`LICENSE`](https://www.google.com/search?q=./LICENSE) file for details.

-----

## Acknowledgments

  * The **Alibaba/Qwen team** for the Qwen/Qwen3-4B-Instruct model.
  * The **Hugging Face** ecosystem for its tools and model hub.
  * The creators of **TransformerLens** and **SAE-Lens** for providing helpful libraries and design patterns.

-----

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@software{superposition_project_2025,
  title  = {superposition_project: SAE steering on Qwen3-4B-Instruct},
  author = {Escobedo, Salvador},
  year   = {2025},
  url    = {https://github.com/svdrecbd/superposition_project}
}
```
