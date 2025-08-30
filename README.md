```markdown
# superposition_project

Sparse-autoencoder (SAE) steering experiments on **Qwen/Qwen3-4B-Instruct-2507**.

We:
1) cache hidden activations from a decoder layer,  
2) train a small SAE,  
3) pick candidate features,  
4) inject a scaled feature vector during forward pass, and  
5) measure **interference** (math logit margin) and formatting validity (code/LaTeX).

All artifacts (JSON/CSV/plots) live in `results/` for full transparency.

---

## TL;DR (what we found)

- Model: **Qwen/Qwen3-4B-Instruct-2507**, hook at **decoder layer L18** (hidden size **2560**).
- SAE: **4096** units trained on cached activations from **112** mixed code/math prompts.
- Steering: learned **code_feature** vs **random_control**, sweeping α across a symmetric range.
- On our prompt set, **code_feature** nudges last-token math margin only slightly; **random_control** often moves it more.
- LaTeX formatting is robust: **leading LaTeX rate ~1.0** across alphas for mixed prompts.

See plots under:
- `results/plots/interference_all_compare.png`
- `results/plots/interference_fine_lfirst.png`

---

## Repository layout

```

data/
prompts.json                     # 112 mixed code/math prompts
logs/
results/
activations/                     # cached activations (*.pt)
saes/                            # SAE checkpoints (sae\_simple.pt, sae\_decoder.pt)
runs/                            # optional per-prompt dumps (*.jsonl)
plots/                           # figures (PNG)
\*.json, \*.csv                    # aggregate outputs
scripts/
01\_activation\_caching.py         # cache L18 activations
02\_train\_sae.py                  # train SimpleSAE (d\_sae=4096)
02b\_pick\_features.py             # pick candidate feature indices
03\_steering\_and\_eval.py          # steering sweeps + metrics
03c\_dump\_lfirst.py               # optional LaTeX-first per-prompt dump
utils/                           # small validators/helpers
README.md
requirements.txt

````

---

## Environment

- **GPU**: H200 (141 GB) for main runs; A100 works too.
- **CUDA/Driver**: CUDA 12.8 / 570.xx.
- **Python**: 3.11–3.12 (VM used 3.12).
- **Key libs**: `torch==2.8.0`, `transformers==4.55.4`, `transformer-lens`, `sae-lens`.

Optional caches (faster HF downloads):

```bash
export HF_HOME=/data/hf
export HF_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_CACHE=/data/hf/transformers
````

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## End-to-end pipeline

> Run from repo root inside the venv. GPU is auto-detected.

### 1) Cache activations

```bash
python scripts/01_activation_caching.py
# → results/activations/prompt_*.pt
```

### 2) Train the SAE

```bash
python scripts/02_train_sae.py
# → results/saes/sae_simple.pt
```

### 3) Pick candidate features

```bash
python scripts/02b_pick_features.py
# → results/feature_candidates.json
```

The JSON has lists like:

```json
{ "code_candidates": [...], "math_candidates": [...] }
```

### 4) Steering + evaluation

Basic sweep (learned code feature vs random control):

```bash
FEATURES=code_feature,random_control \
ALPHAS="-4,-3,-2,-1,0,1,2,3,4" \
MAX_NEW_TOKENS=64 \
python scripts/03_steering_and_eval.py

# → results/interference_results.json
```

Optional: per-prompt LaTeX-first dump + overlay plot:

```bash
FEATURES=code_feature \
MIXED_LIMIT=24 \
ALPHAS="-4,-3.5,-3,-2.5,-2,-1.5,-1,0,1" \
MAX_NEW_TOKENS=64 \
RUN_DUMP=results/runs/fine_lfirst.jsonl \
python scripts/03c_dump_lfirst.py

python - <<'PY'
import json, pandas as pd, matplotlib.pyplot as plt, pathlib
pathlib.Path("results/plots").mkdir(parents=True, exist_ok=True)
agg = pd.DataFrame(json.load(open("results/interference_results.json")))
dd  = pd.DataFrame([json.loads(l) for l in open("results/runs/fine_lfirst.jsonl")])
lfr = dd.groupby(["feature","alpha"], as_index=False)["leading_latex"].mean() \
        .rename(columns={"leading_latex":"latex_first_rate"})
m = agg.merge(lfr, on=["feature","alpha"], how="left")
m.to_csv("results/interference_results_fine_lfirst.csv", index=False)
cdf = m[m.feature=="code_feature"].sort_values("alpha")
fig, ax1 = plt.subplots(figsize=(9,5)); ax2 = ax1.twinx()
ax1.plot(cdf.alpha, cdf.math_margin_mean, marker="o", label="math_margin_last")
ax2.plot(cdf.alpha, cdf.latex_first_rate, marker="x", linestyle="--", label="latex_first")
ax1.set_xlabel("alpha"); ax1.set_ylabel("math margin"); ax2.set_ylabel("latex_first_rate")
ax1.grid(True, linestyle="--", alpha=.5); fig.tight_layout()
fig.savefig("results/plots/interference_fine_lfirst.png", dpi=200, bbox_inches="tight")
PY
```

> **Math feature**: `FEATURES=math_feature,random_control` is supported but experimental; ensure `MATH_FEATURE_IDX` is set (read from `feature_candidates.json` or exported via env var).

---

## Metrics (short)

* **Math logit margin (last token)**:
  For the final context token, take `max(logits[math_token_set]) − max(logits[non_math_set])`, averaged across prompts. See the implementation in `03_steering_and_eval.py`.

* **Code/LaTeX validity**:
  Simple validators for well-formed fenced code blocks and `$$ … $$` blocks.

* **LaTeX-first rate**:
  Whether the assistant’s first tokens are LaTeX (`$$`) at the turn boundary (computed from `03c_dump_lfirst.py` dump).

---

## Steering hook (what actually happens)

We register a forward **pre-hook** at decoder **layer L18** and add $\alpha \cdot v$ to the residual stream, where $v$ is the decoder-column corresponding to a selected SAE feature. The **random control** baseline uses a random unit vector $v$ of the same dimensionality.

---

## Reproducing H200 results quickly

```bash
python scripts/01_activation_caching.py
python scripts/02_train_sae.py
python scripts/02b_pick_features.py

FEATURES=code_feature,random_control \
ALPHAS="-4,-3,-2,-1,0,1,2,3,4" \
MAX_NEW_TOKENS=64 \
python scripts/03_steering_and_eval.py
```

**tmux tip (safe to close your laptop):**

```bash
tmux new -s mats
# run commands
# detach:  Ctrl-b then d
# re-attach later: tmux attach -t mats
```

---

## Artifacts in this repo

* **Aggregates**
  `results/interference_results.json`
  `results/interference_results_all.json`
  `results/interference_all_compare.csv`
  `results/interference_results_fine_lfirst.csv`

* **Per-prompt dump**
  `results/runs/fine_lfirst.jsonl`

* **Plots**
  `results/plots/interference_all_compare.png`
  `results/plots/interference_fine_lfirst.png`

---

## Troubleshooting

* **“No `W_dec` in checkpoint”**
  `02_train_sae.py` saves `sae_simple.pt` with a `state_dict`. If `03_steering_and_eval.py` wants a raw decoder, export once:

  ```python
  import torch
  ck = torch.load("results/saes/sae_simple.pt", map_location="cpu")
  sd = ck.get("state_dict", {})
  W = sd.get("W_dec") or sd.get("decoder.weight")
  if W is None:
      W = max((v for v in sd.values() if getattr(v,"ndim",0)==2), key=lambda t: t.numel())
  W = W.float()
  if W.shape == (2560, 4096):  # ensure (d_sae, d_in)
      W = W.T
  torch.save({"W_dec": W, "d_in": 2560, "d_sae": 4096}, "results/saes/sae_decoder.pt")
  ```

* **Warnings about ignored generation flags**
  Qwen3 may ignore some sampling kwargs during teacher-forced forward passes. Safe to ignore.

* **Long runs on a VM**
  Always use `tmux` (or `nohup`). SSH drops won’t kill jobs.

---

## Changelog & tags

* **v0.1.2** — H200: math-feature sweep vs code-feature; comparison plot
* **v0.1.1** — Fine sweep + LaTeX-first dump, overlay plot
* **v0.1.0** — GPU-ready cut; SimpleSAE refactor, stable paths

---

## License

This project is licensed under the **Apache License 2.0**.
See [`LICENSE`](./LICENSE) for details.

---

## Acknowledgments

* **Qwen/Qwen3-4B-Instruct** (Alibaba/Qwen team)
* **Hugging Face** ecosystem and hub caches
* **TransformerLens** and **SAE-Lens** for helpful patterns/utilities

---

## Citation

```
@software{superposition_project_2025,
  title  = {superposition_project: SAE steering on Qwen3-4B-Instruct},
  author = {Escobedo, Salvador},
  year   = {2025},
  url    = {https://github.com/svdrecbd/superposition_project}
}
```

```
