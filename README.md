# Superposition Stress Test: Code vs. Math Formatting

This project investigates feature superposition in the `Qwen/Qwen3-4B-Instruct-2507` model, for the MATS 9.0 application.

## Hypothesis
Some internal features that support code formatting (e.g., ` ``` `) and math formatting (e.g., `$$`) live in superposition. Steering a discovered "code feature" will causally harm the model's ability to correctly format mathematics in prompts that require both.

## Primary Metrics
1.  **Interference Score**: The average change in the logit margin of math-delimiter tokens when causally steering the code feature on mixed prompts. The margin is defined as `logit(target_math_token) − max_logit(non_math_tokens)`.
2.  **Syntax Validity Rate**: The percentage of generated completions that pass a cheap validator for both Python code blocks (using `ast.parse`) and LaTeX blocks (balanced delimiters).

## Experimental Plan
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Target Layer**: Layer `num_layers // 2`
- **SAE**: 4096 features, trained on activations from diagnostic tokens only.
- **Intervention**: Linear feature steering with `alpha` in `{0, 0.5, 1.0, 1.5, 2.0}`.
- **Controls**: A random feature control and a zero-steer baseline will be used for comparison.

## Results (H200 run)

Model: `Qwen/Qwen3-4B-Instruct-2507` • Hook layer: 18  
SAE: SimpleSAE (d_in=2560, d_sae=4096), trained on cached activations from 112 mixed prompts.

**Metrics:** math logit margin at assistant-start token; plus code_valid_rate, latex_valid_rate.  
We also logged `latex_first_rate` on a fine sweep.

**Key findings (summarized):**
- Baseline (α=0): math_margin_mean ≈ **40.135**; code_valid_rate ≈ **0.179**; latex_valid_rate ≈ **0.949**.
- Steering along the **code_feature** has **modest** effect relative to a random control direction.
- Control direction strongly changes math margin with α; code_feature changes are smaller.  
  Control-adjusted separation score (∑\|Δ vs control\| across α) ≈ **3.885** for code_feature.
- `latex_first_rate` was **1.0** across all α in the fine sweep (24 prompts), so steering didn’t flip the initial token to LaTeX.

**Artifacts:**  
- Wide sweep (all features): `results/interference_results_all.json`, CSV & plot at  
  `results/interference_all_compare.csv`, `results/plots/interference_all_compare.png`.
- Deduped + control-adjusted: `results/interference_results_all_dedup.json`, plot at  
  `results/plots/interference_all_dedup.png`.
- Fine sweep + latex-first: CSV `results/interference_results_fine_lfirst.csv`, plot  
  `results/plots/interference_fine_lfirst.png`, raw dump `results/runs/fine_lfirst.jsonl`.

**Reproduce (GPU):**
```bash
python scripts/01_activation_caching.py
python scripts/02_train_sae.py
python scripts/02b_pick_features.py
ALPHAS="-4,-3,-2,-1,0,1,2,3,4" python scripts/03_steering_and_eval.py
