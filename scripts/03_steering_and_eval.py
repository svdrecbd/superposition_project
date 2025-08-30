import os

# --- injected defaults for feature indices (safe) ---
try:
    import json as _json
    from pathlib import Path as _Path
    _p = _Path("results/feature_candidates.json")
    _d = _json.loads(_p.read_text()) if _p.exists() else {}
    if 'CODE_FEATURE_IDX' not in globals():
        CODE_FEATURE_IDX = int((_d.get('code_candidates') or [0])[0])
    if 'MATH_FEATURE_IDX' not in globals():
        MATH_FEATURE_IDX = int((_d.get('math_candidates') or [0])[0])
except Exception:
    CODE_FEATURE_IDX = globals().get('CODE_FEATURE_IDX', 0)
    MATH_FEATURE_IDX = globals().get('MATH_FEATURE_IDX', 0)
# ----------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import sys
from tqdm import tqdm

def latex_first(text: str) -> bool:
    """
    Return True if a LaTeX block appears before a fenced code block.
    We look for $$, \[, or \begin{align} vs the first ``` (prefer ```python).
    """
    # first code fence
    i_code = text.find("```python")
    if i_code == -1:
        i_code = text.find("```")
    # first LaTeX-ish
    cands = [text.find("$$"), text.find("\\["), text.find("\\begin{align}")]
    cands = [i for i in cands if i != -1]
    i_latex = min(cands) if cands else -1
    if i_latex == -1:
        return False     # no LaTeX found
    if i_code == -1:
        return True      # LaTeX but no code -> LaTeX-first
    return i_latex < i_code

# --- Runtime setup ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# Make "from scripts.utils ..." work when run from project root
sys.path.append(str(Path.cwd()))
from scripts.utils import validate_code, validate_latex

# --- Config ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DECODER_PATH = Path("results/saes/sae_decoder.pt")  # contains W_dec, d_in, d_sae
PROMPTS_PATH = Path("data/prompts.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# --- Feature registry + env parsing (placed before first FEATURES print) ---
name_to_idx = {
    'code_feature': int(os.getenv('CODE_FEATURE_IDX', CODE_FEATURE_IDX)),
    'math_feature': int(os.getenv('MATH_FEATURE_IDX', MATH_FEATURE_IDX)),
    'random_control': None,
}
FEATURES = os.getenv('FEATURES', 'code_feature,random_control')
FEATURES = [f.strip() for f in FEATURES.split(',') if f.strip() in name_to_idx]
print(f"Running with FEATURES: {FEATURES}")
print(f"Running with FEATURES: {FEATURES}")

# Ultra-literal formatting helper
FORMAT_HEADER = (
    "Return exactly two blocks, in this order: (1) a display LaTeX block delimited by $$ on its own lines; "
    "(2) a fenced Python block starting with ```python and ending with ```; no extra text."
)

def encode_with_template(tokenizer, user_prompt):
    try:
        msgs = [
            {"role": "system", "content": "Follow the format requirements exactly."},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt")
    except Exception:
        # Fallback chat formatting
        text = (
            "<|im_start|>system\nFollow the format requirements exactly.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        return tokenizer(text, return_tensors="pt")

# ---------- NEW METRIC (defined before use) ----------
def math_margin_last(logits, math_ids, nonmath_ids):
    """Margin on next-token distribution at the assistant-start boundary (last prompt token)."""
    last = logits[:, -1, :]  # [batch, vocab]
    math_ids_t = torch.tensor(math_ids, device=logits.device)
    non_ids_t  = torch.tensor(nonmath_ids, device=logits.device)
    if math_ids_t.numel() == 0 or non_ids_t.numel() == 0:
        return float("nan")
    math_top = last.index_select(-1, math_ids_t).max(dim=-1).values
    non_top  = last.index_select(-1, non_ids_t).max(dim=-1).values
    return (math_top - non_top).mean().item()
# -----------------------------------------------------

# --- Load model / tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto", trust_remote_code=True).to(DEVICE)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# --- Load prompts ---
with open(PROMPTS_PATH, "r") as f:
    prompts = json.load(f)
assert "mixed" in prompts, "Prompt JSON is missing 'mixed' key."

# --- Token sets (include $$) ---
MATH_TOKEN_STRS = ["$$", "$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\begin", "\\end", "\\sum", "\\alpha"]
NONMATH_TOKEN_STRS = [".", ",", ":", ";", "and", "the", "=", "return", "def"]

MATH_TOKEN_IDS = sum([tokenizer.encode(t, add_special_tokens=False) for t in MATH_TOKEN_STRS], [])
NONMATH_TOKEN_IDS = sum([tokenizer.encode(t, add_special_tokens=False) for t in NONMATH_TOKEN_STRS], [])
MATH_TOKEN_IDS = list(dict.fromkeys(MATH_TOKEN_IDS))
NONMATH_TOKEN_IDS = list(dict.fromkeys(NONMATH_TOKEN_IDS))

# --- Load decoder W_dec ---
ck = torch.load(DECODER_PATH, map_location=DEVICE)
W_dec = ck["W_dec"]  # [d_sae, d_in]
d_sae, d_in = W_dec.shape
print(f"Loaded decoder: d_in={d_in}, d_sae={d_sae}")

# Pick feature indices (prefer auto from feature_candidates.json)
CODE_FEATURE_IDX = None
MATH_FEATURE_IDX = 1902  # auto-filled
cand_path = Path("results/feature_candidates.json")
if cand_path.exists():
    try:
        c = json.load(open(cand_path))
        CODE_FEATURE_IDX = int(c["code_candidates"][0])
        MATH_FEATURE_IDX = int(c["math_candidates"][0])
    except Exception:
        pass
# Fallback to reasonable defaults if still None
if CODE_FEATURE_IDX is None: CODE_FEATURE_IDX = 0
if MATH_FEATURE_IDX is None: MATH_FEATURE_IDX = 1

# --- Steering vectors ---
def norm_row(row):
    v = row / (row.norm(p=2) + 1e-9)
    return v

code_vec = norm_row(W_dec[CODE_FEATURE_IDX]).to(DEVICE)  # [d_in]
# unique random feature != code feature
rand_idx = int(torch.randint(low=0, high=d_sae, size=(1,)))
if rand_idx == CODE_FEATURE_IDX:
    rand_idx = (rand_idx + 1) % d_sae
rand_vec = norm_row(W_dec[rand_idx]).to(DEVICE)

# --- Choose target layer (middle) ---
import os
_default_layer = model.config.num_hidden_layers // 2
try:
    target_layer_idx = int(os.getenv('STEER_LAYER', _default_layer))
except Exception:
    target_layer_idx = _default_layer
target_layer = model.model.layers[target_layer_idx]
current_vec = code_vec
current_alpha = 0.0

def steering_pre_hook(module, inputs):
    (hidden_states, *rest) = inputs
    v = current_vec.to(hidden_states.device).to(hidden_states.dtype).view(1, 1, -1)
    return (hidden_states + current_alpha * v, *rest)

# --- Eval loop ---
import os
alpha_values = [float(x) for x in os.getenv('ALPHAS','-2,-1,-0.5,0,0.5,1,2,3').split(',')]
results = []
run_config = {
    "model": MODEL_NAME,
    "layer": target_layer_idx,
    "alphas": alpha_values,
    "seed": 1337,
    "code_feature_idx": CODE_FEATURE_IDX,
    "rand_feature_idx": rand_idx,
}

with torch.no_grad():
    for which, vec in [("code_feature", code_vec), ("random_control", rand_vec)]:
        current_vec = vec
        print(f"\n--- Steering with: {which} ---")
        for alpha in tqdm(alpha_values, desc=f"Sweeping alpha for {which}"):
            current_alpha = alpha
            handle = target_layer.register_forward_pre_hook(steering_pre_hook)
            try:
                bucket = []
                for prompt in prompts["mixed"]:
                    formatted = f"{FORMAT_HEADER}\n\n{prompt}"
                    enc = encode_with_template(tokenizer, formatted).to(DEVICE)

                    # Compute next-token logits at assistant start
                    out = model(**enc, labels=enc["input_ids"], use_cache=False)
                    mm = math_margin_last(out.logits, MATH_TOKEN_IDS, NONMATH_TOKEN_IDS)

                    # Greedy generate a short completion (no sampling flags that Qwen ignores)
                    gen = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)

                    bucket.append({
                        "math_margin": mm,
                        "code_ok": validate_code(text),
                        "latex_ok": validate_latex(text),
                    })
            finally:
                handle.remove()

            results.append({
                "feature": which,
                "alpha": alpha,
                "math_margin_mean": float(sum(x["math_margin"] for x in bucket) / len(bucket)) if bucket else float("nan"),
                "code_valid_rate": float(sum(x["code_ok"] for x in bucket) / len(bucket)) if bucket else 0.0,
                "latex_valid_rate": float(sum(x["latex_ok"] for x in bucket) / len(bucket)) if bucket else 0.0,
            })

# --- Save ---
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "interference_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(results_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("\nSteering evaluation complete. Results saved to results/interference_results.json")
