import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import sys
from tqdm import tqdm

# --- Determinism / speed ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# Make "from scripts.utils ..." work from project root
sys.path.append(str(Path.cwd()))
from scripts.utils import validate_code, validate_latex

# --- Config ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SAE_PATH   = Path("results/saes/sae_simple.pt")   # <— SimpleSAE from 02_train_sae.py
PROMPTS_FN = Path("data/prompts.json")

# <<< Use your picked features >>>
CODE_FEATURE_IDX = 1598
MATH_FEATURE_IDX = 2000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

FORMAT_HEADER = (
    "Return exactly two blocks, in this order: "
    "(1) a display LaTeX block delimited by $$ on its own lines; "
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
        text = (
            "<|im_start|>system\nFollow the format requirements exactly.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return tokenizer(text, return_tensors="pt")

def math_margin(logits, labels, math_ids, nonmath_ids):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    math_ids = torch.tensor(math_ids, device=logits.device)
    non_ids  = torch.tensor(nonmath_ids, device=logits.device)
    math_top = logits.gather(-1, math_ids.view(1,1,-1).expand(logits.shape[0], logits.shape[1], -1)).max(-1).values
    non_top  = logits.gather(-1, non_ids .view(1,1,-1).expand(logits.shape[0], logits.shape[1], -1)).max(-1).values
    mask = torch.isin(labels, math_ids)
    vals = (math_top - non_top)[mask]
    return vals.mean().item() if vals.numel() > 0 else float("nan")

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
).to(DEVICE).eval()
tokenizer.pad_token = tokenizer.eos_token

# --- Load prompts ---
with open(PROMPTS_FN, "r") as f:
    prompts = json.load(f)
assert "mixed" in prompts, "Prompt JSON missing 'mixed'."

# --- Robust SAE loader (handles SimpleSAE or older state_dict formats) ---
def load_decoder(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    # SimpleSAE direct tensors
    if all(k in ckpt for k in ("W_dec", "b_dec")):
        W_dec = ckpt["W_dec"].to(torch.float32)  # [d_sae, d_in]
        b_dec = ckpt["b_dec"].to(torch.float32)  # [d_in]
        return W_dec, b_dec
    # Nested state_dict variant
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        # Try common names
        for k in ["W_dec", "decoder.weight", "dec.weight"]:
            if k in sd:
                W = sd[k].to(torch.float32)
                # Expect [d_sae, d_in]. If [d_in, d_sae], transpose.
                if W.shape[0] < W.shape[1]:
                    W = W.t().contiguous()
                b_key = "b_dec" if "b_dec" in sd else ("decoder.bias" if "decoder.bias" in sd else None)
                b = sd[b_key].to(torch.float32) if b_key else torch.zeros(W.shape[1], dtype=torch.float32)
                return W, b
    # If here, we couldn't recognize the format
    raise KeyError(
        f"SAE checkpoint at {path} does not contain W_dec/b_dec.\n"
        f"Keys present: {list(ckpt.keys())}. Re-run scripts/02_train_sae.py to create 'sae_simple.pt'."
    )

W_dec, b_dec = load_decoder(SAE_PATH)
d_sae, d_in = W_dec.shape
print(f"Loaded decoder: d_in={d_in}, d_sae={d_sae}")

# --- Build steering vectors ---
code_vec = F.normalize(W_dec[CODE_FEATURE_IDX], dim=0)  # [d_in]
math_vec = F.normalize(W_dec[MATH_FEATURE_IDX], dim=0)  # [d_in]
# Distinct random control
rand_idx = CODE_FEATURE_IDX
while rand_idx == CODE_FEATURE_IDX:
    rand_idx = int(torch.randint(low=0, high=d_sae, size=(1,)))
rand_vec = F.normalize(W_dec[rand_idx], dim=0)

# --- Layer selection & hook ---
target_layer_idx = model.config.num_hidden_layers // 2
target_layer = model.model.layers[target_layer_idx]
current_vec = code_vec
current_alpha = 0.0

def steering_pre_hook(_, inputs):
    (hidden_states, *rest) = inputs
    v = current_vec.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
    return (hidden_states + current_alpha * v, *rest)

# --- Token sets for metric ---
MATH_TOKEN_STRS    = ["$$", "$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\begin", "\\end", "\\sum", "\\alpha"]
NONMATH_TOKEN_STRS = [".", ",", ":", ";", "and", "the", "=", "return", "def"]
MATH_TOKEN_IDS     = sum([tokenizer.encode(t, add_special_tokens=False) for t in MATH_TOKEN_STRS], [])
NONMATH_TOKEN_IDS  = sum([tokenizer.encode(t, add_special_tokens=False) for t in NONMATH_TOKEN_STRS], [])
MATH_TOKEN_IDS     = list(dict.fromkeys(MATH_TOKEN_IDS))
NONMATH_TOKEN_IDS  = list(dict.fromkeys(NONMATH_TOKEN_IDS))
if not MATH_TOKEN_IDS or not NONMATH_TOKEN_IDS:
    raise ValueError("Empty math/nonmath token id sets—add a few more strings.")

# --- Eval ---
alpha_values = [0, 0.5, 1.0, 1.5, 2.0]
results = []
run_config = {
    "model": MODEL_NAME,
    "layer": target_layer_idx,
    "alphas": alpha_values,
    "seed": 1337,
    "code_feature": CODE_FEATURE_IDX,
    "math_feature": MATH_FEATURE_IDX,
    "rand_feature": rand_idx,
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

                    out = model(**enc, labels=enc["input_ids"], use_cache=False)
                    mm = math_margin(out.logits, enc["input_ids"], MATH_TOKEN_IDS, NONMATH_TOKEN_IDS)

                    gen = model.generate(
                        **enc, max_new_tokens=128, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)

                    bucket.append({
                        "math_margin": mm,
                        "code_ok":  validate_code(text),
                        "latex_ok": validate_latex(text),
                    })
            finally:
                handle.remove()

            results.append({
                "feature": which,
                "alpha": alpha,
                "math_margin_mean": sum(x["math_margin"] for x in bucket) / len(bucket) if bucket else float("nan"),
                "code_valid_rate":  sum(x["code_ok"]  for x in bucket) / len(bucket) if bucket else 0.0,
                "latex_valid_rate": sum(x["latex_ok"] for x in bucket) / len(bucket) if bucket else 0.0,
            })

# --- Save ---
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "interference_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(results_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("\nSteering evaluation complete. Results saved to results/interference_results.json")