import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
import json
from pathlib import Path
import sys
from tqdm import tqdm

# --- Final Polish ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Project root & imports ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # ensure "from scripts.utils" works from anywhere
from scripts.utils import validate_code, validate_latex

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SAE_PATH   = ROOT / "results" / "saes" / "sae_model.pt"
# TODO: Set these after running 02b_pick_features.py
CODE_FEATURE_IDX = 1234  # <-- replace
MATH_FEATURE_IDX = 5678  # <-- replace
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Ultra-literal header (keeps outputs consistently formatted)
FORMAT_HEADER = (
    "Return exactly two blocks, in this order: "
    "(1) a display LaTeX block delimited by $$ on its own lines; "
    "(2) a fenced Python block starting with ```python and ending with ```; "
    "no extra text."
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
        print("Warning: `apply_chat_template` failed. Using manual formatting.")
        text = (
            "<|im_start|>system\nFollow the format requirements exactly.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return tokenizer(text, return_tensors="pt")

def math_margin(logits, labels, math_ids, nonmath_ids):
    # align for next-token prediction: logits[t] predicts labels[t+1]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    math_ids = torch.tensor(math_ids, device=logits.device)
    non_ids  = torch.tensor(nonmath_ids, device=logits.device)

    math_top = logits.gather(dim=-1, index=math_ids.view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)).max(dim=-1).values
    non_top  = logits.gather(dim=-1, index=non_ids.view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)).max(dim=-1).values

    mask = torch.isin(labels, math_ids)
    vals = (math_top - non_top)[mask]
    return vals.mean().item() if vals.numel() > 0 else float("nan")

# --- Load models & data ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

ckpt = torch.load(SAE_PATH, map_location=DEVICE)
sae  = SAE(d_in=ckpt["d_model"], d_sae=ckpt["d_sae"]).to(DEVICE).eval()
sae.load_state_dict(ckpt["state_dict"])

with open(ROOT / "data" / "prompts.json", 'r') as f:
    prompts = json.load(f)
assert "mixed" in prompts, "Prompt JSON missing 'mixed' key."

# Token sets (include "$$")
MATH_TOKEN_STRS = ["$$", "$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\begin", "\\end", "\\sum", "\\alpha", "\n$", "\n\\["]
NONMATH_TOKEN_STRS = [".", ",", ":", ";", "and", "the", "=", "return", "def"]
MATH_TOKEN_IDS    = sum([tokenizer.encode(t, add_special_tokens=False) for t in MATH_TOKEN_STRS], [])
NONMATH_TOKEN_IDS = sum([tokenizer.encode(t, add_special_tokens=False) for t in NONMATH_TOKEN_STRS], [])
# de-dup
MATH_TOKEN_IDS    = list(dict.fromkeys(MATH_TOKEN_IDS))
NONMATH_TOKEN_IDS = list(dict.fromkeys(NONMATH_TOKEN_IDS))
if not MATH_TOKEN_IDS or not NONMATH_TOKEN_IDS:
    raise ValueError("Token sets are empty for this tokenizer.")

# --- Steering vectors ---
code_vec = torch.nn.functional.normalize(sae.W_dec[CODE_FEATURE_IDX], dim=0).to(DEVICE)
# (math_vec available if you want mitigations)
# math_vec = torch.nn.functional.normalize(sae.W_dec[MATH_FEATURE_IDX], dim=0).to(DEVICE)

# Random control (not equal to code index)
rand_idx = int(torch.randint(low=0, high=ckpt["d_sae"], size=(1,)))
while rand_idx == CODE_FEATURE_IDX:
    rand_idx = int(torch.randint(low=0, high=ckpt["d_sae"], size=(1,)))
rand_vec = torch.nn.functional.normalize(sae.W_dec[rand_idx], dim=0).to(DEVICE)

target_layer_idx = model.config.num_hidden_layers // 2
target_layer     = model.model.layers[target_layer_idx]
current_vec   = code_vec
current_alpha = 0.0

def steering_pre_hook(module, inputs):
    (hidden_states, *rest) = inputs
    v_broadcast = current_vec.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,d_model]
    return (hidden_states + current_alpha * v_broadcast, *rest)

# --- Main evaluation ---
alpha_values = [0, 0.5, 1.0, 1.5, 2.0]
results = []
run_config = {"model": MODEL_NAME, "layer": target_layer_idx, "alphas": alpha_values, "seed": 1337}

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
                    mm  = math_margin(out.logits, enc["input_ids"], MATH_TOKEN_IDS, NONMATH_TOKEN_IDS)

                    gen  = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
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
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "interference_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(results_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("\nSteering evaluation complete. Results saved.")