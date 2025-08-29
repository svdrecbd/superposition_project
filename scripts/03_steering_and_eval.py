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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import validate_code, validate_latex

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SAE_PATH = Path("../results/saes/sae_model.pt")
# TODO: Set these after running 02b_pick_features.py
CODE_FEATURE_IDX = 1234 # Placeholder
MATH_FEATURE_IDX = 5678 # Placeholder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def math_margin(logits, labels, math_ids, nonmath_ids):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    math_ids = torch.tensor(math_ids, device=logits.device)
    non_ids = torch.tensor(nonmath_ids, device=logits.device)
    math_top_logits = logits.gather(dim=-1, index=math_ids.view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)).max(dim=-1).values
    non_top_logits = logits.gather(dim=-1, index=non_ids.view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)).max(dim=-1).values
    mask = torch.isin(labels, math_ids)
    vals = (math_top_logits - non_top_logits)[mask]
    return vals.mean().item() if vals.numel() > 0 else float("nan")

# --- Load Models & Data ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

ckpt = torch.load(SAE_PATH, map_location=DEVICE)
sae = SAE(d_in=ckpt["d_model"], d_sae=ckpt["d_sae"]).to(DEVICE).eval()
sae.load_state_dict(ckpt["state_dict"])

with open(Path("../data/prompts.json"), 'r') as f:
    prompts = json.load(f)
assert all(k in prompts for k in ["pure_code", "pure_math", "mixed"]), "Prompt JSON is missing required keys."

MATH_TOKEN_STRS = ["$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\begin", "\\end", "\\sum", "\\alpha", "\n$", "\n\\["]
NONMATH_TOKEN_STRS = [".", ",", ":", ";", "and", "the", "=", "return", "def"]
MATH_TOKEN_IDS = sum([tokenizer.encode(t, add_special_tokens=False) for t in MATH_TOKEN_STRS], [])
NONMATH_TOKEN_IDS = sum([tokenizer.encode(t, add_special_tokens=False) for t in NONMATH_TOKEN_STRS], [])
MATH_TOKEN_IDS = list(dict.fromkeys(MATH_TOKEN_IDS))
NONMATH_TOKEN_IDS = list(dict.fromkeys(NONMATH_TOKEN_IDS))

if not MATH_TOKEN_IDS or not NONMATH_TOKEN_IDS:
    raise ValueError("Token sets are empty for this tokenizer.")

# --- Steering Logic ---
code_vec = torch.nn.functional.normalize(sae.W_dec[CODE_FEATURE_IDX], dim=0).to(DEVICE)
rand_idx = int(torch.randint(low=0, high=ckpt["d_sae"], size=(1,)))
while rand_idx == CODE_FEATURE_IDX:
    rand_idx = int(torch.randint(low=0, high=ckpt["d_sae"], size=(1,)))
rand_vec = torch.nn.functional.normalize(sae.W_dec[rand_idx], dim=0).to(DEVICE)

target_layer_idx = model.config.num_hidden_layers // 2
target_layer = model.model.layers[target_layer_idx]
current_vec = code_vec
current_alpha = 0.0

def steering_pre_hook(module, inputs):
    (hidden_states, *rest) = inputs
    v_broadcast = current_vec.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
    return (hidden_states + current_alpha * v_broadcast, *rest)

# --- Main Evaluation Loop ---
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
                    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                    out = model(**enc, labels=enc["input_ids"], use_cache=False)
                    mm = math_margin(out.logits, enc["input_ids"], MATH_TOKEN_IDS, NONMATH_TOKEN_IDS)
                    gen = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)
                    bucket.append({"math_margin": mm, "code_ok": validate_code(text), "latex_ok": validate_latex(text)})
            finally:
                handle.remove()
            
            results.append({
                "feature": which,
                "alpha": alpha,
                "math_margin_mean": sum(x["math_margin"] for x in bucket) / len(bucket) if bucket else float("nan"),
                "code_valid_rate": sum(x["code_ok"] for x in bucket) / len(bucket) if bucket else 0,
                "latex_valid_rate": sum(x["latex_ok"] for x in bucket) / len(bucket) if bucket else 0,
            })

# --- Save Results ---
results_dir = Path("../results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "interference_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(results_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("\nSteering evaluation complete. Results saved.")