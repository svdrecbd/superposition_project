# scripts/02b_pick_features.py
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Config ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path.cwd()
SAE_PATH = ROOT / "results/saes/sae_simple.pt"   # <-- uses the checkpoint you just saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load prompts ---
with open(ROOT / "data/prompts.json", "r") as f:
    PROMPTS = json.load(f)

# --- Load model ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE).eval()

# --- Load SimpleSAE weights ---
ckpt = torch.load(SAE_PATH, map_location=DEVICE)
W_enc = ckpt["state_dict"]["W_enc"].to(DEVICE)  # [d_sae, d_in]
b_enc = ckpt["state_dict"]["b_enc"].to(DEVICE)  # [d_sae]
d_sae = ckpt["d_sae"]
print(f"Loaded SimpleSAE: d_in={ckpt['d_in']}, d_sae={d_sae}")

# --- Target layer ---
target_layer_idx = mdl.config.num_hidden_layers // 2
layer = mdl.model.layers[target_layer_idx]

def mean_feature_activity(texts, max_tokens=4096):
    feats = []
    with torch.no_grad():
        for t in tqdm(texts, desc="Analyzing prompts"):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(DEVICE)
            cache = {}

            def hook(_, __, out):
                h = out if isinstance(out, torch.Tensor) else out[0]  # [1, seq, d_in]
                cache["h"] = h.detach()

            h_handle = layer.register_forward_hook(hook)
            _ = mdl(**enc)
            h_handle.remove()

            if "h" not in cache:
                continue

            # Make sure dtype matches the SAE weights
            H = cache["h"].squeeze(0).to(W_enc.dtype)   # [seq, d_in], cast to float32
            if H.numel() == 0:
                continue

            # Encode via SimpleSAE: z = ReLU(W_enc x + b_enc)
            Z = F.relu(F.linear(H, W_enc, b_enc))       # [seq, d_sae]
            feats.append(Z.mean(dim=0).cpu())           # mean over tokens

    return torch.stack(feats).mean(dim=0) if feats else torch.zeros(d_sae)

# Use a modest subset (or all) for calibration
code_prompts = PROMPTS["pure_code"][:32]
math_prompts = PROMPTS["pure_math"][:32]

code_feat_mean = mean_feature_activity(code_prompts)
math_feat_mean = mean_feature_activity(math_prompts)

# Features strong for code vs math, and vice versa
diff = code_feat_mean - math_feat_mean  # positive => code>math
k = min(10, d_sae)
code_idx = torch.topk(diff, k=k).indices.tolist()
math_idx = torch.topk(-diff, k=k).indices.tolist()

print("\n--- Feature Analysis Complete ---")
print(f"Top {k} candidates for CODE (code>math): {code_idx}")
print(f"Top {k} candidates for MATH (math>code): {math_idx}")

# Save for provenance
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "feature_candidates.json", "w") as f:
    json.dump({"code_candidates": code_idx, "math_candidates": math_idx}, f, indent=2)
print("Candidates saved to results/feature_candidates.json")
