import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from tqdm import tqdm

# --- Final Polish ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parent.parent
SAE_PATH   = ROOT / "results/saes/sae_model.pt"
PROMPTS    = json.load(open(ROOT / "data/prompts.json"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Models & Data ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE).eval()

ckpt = torch.load(SAE_PATH, map_location=DEVICE)
sae  = SAE(d_in=ckpt["d_model"], d_sae=ckpt["d_sae"]).to(DEVICE).eval()
sae.load_state_dict(ckpt["state_dict"])

target_layer_idx = mdl.config.num_hidden_layers // 2
layer = mdl.model.layers[target_layer_idx]

def get_feats(texts, max_tokens=4096):
    acts = []
    with torch.no_grad():
        for t in tqdm(texts, desc="Analyzing prompts"):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(DEVICE)
            cache = {}
            def hook(_, __, out):
                h = out if isinstance(out, torch.Tensor) else out[0]
                cache["h"] = h.detach()
            
            h_handle = layer.register_forward_hook(hook)
            _ = mdl(**enc)
            h_handle.remove()
            
            if "h" in cache:
                H = cache["h"][:, 1:-1, :]
                if H.numel() > 0:
                    _, feat, _, _, _, _ = sae(H.reshape(-1, H.shape[-1]))
                    acts.append(feat.mean(dim=0).cpu())
    
    if len(acts) > 0:
        return torch.stack(acts).mean(dim=0)
    else:
        return torch.zeros(ckpt["d_sae"])

code_prompts = PROMPTS["pure_code"][:16]
math_prompts = PROMPTS["pure_math"][:16]

code_feat_mean = get_feats(code_prompts)
math_feat_mean = get_feats(math_prompts)

diff = (code_feat_mean - math_feat_mean)
code_idx = torch.topk(diff, k=5).indices.tolist()
math_idx = torch.topk(-diff, k=5).indices.tolist()

print("\n--- Feature Analysis Complete ---")
print(f"Top 5 feature candidates for CODE: {code_idx}")
print(f"Top 5 feature candidates for MATH: {math_idx}")

results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "feature_candidates.json", "w") as f:
    json.dump({"code_candidates": code_idx, "math_candidates": math_idx}, f, indent=2)
print("Candidates saved to results/feature_candidates.json")