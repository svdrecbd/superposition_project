import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

# --- Final Polish ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_FILE = Path("../data/prompts.json")
OUTPUT_DIR = Path("../results/activations")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Main Script ---
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE)
model.eval()

num_layers = model.config.num_hidden_layers
target_layer = num_layers // 2

delims = ["```", "$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\\\", "&"]
delim_ids = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in delims}

def keep_rows_for(ids_tensor):
    starts = {ids[0] for ids in delim_ids.values() if len(ids) > 0}
    return torch.isin(ids_tensor, torch.tensor(list(starts), device=ids_tensor.device))

with open(PROMPT_FILE, 'r') as f:
    prompts = json.load(f)
all_prompts = prompts['pure_code'] + prompts['pure_math'] + prompts['mixed']

with torch.no_grad():
    for i, prompt in enumerate(all_prompts):
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        mask = keep_rows_for(enc["input_ids"][0])
        
        cache = {}
        def hook(_, __, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            cache["h"] = h.detach().float().cpu()

        hook_handle = model.model.layers[target_layer].register_forward_hook(hook)
        _ = model(**enc)
        hook_handle.remove()

        if cache.get("h") is None:
            print(f"Warning: Hook did not run for prompt {i+1}. Skipping.")
            continue

        H = cache["h"].squeeze(0)
        H_filtered = H[mask] if mask.any() else H
        
        if H_filtered.numel() > 0:
            torch.save(H_filtered.half(), OUTPUT_DIR / f"prompt_{i}_L{target_layer}.pt")
            print(f"Cached {H_filtered.shape[0]} tokens for prompt {i+1}/{len(all_prompts)}")
        else:
            print(f"Warning: No diagnostic tokens found for prompt {i+1}. Skipping.")

print("Activation caching complete.")