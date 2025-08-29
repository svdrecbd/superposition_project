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

# --- Ultra-Literal Format Headers ---
FORMAT_HEADERS = {
    "pure_code": "Return only a single fenced Python code block that starts with ```python and ends with ```; no prose.",
    "pure_math": "Return only a single display LaTeX block delimited by $$ on its own lines; no prose.",
    "mixed": "Return exactly two blocks, in this order: (1) a display LaTeX block delimited by $$ on its own lines; (2) a fenced Python block starting with ```python and ending with ```; no extra text."
}

def encode_with_template(tokenizer, user_prompt):
    try:
        msgs = [
            {"role": "system", "content": "Follow the format requirements exactly."},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt")
    except Exception: # Fallback for older tokenizer versions
        print("Warning: `apply_chat_template` failed. Using manual formatting.")
        text = f"<|im_start|>system\nFollow the format requirements exactly.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(text, return_tensors="pt")

# --- Main Script ---
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True).to(DEVICE)
model.eval()

num_layers = model.config.num_hidden_layers
target_layer = num_layers // 2

# Added '$$' to delimiter set
delims = ["```", "$$", "$", "\\[", "\\]", "\\(", "\\)", "\\frac", "\\\\", "&"]
delim_ids = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in delims}

def keep_rows_for(ids_tensor):
    starts = {ids[0] for ids in delim_ids.values() if len(ids) > 0}
    return torch.isin(ids_tensor, torch.tensor(list(starts), device=ids_tensor.device))

with open(PROMPT_FILE, 'r') as f:
    prompts = json.load(f)

# Flatten prompts for iteration
all_prompts_with_types = []
for type, prompts_list in prompts.items():
    for p in prompts_list:
        all_prompts_with_types.append((type, p))

with torch.no_grad():
    for i, (prompt_type, prompt_text) in enumerate(all_prompts_with_types):
        formatted_prompt = f"{FORMAT_HEADERS[prompt_type]}\n\n{prompt_text}"
        enc = encode_with_template(tokenizer, formatted_prompt).to(DEVICE)
        mask = keep_rows_for(enc["input_ids"][0])
        
        cache = {}
        def hook(_, __, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            cache["h"] = h.detach().float().cpu()

        hook_handle = model.model.layers[target_layer].register_forward_hook(hook)
        _ = model(**enc)
        hook_handle.remove()

        if cache.get("h") is None:
            continue

        H = cache["h"].squeeze(0)
        H_filtered = H[mask] if mask.any() else H
        
        if H_filtered.numel() > 0:
            torch.save(H_filtered.half(), OUTPUT_DIR / f"prompt_{i}_L{target_layer}.pt")
            print(f"Cached {H_filtered.shape[0]} tokens for prompt {i+1}/{len(all_prompts_with_types)}")

print("Activation caching complete.")