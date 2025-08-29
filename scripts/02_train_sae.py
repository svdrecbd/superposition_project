import torch
from sae_lens import SAE
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Final Polish ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Configuration ---
ACTIVATION_DIR = Path("../results/activations")
SAE_OUTPUT_DIR = Path("../results/saes")
D_SAE = 4096
L1_COEFFICIENT = 1e-3
LEARNING_RATE = 1e-4
EPOCHS = 2
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Main Script ---
SAE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
files = sorted(ACTIVATION_DIR.glob("*.pt"))
if not files:
    raise FileNotFoundError("No activation files found. Run 01_activation_caching.py first.")

tensors = [torch.load(f).to(torch.float32) for f in files]
all_act = torch.cat(tensors, dim=0)
print(f"Loaded activation tensor of shape: {all_act.shape}")

d_model = all_act.shape[-1]
sae = SAE(d_in=d_model, d_sae=D_SAE, l1_coefficient=L1_COEFFICIENT).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
dataset = TensorDataset(all_act)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting minimal SAE training loop...")
for epoch in range(EPOCHS):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        acts = batch[0].to(DEVICE)
        optimizer.zero_grad()
        sae_out, feature_acts, loss, _, _, _ = sae(acts)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Final loss: {loss.item()}")

torch.save({
    "state_dict": sae.state_dict(),
    "d_model": d_model,
    "d_sae": D_SAE,
    "l1_coefficient": L1_COEFFICIENT
}, SAE_OUTPUT_DIR / "sae_model.pt")

print("SAE training complete.")