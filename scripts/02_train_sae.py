import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Final Polish ---
torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)

# --- Configuration ---
ACTIVATION_DIR = Path("results/activations")  # uses cached activations from 01
SAE_OUTPUT_DIR = Path("results/saes")
D_SAE = 4096
L1_COEFFICIENT = 1e-3
LEARNING_RATE = 1e-3
EPOCHS = 3
BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Minimal Sparse Autoencoder ---
class SimpleSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        # Encoder: z = ReLU(W_enc x + b_enc)
        # weight shape (out=d_sae, in=d_in)
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder: x_hat = W_dec z + b_dec
        # IMPORTANT: weight shape (out=d_in, in=d_sae)
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=0.0)   # good for ReLU
        nn.init.xavier_uniform_(self.W_dec)

    def encode(self, x):
        return F.relu(F.linear(x, self.W_enc, self.b_enc))

    def decode(self, z):
        # z: [batch, d_sae], W_dec: [d_in, d_sae] => output [batch, d_in]
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

# --- Load cached activations ---
files = sorted(ACTIVATION_DIR.glob("*.pt"))
if not files:
    raise FileNotFoundError("No activation files found. Run scripts/01_activation_caching.py first.")

tensors = [torch.load(f).to(torch.float32) for f in files]
all_act = torch.cat(tensors, dim=0)  # [N_tokens, d_model]
print(f"Loaded activation tensor of shape: {all_act.shape}")

d_model = all_act.shape[-1]
sae = SimpleSAE(d_in=d_model, d_sae=D_SAE).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

dataset = TensorDataset(all_act)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# --- Train ---
print("Training SimpleSAE...")
for epoch in range(EPOCHS):
    running = 0.0
    count = 0
    for (x_batch,) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x_batch.to(DEVICE)
        optimizer.zero_grad()
        x_hat, z = sae(x)
        mse = F.mse_loss(x_hat, x)
        l1 = z.abs().mean()
        loss = mse + L1_COEFFICIENT * l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * x.size(0)
        count  += x.size(0)
    print(f"Epoch {epoch+1}: loss={running / max(1,count):.6f}  (mse+{L1_COEFFICIENT}*l1)")

# --- Save ---
SAE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
ckpt = {
    "d_in": d_model,
    "d_sae": D_SAE,
    "l1_coefficient": L1_COEFFICIENT,
    "state_dict": {
        "W_enc": sae.W_enc.detach().cpu(),  # [d_sae, d_in]
        "b_enc": sae.b_enc.detach().cpu(),  # [d_sae]
        "W_dec": sae.W_dec.detach().cpu(),  # [d_in, d_sae]  <-- NOTE
        "b_dec": sae.b_dec.detach().cpu(),  # [d_in]
    },
    "seed": 1337,
}
out_path = SAE_OUTPUT_DIR / "sae_simple.pt"
torch.save(ckpt, out_path)
print(f"Saved SimpleSAE checkpoint to {out_path}")
