# VQ-VAE Code Overview

This document provides a comprehensive overview of all components in the VQ-VAE codebase. Use this as a reference to understand what each part does and how they work together.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Model Architecture](#model-architecture)
3. [Quantizers](#quantizers)
4. [Training Pipeline](#training-pipeline)
5. [Metrics](#metrics)
6. [Data Loading](#data-loading)
7. [Experiment Outputs](#experiment-outputs)
8. [Usage Examples](#usage-examples)

---

## Project Structure

```
bavq-research/
├── vqvae.py                  # Complete implementation (all-in-one, 640 lines)
│   ├── ResidualBlock         # Simple 2-conv residual block
│   ├── Encoder               # 32×32×3 → 8×8×256
│   ├── Decoder               # 8×8×256 → 32×32×3
│   ├── VQEMAQuantizer        # Baseline quantizer with EMA updates
│   ├── BAQuantizer           # Blahut-Arimoto quantizer (our method)
│   ├── VQVAE                 # Complete model combining all components
│   ├── get_dataloaders()     # CIFAR-10 data loading
│   ├── train()               # Main training loop
│   ├── train_epoch()         # Single epoch training
│   ├── validate()            # Validation with metrics
│   └── compute_metrics()     # Perplexity, usage rate, PSNR
│
├── train.py                  # CLI wrapper for training (83 lines)
│
├── config.yaml               # Full GPU server experiments (30-100 epochs)
├── config_mac.yaml           # Mac quick tests (3 epochs)
│
├── analyze.ipynb             # Results analysis and visualization
│
├── requirements.txt          # Mac/local dependencies (PyTorch with MPS)
├── requirements-cuda.txt     # GPU server dependencies
│
├── README.md                 # Quick start guide
├── PROJECT_OVERVIEW.md       # Research goals and timeline
├── CLAUDE.md                 # Development guide
└── .gitignore                # Properly configured

# Auto-created directories:
├── data/                     # CIFAR-10 dataset (auto-downloaded)
└── experiments/              # Training results and checkpoints
    └── [run_name]/
        ├── final_model.pt              # Final model weights (~27 MB)
        ├── final_metrics.json          # Final validation metrics
        └── checkpoint_epoch{N}.pt      # Periodic checkpoints (every 10 epochs)
```

---

## Model Architecture

### Overview

The VQ-VAE model consists of three main components:
1. **Encoder**: Compresses images to latent representations
2. **Quantizer**: Maps continuous latents to discrete codes
3. **Decoder**: Reconstructs images from quantized latents

### Encoder

**Purpose:** Compress 32×32 RGB images to 8×8×256 latent representations

**Architecture:**
```
Input: [B, 3, 32, 32]
    ↓
Conv2d(3 → 128, kernel=4, stride=2, padding=1) + ReLU
    → [B, 128, 16, 16]
    ↓
Conv2d(128 → 256, kernel=4, stride=2, padding=1) + ReLU
    → [B, 256, 8, 8]
    ↓
Conv2d(256 → 256, kernel=3, stride=1, padding=1) + ReLU
    → [B, 256, 8, 8]
    ↓
ResidualBlock(256)
    → [B, 256, 8, 8]
    ↓
ResidualBlock(256)
    → [B, 256, 8, 8] (output)
```

**Key Features:**
- 4× spatial compression (32×32 → 8×8)
- 2 residual blocks at bottleneck for expressiveness
- ReLU activations throughout
- ~1.1M parameters

**Code:** `vqvae.py` lines 44-67

### Decoder

**Purpose:** Reconstruct 32×32 RGB images from 8×8×256 quantized latents

**Architecture:**
```
Input: [B, 256, 8, 8]
    ↓
ResidualBlock(256)
    → [B, 256, 8, 8]
    ↓
ResidualBlock(256)
    → [B, 256, 8, 8]
    ↓
Conv2d(256 → 256, kernel=3, stride=1, padding=1) + ReLU
    → [B, 256, 8, 8]
    ↓
ConvTranspose2d(256 → 128, kernel=4, stride=2, padding=1) + ReLU
    → [B, 128, 16, 16]
    ↓
ConvTranspose2d(128 → 3, kernel=4, stride=2, padding=1)
    → [B, 3, 32, 32] (output, NO activation)
```

**Key Features:**
- Mirror of encoder architecture
- No activation on final layer (outputs continuous values)
- Uses transposed convolutions for upsampling
- ~1.1M parameters

**Code:** `vqvae.py` lines 70-93

### ResidualBlock

**Purpose:** Learn complex transformations while maintaining gradient flow

**Architecture:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        self.conv1 = Conv2d(channels, channels, kernel=3, padding=1)
        self.conv2 = Conv2d(channels, channels, kernel=3, padding=1)

    def forward(self, x):
        residual = x
        x = ReLU(self.conv1(x))
        x = self.conv2(x)
        return ReLU(x + residual)
```

**Key Features:**
- Pre-activation style: ReLU before second conv
- Identity shortcut connection
- Preserves spatial dimensions (padding=1)

**Code:** `vqvae.py` lines 30-41

---

## Quantizers

Both quantizers share the same interface but use different algorithms.

### VQEMAQuantizer (Baseline)

**Purpose:** Map continuous latents to discrete codebook entries using hard assignment with EMA updates

**Hyperparameters:**
- `num_embeddings` (K): Codebook size (default: 512)
- `embedding_dim` (D): Vector dimension (default: 256)
- `commitment_cost`: Commitment loss weight (default: 0.25)
- `decay`: EMA decay rate (default: 0.99)
- `epsilon`: Laplace smoothing constant (default: 1e-5)

**Algorithm:**

1. **Initialization:**
   - Codebook vectors: Uniform random in [-1/K, 1/K]
   - EMA buffers: `ema_cluster_size` (initialized to zeros), `ema_w` (copy of embeddings)

2. **Forward Pass:**
   ```python
   # Reshape: [B,D,H,W] → [B*H*W, D]
   z_flat = z.permute(0,2,3,1).reshape(-1, D)  # [N, D] where N=B*H*W

   # Compute L2 distances to all codes
   distances = ||z||² + ||e||² - 2⟨z,e⟩  # [N, K]

   # Hard assignment: nearest neighbor
   indices = argmin(distances, dim=1)  # [N]

   # Lookup quantized vectors
   z_q_flat = embedding(indices)  # [N, D]
   ```

3. **EMA Update (training only):**
   ```python
   # One-hot encode assignments
   encodings = F.one_hot(indices, K)  # [N, K]

   # Update cluster sizes with decay
   ema_cluster_size = decay * ema_cluster_size + (1-decay) * sum(encodings, dim=0)

   # Laplace smoothing for stability
   n = sum(ema_cluster_size)
   ema_cluster_size = (ema_cluster_size + ε) / (n + K*ε) * n

   # Update embedding weights
   dw = encodings.T @ z_flat  # [K, D]
   ema_w = decay * ema_w + (1-decay) * dw

   # Normalize by cluster sizes
   embedding.weight = ema_w / ema_cluster_size.unsqueeze(1)
   ```

4. **Losses:**
   - **Commitment Loss:** `0.25 * MSE(z, z_q.detach())`
     - Encourages encoder to commit to codebook entries
     - Gradient flows only to encoder (z_q detached)

5. **Straight-Through Estimator:**
   ```python
   z_q = z + (z_q - z).detach()
   ```
   - Forward: uses quantized values
   - Backward: gradients bypass quantization

**Returns:**
- `z_q`: Quantized latents [B, D, H, W]
- `loss_dict`: `{'commitment_loss': tensor}`
- `metrics_dict`: `{'perplexity': float, 'usage_rate': float, 'code_indices': tensor}`

**Expected Behavior:**
- **Codebook usage:** 40-60% (significant underutilization)
- **Perplexity:** 250-350 for K=512
- **PSNR:** 27-28 dB

**Code:** `vqvae.py` lines 100-191

---

### BAQuantizer (Our Method)

**Purpose:** Improve codebook utilization using Blahut-Arimoto algorithm with soft assignments

**Hyperparameters:**
- `num_embeddings` (K): Codebook size (default: 512)
- `embedding_dim` (D): Vector dimension (default: 256)
- `beta_start`: Initial temperature (default: 0.1)
- `beta_end`: Final temperature (default: 5.0)
- `commitment_cost`: Commitment loss weight (default: 0.25)
- `ba_iterations`: BA iterations per forward pass (default: 5)

**Algorithm:**

1. **Beta Annealing:**
   ```python
   def set_beta(epoch, max_epochs):
       progress = epoch / max_epochs
       # Cosine schedule: 0.1 → 5.0
       beta = beta_start + (beta_end - beta_start) * (1 - cos(π*progress)) / 2
   ```
   - Low β (early): Soft assignments, exploration
   - High β (late): Hard assignments, exploitation

2. **Blahut-Arimoto Iterations:**
   ```python
   def _blahut_arimoto_step(distances):
       # distances: [N, K] pairwise L2 distances
       N, K = distances.shape

       # Initialize uniform Q(k)
       Q = ones(K) / K

       # Iterate BA algorithm (default: 5 iterations)
       for _ in range(ba_iterations):
           # Compute log-probabilities (numerically stable)
           log_Q = log(Q.clamp(min=1e-10))
           logits = log_Q - beta * distances  # [N, K]

           # Soft assignments: P(k|x) = exp(log Q(k) - β*d(x,k)) / Z
           P = softmax(logits, dim=1)  # [N, K]

           # Update marginal: Q(k) = mean_x P(k|x)
           Q = P.mean(dim=0)  # [K]
           Q = Q.clamp(min=1e-10)  # Numerical stability

       return P  # [N, K] soft assignment probabilities
   ```

   **What BA does:**
   - Alternates between:
     - **E-step:** Compute soft assignments P(k|x) given current Q(k)
     - **M-step:** Update marginal Q(k) = mean(P)
   - Converges to optimal rate-distortion tradeoff
   - β controls hard/soft assignments (temperature parameter)

3. **Forward Pass:**
   ```python
   # Compute distances (same as VQ-EMA)
   distances = ||z||² + ||e||² - 2⟨z,e⟩  # [N, K]

   # Run BA algorithm
   P = _blahut_arimoto_step(distances)  # [N, K] soft assignments

   # Soft quantization: weighted average
   z_q_soft = P @ embeddings  # [N, D]

   # Hard assignment for metrics (argmax)
   indices = argmax(P, dim=1)  # [N]
   ```

4. **Losses:**
   - **Commitment Loss:** `0.25 * MSE(z, z_q_soft.detach())`
   - **Entropy Loss:** `-0.01 * H(Q)` where `Q = mean(P, dim=0)`
     - Encourages uniform code usage
     - Negative weight = maximize entropy

5. **Gradient-Based Codebook Updates:**
   - Unlike VQ-EMA, codebook updated via backprop
   - Gradients flow through soft assignments
   - AdamW optimizer updates embeddings

**Returns:**
- `z_q`: Quantized latents [B, D, H, W]
- `loss_dict`: `{'commitment_loss': tensor, 'entropy_loss': tensor}`
- `metrics_dict`: `{'perplexity': float, 'usage_rate': float, 'beta': float, 'code_indices': tensor}`

**Expected Behavior:**
- **Codebook usage:** 70-90% (improved utilization)
- **Perplexity:** Higher than VQ-EMA (goal of research)
- **PSNR:** Comparable or better than VQ-EMA

**Key Differences from VQ-EMA:**
| Feature | VQ-EMA | BA-VQ |
|---------|--------|-------|
| Assignment | Hard (argmin) | Soft (probabilistic) |
| Codebook update | EMA (no gradients) | Gradient-based (AdamW) |
| Utilization | 40-60% | 70-90% (goal) |
| Losses | Commitment only | Commitment + Entropy |
| Hyperparams | decay, epsilon | beta_start, beta_end, ba_iterations |
| Compute | Fast | Slower (5 BA iterations) |

**Code:** `vqvae.py` lines 194-310

---

## Training Pipeline

### Main Training Loop

**Function:** `train(model, epochs, batch_size, lr, run_name, use_wandb)`

**Code:** `vqvae.py` lines 387-486

**Flow:**

1. **Setup (lines 390-401):**
   ```python
   # Smart device selection (automatic)
   if torch.backends.mps.is_available():      # Mac GPU
       device = torch.device('mps')
   elif torch.cuda.is_available():            # NVIDIA GPU
       device = torch.device('cuda')
   else:
       device = torch.device('cpu')

   # Move model to device
   model = model.to(device)

   # Optimizer
   optimizer = AdamW(model.parameters(), lr=3e-4)

   # Data loaders
   train_loader, val_loader = get_dataloaders(batch_size)
   ```

2. **W&B Initialization (lines 404-411):**
   ```python
   if use_wandb:
       wandb.init(project='vq-codebook', name=run_name, config={
           'quantizer_type': model.quantizer_type,
           'codebook_size': model.quantizer.num_embeddings,
           'epochs': epochs,
           'batch_size': batch_size,
           'lr': lr,
       })
   ```

3. **Create Output Directory (lines 414-415):**
   ```python
   out_dir = Path(f'experiments/{run_name}')
   out_dir.mkdir(parents=True, exist_ok=True)
   ```

4. **Epoch Loop (lines 418-464):**
   ```python
   for epoch in range(epochs):
       # Update beta (BA-VQ only)
       if hasattr(model.quantizer, 'set_beta'):
           model.quantizer.set_beta(epoch, epochs)

       # Training epoch
       model.train()
       train_metrics = train_epoch(model, train_loader, optimizer, device)

       # Validation epoch
       model.eval()
       val_metrics = validate(model, val_loader, device)

       # Print progress
       print(f"Epoch {epoch+1}/{epochs} | "
             f"Train Loss: {train_metrics['loss']:.4f} | "
             f"Val Loss: {val_metrics['loss']:.4f} | "
             f"PSNR: {val_metrics['psnr']:.2f} | "
             f"Perplexity: {val_metrics['perplexity']:.1f} | "
             f"Usage: {val_metrics['usage_rate']*100:.1f}%")

       # Log to W&B
       if use_wandb:
           wandb.log({
               'epoch': epoch,
               'train/loss': train_metrics['loss'],
               'train/recon_loss': train_metrics['recon_loss'],
               'train/commitment_loss': train_metrics.get('commitment_loss', 0),
               'train/entropy_loss': train_metrics.get('entropy_loss', 0),
               'val/loss': val_metrics['loss'],
               'val/recon_loss': val_metrics['recon_loss'],
               'val/psnr': val_metrics['psnr'],
               'val/perplexity': val_metrics['perplexity'],
               'val/usage_rate': val_metrics['usage_rate'],
               'val/commitment_loss': val_metrics.get('commitment_loss', 0),
               'val/entropy_loss': val_metrics.get('entropy_loss', 0),
               'val/beta': val_metrics.get('beta', 0),
           })

       # Save checkpoint every 10 epochs
       if (epoch + 1) % 10 == 0:
           checkpoint = {
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'val_metrics': val_metrics,
           }
           torch.save(checkpoint, out_dir / f'checkpoint_epoch{epoch+1}.pt')
   ```

5. **Final Save (lines 476-481):**
   ```python
   # Save final model
   torch.save(model.state_dict(), out_dir / 'final_model.pt')

   # Save final metrics
   with open(out_dir / 'final_metrics.json', 'w') as f:
       json.dump(val_metrics, f, indent=2, default=float)

   print(f"\nTraining complete! Results saved to {out_dir}")
   if use_wandb:
       wandb.finish()
   ```

---

### Training Epoch

**Function:** `train_epoch(model, loader, optimizer, device)`

**Code:** `vqvae.py` lines 489-525

**Flow:**
```python
def train_epoch(model, loader, optimizer, device):
    total_loss = 0
    total_recon_loss = 0
    total_quant_losses = {}

    for x, _ in tqdm(loader, desc='Training'):
        x = x.to(device)

        # Forward pass
        x_recon, recon_loss, quant_loss_dict, metrics = model(x)

        # Total loss = reconstruction + all quantization losses
        loss = recon_loss + sum(quant_loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        for k, v in quant_loss_dict.items():
            total_quant_losses[k] = total_quant_losses.get(k, 0) + v.item()

    # Return averaged metrics
    n = len(loader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        **{k: v / n for k, v in total_quant_losses.items()}
    }
```

**What's Computed:**
- `loss`: Total loss (reconstruction + quantization)
- `recon_loss`: MSE between original and reconstructed
- `commitment_loss`: Encoder commitment (both quantizers)
- `entropy_loss`: Entropy regularization (BA-VQ only)

**Note:** Perplexity and usage rate NOT computed during training (too expensive)

---

### Validation Epoch

**Function:** `validate(model, loader, device)`

**Code:** `vqvae.py` lines 528-584

**Flow:**
```python
def validate(model, loader, device):
    total_loss = 0
    total_recon_loss = 0
    total_quant_losses = {}
    all_code_indices = []

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, desc='Validation')):
            x = x.to(device)

            # Forward pass
            x_recon, recon_loss, quant_loss_dict, metrics = model(x)
            loss = recon_loss + sum(quant_loss_dict.values())

            # Accumulate
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            for k, v in quant_loss_dict.items():
                total_quant_losses[k] = total_quant_losses.get(k, 0) + v.item()

            # Collect code indices for perplexity/usage
            all_code_indices.append(metrics['code_indices'])

            # Save first batch for PSNR
            if i == 0:
                all_x = x
                all_x_recon = x_recon

    # Compute metrics from ALL validation data
    all_code_indices = torch.cat(all_code_indices)  # [N] where N=10k*64 for CIFAR-10
    perplexity = compute_perplexity(all_code_indices, model.quantizer.num_embeddings)
    usage_rate = compute_usage_rate(all_code_indices, model.quantizer.num_embeddings)

    # PSNR from first batch only (for speed)
    psnr = compute_psnr(all_x, all_x_recon)

    # Build result dictionary
    n = len(loader)
    result = {
        'loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'perplexity': perplexity,
        'usage_rate': usage_rate,
        'psnr': psnr,
        **{k: v / n for k, v in total_quant_losses.items()}
    }

    # Add beta if BA-VQ
    if hasattr(model.quantizer, 'beta'):
        result['beta'] = model.quantizer.beta

    return result
```

**What's Computed:**
- All training losses
- `perplexity`: From ALL validation code indices
- `usage_rate`: From ALL validation code indices
- `psnr`: From FIRST batch only (faster)
- `beta`: Current temperature (BA-VQ only)

---

## Metrics

### Perplexity

**Formula:** `perplexity = exp(entropy)`

**Code:** `vqvae.py` lines 591-607

**Implementation:**
```python
def compute_perplexity(code_indices, num_embeddings):
    """
    Compute perplexity from code indices.

    Args:
        code_indices: [N] tensor of code assignments
        num_embeddings: K (codebook size)

    Returns:
        perplexity: float in [1, K]
    """
    # Count code frequencies
    counts = torch.bincount(code_indices, minlength=num_embeddings).float()
    probs = counts / counts.sum()

    # Entropy (filter out zero probabilities)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))

    # Perplexity = exp(entropy)
    perplexity = torch.exp(entropy)
    return perplexity.item()
```

**Interpretation:**
- Range: [1, K] where K = codebook size
- **Higher = better** codebook utilization
- Uniform distribution: perplexity = K
- One code only: perplexity = 1
- Expected for VQ-EMA K=512: 250-350 (50-70%)
- Goal for BA-VQ: > VQ-EMA perplexity

**When Computed:** Validation only (from ALL validation code indices)

---

### Usage Rate

**Formula:** `usage_rate = # unique codes / K`

**Code:** `vqvae.py` lines 610-617

**Implementation:**
```python
def compute_usage_rate(code_indices, num_embeddings):
    """
    Compute fraction of codes used at least once.

    Args:
        code_indices: [N] tensor of code assignments
        num_embeddings: K (codebook size)

    Returns:
        usage_rate: float in [0, 1]
    """
    unique_codes = torch.unique(code_indices)
    usage_rate = len(unique_codes) / num_embeddings
    return usage_rate
```

**Interpretation:**
- Range: [0, 1]
- **Higher = better** (more codes active)
- 1.0 = all codes used
- 0.0 = no codes used (shouldn't happen)
- Expected for VQ-EMA K=512: 0.6-0.75 (60-75%)
- Goal for BA-VQ: > VQ-EMA usage rate

**When Computed:** Validation only (from ALL validation code indices)

---

### PSNR (Peak Signal-to-Noise Ratio)

**Formula:** `PSNR = 20 * log10(1 / sqrt(MSE))`

**Code:** `vqvae.py` lines 620-627

**Implementation:**
```python
def compute_psnr(x, x_recon):
    """
    Compute PSNR between original and reconstructed images.

    Args:
        x: [B, 3, 32, 32] original images (normalized to [-1, 1])
        x_recon: [B, 3, 32, 32] reconstructed images

    Returns:
        psnr: float in dB
    """
    mse = F.mse_loss(x, x_recon)
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    return psnr.item()
```

**Interpretation:**
- Units: dB (decibels)
- **Higher = better** reconstruction quality
- Range: typically 20-35 dB for VQ-VAE
- Expected for VQ-EMA K=512: 27-28 dB
- Goal for BA-VQ: ≥ VQ-EMA PSNR (don't sacrifice quality)

**When Computed:** Validation only (from FIRST batch only for speed)

**Note:** Uses pixel range [0, 1] but images are normalized to [-1, 1], so formula uses max value = 1.0

---

## Data Loading

### Dataset

**Function:** `get_dataloaders(batch_size, num_workers)`

**Code:** `vqvae.py` lines 354-380

**Dataset:** CIFAR-10
- Training: 50,000 images
- Validation: 10,000 images
- Image size: 32×32 RGB
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Labels not used** (unsupervised learning)

**Preprocessing:**
```python
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1] range, HWC→CHW
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1,1] range
])
```

**DataLoader Configuration:**

```python
# Smart pin_memory: only for CUDA
use_pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=256,        # Default (reduce to 64 for Mac)
    shuffle=True,          # Random order
    num_workers=4,         # Parallel loading (reduce to 2 for Mac)
    pin_memory=use_pin_memory  # False on MPS, True on CUDA
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,         # Sequential order
    num_workers=4,
    pin_memory=use_pin_memory
)
```

**Device-Specific Optimizations:**
- **Mac (MPS):** `pin_memory=False`, `num_workers=2`, `batch_size=64`
- **GPU (CUDA):** `pin_memory=True`, `num_workers=4`, `batch_size=256`

**Auto-Download:**
- First run downloads CIFAR-10 to `./data/` directory (~170 MB)
- Subsequent runs use cached data

---

## Experiment Outputs

### Directory Structure

After training, each experiment creates:

```
experiments/
└── [run_name]/
    ├── final_model.pt              # Final model weights
    ├── final_metrics.json          # Final validation metrics
    └── checkpoint_epoch{N}.pt      # Periodic checkpoints (every 10 epochs)
```

### final_model.pt

**Size:** ~27 MB

**Contents:** Model state dictionary (all trainable parameters)

**How to Load:**
```python
from vqvae import VQVAE

# Create model with same config
model = VQVAE(quantizer_type='vq_ema', codebook_size=512, latent_dim=256)

# Load weights
model.load_state_dict(torch.load('experiments/my_run/final_model.pt'))
model.eval()
```

### final_metrics.json

**Contents:** Dictionary of final validation metrics

**Example:**
```json
{
  "loss": 0.0318,
  "recon_loss": 0.0224,
  "perplexity": 287.3,
  "usage_rate": 0.643,
  "psnr": 27.45,
  "commitment_loss": 0.0094,
  "entropy_loss": -0.0015
}
```

**Usage:**
```python
import json

with open('experiments/my_run/final_metrics.json') as f:
    metrics = json.load(f)

print(f"Perplexity: {metrics['perplexity']:.1f}")
print(f"Usage: {metrics['usage_rate']*100:.1f}%")
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

### checkpoint_epoch{N}.pt

**Saved Every:** 10 epochs

**Contents:**
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'val_metrics': dict
}
```

**How to Resume Training:**
```python
# Load checkpoint
checkpoint = torch.load('experiments/my_run/checkpoint_epoch20.pt')

# Restore model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training from start_epoch...
```

### W&B Tracking

**Project:** `vq-codebook`

**What's Logged (every epoch):**
- Training losses: `train/loss`, `train/recon_loss`, `train/commitment_loss`, `train/entropy_loss`
- Validation losses: Same as training
- Validation metrics: `val/psnr`, `val/perplexity`, `val/usage_rate`
- BA-VQ specific: `val/beta`

**Access:**
```bash
wandb login  # One time only
# Then view at: https://wandb.ai/your-username/vq-codebook
```

**Disable W&B:**
```bash
python train.py --name test --no_wandb
```

---

## Usage Examples

### Quick Test (2 epochs, no W&B)

**Purpose:** Verify everything works

```bash
python train.py --quantizer vq_ema --codebook_size 128 --epochs 2 \
    --batch_size 32 --name smoke_test --no_wandb
```

**Expected Time:** ~5 minutes on Mac M1/M2, ~2 minutes on GPU

**Output:**
- `experiments/smoke_test/final_model.pt`
- `experiments/smoke_test/final_metrics.json`

---

### Mac Quick Test (3 epochs)

**Purpose:** Fast iteration on Mac before GPU deployment

```bash
# VQ-EMA baseline
python train.py --quantizer vq_ema --codebook_size 256 --epochs 3 \
    --batch_size 64 --name mac_vq_k256 --no_wandb

# BA-VQ test
python train.py --quantizer ba_vq --codebook_size 256 --epochs 3 \
    --batch_size 64 --name mac_ba_k256 --no_wandb
```

**Expected Time:** ~15-20 minutes each on Mac M1/M2

**Metrics to Check:**
- No NaN in losses
- Perplexity > 50
- Usage > 20%
- PSNR > 20 dB

---

### GPU Server Full Experiment

**Purpose:** Production experiments for paper

```bash
# Setup (one time)
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cuda.txt
wandb login

# Baseline: VQ-EMA K=512, 100 epochs
python train.py --quantizer vq_ema --codebook_size 512 --epochs 100 \
    --batch_size 256 --name baseline_k512

# Our method: BA-VQ K=512, 100 epochs
python train.py --quantizer ba_vq --codebook_size 512 --epochs 100 \
    --batch_size 256 --name ba_k512

# Multiple seeds for statistical significance
python train.py --quantizer vq_ema --codebook_size 512 --epochs 100 \
    --batch_size 256 --name baseline_k512_seed2

python train.py --quantizer ba_vq --codebook_size 512 --epochs 100 \
    --batch_size 256 --name ba_k512_seed2
```

**Expected Time:** ~4-6 hours per experiment on modern GPU

---

### Resume from Checkpoint

**Purpose:** Continue training after interruption

```bash
# Not directly supported by train.py, need to modify code

# Manual resume:
checkpoint = torch.load('experiments/my_run/checkpoint_epoch50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Then continue training loop from start_epoch...
```

---

## Key Implementation Details

### Automatic Device Selection

**Code:** `vqvae.py` lines 387-394

```python
# Smart device selection: MPS (Mac) > CUDA (GPU server) > CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

**No code changes needed between Mac and GPU server!**

### Loss Computation

**Total Loss:**
```python
# VQ-EMA
loss = recon_loss + commitment_loss

# BA-VQ
loss = recon_loss + commitment_loss + entropy_loss
```

**Weights:**
- Reconstruction: 1.0 (MSE)
- Commitment: 0.25
- Entropy: -0.01 (BA-VQ only, negative to maximize)

### Beta Annealing (BA-VQ)

**Schedule:** Cosine from 0.1 to 5.0 over epochs

**Code:** `vqvae.py` lines 216-220

```python
def set_beta(epoch, max_epochs):
    progress = epoch / max_epochs
    beta = beta_start + (beta_end - beta_start) * (1 - math.cos(math.pi * progress)) / 2
```

**Effect:**
- Epoch 0: β = 0.1 (very soft assignments, exploration)
- Epoch 50 (of 100): β = 2.55 (moderate)
- Epoch 100: β = 5.0 (nearly hard assignments, exploitation)

### Straight-Through Estimator

**Code:** Both quantizers

```python
z_q = z + (z_q - z).detach()
```

**Effect:**
- Forward: uses quantized values `z_q`
- Backward: gradients bypass quantization, flow to `z`
- Essential for training discrete latent models

---

## Common Issues & Solutions

### W&B Login Error

**Error:** `wandb.errors.errors.UsageError: api_key not configured`

**Solution 1:** Disable W&B
```bash
python train.py --name test --no_wandb
```

**Solution 2:** Login once
```bash
wandb login  # Enter API key from https://wandb.ai/authorize
```

### MPS Warnings

**Warning:** `UserWarning: The operator 'aten::...' is not currently supported on the MPS backend`

**Solution:** Ignore - operations automatically fall back to CPU, still works

### Out of Memory

**Error:** `RuntimeError: CUDA out of memory` or MPS equivalent

**Solution:** Reduce batch size
```bash
python train.py --batch_size 32 ...  # or even 16
```

### Low Perplexity/Usage

**Issue:** Perplexity < 50, usage < 20% after many epochs

**Possible Causes:**
- Codebook too large for dataset
- Learning rate too high/low
- Need more training epochs

**Solutions:**
- Reduce codebook size: `--codebook_size 256`
- Adjust learning rate: `--lr 1e-4` or `--lr 1e-3`
- Train longer: `--epochs 200`

---

## Next Steps

1. **Test locally:** Run 3-epoch experiments on Mac to verify
2. **Deploy to GPU:** Copy code to server, install dependencies
3. **Run baselines:** VQ-EMA with K ∈ {256, 512, 1024, 2048}
4. **Run BA-VQ:** Same codebook sizes as baseline
5. **Analyze:** Use `analyze.ipynb` to compare results
6. **Iterate:** Tune BA hyperparameters based on results

**Success Metric:** BA-VQ perplexity > VQ-EMA perplexity for same K

---

## References

- **VQ-VAE Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **Blahut-Arimoto:** [Wikipedia](https://en.wikipedia.org/wiki/Blahut–Arimoto_algorithm)
- **Reference Implementation:** [Karpathy VQ-VAE](https://github.com/nikitus20/deep-vector-quantization)
