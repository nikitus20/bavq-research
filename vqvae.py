"""
VQ-VAE with Blahut-Arimoto Optimization
Complete implementation in one file for easy understanding.

Usage:
    from vqvae import VQVAE, train
    model = VQVAE(quantizer_type='ba_vq', codebook_size=512)
    train(model, epochs=100, run_name='experiment')

Reference: https://github.com/nikitus20/deep-vector-quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import wandb
from pathlib import Path
import json
import math
from tqdm import tqdm


# ============================================================================
# Section 1: Encoder/Decoder (adapted from DeepMind architecture)
# ============================================================================

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class Encoder(nn.Module):
    """
    Encoder: 32x32x3 -> 8x8x256
    Architecture: 2x strided convs (4x spatial reduction) + 2 ResBlocks
    """
    def __init__(self, latent_dim=256, n_hid=128):
        super().__init__()

        # Downsampling: 32x32 -> 16x16 -> 8x8
        self.conv1 = nn.Conv2d(3, n_hid, kernel_size=4, stride=2, padding=1)  # 32->16
        self.conv2 = nn.Conv2d(n_hid, 2*n_hid, kernel_size=4, stride=2, padding=1)  # 16->8
        self.conv3 = nn.Conv2d(2*n_hid, latent_dim, kernel_size=3, padding=1)  # 8->8

        # Residual blocks
        self.res1 = ResidualBlock(latent_dim)
        self.res2 = ResidualBlock(latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)
        x = self.res2(x)
        return x


class Decoder(nn.Module):
    """
    Decoder: 8x8x256 -> 32x32x3
    Architecture: 2 ResBlocks + 2x transposed convs (4x spatial upsampling)
    """
    def __init__(self, latent_dim=256, n_hid=128):
        super().__init__()

        # Residual blocks
        self.res1 = ResidualBlock(latent_dim)
        self.res2 = ResidualBlock(latent_dim)

        # Upsampling: 8x8 -> 16x16 -> 32x32
        self.conv1 = nn.Conv2d(latent_dim, 2*n_hid, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(2*n_hid, n_hid, kernel_size=4, stride=2, padding=1)  # 8->16
        self.conv3 = nn.ConvTranspose2d(n_hid, 3, kernel_size=4, stride=2, padding=1)  # 16->32

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # No activation on final layer
        return x


# ============================================================================
# Section 2: Quantizers
# ============================================================================

class VQEMAQuantizer(nn.Module):
    """
    Standard VQ with EMA updates (baseline)
    Following VQ-VAE paper: https://arxiv.org/abs/1711.00937
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, z):
        """
        Args:
            z: Encoded latents [B, D, H, W]
        Returns:
            z_q: Quantized latents [B, D, H, W]
            loss_dict: Dictionary with 'commitment_loss'
            metrics_dict: Dictionary with 'perplexity', 'usage_rate'
        """
        # Flatten spatial dimensions: [B, D, H, W] -> [B*H*W, D]
        z_flat = z.permute(0, 2, 3, 1).contiguous()
        B, H, W, D = z_flat.shape
        z_flat = z_flat.view(-1, D)

        # Compute distances to codebook: [B*H*W, K]
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Find nearest codes
        encoding_indices = torch.argmin(distances, dim=1)  # [B*H*W]

        # Quantize
        z_q_flat = self.embedding(encoding_indices)  # [B*H*W, D]

        # EMA update (only during training)
        if self.training:
            # One-hot encode assignments: [B*H*W, K]
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

            # Update cluster sizes
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                   (1 - self.decay) * torch.sum(encodings, dim=0)

            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # Update embeddings
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            # Normalize embeddings (add epsilon for numerical stability)
            self.embedding.weight.data = self.ema_w / (self.ema_cluster_size.unsqueeze(1) + self.epsilon)

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back: [B*H*W, D] -> [B, D, H, W]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Compute metrics
        perplexity = compute_perplexity(encoding_indices, self.num_embeddings)
        usage_rate = compute_usage_rate(encoding_indices, self.num_embeddings)

        loss_dict = {'commitment_loss': commitment_loss}
        metrics_dict = {
            'perplexity': perplexity,
            'usage_rate': usage_rate,
            'code_indices': encoding_indices
        }

        return z_q, loss_dict, metrics_dict


class BAQuantizer(nn.Module):
    """
    Blahut-Arimoto VQ (our method)
    Uses soft assignments + entropy regularization
    """
    def __init__(self, num_embeddings, embedding_dim,
                 beta_start=0.1, beta_end=5.0, commitment_cost=0.25, ba_iterations=5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.commitment_cost = commitment_cost
        self.ba_iterations = ba_iterations

        # Codebook (trainable via gradients, not EMA)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        # Current beta (will be updated during training)
        self.register_buffer('beta', torch.tensor(beta_start))

        # Global prior π for dataset-level BA
        self.register_buffer('pi', torch.full((num_embeddings,), 1.0 / num_embeddings))
        self.pi_momentum = 0.99

    def set_beta(self, epoch, max_epochs):
        """Anneal beta from beta_start to beta_end using cosine schedule"""
        progress = epoch / max(max_epochs, 1)
        value = self.beta_start + (self.beta_end - self.beta_start) * \
                (1 - math.cos(progress * math.pi)) / 2
        # Update tensor in-place to avoid rebinding to float
        self.beta.data.fill_(value)

    def _blahut_arimoto_step(self, distances):
        """
        Core BA algorithm for soft assignments

        Args:
            distances: [N, K] pairwise distances
        Returns:
            P: [N, K] soft assignments (probabilities)
            Q: [K] marginal distribution over codes
        """
        N, K = distances.shape

        # Initialize from global prior π (dataset-level BA)
        Q = self.pi.clone()

        # BA iterations
        for _ in range(self.ba_iterations):
            # Compute log-probabilities (numerically stable)
            log_Q = torch.log(Q.clamp(min=1e-10))
            logits = log_Q.unsqueeze(0) - self.beta * distances  # [N, K]

            # Numerical stability: subtract rowwise max before softmax
            logits = logits - logits.max(dim=1, keepdim=True).values

            # Compute soft assignments P(k|x)
            P = F.softmax(logits, dim=1)

            # Update marginal distribution Q(k)
            Q = P.mean(dim=0)
            Q = Q.clamp(min=1e-10)

        return P, Q

    def forward(self, z):
        """
        Args:
            z: Encoded latents [B, D, H, W]
        Returns:
            z_q: Quantized latents [B, D, H, W]
            loss_dict: Dictionary with 'commitment_loss', 'entropy_loss'
            metrics_dict: Dictionary with 'perplexity', 'usage_rate', 'beta'
        """
        # Flatten spatial dimensions: [B, D, H, W] -> [B*H*W, D]
        z_flat = z.permute(0, 2, 3, 1).contiguous()
        B, H, W, D = z_flat.shape
        z_flat = z_flat.view(-1, D)

        # Compute distances to codebook: [B*H*W, K]
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # BA algorithm for soft assignments
        P, Q = self._blahut_arimoto_step(distances)  # [B*H*W, K], [K]

        # Update global prior π with EMA (dataset-level BA)
        with torch.no_grad():
            self.pi.mul_(self.pi_momentum).add_((1 - self.pi_momentum) * Q)
            self.pi.div_(self.pi.sum())  # Renormalize for safety

        # Soft quantization: weighted average of codes
        z_q_soft = torch.matmul(P, self.embedding.weight)  # [B*H*W, D]

        # For metrics, use hard assignment (argmax)
        encoding_indices = torch.argmax(P, dim=1)

        # Commitment loss (encourage encoder to commit to codes)
        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q_soft.detach())

        # Codebook loss (move embeddings toward encoder outputs)
        codebook_loss = F.mse_loss(z_q_soft, z_flat.detach())

        # Entropy regularization (encourage uniform usage)
        entropy = -torch.sum(Q * torch.log(Q.clamp(min=1e-10)))
        entropy_loss = -0.01 * entropy  # Negative because we want to maximize entropy

        # Soft→hard STE: forward uses hard quantization, backward uses soft
        z_q_hard = self.embedding(encoding_indices)
        z_q = z_q_soft + (z_q_hard - z_q_soft).detach()

        # Reshape back: [B*H*W, D] -> [B, D, H, W]
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Compute metrics
        perplexity = compute_perplexity(encoding_indices, self.num_embeddings)
        usage_rate = compute_usage_rate(encoding_indices, self.num_embeddings)

        loss_dict = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'entropy_loss': entropy_loss
        }
        metrics_dict = {
            'perplexity': perplexity,
            'usage_rate': usage_rate,
            'beta': self.beta.item(),
            'code_indices': encoding_indices
        }

        return z_q, loss_dict, metrics_dict


# ============================================================================
# Section 3: Complete VQ-VAE Model
# ============================================================================

class VQVAE(nn.Module):
    """Complete VQ-VAE - just plug in different quantizers"""
    def __init__(self, quantizer_type='vq_ema', codebook_size=512, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Choose quantizer
        if quantizer_type == 'vq_ema':
            self.quantizer = VQEMAQuantizer(codebook_size, latent_dim)
        elif quantizer_type == 'ba_vq':
            self.quantizer = BAQuantizer(codebook_size, latent_dim)
        else:
            raise ValueError(f"Unknown quantizer: {quantizer_type}")

        self.quantizer_type = quantizer_type

    def forward(self, x):
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, quant_loss_dict, metrics = self.quantizer(z)

        # Decode
        x_recon = self.decoder(z_q)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        return x_recon, recon_loss, quant_loss_dict, metrics


# ============================================================================
# Section 4: Data Loading
# ============================================================================

def get_dataloaders(batch_size=256, num_workers=4):
    """Returns CIFAR-10 train/val loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # pin_memory only helps with CUDA, not MPS
    use_pin_memory = torch.cuda.is_available()

    # persistent_workers speeds up training when num_workers > 0
    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=use_persistent
    )

    return train_loader, val_loader


# ============================================================================
# Section 5: Training
# ============================================================================

def train(model, epochs=100, batch_size=256, lr=3e-4, run_name='experiment', use_wandb=True):
    """Simple training loop with W&B logging"""

    # Setup - Smart device selection: MPS (Mac) > CUDA (GPU server) > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader, val_loader = get_dataloaders(batch_size)

    # W&B
    if use_wandb:
        wandb.init(project='vq-codebook', name=run_name, config={
            'quantizer_type': model.quantizer_type,
            'codebook_size': model.quantizer.num_embeddings,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
        })

    # Output directory
    out_dir = Path(f'experiments/{run_name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        # Update beta for BA-VQ
        if hasattr(model.quantizer, 'set_beta'):
            model.quantizer.set_beta(epoch, epochs)

        # Training
        model.train()
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validation
        model.eval()
        val_metrics = validate(model, val_loader, device)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"PSNR: {val_metrics['psnr']:.2f} | "
              f"Perplexity: {val_metrics['perplexity']:.1f} | "
              f"Usage: {val_metrics['usage_rate']:.1%}")

        # Log to W&B
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/recon_loss': train_metrics['recon_loss'],
                'val/loss': val_metrics['loss'],
                'val/recon_loss': val_metrics['recon_loss'],
                'val/psnr': val_metrics['psnr'],
                'val/perplexity': val_metrics['perplexity'],
                'val/usage_rate': val_metrics['usage_rate'],
            }

            # Add quantizer-specific losses
            for k, v in train_metrics.items():
                if k.endswith('_loss') and k not in ['loss', 'recon_loss']:
                    log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                if k.endswith('_loss') and k not in ['loss', 'recon_loss']:
                    log_dict[f'val/{k}'] = v

            # Add beta for BA-VQ
            if 'beta' in val_metrics:
                log_dict['val/beta'] = val_metrics['beta']

            wandb.log(log_dict)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, out_dir / f'checkpoint_epoch{epoch+1}.pt')

    # Final save
    torch.save(model.state_dict(), out_dir / 'final_model.pt')

    # Save final metrics
    with open(out_dir / 'final_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2, default=float)

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Results saved to {out_dir}")


def train_epoch(model, loader, optimizer, device):
    """Single training epoch"""
    total_loss = 0
    total_recon_loss = 0
    total_quant_losses = {}
    n_batches = 0

    for x, _ in tqdm(loader, desc='Training', leave=False):
        x = x.to(device)

        # Forward
        x_recon, recon_loss, quant_loss_dict, metrics = model(x)

        # Total loss
        loss = recon_loss + sum(quant_loss_dict.values())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        for k, v in quant_loss_dict.items():
            total_quant_losses[k] = total_quant_losses.get(k, 0) + v.item()
        n_batches += 1

    # Average
    metrics_dict = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
    }
    for k, v in total_quant_losses.items():
        metrics_dict[k] = v / n_batches

    return metrics_dict


def validate(model, loader, device):
    """Validation loop - returns metrics dict"""
    total_loss = 0
    total_recon_loss = 0
    total_quant_losses = {}
    all_code_indices = []
    sum_mse_psnr = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, _ in tqdm(loader, desc='Validation', leave=False):
            x = x.to(device)

            # Forward
            x_recon, recon_loss, quant_loss_dict, metrics = model(x)

            # Total loss
            loss = recon_loss + sum(quant_loss_dict.values())

            # Accumulate
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            for k, v in quant_loss_dict.items():
                total_quant_losses[k] = total_quant_losses.get(k, 0) + v.item()

            all_code_indices.append(metrics['code_indices'])

            # Accumulate MSE for PSNR over all batches
            x_denorm = (x * 0.5 + 0.5).clamp(0.0, 1.0)
            x_recon_denorm = (x_recon * 0.5 + 0.5).clamp(0.0, 1.0)
            sum_mse_psnr += F.mse_loss(x_denorm, x_recon_denorm, reduction='mean').item()

            n_batches += 1

    # Compute metrics
    all_code_indices = torch.cat(all_code_indices)
    perplexity = compute_perplexity(all_code_indices, model.quantizer.num_embeddings)
    usage_rate = compute_usage_rate(all_code_indices, model.quantizer.num_embeddings)

    # Compute PSNR from mean MSE over all batches
    mean_mse = sum_mse_psnr / max(1, n_batches)
    psnr = 10.0 * math.log10(1.0 / (mean_mse + 1e-12))

    metrics_dict = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'perplexity': perplexity,
        'usage_rate': usage_rate,
        'psnr': psnr,
    }

    for k, v in total_quant_losses.items():
        metrics_dict[k] = v / n_batches

    # Add beta for BA-VQ
    if hasattr(model.quantizer, 'beta'):
        metrics_dict['beta'] = model.quantizer.beta.item()

    # Add π (global prior) metrics for BA-VQ
    if hasattr(model.quantizer, 'pi'):
        pi = model.quantizer.pi
        H_pi = -(pi * (pi + 1e-10).log()).sum().item()
        ppl_pi = math.exp(H_pi)
        metrics_dict['pi_entropy'] = H_pi
        metrics_dict['pi_perplexity'] = ppl_pi

    return metrics_dict


# ============================================================================
# Section 6: Metrics
# ============================================================================

def compute_perplexity(code_indices, num_embeddings):
    """
    Perplexity = exp(entropy) of code distribution
    Higher is better (more codes being used)
    """
    # Count code frequencies
    counts = torch.bincount(code_indices, minlength=num_embeddings).float()
    probs = counts / counts.sum()

    # Entropy (filter out zero probabilities)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))

    # Perplexity
    perplexity = torch.exp(entropy)

    return perplexity.item()


def compute_usage_rate(code_indices, num_embeddings):
    """
    Usage rate = % of codes used at least once
    Higher is better (less dead codes)
    """
    unique_codes = torch.unique(code_indices)
    usage_rate = len(unique_codes) / num_embeddings
    return usage_rate


def compute_psnr(x, x_recon):
    """
    Peak Signal-to-Noise Ratio
    Higher is better (better reconstruction quality)

    De-normalizes from [-1,1] to [0,1] for proper PSNR calculation
    """
    # De-normalize from [-1, 1] to [0, 1]
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    x_recon = (x_recon * 0.5 + 0.5).clamp(0.0, 1.0)

    mse = F.mse_loss(x, x_recon)
    # Use device-aware max_i and add epsilon for numerical stability
    max_i = torch.ones((), device=mse.device)
    psnr = 20 * torch.log10(max_i / torch.sqrt(mse + 1e-12))
    return psnr.item()


if __name__ == '__main__':
    # Quick test
    print("Testing VQ-VAE implementation...")

    # Create model
    model = VQVAE(quantizer_type='vq_ema', codebook_size=256)

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    x_recon, recon_loss, quant_loss_dict, metrics = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Quantization losses: {quant_loss_dict}")
    print(f"Metrics: {metrics}")
    print("\n✓ VQ-VAE implementation test passed!")
