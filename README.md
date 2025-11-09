# VQ Codebook Utilization Research

Investigating codebook utilization in Vector Quantized Variational Autoencoders (VQ-VAE) through BA optimization.

## Problem

VQ-VAE and VQ-GAN STE training suffer from slow updates of unused codebooks, and this results in severe codebook underutilization - sometimes only 11-12% of codes are utilized at scale (K=16,384). This limits representational capacity and reconstruction quality.

## Approach

Apply rate-distortion theory via BA algorithm for principled codebook optimization with soft assignments and entropy regularization.

## Documentation

📖 **[CODE_OVERVIEW.md](CODE_OVERVIEW.md)** - Comprehensive guide to all components, metrics, and implementation details

## Project Status

✅ **Phase 1: Foundation Complete** - Minimal, working implementation ready for experiments

## Quick Start

### ⚡ GPU Cloud (A100 - 1 Hour)

**Recommended for initial experiments.** Runs 8 core experiments in ~50-60 minutes.

```bash
# 1. Clone repo
git clone https://github.com/nikitus20/bavq-research.git
cd bavq-research

# 2. Install dependencies
pip install torch torchvision tqdm pyyaml clean-fid Pillow scipy

# 3. Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Configure CUDA allocator for efficient memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 5. Run all experiments (8 runs in ~50-60 min)
python run_gpu_experiments.py
```

**What it does:**
- VQ-EMA vs BA-VQ at K=256, K=512, K=1024
- Euclidean vs Cosine distance metrics
- Saves results to `experiments/` and summary to `experiments/gpu_run_summary.json`

### GPU Server Setup (Extended Experiments)

```bash
# 1. Clone repo
git clone https://github.com/nikitus20/bavq-research.git
cd bavq-research

# 2. Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Configure CUDA allocator for efficient memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 6. Quick test (2 epochs, ~2 minutes)
python train.py --quantizer vq_ema --codebook_size 128 --epochs 2 --batch_size 256 --name test --no_wandb

# 7. Run all experiments (~3-4 hours)
./run_experiments.sh
```

### Local Mac Setup (Development Only)

```bash
# Install dependencies (uses MPS GPU automatically)
pip install -r requirements.txt

# Quick test
python train.py --quantizer vq_ema --codebook_size 128 --epochs 2 --batch_size 64 --name test --no_wandb
```

### Individual Experiments

```bash
# VQ-EMA K=256 baseline
python train.py --quantizer vq_ema --codebook_size 256 --epochs 30 --batch_size 256 --name vq_ema_k256 --no_wandb

# BA-VQ K=256 (our method)
python train.py --quantizer ba_vq --codebook_size 256 --epochs 30 --batch_size 256 --name ba_vq_k256 --no_wandb

# VQ-EMA K=512 baseline
python train.py --quantizer vq_ema --codebook_size 512 --epochs 30 --batch_size 256 --name vq_ema_k512 --no_wandb

# BA-VQ K=512 (our method)
python train.py --quantizer ba_vq --codebook_size 512 --epochs 30 --batch_size 256 --name ba_vq_k512 --no_wandb
```

## Project Structure

```
bavq-research/
├── vqvae.py                  # Complete VQ-VAE implementation (~650 lines)
│                             #   - Encoder/Decoder
│                             #   - VQ-EMA quantizer (baseline)
│                             #   - BA-VQ quantizer (our method)
│                             #   - Training loop + metrics
│
├── train.py                  # CLI wrapper
├── eval_cifar.py             # Evaluation script (r-FID, r-IS)
├── run_gpu_experiments.py    # ⚡ GPU cloud runner (1 hour, 8 experiments)
├── run_experiments.sh        # Extended experiments (3-4 hours)
│
├── config.yaml               # Default hyperparameters
├── config_gpu.yaml           # GPU server config (batch=256)
│
├── requirements.txt       # Dependencies (Mac/CPU)
├── requirements-cuda.txt  # Dependencies (GPU server)
│
├── analyze.ipynb          # Results analysis notebook
├── CODE_OVERVIEW.md       # Detailed implementation guide
│
├── experiments/           # Created during training (gitignored)
│   └── [run_name]/
│       ├── final_model.pt
│       ├── final_metrics.json
│       └── checkpoint_epoch_*.pt
│
├── logs/                  # Created by run_experiments.sh
│   └── [run_name].log
│
└── data/                  # CIFAR-10 (auto-downloaded, gitignored)
```

## Viewing Results

After experiments complete:

```bash
# Quick summary (command line)
for exp in experiments/*/final_metrics.json; do
  echo "$(dirname $exp):"
  python -c "import json; m=json.load(open('$exp')); print(f\"  PSNR: {m['psnr']:.2f} dB, Perplexity: {m['perplexity']:.1f}, Usage: {m['usage_rate']*100:.1f}%\")"
done

# Detailed analysis (Jupyter)
jupyter notebook analyze.ipynb
```

## Expected Results

**Baseline VQ-EMA (K=512):**
- Perplexity: ~250-350 (lower = more dead codes)
- Usage rate: ~60-70% (lower = more underutilization)
- PSNR: ~27-28 dB

**BA-VQ (K=512):**
- Perplexity: Higher than baseline (goal: +10-20%)
- Usage rate: Higher than baseline (goal: +10-20%)
- PSNR: Similar or better (maintain quality)

## Results

Coming soon after running experiments...

## References

- VQ-VAE: [van den Oord et al., 2017]
- VQ-GAN: [Esser et al., 2021]
- Blahut-Arimoto: [Blahut, 1972]
- VQGAN-LC: [Zhu et al., 2024] - shows utilization crisis at scale

## License

MIT
