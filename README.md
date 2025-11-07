# VQ Codebook Utilization Research

Investigating codebook utilization in Vector Quantized Variational Autoencoders (VQ-VAE) through BA optimization.

## Problem

VQ-VAE and VQ-GAN STE training suffer from slow updates of unused codebooks, and this results in severe codebook underutilization - sometimes only 11-12% of codes are utilized at scale (K=16,384). This limits representational capacity and reconstruction quality.

## Approach

Apply rate-distortion theory via BA algorithm for principled codebook optimization with soft assignments and entropy regularization.

## Project Status

✅ **Phase 1: Foundation Complete** - Minimal, working implementation ready for experiments

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Login to W&B (optional, use --no_wandb flag to skip)
wandb login
```

### 2. Run Quick Validation Experiments (~90 minutes total)

**Experiment 1: Baseline VQ-EMA (K=256, 30 epochs)**
```bash
python train.py --quantizer vq_ema --codebook_size 256 --epochs 30 --name baseline_test
```

**Experiment 2: BA-VQ (K=256, 30 epochs)**
```bash
python train.py --quantizer ba_vq --codebook_size 256 --epochs 30 --name ba_test
```

**Experiment 3: Baseline at K=512**
```bash
python train.py --quantizer vq_ema --codebook_size 512 --epochs 30 --name vq_k512
```

**Experiment 4: BA-VQ at K=512**
```bash
python train.py --quantizer ba_vq --codebook_size 512 --epochs 30 --name ba_k512
```

### 3. Analyze Results
```bash
# Open Jupyter notebook
jupyter notebook analyze.ipynb

# Or check W&B dashboard
# https://wandb.ai/your-username/vq-codebook
```

## Project Structure

```
bavq-research/
├── vqvae.py           # Complete VQ-VAE implementation (~640 lines)
│                      #   - Encoder/Decoder
│                      #   - VQ-EMA quantizer (baseline)
│                      #   - BA-VQ quantizer (our method)
│                      #   - Training loop
│                      #   - Metrics
│
├── train.py           # Simple CLI wrapper
├── config.yaml        # Default hyperparameters
├── analyze.ipynb      # Results analysis notebook
├── requirements.txt   # Dependencies
│
├── experiments/       # Auto-created during training (gitignored)
│   └── [run_name]/
│       ├── final_model.pt
│       ├── final_metrics.json
│       └── checkpoint_epoch*.pt
│
└── data/              # CIFAR-10 (auto-downloaded, gitignored)
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
