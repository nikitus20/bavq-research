# VQ Codebook Utilization Research

Investigating codebook utilization in Vector Quantized Variational Autoencoders (VQ-VAE) through BA optimization.

## Problem

VQ-VAE and VQ-GAN STE training suffer from slow updates of unused codebooks, and this results in severe codebook underutilization - sometimes only 11-12% of codes are utilized at scale (K=16,384). This limits representational capacity and reconstruction quality.

## Approach

Apply rate-distortion theory via BA algorithm for principled codebook optimization with soft assignments and entropy regularization.

## Project Status

🚧 **Phase 1: Foundation** - Building baseline VQ-VAE and BA-VQ implementations

## Quick Start
```bash
# Setup
pip install -r requirements.txt
wandb login

# Train baseline VQ-VAE
python scripts/train.py --config configs/base.yaml

# Train BA-VQ
python scripts/train.py --config configs/ba_vq.yaml
```

## Results

Coming soon...

## References

- VQ-VAE: [van den Oord et al., 2017]
- VQ-GAN: [Esser et al., 2021]
- Blahut-Arimoto: [Blahut, 1972]
- VQGAN-LC: [Zhu et al., 2024] - shows utilization crisis at scale

## License

MIT
