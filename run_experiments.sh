#!/bin/bash
# GPU Server Experiment Runner
# Runs all 4 validation experiments sequentially with proper logging

set -e  # Exit on error

echo "============================================================"
echo "VQ-VAE Codebook Utilization Experiments"
echo "============================================================"
echo "Device: $(python -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Create logs directory
mkdir -p logs

# Experiment 1: VQ-EMA K=256 (Baseline)
echo "[1/4] Running VQ-EMA K=256..."
python train.py \
    --quantizer vq_ema \
    --codebook_size 256 \
    --epochs 30 \
    --batch_size 256 \
    --name vq_ema_k256 \
    --no_wandb \
    2>&1 | tee logs/vq_ema_k256.log

echo ""
echo "✓ Experiment 1/4 complete"
echo ""

# Experiment 2: BA-VQ K=256 (Our method)
echo "[2/4] Running BA-VQ K=256..."
python train.py \
    --quantizer ba_vq \
    --codebook_size 256 \
    --epochs 30 \
    --batch_size 256 \
    --name ba_vq_k256 \
    --no_wandb \
    2>&1 | tee logs/ba_vq_k256.log

echo ""
echo "✓ Experiment 2/4 complete"
echo ""

# Experiment 3: VQ-EMA K=512 (Baseline)
echo "[3/4] Running VQ-EMA K=512..."
python train.py \
    --quantizer vq_ema \
    --codebook_size 512 \
    --epochs 30 \
    --batch_size 256 \
    --name vq_ema_k512 \
    --no_wandb \
    2>&1 | tee logs/vq_ema_k512.log

echo ""
echo "✓ Experiment 3/4 complete"
echo ""

# Experiment 4: BA-VQ K=512 (Our method)
echo "[4/4] Running BA-VQ K=512..."
python train.py \
    --quantizer ba_vq \
    --codebook_size 512 \
    --epochs 30 \
    --batch_size 256 \
    --name ba_vq_k512 \
    --no_wandb \
    2>&1 | tee logs/ba_vq_k512.log

echo ""
echo "============================================================"
echo "✓ All experiments complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - experiments/vq_ema_k256/"
echo "  - experiments/ba_vq_k256/"
echo "  - experiments/vq_ema_k512/"
echo "  - experiments/ba_vq_k512/"
echo ""
echo "Logs saved to: logs/"
echo ""
echo "To analyze results, run:"
echo "  jupyter notebook analyze.ipynb"
