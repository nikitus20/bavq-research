#!/usr/bin/env python3
"""
Simple CLI wrapper for training VQ-VAE models

Usage:
    # Train baseline VQ-EMA
    python train.py --quantizer vq_ema --codebook_size 512 --epochs 100 --name baseline_k512

    # Train BA-VQ
    python train.py --quantizer ba_vq --codebook_size 512 --epochs 100 --name ba_k512

    # Quick test (10 epochs)
    python train.py --quantizer vq_ema --codebook_size 256 --epochs 10 --name test
"""

import argparse
from vqvae import VQVAE, train


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE models')

    # Model config
    parser.add_argument('--quantizer', type=str, default='vq_ema',
                       choices=['vq_ema', 'ba_vq'],
                       help='Quantizer type: vq_ema (baseline) or ba_vq (ours)')
    parser.add_argument('--codebook_size', type=int, default=512,
                       help='Number of codes in codebook (default: 512)')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension (default: 256)')

    # Training config
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')

    # Experiment config
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name (required)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')

    args = parser.parse_args()

    # Print config
    print("=" * 60)
    print("VQ-VAE Training")
    print("=" * 60)
    print(f"Quantizer:      {args.quantizer}")
    print(f"Codebook size:  {args.codebook_size}")
    print(f"Latent dim:     {args.latent_dim}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Experiment:     {args.name}")
    print(f"W&B logging:    {not args.no_wandb}")
    print("=" * 60)
    print()

    # Create model
    model = VQVAE(
        quantizer_type=args.quantizer,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim
    )

    # Train
    train(
        model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.name,
        use_wandb=not args.no_wandb
    )


if __name__ == '__main__':
    main()
