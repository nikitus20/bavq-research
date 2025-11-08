#!/usr/bin/env python3
"""
Evaluate VQ-VAE on CIFAR-10 with reconstruction FID and IS

Usage:
    python eval_cifar.py --checkpoint experiments/vq_ema_k512/final_model.pt
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from cleanfid import fid
from scipy import linalg

from vqvae import VQVAE


def compute_inception_score(images, cuda=True, batch_size=32, splits=10):
    """
    Computes the inception score of generated images
    images: Tensor of shape [N, 3, H, W] in range [0, 1]
    """
    from torchvision.models import inception_v3

    N = len(images)
    assert batch_size > 0
    assert N > batch_size

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    if cuda:
        inception_model = inception_model.cuda()

    # Upsample images to 299x299 (Inception input size)
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def get_pred(x):
        if cuda:
            x = x.cuda()
        x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        batch_size_i = batch.shape[0]
        preds[i:i+batch_size_i] = get_pred(batch)

    # Compute the mean KL divergence
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def entropy(p, q):
    return np.sum(p * np.log(p / q))


@torch.no_grad()
def reconstruct_dataset(model, loader, device):
    """Reconstruct entire dataset"""
    originals = []
    reconstructions = []

    model.eval()
    for x, _ in loader:
        x = x.to(device)

        # Reconstruct
        x_recon, _, _, _ = model(x)

        # Denormalize from [-1, 1] to [0, 1]
        x = (x * 0.5 + 0.5).clamp(0, 1)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)

        originals.append(x.cpu())
        reconstructions.append(x_recon.cpu())

    originals = torch.cat(originals, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0)

    return originals, reconstructions


def save_images_for_fid(images, save_dir):
    """Save images as PNG files for FID computation"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy uint8 [0, 255]
    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    for i, img in enumerate(images_np):
        from PIL import Image
        Image.fromarray(img).save(os.path.join(save_dir, f'{i:05d}.png'))


def main():
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE on CIFAR-10')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, mps, or cpu')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    model = VQVAE(
        quantizer_type=checkpoint['config']['quantizer_type'],
        codebook_size=checkpoint['config']['codebook_size'],
        latent_dim=checkpoint['config']['latent_dim']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load CIFAR-10 test set
    print("Loading CIFAR-10 test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda')
    )

    # Reconstruct dataset
    print("Reconstructing test set...")
    originals, reconstructions = reconstruct_dataset(model, test_loader, device)

    print(f"Reconstructed {len(originals)} images")

    # Prepare directories for FID
    exp_dir = os.path.dirname(args.checkpoint)
    orig_dir = os.path.join(exp_dir, 'eval_originals')
    recon_dir = os.path.join(exp_dir, 'eval_reconstructions')

    print("Saving images for FID computation...")
    save_images_for_fid(originals, orig_dir)
    save_images_for_fid(reconstructions, recon_dir)

    # Compute r-FID (reconstruction FID)
    print("Computing r-FID...")
    try:
        rfid = fid.compute_fid(orig_dir, recon_dir, mode='clean', device=device)
        print(f"r-FID: {rfid:.2f}")
    except Exception as e:
        print(f"Failed to compute r-FID: {e}")
        rfid = None

    # Compute r-IS (reconstruction Inception Score)
    print("Computing r-IS...")
    try:
        ris_mean, ris_std = compute_inception_score(
            reconstructions,
            cuda=(device.type == 'cuda'),
            batch_size=32,
            splits=10
        )
        print(f"r-IS: {ris_mean:.2f} ± {ris_std:.2f}")
    except Exception as e:
        print(f"Failed to compute r-IS: {e}")
        ris_mean, ris_std = None, None

    # Save results
    results = {
        'r_fid': rfid,
        'r_is_mean': ris_mean,
        'r_is_std': ris_std,
        'num_images': len(originals)
    }

    results_path = os.path.join(exp_dir, 'eval_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nEvaluation Summary:")
    print(f"  r-FID: {rfid:.2f}" if rfid is not None else "  r-FID: N/A")
    print(f"  r-IS:  {ris_mean:.2f} ± {ris_std:.2f}" if ris_mean is not None else "  r-IS: N/A")

    # Clean up temporary directories
    import shutil
    if os.path.exists(orig_dir):
        shutil.rmtree(orig_dir)
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
