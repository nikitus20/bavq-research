#!/usr/bin/env python3
"""
GPU Experiment Runner for BA-VQ Research
=========================================

Run a comprehensive suite of VQ-VAE experiments on GPU in ~1 hour.
Designed for A100 with CUDA.

Usage:
    python run_gpu_experiments.py

This script will:
1. Run 8 core experiments comparing VQ-EMA vs BA-VQ across codebook sizes
2. Save results to experiments/ directory
3. Generate a summary report

Expected runtime: ~50-60 minutes on A100
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


# Experiment configuration
# Target: ~6-7 min per experiment on A100 = 8 experiments in 1 hour
EXPERIMENTS = [
    # Core comparison: VQ-EMA vs BA-VQ at K=256
    {
        'name': 'vq_ema_k256',
        'quantizer': 'vq_ema',
        'codebook_size': 256,
        'epochs': 25,
        'batch_size': 256,
        'metric': 'euclid',
    },
    {
        'name': 'ba_vq_k256',
        'quantizer': 'ba_vq',
        'codebook_size': 256,
        'epochs': 25,
        'batch_size': 256,
        'metric': 'euclid',
    },

    # Core comparison: VQ-EMA vs BA-VQ at K=512
    {
        'name': 'vq_ema_k512',
        'quantizer': 'vq_ema',
        'codebook_size': 512,
        'epochs': 25,
        'batch_size': 256,
        'metric': 'euclid',
    },
    {
        'name': 'ba_vq_k512',
        'quantizer': 'ba_vq',
        'codebook_size': 512,
        'epochs': 25,
        'batch_size': 256,
        'metric': 'euclid',
    },

    # Larger codebook: K=1024
    {
        'name': 'vq_ema_k1024',
        'quantizer': 'vq_ema',
        'codebook_size': 1024,
        'epochs': 20,
        'batch_size': 256,
        'metric': 'euclid',
    },
    {
        'name': 'ba_vq_k1024',
        'quantizer': 'ba_vq',
        'codebook_size': 1024,
        'epochs': 20,
        'batch_size': 256,
        'metric': 'euclid',
    },

    # Metric comparison: Cosine distance at K=256
    {
        'name': 'ba_vq_k256_cosine',
        'quantizer': 'ba_vq',
        'codebook_size': 256,
        'epochs': 20,
        'batch_size': 256,
        'metric': 'cosine',
    },
    {
        'name': 'vq_ema_k256_cosine',
        'quantizer': 'vq_ema',
        'codebook_size': 256,
        'epochs': 20,
        'batch_size': 256,
        'metric': 'cosine',
    },
]


def run_experiment(exp_config):
    """Run a single experiment"""
    print("\n" + "="*70)
    print(f"Running: {exp_config['name']}")
    print(f"  Quantizer: {exp_config['quantizer']}")
    print(f"  Codebook:  K={exp_config['codebook_size']}")
    print(f"  Metric:    {exp_config['metric']}")
    print(f"  Epochs:    {exp_config['epochs']}")
    print("="*70 + "\n")

    # Build command
    cmd = [
        'python', 'train.py',
        '--quantizer', exp_config['quantizer'],
        '--codebook_size', str(exp_config['codebook_size']),
        '--epochs', str(exp_config['epochs']),
        '--batch_size', str(exp_config['batch_size']),
        '--metric', exp_config['metric'],
        '--name', exp_config['name'],
        # W&B enabled for tracking GPU experiments
    ]

    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    # Check if successful
    success = result.returncode == 0

    return {
        'name': exp_config['name'],
        'success': success,
        'elapsed_time': elapsed,
        'config': exp_config,
    }


def load_experiment_results(exp_name):
    """Load final metrics from an experiment"""
    metrics_file = Path(f'experiments/{exp_name}/final_metrics.json')
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def generate_summary_report(results):
    """Generate a summary report of all experiments"""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70 + "\n")

    # Print results table
    print(f"{'Experiment':<25} {'Status':<10} {'Time':<10} {'Perplexity':<12} {'Usage':<10} {'PSNR':<8}")
    print("-" * 85)

    for result in results:
        metrics = load_experiment_results(result['name'])

        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        elapsed = f"{result['elapsed_time']/60:.1f}m"

        if metrics:
            ppl = f"{metrics.get('perplexity', 0):.1f}"
            usage = f"{metrics.get('usage_rate', 0)*100:.1f}%"
            psnr = f"{metrics.get('psnr', 0):.2f}"
        else:
            ppl = usage = psnr = "N/A"

        print(f"{result['name']:<25} {status:<10} {elapsed:<10} {ppl:<12} {usage:<10} {psnr:<8}")

    print("\n" + "="*70)

    # Save summary to file
    summary_file = Path('experiments/gpu_run_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_time': sum(r['elapsed_time'] for r in results),
            'results': results,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Key comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70 + "\n")

    # VQ-EMA vs BA-VQ at K=256
    vq_ema_256 = load_experiment_results('vq_ema_k256')
    ba_vq_256 = load_experiment_results('ba_vq_k256')

    if vq_ema_256 and ba_vq_256:
        print("📊 K=256 Comparison:")
        print(f"  VQ-EMA: Perplexity={vq_ema_256['perplexity']:.1f}, Usage={vq_ema_256['usage_rate']*100:.1f}%, PSNR={vq_ema_256['psnr']:.2f}")
        print(f"  BA-VQ:  Perplexity={ba_vq_256['perplexity']:.1f}, Usage={ba_vq_256['usage_rate']*100:.1f}%, PSNR={ba_vq_256['psnr']:.2f}")

        ppl_improvement = ((ba_vq_256['perplexity'] - vq_ema_256['perplexity']) / vq_ema_256['perplexity']) * 100
        usage_improvement = ((ba_vq_256['usage_rate'] - vq_ema_256['usage_rate']) / vq_ema_256['usage_rate']) * 100

        print(f"  → Perplexity improvement: {ppl_improvement:+.1f}%")
        print(f"  → Usage rate improvement: {usage_improvement:+.1f}%")

    # VQ-EMA vs BA-VQ at K=512
    vq_ema_512 = load_experiment_results('vq_ema_k512')
    ba_vq_512 = load_experiment_results('ba_vq_k512')

    if vq_ema_512 and ba_vq_512:
        print("\n📊 K=512 Comparison:")
        print(f"  VQ-EMA: Perplexity={vq_ema_512['perplexity']:.1f}, Usage={vq_ema_512['usage_rate']*100:.1f}%, PSNR={vq_ema_512['psnr']:.2f}")
        print(f"  BA-VQ:  Perplexity={ba_vq_512['perplexity']:.1f}, Usage={ba_vq_512['usage_rate']*100:.1f}%, PSNR={ba_vq_512['psnr']:.2f}")

        ppl_improvement = ((ba_vq_512['perplexity'] - vq_ema_512['perplexity']) / vq_ema_512['perplexity']) * 100
        usage_improvement = ((ba_vq_512['usage_rate'] - vq_ema_512['usage_rate']) / vq_ema_512['usage_rate']) * 100

        print(f"  → Perplexity improvement: {ppl_improvement:+.1f}%")
        print(f"  → Usage rate improvement: {usage_improvement:+.1f}%")

    print("\n" + "="*70)


def main():
    """Run all experiments"""
    print("\n" + "="*70)
    print("BA-VQ GPU EXPERIMENT SUITE")
    print("="*70)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Expected runtime: ~50-60 minutes on A100")
    print("="*70 + "\n")

    # Create experiments directory
    Path('experiments').mkdir(exist_ok=True)

    # Run all experiments
    results = []
    total_start = time.time()

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}]", end=" ")
        result = run_experiment(exp_config)
        results.append(result)

        # Print progress
        elapsed_total = time.time() - total_start
        avg_time = elapsed_total / i
        remaining = (len(EXPERIMENTS) - i) * avg_time

        print(f"\nProgress: {i}/{len(EXPERIMENTS)} completed")
        print(f"Elapsed: {elapsed_total/60:.1f}m | Remaining: ~{remaining/60:.1f}m")

    # Generate summary
    total_elapsed = time.time() - total_start
    print(f"\n\nAll experiments completed in {total_elapsed/60:.1f} minutes")

    generate_summary_report(results)

    print("\n✓ GPU experiment suite complete!")
    print("  Results saved to: experiments/")
    print("  Summary saved to: experiments/gpu_run_summary.json")


if __name__ == '__main__':
    main()
