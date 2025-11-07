# Claude Code Development Guide

## Project Context

Building VQ-VAE research codebase to study codebook utilization via Blahut-Arimoto optimization.

**Reference implementation:** https://github.com/nikitus20/deep-vector-quantization (Karpathy's clean VQ-VAE)

## Code Style

**Keep it simple:**
- Minimal abstractions
- Clear variable names
- Docstrings only where needed (not every function)
- No over-engineering
- Single file per component where possible

**Structure:**
```
src/
  data.py          # CIFAR-10 loading
  model.py         # Encoder/Decoder
  quantizers.py    # All quantizers (VQ-EMA, BA-VQ, etc.)
  train.py         # Training loop
  metrics.py       # Utilization metrics
  utils.py         # Config, logging, viz
```

## Development Sessions

### Session 1: Foundation
**Goal:** Working training pipeline with identity quantizer

**Tasks:**
1. Project structure + requirements.txt
2. Data loading (adapt from Karpathy)
3. Encoder/Decoder (adapt from Karpathy)
4. Training loop with W&B logging
5. Checkpointing

**Success:** One full training run completes, logs to W&B

---

### Session 2: VQ-EMA Baseline
**Goal:** Working VQ-EMA quantizer with metrics

**Tasks:**
1. Implement VQEMAQuantizer (follow Karpathy closely)
2. Utilization metrics (perplexity, usage rate)
3. Integrate into training loop
4. Test run (10 epochs)

**Success:** Perplexity ~250-350 for K=512, PSNR ~27-28 dB

---

### Session 3: Config System & Quick Tests
**Goal:** Run quick validation experiments

**Tasks:**
1. YAML config system
2. Sweep script for multiple runs
3. Quick tests: K ∈ {256, 512}, 2 seeds, 30 epochs
4. Analysis script with plots

**Success:** 4 runs complete in ~1 hour, clear metrics

---

### Session 4: BA-VQ Implementation
**Goal:** Blahut-Arimoto quantizer working

**Tasks:**
1. Implement BAQuantizer with soft assignments
2. β annealing schedules
3. Numerical stability (log-sum-exp)
4. Unit tests

**Success:** BA-VQ trains without NaN, perplexity > baseline

---

### Session 5: BA-VQ Comparison
**Goal:** Full comparison experiments

**Tasks:**
1. Refine BA-VQ based on Session 4
2. Run comparison sweep
3. Generate analysis plots
4. Statistical tests

**Success:** Clear evidence BA-VQ helps (or doesn't)

## Key Implementation Details

### VQ-EMA Quantizer
```python
# Follow Karpathy's implementation:
# - EMA updates for codebook
# - Commitment loss: 0.25
# - Straight-through estimator
# - Laplace smoothing for stability
```

### BA-VQ Quantizer
```python
# Core BA iterations:
for _ in range(ba_iterations):
    logits = log(Q) - beta * distances
    P = softmax(logits, dim=1)
    Q = P.mean(dim=0)

# Use log-sum-exp for numerical stability
# Clamp Q to [1e-10, 1.0]
# β annealing: 0.1 → 5.0 over epochs
```

### Metrics
```python
# Perplexity: exp(entropy)
# Usage rate: % codes used ≥1 time
# Dead codes: codes never used
# All computed from code_indices tensor
```

## Testing Strategy

**Unit tests for:**
- Soft assignments sum to 1
- β→∞ recovers hard VQ
- No NaN in gradients

**Integration tests:**
- Full training run (10 epochs)
- Metrics make sense
- Checkpointing works

## When Things Go Wrong

**NaN in training:**
- Check gradient clipping
- Verify log-sum-exp in softmax
- Inspect β values

**Low utilization:**
- Expected for VQ-EMA (this is the problem!)
- BA-VQ should improve

**Poor PSNR:**
- Check reconstruction loss weight
- Verify encoder/decoder architecture
- Compare to Karpathy's numbers

## Quick Commands
```bash
# Quick test (10 epochs)
python scripts/train.py --config configs/base.yaml --epochs 10 --name test

# Full training
python scripts/train.py --config configs/base.yaml

# Resume from checkpoint
python scripts/train.py --resume experiments/run_name/checkpoints/latest.pt

# Analyze results
python scripts/analyze.py --experiment run_name
```

## External Resources

- Karpathy VQ-VAE: https://github.com/nikitus20/deep-vector-quantization
- VQ-VAE paper: https://arxiv.org/abs/1711.00937
- BA algorithm: https://en.wikipedia.org/wiki/Blahut–Arimoto_algorithm
