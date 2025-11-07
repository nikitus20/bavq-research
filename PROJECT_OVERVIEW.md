# VQ Codebook Utilization: Project Overview

## Research Goal

Improve codebook utilization in VQ-VAE using Blahut-Arimoto algorithm

## Key Metrics

**Utilization:**
- Perplexity: exp(H(C)) - higher is better
- Usage rate: % of codes used ≥1 time per epoch
- Dead codes: count of unused codes

**Quality:**
- PSNR: reconstruction quality
- LPIPS: perceptual quality
- FID: generation quality

## Success Criteria

- ✅ +20-30% perplexity improvement over VQ-EMA baseline
- ✅ Utilization >70% at K=1024 (baseline: ~50%)
- ✅ Quality maintained (PSNR drop ≤0.3 dB)
- ✅ Stable across seeds

## Experimental Plan

### Phase 1: Foundation (Week 1-2)
- Baseline VQ-EMA implementation
- BA-VQ quantizer
- CIFAR-10 experiments: K ∈ {256, 512, 1024, 2048}
- **Target:** Establish if BA-VQ helps

### Phase 2: Scaling (Week 3) *[if Phase 1 promising]*
- CelebA-HQ 128×128
- Larger codebooks: K ∈ {4096, 8192}
- Test scaling behavior

### Phase 3: Ablations (Week 4) *[if Phase 2 successful]*
- β schedule variations
- Learned Q(c) vs uniform
- Distance metrics
- Rate-distortion curves

## Timeline

- Week 1: Infrastructure + baseline (Sessions 1-3)
- Week 2: BA-VQ implementation + comparison (Sessions 4-5)
- Week 3+: Iterate based on results

## References

See `docs/LITERATURE.md` for detailed background.
