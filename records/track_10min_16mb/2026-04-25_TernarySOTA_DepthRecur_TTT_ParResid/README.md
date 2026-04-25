# Record: TernarySOTA — Ternary + Depth Recurrence + Parallel Residuals + Score-First TTT

**Expected val_bpb: ~1.090–1.110** (estimate; requires H100 validation) | **≤16MB** | 8×H100 SXM

## The Core Bet

Two tracks in this competition were never merged:

| Track | BPB | Params | Compression | Tricks |
|-------|-----|--------|-------------|--------|
| **A** (SOTA) | **1.0810** | ~30M | GPTQ int6/int8 + Brotli | Depth recurrence, par. residuals, TTT, QK-gain 5.25 |
| **B** (abandoned) | 1.1570 | 73.7M | Ternary + LZMA | U-Net, relu², YaRN, NeoMuon |

Track B was abandoned at 1.1570 — not because the architecture is inferior, but because **nobody applied Track A's winning techniques to it**. With 2.4× more parameters per byte (ternary bit-packing is ~1.6 bits/param vs ~4 bits/param for int6+Brotli), Track B has a fundamentally better compression ratio. The open question is whether the parameter efficiency gap closes when you apply the same tricks.

This submission applies all Track A techniques to Track B's architecture.

## Key Techniques

### From Track B (base architecture)
1. **BitNet b1.58 ternary quantization** — weights {-1, 0, +1} with per-group (128) absmean scaling, STE gradients
2. **U-Net skip connections** — encoder/decoder with learned skip gates (sigmoid-gated), per-block input mix
3. **Factored tied embedding** — 8192×254 bottleneck + 254↔768 projections (saves ~4MB for wider MLP)
4. **relu² MLP, 4x width** — hidden=3072, fused projections
5. **NeoMuon (3 Newton-Schulz steps)** — row-normalized; compensates ternary STE gradient attenuation
6. **YaRN positional encoding** — max_len=2048, ROPE_BASE=5000
7. **FP8 QAT** for embedding scalars + projections; Base-3 + LZMA compression
8. **Temperature scaling** — grid search on training tokens (ternary models slightly underconfident)

### From Track A (new additions)
9. **Depth recurrence** — encoder layers 2-4 looped 3× (activated at 35% of training), virtual depth 16 from 10 physical
10. **Parallel residuals** — GPT-J style for decoder layers 7-9: attn + MLP both read from same pre-residual state
11. **QK-Gain 5.25** — learnable per-head query scaling (Track A finding: monotonic improvement 4.0→5.25)
12. **Legal Score-First TTT** — SGD on ternary model, lr=0.02 (higher than Track A's 0.005 to compensate ternary STE damping), 3 epochs/chunk, 32K-token chunks, cosine LR decay

### Not included / disabled
- **EMA**: Track B found EMA incompatible with ternary weights. The issue: EMA averages float weights, but behavior is determined by the ternary quantization. Averaged float weights can sit on wrong sides of quantization boundaries, producing worse ternary representations than training weights. Default: `EMA_ENABLED=0`.
- **Tversky projection**: disabled (TVERSKY_NUM_FEATURES=0)
- **Diff attention**: disabled
- **SMEAR gate**: disabled
- **MTP heads**: disabled

## Architecture

```
10 transformer layers, dim=768, 8H, 4KV, head_dim=96
Embedding: 8192×254 → FP8, 254→768 projection
MLP: 768→3072×2 (fused gate+up) → relu² → NormedTernaryLinear 3072→768
Attention: fused QKV TernaryLinear, NormedTernaryLinear output proj
QK normalization + RoPE (YaRN variant) + QK-Gain 5.25
U-Net: encoder [0,1,2,3,4] + decoder [5,6,7,8,9], 5 learned skip weights
Depth recurrence: loop encoder layers [2,3,4] × 3 → 16 virtual layers
Parallel residuals: layers 7, 8, 9 (decoder)
Poly5 softcap=10 + Z-loss (1e-4)
Tied embeddings, vocab_bias

Ternary: ~64.9M params at ~1.6 bits/param
FP8 scalars: ~2.5M params
Code: ~70KB
Total artifact: ~15.9MB
```

## Compatibility Analysis

### Depth recurrence + ternary STE ✅
Depth recurrence loops the same physical layers multiple times. Each pass calls `TernaryLinear.forward()` which applies the STE fresh. Gradients accumulate correctly through multiple passes. **Compatible.**

### Parallel residuals + ternary STE ✅
Changes the computation graph (both attn and MLP read from same pre-residual x) without affecting weight representation. **Compatible.**

### TTT + ternary STE ✅ (with higher LR)
TTT runs SGD on float weights. The STE applies during TTT forward passes. Small SGD updates may not immediately change the ternary representation (need to cross scale/2 threshold), but accumulated over 3 epochs with lr=0.02 they typically do. The key insight: the GROUP SCALE also adapts, so effective representation can shift even with sub-threshold weight updates. **Compatible, but use lr=0.02 vs Track A's 0.005.**

### GPTQ + ternary ❌ (not attempted)
GPTQ is a post-training quantization method for calibrating integer quantization. Ternary quantization is done during training via STE. They serve the same purpose via different methods — no reason to combine. **Track B's ternary STE already handles quantization natively.**

### EMA + ternary ⚠️ (disabled by default)
EMA averages float weights. But behavioral fidelity depends on the ternary quantization of those weights. Weights oscillating near quantization boundaries (common in converged ternary models) get averaged to the boundary, producing 0 in ternary space rather than ±1. Track B author confirmed EMA hurts. **Available as EMA_ENABLED=1 for experimentation.**

## Byte Budget

| Component | Params | Bytes (compressed) |
|-----------|--------|-------------------|
| Ternary weights (lin. layers >65K params) | ~64.9M | ~13.4MB |
| FP8 weights (embed, projections) | ~2.5M | ~2.5MB |
| Code (this file) | — | ~70KB |
| **Total** | **~67.4M** | **~15.97MB** |

**Ternary compression ratio**: log2(3) ≈ 1.585 bits/param → Base-3 packs 5 trits/byte → ~2.4M bytes per 12M params → LZMA reduces by ~39% → **~1.46MB per 10M ternary params**

**vs Track A (int6 GPTQ + Brotli)**: ~4 bits/param effective → **~0.5MB per 1M params**

**Track B wins on bits/param**: 1.6 bits vs 4 bits → 2.5× advantage. At 16MB total, this allows:
- Track B: 73.7M params
- Track A: ~30M params

The open question: do 73.7M ternary params represent as much information as 30M int6 params? Empirically Track B has ~0.077 bpb gap to close.

## Expected Performance

| Component | Expected BPB gain |
|-----------|------------------|
| Track B baseline | 1.1570 |
| Depth recurrence (3× loop layers 2-4) | −0.010 to −0.020 |
| Parallel residuals (layers 7-9) | −0.005 to −0.010 |
| QK-Gain tuned to 5.25 | −0.005 to −0.010 |
| TTT (score-first, 3 epochs) | −0.015 to −0.025 |
| **Projected range** | **~1.090–1.110** |

**Honest assessment**: This likely does NOT beat Track A's 1.0810. The ternary architecture is at a fundamental disadvantage because 1.6-bit quantization reduces model expressiveness. Track A's 30M int6 params represent information more densely than Track B's 74M ternary params. However, this submission explores an entirely different optimization direction that could matter for future architectures. If it reaches 1.09-1.10, that would be a significant result showing ternary models can close the gap with proper techniques.

## Reproduction

```bash
# Setup (same as Track B)
pip install brotli sentencepiece flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 \
  data/cached_challenge_fineweb.py --variant sp8192

# Run (3 seeds)
SEED=42 TTT_ENABLED=1 TTT_LR=0.02 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

SEED=314 TTT_ENABLED=1 TTT_LR=0.02 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

SEED=999 TTT_ENABLED=1 TTT_LR=0.02 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What to Watch For (H100 engineers)

1. **Step timing jump at ~35% training**: Depth recurrence activation will increase step time by ~1.5-2×. This is expected. Training time is wall-clock capped at 599s.

2. **TTT eval time budget**: TTT needs to complete within 600s eval budget. With 32K chunks and 3 epochs, estimate ~400-450s. If timing is tight, reduce `TTT_CHUNK_SIZE` to 16384 or `TTT_EPOCHS` to 2.

3. **Ternary zero fraction**: During training, log lines show `zero_frac`. Healthy range: 0.30-0.50. If it climbs above 0.60, the model is collapsing to sparse representations.

4. **TTT effectiveness check**: If `ttt_bpb` ≈ `sw_bpb` (no improvement), the ternary STE is absorbing TTT updates. Try `TTT_LR=0.05` for the next run.

5. **Budget check**: The script prints `total:X/16000000`. If over budget, reduce `embed_dim` from 254 to 252 (saves ~6MB raw, ~3.5MB compressed).

6. **EMA experiment**: If baseline results are disappointing, try `EMA_ENABLED=1 EMA_DECAY=0.9965` to see if the Track B author's finding still holds with the new architecture.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. See `eval_val_ttt()`.
- **Condition 4 (Single pass):** Each token scored exactly once.

Additional:
- No SLOT, no ETLB, no n-gram cache
- No TTT on validation data before scoring (score-first strictly enforced)
- All artifacts ≤ 16,000,000 bytes (expected ~15.97MB)
- Training ≤ 600s on 8×H100 SXM (wall-clock capped at 599s)
- Eval ≤ 600s on 8×H100 SXM (TTT estimated ~400-450s)

## Credits

- **Ciprian-Florin Ifrim** — Track B ternary architecture, U-Net, NeoMuon, FP8 QAT, Base-3 LZMA
- **@clarkkev** — Track A SOTA: SP8192, GPTQ, MuonEq-R, depth recurrence
- **@dexhunter** — 3-layer depth recurrence, legal TTT on SP8192
- **@abaybektursun** — Score-first TTT framework
- **@Robby955, @msisovic** — Parallel residuals
- **@X-Abhishek-X** — Hyperparameter tuning (QK-gain 5.25, WD, EMA decay)
- **LeeLoo / Andrew Aldere** — Synthesis and merged implementation
