"""
TernarySOTA: Ternary (BitNet b1.58) architecture + Depth Recurrence + Parallel Residuals
             + Score-First Legal TTT

Merges:
  - Track B (1.1570 bpb): 73.7M ternary params, U-Net skips, NeoMuon, relu², FP8 scalars
  - Track A (1.0810 bpb): 3-layer depth recurrence, parallel residuals, QK-Gain 5.25,
                          Legal Score-First TTT, EMA (optional), SP8192

Key design decisions (see README.md for full analysis):
  - Depth recurrence loops encoder layers 2-4 three times → 16 virtual layers from 10 physical
  - Parallel residuals for decoder layers 7-9 (GPT-J style)
  - TTT uses SGD on float weights with ternary STE applied in forward pass
  - EMA disabled by default (Track B author found it hurts ternary models due to
    quantization boundary interference; available as EMA_ENABLED=1 experiment)
  - LZMA + Base-3 ternary compression retained from Track B (beats Brotli+GPTQ on bits/param)

Author: LeeLoo / Andrew Aldere — April 25, 2026
Based on: train_gpt_cuda_ternary.py (Ciprian-Florin Ifrim, March 2026)
          train_gpt.py SOTA (clarkkev et al., April 2026)
"""

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func

# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via environment variables)
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    # Data
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")

    # Reproducibility
    seed = _e("SEED", 42, int)
    compile_mode = _e("COMPILE_MODE", "default")

    # Architecture
    vocab_size = _e("VOCAB_SIZE", 8192, int)
    num_layers = _e("NUM_LAYERS", 10, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 768, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    embed_dim = _e("EMBED_DIM", 254, int)        # factored embedding bottleneck
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 10.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    qk_gain_init = _e("QK_GAIN_INIT", 5.25, float)  # Track A finding: 5.25 optimal
    activation_type = _e("ACTIVATION", "relu2")
    rope_base = _e("ROPE_BASE", 5000.0, float)
    rope_type = _e("ROPE_TYPE", "yarn")
    yarn_max_len = _e("YARN_MAX_LEN", 2048, int)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 128, int)

    # Depth recurrence (Track A technique applied to Track B)
    depth_recurrence_iters = _e("DEPTH_RECURRENCE_ITERS", 3, int)   # loop count for looped layers
    depth_recurrence_start = _e("DEPTH_RECURRENCE_START", 2, int)   # which encoder layer to start looping
    depth_recurrence_start_frac = _e("DEPTH_RECURRENCE_START_FRAC", 0.35, float)  # when to activate

    # Parallel residuals (GPT-J style, Track A technique)
    parallel_residuals_from = _e("PARALLEL_RESIDUALS_FROM", 7, int)  # physical layer idx

    # EMA (Track B found incompatible with ternary; disabled by default but configurable)
    ema_enabled = _e("EMA_ENABLED", 0, bool)
    ema_decay = _e("EMA_DECAY", 0.9965, float)

    # TTT: Legal Score-First Test-Time Training (Track A technique)
    ttt_enabled = _e("TTT_ENABLED", 1, bool)
    ttt_lr = _e("TTT_LR", 0.02, float)          # higher than Track A's 0.005 due to ternary STE damping
    ttt_momentum = _e("TTT_MOMENTUM", 0.9, float)
    ttt_epochs = _e("TTT_EPOCHS", 3, int)
    ttt_chunk_size = _e("TTT_CHUNK_SIZE", 32768, int)  # 32K tokens per chunk

    # Training schedule
    iterations = _e("ITERATIONS", 10000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.2, float)
    warmup_steps = _e("WARMUP_STEPS", 5, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 599.0, float)

    # Batch/sequence scheduling
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.33, float)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.0, float)

    # Optimizer
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    scalar_lr = _e("SCALAR_LR", 0.02, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.02, float)
    adam_lr = _e("ADAM_LR", 0.05, float)
    adam_wd = _e("ADAM_WD", 0.05, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 3, int)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    muon_wd = _e("MUON_WD", 0.0, float)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.0, float)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    head_lr = _e("HEAD_LR", 0.02, float)

    # Evaluation
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 0, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 1000, int)
    val_max_tokens = _e("VAL_MAX_TOKENS", 0, int)
    sliding_eval = _e("SLIDING_EVAL", 1, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 16, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 256, int)
    temp_scaling = _e("TEMP_SCALING", 1, bool)

    # Storage
    _fp_raw = os.environ.get("FP_STORAGE", "FP8")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)


# CTP: scalar/low-dimensional params that use Adam (not Muon)
CTP = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes",
       "q_gain", "diff_lambda", "skip_weight", "skip_weights", "vocab_bias")

# ---------------------------------------------------------------------------
# Ternary packing — Base-3 encoding (5 trits/byte)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:, 0] + g[:, 1] * 3 + g[:, 2] * 9 + g[:, 3] * 27 + g[:, 4] * 81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:, i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

def pack_ternary_bitmask(q: Tensor):
    f = q.reshape(-1).to(torch.int8).numpy(); n = len(f)
    nz = (f != 0)
    return np.packbits(nz).tobytes() + np.packbits(f[nz] > 0).tobytes(), n

def unpack_ternary_bitmask(data: bytes, n: int) -> Tensor:
    ms = (n + 7) // 8
    nz = np.unpackbits(np.frombuffer(data[:ms], dtype=np.uint8))[:n].astype(bool)
    s = np.unpackbits(np.frombuffer(data[ms:], dtype=np.uint8))[:int(nz.sum())].astype(bool)
    w = np.zeros(n, dtype=np.int8); w[nz] = np.where(s, 1, -1)
    return torch.from_numpy(w)

# ---------------------------------------------------------------------------
# State dict serialization (ternary + fp16/fp8)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 128, fp_storage=False,
         ternary_method: str = "standard") -> tuple[dict, dict]:
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}
    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary_candidate = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name
            and "embed_proj" not in name and "lm_head_U" not in name
            and "lm_head_V" not in name
        )
        if is_ternary_candidate:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
            q = (t_grouped / scale).round().clamp(-1, 1).to(torch.int8)
            if ternary_method == "standard":
                packed_bytes, n_trits = pack_ternary(q)
                entry_type = "ternary"
            else:
                packed_bytes, n_trits = pack_ternary_bitmask(q)
                entry_type = "ternary_bitmask"
            quantized[name] = {
                "type": entry_type, "packed": packed_bytes,
                "scale": scale.half().squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits,
                "orig_shape": t_orig_shape,
            }
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2
    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    out = {}
    for name, entry in quantized.items():
        if entry["type"] in ("ternary", "ternary_bitmask"):
            if entry["type"] == "ternary":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
            q = q.float().reshape(-1, entry["group_size"])
            scale = entry["scale"].float().unsqueeze(-1)
            q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            t = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        else:
            out[name] = entry["data"].to(target_dtype).contiguous()
    return out

def tern_stats(model: nn.Module, group_size: int = 128):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "weight" in name and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

# ---------------------------------------------------------------------------
# Muon optimizer (NeoMuon: Newton-Schulz orthogonalized momentum)
# ---------------------------------------------------------------------------
def ns_orth(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = F.rms_norm(g.float(), (g.size(-1),)).bfloat16()
                    g = ns_orth(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---------------------------------------------------------------------------
# EMA (optional; disabled by default due to ternary quantization boundary issues)
# ---------------------------------------------------------------------------
class EMAHelper:
    """
    EMA over float weights. Note: Track B found EMA hurts ternary models because
    the EMA average can push float weights across ternary quantization thresholds
    in the wrong direction. Use with caution (EMA_ENABLED=1).
    """
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().float().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Swap in EMA weights (save training weights to backup)."""
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters()
                        if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore training weights from backup."""
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].pin_memory().to(
            self.device, non_blocking=True).to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_qat_ste(w: Tensor, fp_storage) -> Tensor:
    if not fp_storage: return w
    if fp_storage == "fp4":
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w / scale), -7.0, 7.0)
        return (q * scale - w).detach() + w
    w_sim = w.to(torch.float8_e4m3fn).to(w.dtype)
    return (w_sim - w).detach() + w

class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, fp_storage=False):
        super().__init__(in_features, out_features, bias=bias)
        self.fp_storage = fp_storage

    def forward(self, x: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.linear(x, w_qat.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class QATEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, fp_storage=False):
        super().__init__(num_embeddings, embedding_dim)
        self.fp_storage = fp_storage

    def forward(self, input: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.embedding(input, w_qat, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

class TernaryLinear(nn.Linear):
    """BitNet b1.58: weights quantized to {-1, 0, +1} via STE. Gradients flow to float weights."""
    def __init__(self, in_features, out_features, bias=False, group_size=128):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        # STE: ternary values in forward, full gradient in backward
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary, self.bias.to(x.dtype) if self.bias is not None else None)

class NormedTernaryLinear(TernaryLinear):
    """Ternary linear with RMSNorm on input — prevents activation explosion in output projections."""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, no_cache: bool = False,
                 rope_type: str = "rope", yarn_max_len: int = 4096, train_seq_len: int = 1024):
        super().__init__()
        self.no_cache = no_cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if rope_type == "yarn":
            scale = train_seq_len / yarn_max_len
            freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
            ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
            inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len, device, dtype):
        if self.no_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            return freqs.cos()[None, :, None, :].to(dtype=dtype), freqs.sin()[None, :, None, :].to(dtype=dtype)
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=128, no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size,
                                   bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, no_cache=no_cache,
                             rope_type=rope_type, yarn_max_len=yarn_max_len,
                             train_seq_len=train_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.c_qkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=128, activation="relu2"):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        if activation == "swiglu":
            self.gate_up = TernaryLinear(dim, hidden * 2, bias=False, group_size=group_size)
        else:
            self.fc = TernaryLinear(dim, hidden, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(hidden, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.proj(F.silu(gate) * up)
        elif self.activation == "relu2":
            return self.proj(torch.relu(self.fc(x)).square())
        else:
            return self.proj(torch.relu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, group_size: int = 128,
                 activation: str = "relu2", no_cache: bool = False,
                 rope_type: str = "rope", yarn_max_len: int = 4096,
                 train_seq_len: int = 1024, parallel_residual: bool = False):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        group_size, no_cache, rope_type, yarn_max_len, train_seq_len)
        self.mlp = MLP(dim, mlp_mult, group_size, activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        # U-Net residual mix: blend current hidden state with input embedding
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0

        if self.parallel_residual:
            # GPT-J style: attn and MLP both read from same pre-residual normalized input
            # Benefit: more gradient flow, better optimization; effective at decoder layers
            n = self.attn_norm(x)  # single normalization shared by both
            x = x + (self.attn_scale.to(dtype=x.dtype) * self.attn(n)
                     + self.mlp_scale.to(dtype=x.dtype) * self.mlp(n))
        else:
            # Standard sequential residual
            n = self.attn_norm(x)
            x = x + self.attn_scale.to(dtype=x.dtype) * self.attn(n)
            x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 group_size=128, activation="relu2", embed_dim=0, fp_storage=False,
                 softcap_type="poly", no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024,
                 depth_recurrence_iters=1, depth_recurrence_start=2,
                 parallel_residuals_from=7):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.depth_recurrence_iters = depth_recurrence_iters
        self.depth_recurrence_start = depth_recurrence_start
        self.use_depth_recurrence = False  # activated partway through training
        self.parallel_residuals_from = parallel_residuals_from

        # Factored embedding bottleneck (saves ~4MB vs full embedding)
        self.tok_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage)
        self.embed_proj = QATLinear(self.embed_dim, model_dim, bias=False,
                                    fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.embed_proj_rev = QATLinear(model_dim, self.embed_dim, bias=False,
                                        fp_storage=fp_storage) if self.embed_dim != model_dim else None

        # U-Net architecture: first half = encoder, second half = decoder
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Build blocks with parallel residuals for later decoder layers
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  group_size, activation, no_cache, rope_type, yarn_max_len, train_seq_len,
                  parallel_residual=(i >= parallel_residuals_from))
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head = QATLinear(model_dim, vocab_size, bias=False, fp_storage=fp_storage)
        self.lm_head._zero_init = True
        if tie_embeddings:
            self.lm_head.weight.requires_grad_(False)
        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, TernaryLinear) and not getattr(module, "_zero_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            proj = self.embed_proj_rev(x) if self.embed_proj_rev is not None else x
            logits_raw = F.linear(proj, self.tok_emb.weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        return logits_raw + self.vocab_bias.to(x.dtype)

    def _softcap(self, logits: Tensor) -> Tensor:
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        # Polynomial degree-5 approximation (cheaper, differentiable)
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                reduction: str = "mean", temperature: float = 1.0) -> Tensor:
        x = self.tok_emb(input_ids).float()
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x  # input embedding for U-Net residual mix

        # ---- Encoder with depth recurrence ----
        # Strategy: layers 0..recur_start-1 run once (store skips immediately)
        #           layers recur_start..num_encoder_layers-1 loop depth_recurrence_iters times
        #           skips are captured from the FINAL pass through each encoder layer
        recur_start = self.depth_recurrence_start
        iters = self.depth_recurrence_iters if self.use_depth_recurrence else 1

        skips = []

        # Phase 1: non-looped encoder layers (no recurrence)
        for i in range(min(recur_start, self.num_encoder_layers)):
            x = self.blocks[i](x, x0)
            skips.append(x.clone())

        # Phase 2: looped encoder layers
        # Run all [recur_start..num_encoder_layers-1] layers `iters` times
        # Only store skips on the LAST iteration (final states are most refined)
        for loop_iter in range(iters):
            is_last = (loop_iter == iters - 1)
            for i in range(recur_start, self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                if is_last:
                    skips.append(x.clone())

        # ---- Decoder with U-Net skip connections ----
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            # Skip connection from corresponding encoder layer (in reverse)
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips.pop()
            x = self.blocks[bi](x, x0)

        x_normed = self.final_norm(x)
        x_flat = x_normed.reshape(-1, x_normed.size(-1))
        targets = target_ids.reshape(-1)
        logits = self._softcap(self._compute_logits(x_flat))

        if reduction == "none":
            if temperature != 1.0:
                logits = logits / temperature
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(input_ids.shape)

        logits_f = logits.float()
        if temperature != 1.0:
            logits_f = logits_f / temperature
        # Fused CE + Z-loss (single logsumexp)
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        main_loss = (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()
        return main_loss

def restore_scalar_params_to_fp32(model: nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()

# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------
def build_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def ld_val(pattern, seq_len, max_tok=0):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val_sliding(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 16, temperature: float = 1.0):
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens - seq_len, stride))
    my_starts = all_starts[rank::world_size]
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]
            starts_t = torch.tensor(batch_starts, dtype=torch.int64)
            offsets = torch.arange(seq_len + 1, dtype=torch.int64)
            indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)
            local_batch = val_tokens[indices].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local_batch[:, :-1], local_batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_token_loss = model(x, y, reduction="none", temperature=temperature).detach()
            for b, start in enumerate(batch_starts):
                score_from = 0 if start == 0 else seq_len - stride
                scored = per_token_loss[b, score_from:]
                sx, sy = x[b, score_from:], y[b, score_from:]
                loss_sum += scored.to(torch.float64).sum()
                token_count += scored.numel()
                tok_bytes = base_bytes_lut[sy].to(torch.int16)
                tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tok_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    bpb = (loss_sum.item() / token_count.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(loss_sum.item() / token_count.item()), float(bpb)

def find_temp(args, model, rank, world_size, device, grad_accum_steps,
              calibration_tokens, base_bytes_lut, has_leading_space_lut,
              is_boundary_token_lut):
    """Grid search for optimal temperature scaling."""
    best_t, best_loss = 1.0, float("inf")
    for t in [0.85, 0.90, 0.95, 1.00, 1.05]:
        loss, _ = eval_val_sliding(args, model, rank, world_size, device, grad_accum_steps,
                                   calibration_tokens, base_bytes_lut, has_leading_space_lut,
                                   is_boundary_token_lut, stride=args.sliding_eval_stride,
                                   temperature=t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t

# ---------------------------------------------------------------------------
# Legal Score-First TTT (Test-Time Training)
# ---------------------------------------------------------------------------
def eval_val_ttt(args, base_model, rank, world_size, device, val_tokens,
                 base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                 ttt_lr: float, ttt_momentum: float, ttt_epochs: int,
                 ttt_chunk_size: int, stride: int, temperature: float = 1.0,
                 log_fn=None):
    """
    Legal Score-First Test-Time Training evaluation.

    Compliance (per Issue #1017):
      - Condition 1 (Causality): Sliding-window eval is strictly causal.
      - Condition 2 (Normalized distribution): Standard softmax over full vocab.
      - Condition 3 (Score before update): Each chunk scored under no_grad BEFORE
        any SGD update on that chunk's tokens. Training happens after scoring.
      - Condition 4 (Single pass): Each token scored exactly once.

    Note on ternary STE + TTT:
      The float weights are updated by SGD. The ternary STE is applied during
      each forward pass, so the model sees quantized weights. Small SGD steps
      (< group_scale/2) may not change the ternary representation, but the
      accumulated effect over 3 epochs typically does. Use a slightly higher LR
      than Track A's 0.005 (default: 0.02) to account for ternary quantization damping.
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    all_starts = list(range(0, total_tokens - seq_len, stride))

    # Partition all sliding windows into sequential chunks of ~ttt_chunk_size tokens
    chunks = []
    chunk_begin = 0
    while chunk_begin < total_tokens:
        chunk_end = min(chunk_begin + ttt_chunk_size, total_tokens)
        # All window starts that fall within this chunk
        window_starts = [s for s in all_starts if chunk_begin <= s < chunk_end]
        if window_starts:
            chunks.append((chunk_begin, chunk_end, window_starts))
        chunk_begin = chunk_end

    # Setup TTT optimizer: SGD on all matrix weights (ternary and FP8 parameters)
    # We EXCLUDE scalar/low-dim params to focus adaptation on the main weights
    ttt_params = [p for n, p in base_model.named_parameters()
                  if p.requires_grad and p.ndim == 2
                  and not any(pat in n for pat in CTP)]
    ttt_opt = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=ttt_momentum)

    total_update_steps = len(chunks) * ttt_epochs

    # Accumulators for final BPB
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    if log_fn:
        log_fn(f"ttt: {len(chunks)} chunks, {ttt_epochs} epochs, lr={ttt_lr}, "
               f"chunk_size={ttt_chunk_size}, total_update_steps={total_update_steps}")

    base_model.eval()

    for chunk_idx, (chunk_begin, chunk_end, window_starts) in enumerate(chunks):

        # ===== STEP 1: SCORE — strictly under no_grad, BEFORE any update =====
        # Compliance: model has NOT been trained on these tokens yet (only on prior chunks)
        my_starts = window_starts[rank::world_size]
        with torch.inference_mode():
            for i in range(0, len(my_starts), args.sliding_batch_size):
                batch_starts = my_starts[i:i + args.sliding_batch_size]
                starts_t = torch.tensor(batch_starts, dtype=torch.int64)
                offsets = torch.arange(seq_len + 1, dtype=torch.int64)
                indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)
                local_batch = val_tokens[indices].to(device=device, dtype=torch.int64)
                x, y = local_batch[:, :-1], local_batch[:, 1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    per_token_loss = base_model(x, y, reduction="none", temperature=temperature).detach()
                for b, start in enumerate(batch_starts):
                    score_from = 0 if start == 0 else seq_len - stride
                    scored = per_token_loss[b, score_from:]
                    sx, sy = x[b, score_from:], y[b, score_from:]
                    loss_sum += scored.to(torch.float64).sum()
                    token_count += scored.numel()
                    tok_bytes = base_bytes_lut[sy].to(torch.int16)
                    tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                    byte_count += tok_bytes.to(torch.float64).sum()

        # ===== STEP 2: TRAIN — update model on this chunk's tokens =====
        # This only affects FUTURE chunks' scoring (current chunk already scored above)
        chunk_tokens = val_tokens[chunk_begin:chunk_end + 1].to(device=device, dtype=torch.int64)
        chunk_len = chunk_tokens.numel() - 1
        n_seqs = chunk_len // seq_len

        if n_seqs == 0:
            continue

        base_model.train()
        cx = chunk_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)
        cy = chunk_tokens[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len)

        for epoch in range(ttt_epochs):
            # Cosine LR decay across all update steps
            global_step = chunk_idx * ttt_epochs + epoch
            cosine_frac = global_step / max(total_update_steps - 1, 1)
            cur_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * cosine_frac))
            for g in ttt_opt.param_groups:
                g["lr"] = cur_lr

            ttt_opt.zero_grad(set_to_none=True)

            # Process in micro-batches to avoid OOM
            mb_size = max(1, args.val_batch_size // (world_size * seq_len))
            total_loss = torch.zeros((), device=device)
            for mb_start in range(0, n_seqs, mb_size):
                mb_end = min(mb_start + mb_size, n_seqs)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(cx[mb_start:mb_end], cy[mb_start:mb_end])
                (loss / max(1, n_seqs // mb_size + 1)).backward()
                total_loss += loss.detach()

            # Gradient clip + distributed reduce
            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            if dist.is_available() and dist.is_initialized():
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
            ttt_opt.step()

        base_model.eval()

    # All-reduce accumulated metrics
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = loss_sum.item() / max(token_count.item(), 1)
    bpb = (val_loss / math.log(2.0)) * (token_count.item() / max(byte_count.item(), 1))
    base_model.train()
    return float(val_loss), float(bpb)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    if args.matrix_optimizer != "adamw":
        global ns_orth
        ns_orth = torch.compile(ns_orth)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None

    def log0(msg: str, console: bool = True) -> None:
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_max_tok = args.val_max_tokens if args.val_max_tokens > 0 else 0
    val_tokens = ld_val(args.val_files, args.train_seq_len, max_tok=val_max_tok)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(
        sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size,
        activation=args.activation_type,
        embed_dim=args.embed_dim,
        fp_storage=args.fp_storage,
        softcap_type=args.softcap_type,
        no_cache=(args.compile_mode == "reduce-overhead"),
        rope_type=args.rope_type,
        yarn_max_len=args.yarn_max_len,
        train_seq_len=args.train_seq_len,
        depth_recurrence_iters=args.depth_recurrence_iters,
        depth_recurrence_start=args.depth_recurrence_start,
        parallel_residuals_from=args.parallel_residuals_from,
    ).to(device).bfloat16()

    # Keep linear layers in float32 for stable ternary STE gradient flow
    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            module.float()
    restore_scalar_params_to_fp32(base_model)
    if base_model.tie_embeddings:
        base_model.lm_head.weight.requires_grad_(False)

    torch._dynamo.config.optimize_ddp = False
    compiled_model = torch.compile(
        base_model, mode=args.compile_mode if args.compile_mode != "default" else None)
    use_find_unused = (args.untie_at_fraction > 0) or (not args.tie_embeddings)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=use_find_unused,
                static_graph=not use_find_unused,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight"}
    all_other_params = [(n, p) for n, p in base_model.named_parameters()
                        if not any(eh in n for eh in _excl) and p.requires_grad]
    matrix_params = [p for n, p in all_other_params
                     if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other_params
                     if p.ndim < 2 or any(pat in n for pat in CTP)]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.adam_lr
    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    if args.matrix_optimizer == "adamw":
        opt_muon = torch.optim.AdamW(
            [{"params": matrix_params, "lr": args.adam_lr, "base_lr": args.adam_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps,
            weight_decay=args.adam_wd, fused=True)
    else:
        opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                        backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr

    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": 0.0, "base_lr": 0.0}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar, opt_head]

    # EMA (optional; typically hurts ternary models, see header comment)
    ema = EMAHelper(base_model, args.ema_decay) if args.ema_enabled else None
    if ema and master_process:
        log0(f"EMA enabled with decay={args.ema_decay} (experimental for ternary)")

    # --- Log config ---
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers} d:{args.model_dim} "
         f"h:{args.num_heads} kv:{args.num_kv_heads} "
         f"recur_iters:{args.depth_recurrence_iters} recur_start:{args.depth_recurrence_start} "
         f"par_resid_from:{args.parallel_residuals_from} "
         f"ws:{world_size} ga:{grad_accum_steps} s:{args.seed}")

    # --- Data loader ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.warmdown_fraction <= 0: return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0) \
                if step >= warmdown_start else 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    _seq_switched = False
    _batch_switched = False
    active_seq_len = args.seq_len_start if args.seq_len_start > 0 else args.train_seq_len
    active_batch_tokens = args.batch_tokens_start if args.batch_tokens_start > 0 else args.train_batch_tokens

    # --- Compiler warmup ---
    if args.warmup_steps > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = (mi == grad_accum_steps - 1)
                x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(_ms, strict=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    _untied = False
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (step == args.iterations or
                     (stop_after_step is not None and step >= stop_after_step))

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Quick chunked validation for monitoring
            local_batch = train_loader.stream.take(args.train_seq_len * 16)
            valx = local_batch[:-1].reshape(-1, args.train_seq_len).to(device)
            valy = local_batch[1:].reshape(-1, args.train_seq_len).to(device)
            base_model.eval()
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vl = base_model(valx, valy).item()
            base_model.train()
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} approx_val:{vl:.4f} "
                 f"train_time:{training_time_ms:.0f}ms "
                 f"zero_frac:{tstats['zero_frac']:.3f} "
                 f"recur_active:{base_model.use_depth_recurrence}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Activate depth recurrence partway through training (Track A finding: 35%)
        if not base_model.use_depth_recurrence and args.depth_recurrence_iters > 1:
            if max_wallclock_ms is not None:
                should_activate = elapsed_ms >= args.depth_recurrence_start_frac * max_wallclock_ms
            else:
                should_activate = step >= int(args.iterations * args.depth_recurrence_start_frac)
            if should_activate:
                base_model.use_depth_recurrence = True
                torch._dynamo.reset()
                log0(f"step:{step} depth_recurrence activated "
                     f"(iters={args.depth_recurrence_iters}, start={args.depth_recurrence_start})")

        # Sequence length schedule
        if args.seq_len_start > 0 and not _seq_switched:
            should_switch = (elapsed_ms >= args.seq_schedule_fraction * max_wallclock_ms
                             if max_wallclock_ms else step >= int(args.iterations * args.seq_schedule_fraction))
            if should_switch:
                active_seq_len = args.train_seq_len
                _seq_switched = True
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} seq_len_switch:{args.seq_len_start}->{active_seq_len}")

        # Batch size schedule
        if args.batch_tokens_start > 0 and not _batch_switched:
            should_switch = (elapsed_ms >= args.batch_schedule_fraction * max_wallclock_ms
                             if max_wallclock_ms else step >= int(args.iterations * args.batch_schedule_fraction))
            if should_switch:
                active_batch_tokens = args.train_batch_tokens
                _batch_switched = True
                log0(f"step:{step} batch_switch:{args.batch_tokens_start}->{active_batch_tokens}")

        zero_grad_all()
        train_loss.zero_()

        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
            x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        if args.matrix_optimizer != "adamw":
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) \
                if args.muon_momentum_warmup_steps > 0 else 1.0
            for g in opt_muon.param_groups:
                g["momentum"] = ((1 - frac) * args.muon_momentum_warmup_start
                                 + frac * args.muon_momentum)

        # LR scheduling + optimizer step
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
        zero_grad_all()

        # EMA update (optional)
        if ema is not None:
            ema.update(base_model)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and step % args.train_log_every == 0:
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} "
                 f"t:{approx_ms:.0f}ms avg:{approx_ms/step:.1f}ms/step")

        # Wallclock cap check
        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap:
                stop_after_step = step

    # --- Serialization ---
    if master_process:
        # Apply EMA if enabled
        if ema is not None:
            ema.apply(base_model)
            log0("Applied EMA weights for serialization")

        sd = base_model.state_dict()
        if base_model.tie_embeddings:
            sd.pop("lm_head.weight", None)

        # Compare Base-3 vs Bitmask ternary encoding, pick smaller
        methods = {}
        for method in ("standard", "bitmask"):
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size,
                                fp_storage=args.fp_storage, ternary_method=method)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            compressed = lzma.compress(buf.getvalue(), preset=9)
            methods[method] = {"blob": compressed, "stats": stats}
        best = min(methods, key=lambda m: len(methods[m]["blob"]))
        final_blob = methods[best]["blob"]
        q_stats = methods[best]["stats"]

        with open("final_model.ternary.ptz", "wb") as f:
            f.write(final_blob)

        artifact_bytes = len(final_blob)
        code_bytes = len(code.encode("utf-8"))
        total = artifact_bytes + code_bytes
        log0(f"serialization method:{best}")
        log0(f"artifact:{artifact_bytes/1e6:.3f}MB "
             f"ternary_params:{q_stats['ternary_params']}({q_stats['ternary_bytes']/1e6:.2f}MB) "
             f"fp_params:{q_stats['fp_params']}({q_stats['fp_bytes']/1e6:.2f}MB) "
             f"code:{code_bytes/1e3:.1f}KB")
        log0(f"total:{total}/{16_000_000} ({total/1e6:.3f}/{16.0:.2f}MB) "
             f"{'FITS' if total <= 16_000_000 else 'OVER BUDGET'}")

        if ema is not None:
            ema.restore(base_model)

    if distributed:
        dist.barrier()

    # --- Roundtrip load and evaluate ---
    with open("final_model.ternary.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())),
                            map_location="cpu", weights_only=False)
    base_model.load_state_dict(deq_sd(loaded), strict=False)
    torch._dynamo.reset()

    # Full depth recurrence for eval
    base_model.use_depth_recurrence = True
    log0(f"eval: depth_recurrence active (iters={args.depth_recurrence_iters})")

    # Temperature scaling (grid search on a training sample)
    opt_temp = 1.0
    if args.temp_scaling:
        torch.cuda.synchronize()
        t_temp = time.perf_counter()
        calib_tokens = train_loader.stream.take(65536).contiguous()
        opt_temp = find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                             calib_tokens, base_bytes_lut, has_leading_space_lut,
                             is_boundary_token_lut)
        torch.cuda.synchronize()
        temp_ms = 1000.0 * (time.perf_counter() - t_temp)
        log0(f"temp_scaling optimal_T:{opt_temp:.2f} time:{temp_ms:.0f}ms")

    # Sliding window evaluation
    if args.sliding_eval:
        torch.cuda.synchronize()
        t_sliding = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, base_model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.sliding_eval_stride, temperature=opt_temp)
        torch.cuda.synchronize()
        sliding_ms = 1000.0 * (time.perf_counter() - t_sliding)
        log0(f"sliding_eval val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"(stride={args.sliding_eval_stride} T={opt_temp:.2f}) time:{sliding_ms:.0f}ms")

    # Score-First Legal TTT (if enabled)
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ttt(
            args, base_model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            ttt_lr=args.ttt_lr, ttt_momentum=args.ttt_momentum,
            ttt_epochs=args.ttt_epochs, ttt_chunk_size=args.ttt_chunk_size,
            stride=args.sliding_eval_stride, temperature=opt_temp,
            log_fn=log0)
        torch.cuda.synchronize()
        ttt_ms = 1000.0 * (time.perf_counter() - t_ttt)
        log0(f"ttt_eval val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"(lr={args.ttt_lr} epochs={args.ttt_epochs} chunk={args.ttt_chunk_size}) "
             f"time:{ttt_ms:.0f}ms")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
