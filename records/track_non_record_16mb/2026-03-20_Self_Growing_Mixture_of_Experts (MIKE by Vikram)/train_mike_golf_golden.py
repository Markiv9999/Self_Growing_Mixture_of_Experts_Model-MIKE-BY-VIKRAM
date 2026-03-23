"""
train_mike_golf.py
------------------
train_gpt.py infrastructure + trainmike.py MoE architecture.

Key changes vs trainmike.py:
  - Data/tokenizer: FineWeb binary shards + SentencePiece (vocab 1024)
  - Model signature: forward(input_ids, target_ids, lora=None) to match train_gpt.py harness
  - Causal attention: replaced bidirectional TransformerEncoderLayer with CausalSelfAttention
    (RoPE, GQA, flash-attn) — Mike's backbone was non-causal (bidirectional encoder)
  - Dynamic expert growth runs during training; model is frozen before int8 export
  - Optimizer: Muon for 2D matrix params, Adam for embeddings/scalars
  - TTT-LoRA eval harness preserved exactly from train_gpt.py
  - int8+zlib serialization preserved exactly from train_gpt.py
  - fullgraph=False for torch.compile (model grows dynamically during training)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import faiss
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# =====================================================================
# HYPERPARAMETERS
# =====================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    max_val_batches = int(os.environ.get("MAX_VAL_BATCHES", 0))  # 0 = full val (for submission); set ~100 for local
    spawn_delay_steps = int(os.environ.get("SPAWN_DELAY_STEPS", 500))  # backbone-only steps before expert spawning
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 1))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Backbone shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 6))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))  # match num_heads; GQA requires flash attn (unavailable on Windows)
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # MoE shape
    max_experts = int(os.environ.get("MAX_EXPERTS", 8))        # small for 16MB budget
    top_k = int(os.environ.get("TOP_K", 2))
    res_scale = float(os.environ.get("RES_SCALE", 0.8))
    num_hops = int(os.environ.get("NUM_HOPS", 5))               # was 3, original used 5
    moe_dim = int(os.environ.get("MOE_DIM", 256))              # expert internal width
    moe_heads = int(os.environ.get("MOE_HEADS", 4))
    expert_depth = int(os.environ.get("EXPERT_DEPTH", 1))      # stacked MLP blocks per expert

    # Expert growth (training only)
    grow_interval = int(os.environ.get("GROW_INTERVAL", 500))   # was 2000
    expert_min_uses = int(os.environ.get("EXPERT_MIN_USES", 300))  # was 800
    expert_warmup = int(os.environ.get("EXPERT_WARMUP", 100))   # was 300
    grad_eps = float(os.environ.get("GRAD_EPS", 0.1))           # was 0.05

    # Attractor
    base_radius = float(os.environ.get("BASE_RADIUS", 0.9))
    min_hits = int(os.environ.get("MIN_HITS", 15))              # was 25
    max_drift = float(os.environ.get("MAX_DRIFT", 0.1))         # was 0.07
    min_sep = float(os.environ.get("MIN_SEP", 0.3))             # was 0.4
    attractor_history = int(os.environ.get("ATTRACTOR_HISTORY", 2000))
    attractor_interval = int(os.environ.get("ATTRACTOR_INTERVAL", 8))   # every 8 micro-steps (~1 per train step)
    attractor_perc = float(os.environ.get("ATTRACTOR_PERC", 0.2))
    max_protos = int(os.environ.get("MAX_PROTOS", 200))  # unused with FAISS but kept for compat

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # TTT-LoRA
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))


# =====================================================================
# MUON OPTIMIZER (unchanged from train_gpt.py)
# =====================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
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
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# =====================================================================
# BPB / VALIDATION (unchanged from train_gpt.py except for batched eval additions added to allow for fast eval on 3060 laptop gpu)
# =====================================================================

def build_sentencepiece_luts(sp, vocab_size: int, device):
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
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# =====================================================================
# INT8+ZLIB QUANTIZATION (unchanged from train_gpt.py)
# =====================================================================

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(
        ("param_count","num_tensors","num_float_tensors","num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# =====================================================================
# DATA LOADING (unchanged from train_gpt.py) short read case was changed by ai, now fixed 
# =====================================================================

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# =====================================================================
# TRANSFORMER PRIMITIVES (from train_gpt.py)
# =====================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    # Re added self_sin_cached check here, ai removed it... sed
    def forward(self, seq_len: int, device, dtype):
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Causal flash-attention with RoPE + GQA from train_gpt.py.
    Replaces Mike's bidirectional TransformerEncoderLayer.
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        # AI Switched if chekcs to assert, changed it back
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        
        
        #  AI Added this:  expand KV heads to match Q heads if using GQA without flash attn for windows compatibility. now we dont need it: comment
        #if self.num_kv_heads != self.num_heads:
        #    repeat = self.num_heads // self.num_kv_heads
        #    k = k.repeat_interleave(repeat, dim=1)
        #    v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# =====================================================================
# MIKE'S MOE COMPONENTS (adapted)
# =====================================================================

class Proto:
    """EMA centroid tracker for attractor detection (from trainmike.py)."""
    def __init__(self, x: Tensor):
        self.center = x.detach().cpu().float().clone()
        self.prev = self.center.clone()
        self.hits = 1
        self.ema = 0.98

    def update(self, x: Tensor):
        xc = x.detach().cpu().float()
        self.prev = self.center.clone()
        self.center = self.ema * self.center + (1 - self.ema) * xc
        self.hits += 1

    def drift(self) -> float:
        return float(torch.norm(self.center - self.prev).item())


class ANNAttractor:
    """
    Semantic cluster detector using FAISS IndexHNSWFlat.
    True O(log N) search with incremental adds — no rebuild needed on every add.
    Rebuild only happens on promotion (rare).
    """
    def __init__(self, args: Hyperparameters, dim: int):
        self.args = args
        self.dim = dim
        self.protos: list[Proto] = []
        self.clusters: list[dict] = []
        self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbours per node
        self.index.hnsw.efSearch = 16              # search beam width
        self.hist: list[float] = []

    def _new_index(self) -> faiss.IndexHNSWFlat:
        idx = faiss.IndexHNSWFlat(self.dim, 32)
        idx.hnsw.efSearch = 16
        return idx

    def _thresh(self) -> float:
        # Always use base_radius as a floor — prevents one mega-proto absorbing everything.
        # Percentile mode only tightens the threshold, never loosens it below base_radius^2.
        floor = self.args.base_radius ** 2
        if len(self.hist) < 200:
            return floor
        return max(floor, float(np.percentile(self.hist, self.args.attractor_perc * 100)))

    def add(self, x: Tensor):
        x_np = F.normalize(x.detach().cpu().float(), dim=-1).numpy().astype("float32")[None]

        if not self.protos:
            self.protos.append(Proto(torch.from_numpy(x_np[0])))
            self.index.add(x_np)  # incremental add — no rebuild
            return

        D, I = self.index.search(x_np, 1)
        d, i = float(D[0][0]), int(I[0][0])

        self.hist.append(d)
        if len(self.hist) > self.args.attractor_history:
            self.hist.pop(0)

        if d < self._thresh():
            # Update proto EMA — HNSW index keeps old position but that's fine,
            # centroid only drifts slightly and search still finds correct bucket
            self.protos[i].update(torch.from_numpy(x_np[0]))
        else:
            # New proto — incremental add, no rebuild needed
            self.protos.append(Proto(torch.from_numpy(x_np[0])))
            self.index.add(x_np)

    def try_promote(self, device) -> Tensor | None:
        promoted = []
        for i, p in enumerate(self.protos):
            if p.hits < self.args.min_hits or p.drift() > self.args.max_drift:
                continue
            too_close = any(
                float(torch.norm(p.center - c["center"].cpu()).item()) < self.args.min_sep
                for c in self.clusters
            )
            if too_close:
                continue
            self.clusters.append({"center": p.center.clone().to(device), "count": p.hits})
            promoted.append(i)
            print(f"\n🌱 Attractor formed | hits={p.hits}\n")
        if not promoted:
            return None
        for i in reversed(promoted):
            del self.protos[i]
        # Rebuild index from remaining protos
        self.index = self._new_index()
        if self.protos:
            remaining = np.vstack([p.center.numpy().astype("float32") for p in self.protos])
            self.index.add(remaining)
        # Return ALL new centroids so all promoted experts can spawn
        return [self.clusters[i]["center"] for i in range(len(self.clusters) - len(promoted), len(self.clusters))]

    def stats(self):
        return len(self.protos), len(self.clusters)


class LightweightExpertBlock(nn.Module):
    """Lightweight expert block: RMSNorm → MLP (relu² activation), no attention.
    The routing already provides semantic context; the expert just needs to
    apply domain-specific computation."""
    def __init__(self, dim: int, mlp_mult: int = 2):
        super().__init__()
        self.norm = RMSNorm()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        n = self.norm(x)
        h = torch.relu(self.fc(n))
        return x + self.scale.to(dtype=x.dtype)[None, None, :] * self.proj(h.square())


class CausalExpertLayer(nn.Module):
    """
    Single expert: proj_in → LightweightExpertBlock(s) at moe_dim → proj_out.
    No attention inside experts — routing provides semantic context,
    experts apply specialized computation via MLP blocks.
    """
    def __init__(self, model_dim: int, moe_dim: int, moe_heads: int, rope_base: float, qk_gain_init: float, mlp_mult: int = 2, num_expert_blocks: int = 1):
        super().__init__()
        self.proj_in = CastedLinear(model_dim, moe_dim, bias=False)
        self.blocks = nn.ModuleList([LightweightExpertBlock(moe_dim, mlp_mult) for _ in range(num_expert_blocks)])
        self.extra_blocks = nn.ModuleList()  # can still grow dynamically
        self.proj_out = CastedLinear(moe_dim, model_dim, bias=False)
        self.proj_out._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h)
        for blk in self.extra_blocks:
            h = blk(h)
        return self.proj_out(h)


class MoERouter(nn.Module):
    """
    Per-token router. Each token independently selects its top-k experts
    based on cosine similarity between a learned projection and expert centroids.
    This allows different parts of a sequence to route to different experts.
    """
    def __init__(self, model_dim: int, seq_len: int = 1024, chunk_size: int = 0):
        super().__init__()
        self.router_proj = CastedLinear(model_dim, model_dim, bias=False)
        self.register_buffer("centroids", torch.zeros(0, model_dim))

    def add_centroid(self, c: Tensor):
        c_norm = F.normalize(c.detach().reshape(1, -1).to(self.centroids.device), dim=-1)
        self.centroids = torch.cat([self.centroids, c_norm], dim=0)

    def forward(self, h: Tensor, top_k: int):
        """Per-token routing. Returns (weights, ids) both (bsz, seq, k), or (None, None)."""
        n = self.centroids.size(0)
        if n == 0:
            return None, None
        bsz, seq, dim = h.shape
        r = F.normalize(self.router_proj(h), dim=-1)       # (bsz, seq, dim)
        sim = r @ self.centroids.T                           # (bsz, seq, n)
        k = min(top_k, n)
        weights, ids = torch.topk(sim, k, dim=-1)           # (bsz, seq, k)
        return F.softmax(weights, dim=-1), ids


# =====================================================================
# MIKE'S MOE MODEL — drop-in replacement for train_gpt.py's GPT
# =====================================================================

class MikeMoE(nn.Module):
    """
    Growing Mixture-of-Experts language model.

    Forward pass:
      1. Token embedding → RMS-norm
      2. Backbone: num_layers causal Block stack with U-Net skips (from train_gpt.py)
      3. MoE refinement: num_hops passes of top-k expert routing (residual blend)
      4. Final RMS-norm → tied LM head → logit softcap → cross-entropy loss

    Training-only:
      - ANNAttractor detects stable semantic clusters in routing embeddings
      - Clusters trigger expert spawning (up to max_experts)
      - Expert depth grows when gradient signal is weak but usage is high
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        V, D = args.vocab_size, args.model_dim
        self.logit_softcap = args.logit_softcap
        self.tie_embeddings = args.tie_embeddings

        # Embedding
        self.tok_emb = nn.Embedding(V, D)
        if args.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, 0.0, args.tied_embed_init_std)

        # Backbone with U-Net skips
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, D, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(D, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init)
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(D, V, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # MoE — experts allocated dynamically at spawn time
        self.router = MoERouter(D, seq_len=args.train_seq_len)
        self.experts = nn.ModuleList()  # starts empty, experts added at spawn
        self._n_active: int = 0

        # Training-time state
        self.attractor = ANNAttractor(args, dim=args.model_dim)
        self._expert_age: list[int] = [0] * args.max_experts
        self._expert_uses: list[int] = [0] * args.max_experts
        self._expert_loss_ema: list[float] = [10.0] * args.max_experts
        self._expert_grad_ema: list[float] = [1.0] * args.max_experts
        self._expert_last_grow: list[int] = [0] * args.max_experts
        self._train_step: int = 0
        self._pending_centroid = None
        self._pending_signal = None

    # ------------------------------------------------------------------
    # Expert lifecycle
    # ------------------------------------------------------------------

    def spawn(self, centroid: Tensor, device):
        if self._n_active >= self.args.max_experts:
            return
        ei = self._n_active
        # Create new expert on demand
        e = CausalExpertLayer(
            self.args.model_dim, self.args.moe_dim, self.args.moe_heads,
            self.args.rope_base, self.args.qk_gain_init, self.args.mlp_mult,
            num_expert_blocks=self.args.expert_depth,
        ).to(device)
        # Match dtype of existing model
        for module in e.modules():
            if isinstance(module, CastedLinear):
                module.float()
        # Inherit from a random active parent for warm start
        if ei > 0:
            parent = self.experts[random.randrange(ei)]
            with torch.no_grad():
                for p, pp in zip(e.parameters(), parent.parameters()):
                    p.data.copy_(pp.data)
                    p.data += 0.02 * torch.randn_like(p)
        self.experts.append(e)
        self.router.add_centroid(centroid)
        self._expert_age[ei] = 0
        self._expert_uses[ei] = 0
        self._expert_loss_ema[ei] = 10.0
        self._expert_grad_ema[ei] = 1.0
        self._expert_last_grow[ei] = 0
        self._n_active += 1
        print(f"🌱 Spawned expert {ei} ({self._n_active}/{self.args.max_experts} active)")

    def try_grow_experts(self, step: int):
        """Deepen any expert that is heavily used but has weak gradient signal."""
        for i, e in enumerate(self.experts[:self._n_active]):
            if (self._expert_grad_ema[i] < self.args.grad_eps
                    and step - self._expert_last_grow[i] > self.args.grow_interval
                    and self._expert_uses[i] > self.args.expert_min_uses):
                new_block = LightweightExpertBlock(
                    self.args.moe_dim, self.args.mlp_mult,
                ).to(next(e.parameters()).device)
                e.extra_blocks.append(new_block)
                self._expert_last_grow[i] = step
                print(f"🧬 Expert {i} grew a new layer")

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _backbone(self, input_ids: Tensor, lora=None) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.args.model_dim,))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
        return x

    # ------------------------------------------------------------------
    # MoE refinement
    # ------------------------------------------------------------------

    def _moe_refine(self, h: Tensor) -> Tensor:
        """NUM_HOPS iterations of sparse per-token expert routing as residual corrections.
        Each token independently selects top-k experts. Only unique experts are executed.
        """
        for hop in range(self.args.num_hops):
            h_prev = h
            weights, ids = self.router(h, self.args.top_k)
            if ids is None:
                break

            bsz, seq, dim = h.shape
            k = ids.shape[-1]
            # ids: (bsz, seq, k), weights: (bsz, seq, k)

            # Find unique expert ids across all tokens
            selected = ids.unique().tolist()

            # Run only selected experts on full sequence
            expert_outs = {}
            for ei in selected:
                expert_outs[ei] = self.experts[ei](h)  # (bsz, seq, dim)

            # Update diagnostics
            if self.training:
                for ei in selected:
                    uses = int((ids == ei).sum().item())
                    self._expert_uses[ei] += uses

            # Build per-token weighted residual
            # Stack selected expert outputs for gathering
            eid_to_idx = {eid: idx for idx, eid in enumerate(selected)}
            stacked = torch.stack([expert_outs[eid] for eid in selected], dim=0)  # (n_sel, bsz, seq, dim)

            # Remap ids to local indices
            local_ids = ids.clone()
            for eid, idx in eid_to_idx.items():
                local_ids[local_ids == eid] = idx

            # Per-token combination: for each (batch, seq_pos, k), gather and weight
            res = torch.zeros_like(h)
            for k_idx in range(k):
                eidx = local_ids[:, :, k_idx]  # (bsz, seq)
                # Gather: for each (batch, seq_pos), pick the expert output
                # eidx indexes into stacked dim 0, we need stacked[eidx[b,s], b, s, :]
                gathered = stacked[
                    eidx.reshape(-1),
                    torch.arange(bsz, device=h.device).unsqueeze(1).expand(-1, seq).reshape(-1),
                    torch.arange(seq, device=h.device).unsqueeze(0).expand(bsz, -1).reshape(-1),
                ].reshape(bsz, seq, dim)  # (bsz, seq, dim)
                res += weights[:, :, k_idx].unsqueeze(-1) * gathered

            h = h + res
            if (h - h_prev).norm(dim=-1).mean() < 1e-3:
                break
        return h

    # ------------------------------------------------------------------
    # forward — matches train_gpt.py GPT.forward(input_ids, target_ids, lora)
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        # 1. Backbone
        h = self._backbone(input_ids, lora)

        # 2. Collect routing signal — FAISS/attractor runs in training loop to avoid GPU sync
        if self.training and self._train_step >= 0:
            self._train_step += 1
            if self._train_step % self.args.attractor_interval == 0:
                # Per-token router signal averaged for attractor clustering
                with torch.no_grad():
                    r = F.normalize(self.router.router_proj(h), dim=-1)  # (bsz, seq, dim)
                    self._pending_signal = r.mean(dim=1).detach().cpu()  # (bsz, dim)
            # Tick expert age
            for i in range(len(self._expert_age)):
                self._expert_age[i] += 1

        # 3. MoE refinement
        if self._n_active > 0:
            h = self._moe_refine(h)

        # 3. Head
        h = self.final_norm(h)
        if self.tie_embeddings:
            logits = F.linear(h, self.tok_emb.weight)
        else:
            logits = self.lm_head(h)
        logits = logits + (lora.lm_head_lora(h) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
            ).reshape(bsz, sl)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean"
        )

    def expert_diagnostics(self) -> dict:
        return {
            "n": self._n_active,
            "age": list(self._expert_age),
            "uses": list(self._expert_uses),
            "grad_ema": list(self._expert_grad_ema),
        }


# =====================================================================
# TTT-LORA (unchanged from train_gpt.py)
# =====================================================================

BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: MikeMoE, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _build_ttt_optimizer(lora, args: Hyperparameters):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr,
                            betas=(args.beta1, args.beta2), eps=1e-10)


def _find_docs(all_tokens: Tensor, include_next_bos: bool = True):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(ptl, x, y, batch_i, chunk_offset, chunk_len,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                    loss_sum, byte_sum, token_count):
    lbl = ptl[batch_i, chunk_offset:chunk_offset + chunk_len].to(torch.float64)
    prev = x[batch_i, chunk_offset:chunk_offset + chunk_len]
    tgt = y[batch_i, chunk_offset:chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
    loss_sum += lbl.sum()
    byte_sum += tok_bytes.sum()
    token_count += chunk_len


# Identical to template, removed base model hint as we use a different one
def eval_val_ttt_lora(
    args: Hyperparameters,
    base_model,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate with batched LoRA test-time training. Returns (val_loss, val_bpb)."""
    # Load validation tokens and find document boundaries
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)

    # Each rank takes a contiguous slice of documents
    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    lora_rank = args.ttt_lora_rank

    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi:bi + batch_size]
        bsz = len(batch)

        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt = _build_ttt_optimizer(cur_lora, args)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            chunk_stats = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            context_size, chunk_offset = chunk_stats[1], chunk_stats[2]

            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []  # (chunk_offset, chunk_len) per doc
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws: ds + ws + wl + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            # Forward pass (keep grad graph alive only when we need to train)
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)

            # Score: accumulate loss and byte counts for BPB (before training on chunk)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    _accumulate_bpb(
                        ptl, x, y, b, co, cl, base_bytes_lut, has_leading_space_lut,
                        is_boundary_token_lut, loss_sum, byte_sum, token_count)

            # Train: one Adam step on the LoRA params using this chunk's loss
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # Skips torch compile if no compile flag is set
    if not int(os.environ.get("NO_COMPILE", "0")):
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed + CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", str(8 // world_size)))
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
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                      enable_math_sdp, enable_mem_efficient_sdp)
    # Flash attention is not compiled into Windows PyTorch builds.
    # Fall back to math kernel (slower but correct). On Linux/H100 for submission,
    # swap these back: flash=True, math=False, and set NUM_KV_HEADS=4 for GQA.
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # Tokenizer + val metric setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only supports SentencePiece .model: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # Model + optimizer setup
    base_model = MikeMoE(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)

    log0("model initialized (experts spawn dynamically)")

    # fullgraph=False because MoE grows dynamically (new experts appended during training)
    # NO_COMPILE=1 skips torch.compile — useful for local debugging on small GPUs
    # where repeated recompilation from graph breaks (fullgraph=False) dominates runtime
    if int(os.environ.get("NO_COMPILE", "0")):
        compiled_model = base_model
    else:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # Optimizer: Muon for 2D matrix params, Adam for embeddings/scalars, 
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # Router proj is a matrix param too
    if base_model.router.router_proj.weight.ndim == 2:
        matrix_params.append(base_model.router.router_proj.weight)
    
    # Expert params are registered dynamically at spawn time (see training loop)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr,
                          momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"MikeMoE backbone_layers:{args.num_layers} model_dim:{args.model_dim} "
         f"moe_dim:{args.moe_dim} max_experts:{args.max_experts} top_k:{args.top_k} num_hops:{args.num_hops}")
    
    # base logs 
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # Data loader + warmup
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        base_model._train_step = 0  # reset so warmup steps dont count toward attractor

    # Main training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = (last_step and step > 0) or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            diag = base_model.expert_diagnostics()
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"experts:{diag['n']} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Update expert grad EMAs — must happen before zero_grad clears grads
        for i, e in enumerate(base_model.experts[:base_model._n_active]):
            g = sum(p.grad.norm().item() for p in e.parameters() if p.grad is not None)
            base_model._expert_grad_ema[i] = 0.99 * base_model._expert_grad_ema[i] + 0.01 * g
            base_model._expert_loss_ema[i] = 0.99 * base_model._expert_loss_ema[i] + 0.01 * train_loss.item()

        # Process attractor signal — add to index when available
        if base_model._pending_signal is not None:
            for sig in base_model._pending_signal:
                base_model.attractor.add(sig)
            base_model._pending_signal = None

        # Try promote every training step — but only after spawn_delay_steps
        if base_model._n_active < args.max_experts and step >= args.spawn_delay_steps:
            prev_active = base_model._n_active
            new_centroids = base_model.attractor.try_promote(device)
            if new_centroids is not None:
                for c in new_centroids:
                    if base_model._n_active < args.max_experts:
                        base_model.spawn(c, device)
            # Register newly spawned expert params with optimizers
            if base_model._n_active > prev_active:
                existing_muon_ids = {id(p) for p in optimizer_muon.param_groups[0]["params"]}
                existing_scalar_ids = {id(p) for pg in optimizer_scalar.param_groups for p in pg["params"]}
                for e in base_model.experts[prev_active:base_model._n_active]:
                    for name, p in e.named_parameters():
                        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                            if id(p) not in existing_muon_ids:
                                optimizer_muon.param_groups[0]["params"].append(p)
                                existing_muon_ids.add(id(p))
                        else:
                            if id(p) not in existing_scalar_ids:
                                optimizer_scalar.param_groups[0]["params"].append(p)
                                existing_scalar_ids.add(id(p))

        # Expert growth check — register any new extra_block params with optimizers
        if step % args.grow_interval == 0 and step > 0:
            base_model.try_grow_experts(step)
            existing_muon_ids = {id(p) for p in optimizer_muon.param_groups[0]["params"]}
            existing_scalar_ids = {id(p) for pg in optimizer_scalar.param_groups for p in pg["params"]}
            for e in base_model.experts[:base_model._n_active]:
                for name, p in e.extra_blocks.named_parameters():
                    if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                        if id(p) not in existing_muon_ids:
                            optimizer_muon.param_groups[0]["params"].append(p)
                            existing_muon_ids.add(id(p))
                    else:
                        if id(p) not in existing_scalar_ids:
                            optimizer_scalar.param_groups[0]["params"].append(p)
                            existing_scalar_ids.add(id(p))


        #continues as template from here 
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"experts:{base_model._n_active} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # Serialization + roundtrip validation
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int8+zlib: {quant_file_bytes} bytes "
             f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # LoRA TTT evaluation (the competition score)
    torch._dynamo.reset()
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
        args, base_model, rank, world_size, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int8_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
