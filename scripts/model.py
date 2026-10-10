"""Small byte-level transformer encoder for document / driver-note classification.

The attention module is chosen by name from ATTENTION so architecture variants
(e.g. linear attention) can be swapped in without touching the rest of the model.
load_for_inference() also applies post-training 4-bit quantization when a config asks for it.
"""
from __future__ import annotations

import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MAX_LEN, PAD_ID, VOCAB_SIZE


class SoftmaxAttention(nn.Module):
    """Standard multi-head scaled dot-product attention: O(T^2) in sequence length."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        mask = keep[:, None, None, :]  # (B, 1, 1, T): True = real token
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class LinearAttention(nn.Module):
    """Linear attention (Katharopoulos et al., 2020): softmax(QK^T)V is replaced by
    phi(Q) (phi(K)^T V) with phi(x) = elu(x) + 1, so cost grows O(T) instead of O(T^2).
    Same parameters as SoftmaxAttention; only the token-mixing math changes."""

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.eps = eps
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        # Sums over T must stay in fp32: under fp16 autocast they overflow and training goes NaN.
        # (autocast would re-cast einsum to fp16 even after .float(), so it is disabled here.)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = F.elu(q) + 1
            k = (F.elu(k) + 1) * keep[:, None, :, None]  # padded keys contribute nothing
            kv = torch.einsum("bhtd,bhte->bhde", k, v)  # (B, H, dh, dh) summary of the sequence
            z = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k.sum(2)) + self.eps)
            y = torch.einsum("bhtd,bhde->bhte", q, kv) * z.unsqueeze(-1)
        return self.out(y.to(x.dtype).transpose(1, 2).reshape(B, T, D))


ATTENTION = {"softmax": SoftmaxAttention, "linear": LinearAttention}


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attention: str, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ATTENTION[attention](d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        x = x + self.drop(self.attn(self.ln1(x), keep))
        return x + self.drop(self.ff(self.ln2(x)))


class TextClassifier(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 1024, attention: str = "softmax", dropout: float = 0.1, max_len: int = MAX_LEN):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, attention, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        keep = ids != PAD_ID
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.drop(self.tok(ids) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.ln(x)
        # Mean-pool over real tokens only.
        w = keep.unsqueeze(-1).to(x.dtype)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.head(pooled)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("tok.", "pos.")))
    return {"params_total": total, "params_non_embedding": total - embed}


def build_model(cfg: dict, n_classes: int) -> TextClassifier:
    keys = ["d_model", "n_layers", "n_heads", "d_ff", "attention", "dropout"]
    return TextClassifier(n_classes, **{k: cfg[k] for k in keys if k in cfg})


def quantize_4bit(model: nn.Module, skip: tuple[str, ...] = ("head",)) -> nn.Module:
    """Swap every nn.Linear (except the classifier head) for a bitsandbytes NF4 Linear4bit.
    Weights are quantized when the model is moved to the GPU; matmuls run in fp16."""
    import bitsandbytes as bnb

    for name, child in list(model.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            q = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                  compute_dtype=torch.float16, quant_type="nf4")
            q.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                q.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(model, name, q)
        else:
            quantize_4bit(child, skip)
    return model


def load_for_inference(name: str, device: torch.device) -> tuple[nn.Module, dict, list[str], dict]:
    """Build a benchmark-ready model for config `name`.

    Trained configs load their own checkpoint. Post-training configs (with a "source" key)
    load the source checkpoint and transform it, e.g. quantize to 4-bit."""
    from common import CKPT_DIR
    from configs import get_config

    cfg = get_config(name)
    ckpt = torch.load(CKPT_DIR / f"{cfg.get('source', name)}.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    params = count_params(model)  # counted before quantization packs two weights per byte
    if cfg.get("quantize") == "nf4":
        if device.type != "cuda":
            raise SystemExit("4-bit quantization (bitsandbytes) needs a CUDA GPU")
        model = quantize_4bit(model)
    return model.to(device).eval(), cfg, ckpt["labels"], params


def state_dict_mb(model: nn.Module) -> float:
    """Serialized size of the weights as they sit in memory (what ships to the tablet)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20
