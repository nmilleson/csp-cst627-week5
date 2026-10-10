"""Benchmark configurations. Each entry is one row on Sandra's trade-off board."""

CONFIGS = {
    "baseline": {
        "description": "4-layer softmax-attention transformer, FP32",
        "variant_type": "baseline",
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 1024,
        "attention": "softmax",
        "dropout": 0.1,
        "epochs": 8,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.05,
    },
    "linear_attention": {
        "description": "Architecture variant: same 4-layer model, linear (elu+1) attention, FP32",
        "variant_type": "architecture",
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 1024,
        "attention": "linear",
        "dropout": 0.1,
        "epochs": 8,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.05,
    },
    "quant_4bit": {
        "description": "Baseline weights quantized post-training to 4-bit NF4 (bitsandbytes), FP16 compute",
        "variant_type": "quantized",
        "source": "baseline",  # no training: transforms the baseline checkpoint at load time
        "quantize": "nf4",
        "precision": "nf4 weights / fp16 compute",
    },
    "distilled_student": {
        "description": "Distilled student: 2-layer, d=128 transformer trained on baseline's soft labels, FP32",
        "variant_type": "distilled",
        "teacher": "baseline",
        "kd_alpha": 0.7,  # weight on the teacher's soft targets vs. the hard labels
        "kd_temperature": 2.0,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 512,
        "attention": "softmax",
        "dropout": 0.1,
        "epochs": 12,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.05,
    },
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return {"name": name, **CONFIGS[name]}
