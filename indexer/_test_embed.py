"""
_test_embed.py — 本地 embedding 模型加载诊断脚本
================================================
独立运行，不依赖 indexer 任何模块。逐步测试 import / 加载 / 编码。
用法:
    python indexer/_test_embed.py
    python indexer/_test_embed.py --model BGE/bge-small-en-v1.5 --device cpu
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL = "Qwen3_model/Qwen3-Embedding-0.6B"
_DEVICE = "cuda"


def step(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    sys.stdout.flush()


def check(label: str):
    print(f"  [{label}]...", end=" ", flush=True)


def ok(msg: str = "OK"):
    print(msg)
    sys.stdout.flush()


def fail(exc: Exception):
    print(f"FAILED: {exc}")
    sys.stdout.flush()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    device = sys.argv[2] if len(sys.argv) > 2 else _DEVICE

    print("=" * 60)
    print("  Local Embedding Model Diagnostic")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print("=" * 60)
    sys.stdout.flush()

    # ---- Step 1: resolve path ----
    step("1/6 Resolve model path")
    model_path = model_name
    if not os.path.isabs(model_name) and not model_name.startswith(("./", "../", ".\\")):
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / model_name
        if candidate.exists():
            model_path = str(candidate)
    print(f"  Resolved: {model_path}")
    print(f"  Exists: {os.path.isdir(model_path)}")
    if not os.path.isdir(model_path):
        print(f"  [ERROR] Model directory not found!")
        sys.exit(1)
    sys.stdout.flush()

    # ---- Step 2: System info ----
    step("2/6 System info")
    print(f"  Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
    sys.stdout.flush()

    # ---- Step 3: import torch ----
    step("3/6 Import torch")
    t0 = time.time()
    try:
        import torch
        dt = (time.time() - t0) * 1000
        print(f"  torch {torch.__version__} ({dt:.0f}ms)")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            try:
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except AttributeError:
                mem = 0
            print(f"  CUDA mem: {mem:.1f} GB")
        sys.stdout.flush()
    except Exception as e:
        fail(e)
        sys.exit(1)

    # ---- Step 4: import transformers ----
    step("4/6 Import transformers")
    print(f"  (bypassing sentence_transformers to avoid DLL segfault)...")
    sys.stdout.flush()
    t0 = time.time()
    try:
        from transformers import AutoModel, AutoTokenizer
        dt = (time.time() - t0) * 1000
        print(f"  transformers imported ({dt:.0f}ms)")
        sys.stdout.flush()
    except Exception as e:
        fail(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- Step 5: load model ----
    step("5/6 Load model via AutoModel")
    print(f"  Loading: {model_path}")
    print(f"  (loading to {device}, may take 10-30s)...")
    sys.stdout.flush()
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        print(f"  Tokenizer loaded ({(time.time()-t0)*1000:.0f}ms)")
        sys.stdout.flush()

        dtype = torch.float16 if device.startswith("cuda") else None
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).to(device)
        model.eval()
        dt = (time.time() - t0) * 1000
        print(f"  Model loaded ({dt:.0f}ms)")

        dim = model.config.hidden_size
        print(f"  Dimension: {dim}")
        sys.stdout.flush()
    except Exception as e:
        fail(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- Step 6: test encode ----
    step("6/6 Test encode with last-token pooling")
    texts = [
        "Hello world, this is a test sentence for embedding.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    print(f"  Texts: {len(texts)} items")
    sys.stdout.flush()

    def encode(texts_batch):
        encoded = tokenizer(texts_batch, padding=True, truncation=True,
                            max_length=8192, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state          # [B, L, D]
        attn = encoded["attention_mask"]             # [B, L]
        seq_lens = attn.sum(dim=1) - 1              # [B]
        batch_indices = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_indices, seq_lens]     # [B, D]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.cpu()

    try:
        print(f"  Warmup encode...")
        sys.stdout.flush()
        t_warm = time.time()
        encode(["warmup"])
        print(f"  Warmup done ({(time.time()-t_warm)*1000:.0f}ms)")

        print(f"  Encoding {len(texts)} texts...")
        sys.stdout.flush()
        t_enc = time.time()
        embeddings = encode(texts)
        dt = (time.time() - t_enc) * 1000

        import numpy as np
        norms = np.linalg.norm(embeddings.numpy(), axis=1)
        print(f"  Encoded: {embeddings.shape} ({dt:.0f}ms, {dt/len(texts):.1f}ms/text)")
        print(f"  Norms: {norms}")
        print(f"  Sample[0][:5]: {embeddings[0][:5].tolist()}")
        sys.stdout.flush()
    except Exception as e:
        fail(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  [OK] All checks passed — model is working correctly")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
