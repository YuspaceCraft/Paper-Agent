"""
pdf_pipeline config — 独立于项目根 config.py 的最小化配置。
仅包含 PDF 管道需要的参数，不依赖 agent/ 或项目级 config。
"""

import os
from pathlib import Path

# 加载 .env（管道独立运行时也能读取 PAPER_CHUNK_SIZE 等环境变量）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_path)

PAPER_CHUNK_SIZE = int(os.getenv("PAPER_CHUNK_SIZE", "800"))
PAPER_CHUNK_OVERLAP = int(os.getenv("PAPER_CHUNK_OVERLAP", "100"))
