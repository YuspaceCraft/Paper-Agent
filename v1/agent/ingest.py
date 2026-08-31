"""
ingest.py — 文档导入流水线
========
将 PDF 论文文件加载、切分、嵌入并存入向量数据库。

这是文档写入路径 — 与 agent/ 中的读取路径分离。

使用方式:
  from agent.ingest import ingest
  chunk_count = ingest(file_path="./data", force=False)
"""

from pathlib import Path

from config import DATA_DIR, CHROMA_PERSIST_DIR
from agent.loader import load_documents
from agent.splitter import split_documents
from agent.embedder import get_embeddings
from agent.store import create_vector_store, store_exists


def ingest(file_path: str | None = None, force: bool = False) -> int:
    """
    文档导入流水线：加载 → 切分 → 嵌入 → 存储

    步骤：
      1. 检查是否已有向量数据库（有就跳过，除非 force=True）
      2. 从文件加载原始文档
      3. 将文档切分成小块
      4. 创建嵌入模型
      5. 为每个块生成向量并存入 Chroma
      6. 输出导入摘要

    参数:
        file_path: 文档路径，默认为 data/ 目录
        force:     即使已有向量数据库也强制重建

    返回:
        int: 存储的文档块数量
    """
    # 解析文件路径
    if file_path is None:
        file_path = str(DATA_DIR)
    else:
        file_path = str(Path(file_path))

    # 如果已有向量数据库且不是强制模式，跳过
    if store_exists() and not force:
        print("[INGEST] [SKIP] 向量数据库已存在，跳过导入。")
        print("[INGEST] [TIP] 如需重建请使用: python -m web.cli ingest --rebuild")
        return 0

    if force and store_exists():
        print("[INGEST] [REBUILD] 强制重建模式：正在删除旧的向量数据库...")
        import shutil
        shutil.rmtree(str(CHROMA_PERSIST_DIR), ignore_errors=True)
        print("[INGEST] [REBUILD] 旧数据库已删除，开始重建...")

    print(f"[INGEST] [START] 开始文档导入: {file_path}")
    print("[INGEST] " + "─" * 50)

    # 步骤 1: 加载文档
    docs = load_documents(file_path)
    if not docs:
        print("[INGEST] [ERR] 导入失败: 文件中没有内容")
        return 0

    # 步骤 2: 切分文档
    chunks = split_documents(docs)
    if not chunks:
        print("[INGEST] [ERR] 导入失败: 切分后没有内容")
        return 0

    # 步骤 3: 创建嵌入模型
    embeddings = get_embeddings()

    # 步骤 4: 创建向量数据库
    vector_store = create_vector_store(chunks, embeddings)

    # 导入完成
    chunk_count = vector_store._collection.count()
    print("[INGEST] " + "─" * 50)
    print(f"[INGEST] [OK] 导入完成！已存储 {chunk_count} 个文档块到向量数据库")
    print(f"[INGEST] [TIP] 现在可以提问了: python -m web.cli interactive")
    return chunk_count
