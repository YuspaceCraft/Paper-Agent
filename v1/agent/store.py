"""
store.py — 向量数据库(Chroma)
=======
负责创建、持久化和加载向量数据库。

Chroma 是什么？
  一个轻量级的向量数据库，数据存储在本地文件中，不需要额外的服务器。
  它支持：
    1. 存储文档块的嵌入向量
    2. 基于向量相似度的语义搜索
    3. 数据持久化（关掉程序后数据不丢失）

为什么要持久化？
  生成向量需要调用 OpenAI API(钱+耗时)，所以我们把它存到磁盘。
  第一次运行 ingest 后，之后的查询都可以直接从磁盘加载，又快又省钱。
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
) -> Chroma:
    """
    创建向量数据库：将文档块的嵌入向量存入 Chroma。

    这个过程会：
      1. 对每个文档块调用 OpenAI API 生成嵌入向量
      2. 将向量和原始文本一起存入 Chroma
      3. 将数据持久化到 chroma_db/ 目录

    参数:
        documents:  切分后的文档块列表
        embeddings: 嵌入模型实例

    返回:
        Chroma: 创建好的向量数据库对象
    """
    print(f"[STORE] [SETUP] 正在创建向量数据库...（将为 {len(documents)} 个块生成嵌入向量）")

    # Chroma.from_documents() 会：
    #   1. 对每个 document 调用 embeddings 生成向量
    #   2. 存储向量 + 元数据 + 原文到 chroma_db/
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    print(f"[STORE] [OK] 向量数据库已创建并保存到: {CHROMA_PERSIST_DIR}")
    return vector_store


def load_vector_store(embeddings: Embeddings) -> Chroma:
    """
    从磁盘加载已有的向量数据库。

    参数:
        embeddings: 嵌入模型实例（查询时需要用它把问题转为向量）

    返回:
        Chroma: 加载的向量数据库对象
    """
    if not store_exists():
        raise FileNotFoundError(
            "[ERR] 未找到向量数据库！\n"
            "请先运行文档导入: python -m src.cli ingest"
        )

    print(f"[STORE] [LOAD] 从磁盘加载向量数据库: {CHROMA_PERSIST_DIR}")

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    return vector_store


def store_exists() -> bool:
    """
    检查向量数据库是否已存在。

    返回:
        bool: True 表示已有持久化的数据，可以直接 load
    """
    # Chroma 会在 persist_directory 下创建 chroma.sqlite3 文件
    chroma_file = CHROMA_PERSIST_DIR / "chroma.sqlite3"
    return chroma_file.exists()
