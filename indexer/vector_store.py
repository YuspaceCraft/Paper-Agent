"""
vector_store.py — 向量库适配器
==============================

抽象接口 + Chroma (本地调试) + Qdrant (生产级, langchain_qdrant) 实现。

接口:
  - upsert(units): 批量写入 IndexUnit（含向量 + 元数据）
  - search(query_vector, filters, top_k): 向量检索
  - delete(chunk_ids): 按 ID 删除
  - count(): 集合大小
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ================================================================
# Abstract Interface
# ================================================================

class VectorStoreAdapter(ABC):
    """向量库适配器抽象接口。切换后端仅需更换实现类。"""

    @abstractmethod
    def upsert(self, units: list[dict]) -> int:
        """批量写入 IndexUnit。返回成功写入数。"""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """向量检索。返回 list of {chunk_id, metadata, score}。"""
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """按 chunk_id 删除。返回删除数。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回集合中的记录数。"""
        ...

    @abstractmethod
    def get_existing_hashes(self) -> dict[str, str]:
        """返回 {chunk_id: content_hash} 映射（供去重使用）。"""
        ...


# ================================================================
# Chroma Implementation
# ================================================================

class ChromaVectorStore(VectorStoreAdapter):
    """Chroma 向量库实现 — 本地调试首选。

    自动持久化到磁盘，支持 metadata 过滤。
    """

    def __init__(self, persist_dir: str = "./indexer/data/chroma",
                 collection_name: str = "rag_chunks"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    # ---- lazy init ----

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "pip install chromadb"
            )

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # 获取或创建 collection
        try:
            self._collection = self._client.get_collection(self._collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    # ---- public API ----

    def upsert(self, units: list[dict]) -> int:
        self._ensure_client()
        if not units:
            return 0

        ids, embeddings, metadatas, documents = [], [], [], []

        for u in units:
            ids.append(u["chunk_id"])
            vec = u.get("dense_vector")
            if vec is not None and len(vec) > 0:
                embeddings.append(vec)
            else:
                # Chroma 要求非空 embedding；对缺失的填零向量
                dim = self._get_dim()
                embeddings.append([0.0] * dim)

            # 扁平化 metadata（Chroma 要求标量值）
            meta = self._flatten_metadata(u)
            metadatas.append(meta)

            # generation_text 作为 document 字段存储（便于调试）
            documents.append(u.get("generation_text", ""))

        # 去重：已存在 ID 标记为 update
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        except Exception as exc:
            print(f"  [VECTOR-STORE] Upsert FAILED: {exc}")
            # 降级: 逐条 upsert
            success = 0
            for i, uid in enumerate(ids):
                try:
                    self._collection.upsert(
                        ids=[uid],
                        embeddings=[embeddings[i]],
                        metadatas=[metadatas[i]],
                        documents=[documents[i]],
                    )
                    success += 1
                except Exception as e2:
                    print(f"  [VECTOR-STORE] Single upsert {uid} FAILED: {e2}")
            return success

        return len(ids)

    def search(
        self,
        query_vector: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        self._ensure_client()

        where = self._build_filter(filters) if filters else None

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            print(f"  [VECTOR-STORE] Search FAILED: {exc}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, cid in enumerate(results["ids"][0]):
            output.append({
                "chunk_id": cid,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "document": results["documents"][0][i] if results["documents"] else "",
                "score": 1.0 - results["distances"][0][i] if results["distances"] else 0,
            })
        return output

    def delete(self, chunk_ids: list[str]) -> int:
        self._ensure_client()
        if not chunk_ids:
            return 0
        try:
            self._collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as exc:
            print(f"  [VECTOR-STORE] Delete FAILED: {exc}")
            return 0

    def count(self) -> int:
        self._ensure_client()
        return self._collection.count()

    def get_existing_hashes(self) -> dict[str, str]:
        """获取已有 chunk 的 content_hash 映射。

        ponytail: Chroma metadata 过滤 + 全量获取。
        大量数据时考虑分页。
        """
        self._ensure_client()
        try:
            existing = self._collection.get(include=["metadatas"])
            result = {}
            for cid, meta in zip(existing["ids"], existing["metadatas"]):
                h = meta.get("content_hash", "")
                if h:
                    result[cid] = h
            return result
        except Exception as exc:
            print(f"  [VECTOR-STORE] get_existing_hashes FAILED: {exc}")
            return {}

    # ---- helpers ----

    def _get_dim(self) -> int:
        """尝试推断 embedding 维度，fallback 1024。"""
        try:
            n = self._collection.count()
            if n > 0:
                sample = self._collection.get(limit=1, include=["embeddings"])
                if sample["embeddings"] and sample["embeddings"][0]:
                    return len(sample["embeddings"][0])
        except Exception:
            pass
        return 1024

    @staticmethod
    def _flatten_metadata(unit: dict) -> dict:
        """将 IndexUnit metadata 展开为 Chroma 兼容的标量字段。

        Chroma 不支持嵌套 dict/list → 序列化为 JSON 字符串。
        """
        meta = dict(unit.get("metadata", {}))
        meta["content_hash"] = unit.get("content_hash", "")
        meta["content_type"] = meta.get("content_type", "")
        meta["section_path"] = meta.get("section_path", "")
        meta["chunk_id"] = unit["chunk_id"]
        meta["schema_version"] = unit.get("schema_version", "1.0")
        meta["pii_flagged"] = unit.get("pii_flagged", False)

        # 序列化 list 字段
        for f in ("sparse_keywords", "hyde_questions", "ref_ids",
                   "bound_elements", "pii_findings"):
            val = unit.get(f, [])
            if isinstance(val, list):
                import json
                meta[f] = json.dumps(val, ensure_ascii=False)

        # 移除 Chroma 不支持的 None 值
        return {k: (v if v is not None else "") for k, v in meta.items()}

    @staticmethod
    def _build_filter(filters: dict) -> dict | None:
        """将用户 filter dict 转换为 Chroma where 语法。

        支持: {"content_type": "body", "section_path": "Methods"}
        → {"$and": [{"content_type": "body"}, {"section_path": "Methods"}]}
        """
        if not filters:
            return None
        if len(filters) == 1:
            k, v = next(iter(filters.items()))
            return {k: v}
        return {"$and": [{k: v} for k, v in filters.items()]}


# ================================================================
# Qdrant Implementation (LangChain-integrated)
# ================================================================

class QdrantVectorStore(VectorStoreAdapter):
    """Qdrant 向量库 — 全线使用 langchain_qdrant.QdrantVectorStore。

    Qdrant 要求 point ID 为 unsigned integer 或 UUID。
    我们用 uuid5(chunk_id) 将字符串 ID 转换为确定性 UUID，
   原始 chunk_id 保存在 payload 中。

    所有操作通过 LangChain wrapper：
    - search → _lc_store.similarity_search_by_vector()
    - delete → _lc_store.delete()
    - upsert → _lc_store.client.upsert()   （LangChain 不暴露预计算向量写入）
    - count/get_existing_hashes → _lc_store.client （LangChain 未覆盖的只读操作）
    """

    def __init__(self, url: str = "http://localhost:6333",
                 collection_name: str = "rag_chunks",
                 vector_size: int = 0):
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size  # 0 = auto-detect from first upsert
        self._lc_store = None

    # ----------------------------------------------------------------
    # ID mapping
    # ----------------------------------------------------------------

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).hex

    # ----------------------------------------------------------------
    # init — 全线通过 LangChain
    # ----------------------------------------------------------------

    def _ensure_store(self):
        if self._lc_store is not None:
            return

        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore as LCQdrant

        client = QdrantClient(url=self._url)
        self._lc_store = LCQdrant(
            client=client,
            collection_name=self._collection_name,
            embedding=None,
            validate_embeddings=False,
            validate_collection_config=False,
        )

    def _ensure_collection(self, vector_dim: int):
        """确保 collection 存在且维度匹配。ponytail: 维度不匹配时重建。"""
        from qdrant_client.models import Distance, VectorParams

        client = self._lc_store.client
        try:
            info = client.get_collection(self._collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != vector_dim:
                print(f"  [VECTOR-STORE] Dimension mismatch: collection={existing_dim}d, "
                      f"vectors={vector_dim}d → recreating collection")
                client.delete_collection(self._collection_name)
                raise Exception("recreate")
        except Exception:
            # 不存在或已删除 → 创建
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                ),
            )
        self._vector_size = vector_dim

    # ----------------------------------------------------------------
    # public API — 全部通过 _lc_store
    # ----------------------------------------------------------------

    def upsert(self, units: list[dict]) -> int:
        self._ensure_store()
        if not units:
            return 0

        # 从数据中推断实际向量维度
        sample_vec = next(
            (u["dense_vector"] for u in units
             if u.get("dense_vector") and len(u["dense_vector"]) > 0),
            None,
        )
        if sample_vec is None:
            print("  [VECTOR-STORE] No valid vectors in units, skipping")
            return 0
        dim = len(sample_vec)
        self._ensure_collection(dim)

        from qdrant_client.models import PointStruct

        points = []
        for u in units:
            vec = u.get("dense_vector")
            if vec is None or len(vec) == 0:
                vec = [0.0] * dim

            meta = u.get("metadata", {})
            payload = {
                "chunk_id": u["chunk_id"],
                "content_hash": u.get("content_hash", ""),
                "content_type": meta.get("content_type", ""),
                "section_path": meta.get("section_path", ""),
                "sparse_keywords": u.get("sparse_keywords", []),
                "generation_text": u.get("generation_text", ""),
                "schema_version": u.get("schema_version", "1.0"),
            }
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    payload.setdefault(k, v)

            points.append(PointStruct(
                id=self._to_uuid(u["chunk_id"]),
                vector=vec,
                payload=payload,
            ))

        # ponytail: LangChain 不暴露预计算向量的批量写入，
        # 使用其底层 client（_lc_store.client === LangChain 内部的同个 QdrantClient）
        try:
            self._lc_store.client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            return len(points)
        except Exception as exc:
            print(f"  [VECTOR-STORE] Qdrant upsert FAILED: {exc}")
            return 0

    def search(
        self,
        query_vector: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        self._ensure_store()

        # ponytail: raw Qdrant client — LangChain similarity_search_by_vector
        # drops payload fields from metadata in some versions.
        from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue

        qfilter = None
        if filters:
            qfilter = QFilter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ])

        results = self._lc_store.client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )

        return [
            {
                "chunk_id": (r.payload or {}).get("chunk_id", ""),
                "metadata": r.payload or {},
                "document": (r.payload or {}).get("generation_text", ""),
                "score": r.score,
            }
            for r in results.points
        ]

    def delete(self, chunk_ids: list[str]) -> int:
        self._ensure_store()
        if not chunk_ids:
            return 0
        uuids = [self._to_uuid(cid) for cid in chunk_ids]
        self._lc_store.delete(ids=uuids)
        return len(chunk_ids)

    def count(self) -> int:
        self._ensure_store()
        # ponytail: collection may not exist yet (pre-upsert)
        try:
            info = self._lc_store.client.get_collection(self._collection_name)
            return info.points_count
        except Exception:
            return 0

    def get_existing_hashes(self) -> dict[str, str]:
        self._ensure_store()
        result = {}
        try:
            offset = None
            while True:
                points, next_offset = self._lc_store.client.scroll(
                    collection_name=self._collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["content_hash", "chunk_id"],
                )
                for pt in points:
                    payload = pt.payload or {}
                    h = payload.get("content_hash", "")
                    cid = payload.get("chunk_id", "")
                    if h and cid:
                        result[cid] = h
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            print(f"  [VECTOR-STORE] Qdrant get_existing_hashes FAILED: {exc}")
        return result

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _build_qdrant_filter(filters: dict):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions) if conditions else None
