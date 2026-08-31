"""
dedup_manager.py — 去重与版本管理器
====================================

通过 Content Hash 实现增量同步:
  - 相同 hash → 跳过（幂等）
  - 不同 hash → 标记更新
  - 不存在   → 标记新增

保留 schema_version 字段，支持索引策略升级后的灰度验证。
"""

from __future__ import annotations

import hashlib
import time


def compute_hash(chunk_id: str, retrieval_text: str, generation_text: str,
                 algorithm: str = "sha256") -> str:
    """计算 IndexUnit 的内容指纹。

    指纹＝sha256(chunk_id | retrieval_text | generation_text)，截断至 32 hex 字符。
    """
    payload = f"{chunk_id}|{retrieval_text}|{generation_text}"
    h = hashlib.new(algorithm)
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:32]


class DedupManager:
    """去重与版本管理器。

    用法:
        mgr = DedupManager(store)
        stats = mgr.sync(units)  # 自动分类 new/updated/skipped
    """

    def __init__(self, store, schema_version: str = "1.0"):
        """
        Args:
            store: VectorStoreAdapter 实例
            schema_version: 当前索引 schema 版本
        """
        self._store = store
        self.schema_version = schema_version

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def sync(self, units: list[dict]) -> dict:
        """对 units 执行增量同步分类。

        返回:
            {
                "new": list[dict],       # 新 chunk → 需 embed + insert
                "updated": list[dict],   # hash 变更 → 需 re-embed + update
                "skipped": list[str],    # hash 不变 → 跳过
                "orphaned": list[str],   # 在 store 中存在但 units 中不存在
                "counts": {total, new, updated, skipped, orphaned},
            }
        """
        t0 = time.time()

        # 计算所有 units 的 hash
        for u in units:
            u["content_hash"] = compute_hash(
                u["chunk_id"],
                u.get("retrieval_text", ""),
                u.get("generation_text", ""),
            )
            u["schema_version"] = self.schema_version

        # 获取存储中已存在的 hash
        existing_hashes = self._store.get_existing_hashes()

        # 当前 units 的 chunk_id 集合
        current_ids = {u["chunk_id"] for u in units}

        # 分类
        new_units = []
        updated_units = []
        skipped_ids = []

        for u in units:
            cid = u["chunk_id"]
            h = u["content_hash"]

            if cid not in existing_hashes:
                new_units.append(u)
            elif existing_hashes[cid] != h:
                updated_units.append(u)
            else:
                skipped_ids.append(cid)

        # 孤儿 chunk：仅在「当前 paper」作用域内检测。
        # chunk_id 格式为 {paper}__{chunk}，single-paper 增量索引时若用全库
        # existing_hashes 直接做差集，会把其他 paper 的 chunk 误判为孤儿删掉。
        # 无 __ 前缀的旧格式 chunk 视为同一作用域，保持全局孤儿语义。
        def _scope(cid: str) -> str:
            return cid.rsplit("__", 1)[0] if "__" in cid else ""

        scope = {_scope(cid) for cid in current_ids}
        orphaned = [
            cid for cid in existing_hashes
            if cid not in current_ids and _scope(cid) in scope
        ]

        elapsed = (time.time() - t0) * 1000
        print(f"  [DEDUP] {len(units)} units: {len(new_units)} new, "
              f"{len(updated_units)} updated, {len(skipped_ids)} skipped, "
              f"{len(orphaned)} orphaned ({elapsed:.0f}ms)")

        return {
            "new": new_units,
            "updated": updated_units,
            "skipped": skipped_ids,
            "orphaned": orphaned,
            "counts": {
                "total": len(units),
                "new": len(new_units),
                "updated": len(updated_units),
                "skipped": len(skipped_ids),
                "orphaned": len(orphaned),
            },
        }

    def apply(self, classification: dict, embed_fn=None) -> dict:
        """将分类结果写入存储。

        Args:
            classification: self.sync() 的返回值
            embed_fn: embedding 函数 (仅在需要时调用)

        Returns: 写入统计
        """
        to_upsert = classification["new"] + classification["updated"]

        if not to_upsert:
            print("  [DEDUP] Nothing to upsert — all chunks up to date")
            return {"inserted": 0, "updated": 0, "deleted": 0, "skipped": len(classification["skipped"])}

        # 写入
        n_written = self._store.upsert(to_upsert)

        # 清理孤儿
        n_deleted = 0
        if classification["orphaned"]:
            n_deleted = self._store.delete(classification["orphaned"])

        return {
            "inserted": n_written,  # 实际写入数，非期望数
            "updated": 0,
            "deleted": n_deleted,
            "skipped": len(classification["skipped"]),
        }
