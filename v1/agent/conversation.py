"""
conversation.py — 对话持久化管理
==============
提供对话的保存、加载、列表、删除功能。

对话以 JSON 文件形式存储在 MEMORY_PERSIST_DIR 目录下，
每个文件用 UUID 命名（避免冲突），同时维护 name → id 的映射。

使用方式：
  store = ConversationStore(Path("conversations"))
  store.save("我的会话", memory)          # 保存
  memory = store.load("我的会话")         # 加载
  records = store.list_all()              # 列表
  store.delete("我的会话")                # 删除
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.memory import BaseMemory, create_memory


# ============================================================
#  ConversationRecord — 对话元信息
# ============================================================

@dataclass
class ConversationRecord:
    """对话的摘要信息（用于 /list 展示）。"""
    id: str              # UUID
    name: str            # 用户友好的名称
    created_at: str      # ISO 8601
    updated_at: str      # ISO 8601
    memory_type: str     # "buffer" 或 "window"
    message_count: int   # 消息总数


# ============================================================
#  ConversationStore — 对话仓库
# ============================================================

class ConversationStore:
    """
    管理对话的持久化存储。

    每个对话保存为一个 JSON 文件，文件名 = {id}.json。
    目录下的 _index.json 维护 name → id 的映射（加速查找）。
    """

    INDEX_FILE = "_index.json"

    def __init__(self, persist_dir: Path):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, str] = {}  # name → id
        self._rebuild_index()

    # ----- 公共 API -----

    def save(self, name: str, memory: BaseMemory) -> str:
        """
        保存对话到磁盘。

        如果 name 已存在则覆盖（更新同一文件）。

        参数:
            name:   对话名称（用户友好标签）
            memory: 记忆实例

        返回:
            str: 对话的 UUID
        """
        name = name.strip()
        if not name:
            raise ValueError("对话名称不能为空")

        now = datetime.now(timezone.utc).isoformat()

        # 检查是否已存在同名对话
        if name in self._index:
            conv_id = self._index[name]
            created_at = self._read_created_at(conv_id) or now
        else:
            conv_id = uuid.uuid4().hex[:12]  # 12 位短 ID，方便展示
            created_at = now

        # 构建 JSON 数据
        data = {
            "id": conv_id,
            "name": name,
            "created_at": created_at,
            "updated_at": now,
            **memory.to_dict(),  # memory_type, memory_config, messages
        }

        # 写入文件
        file_path = self._persist_dir / f"{conv_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 更新索引
        self._index[name] = conv_id
        self._write_index()

        print(f"[CONV] [OK] 对话已保存: '{name}' (id={conv_id}, "
              f"{memory.message_count()} 条消息)")
        return conv_id

    def load(self, name_or_id: str) -> BaseMemory | None:
        """
        从磁盘加载对话。

        参数:
            name_or_id: 对话名称或 UUID

        返回:
            BaseMemory | None: 加载的记忆实例，找不到返回 None
        """
        file_path = self._resolve_path(name_or_id)
        if file_path is None:
            print(f"[CONV] [ERR] 未找到对话: '{name_or_id}'")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 根据 memory_type 调用对应的 from_dict
        memory_type = data.get("memory_type", "buffer")
        memory = create_memory(
            memory_type=memory_type,
            window_size=data.get("memory_config", {}).get("max_turns", 5),
        )
        # 用 from_dict 恢复消息
        restored = memory.from_dict(data)

        name = data.get("name", name_or_id)
        print(f"[CONV] [OK] 对话已加载: '{name}' "
              f"({restored.message_count()} 条消息, 类型={memory_type})")
        return restored

    def list_all(self) -> list[ConversationRecord]:
        """
        列出所有已保存的对话。

        返回:
            list[ConversationRecord]: 按更新时间降序排列
        """
        records = []
        for json_file in sorted(
            self._persist_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if json_file.name == self.INDEX_FILE:
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(ConversationRecord(
                    id=data.get("id", json_file.stem),
                    name=data.get("name", "(未命名)"),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    memory_type=data.get("memory_type", "buffer"),
                    message_count=len(data.get("messages", [])),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return records

    def delete(self, name_or_id: str) -> bool:
        """
        删除一个已保存的对话。

        参数:
            name_or_id: 对话名称或 UUID

        返回:
            bool: 是否成功删除
        """
        file_path = self._resolve_path(name_or_id)
        if file_path is None:
            print(f"[CONV] [ERR] 未找到对话: '{name_or_id}'")
            return False

        # 从索引中移除
        name_to_remove = None
        for name, cid in self._index.items():
            if cid == file_path.stem:
                name_to_remove = name
                break
        if name_to_remove:
            del self._index[name_to_remove]
            self._write_index()

        file_path.unlink()
        print(f"[CONV] [OK] 已删除对话: '{name_or_id}'")
        return True

    # ----- 内部方法 -----

    def _resolve_path(self, name_or_id: str) -> Path | None:
        """根据 name 或 id 找到对应的 JSON 文件路径。"""
        name_or_id = name_or_id.strip()

        # 先按名称查找
        if name_or_id in self._index:
            conv_id = self._index[name_or_id]
            file_path = self._persist_dir / f"{conv_id}.json"
            if file_path.exists():
                return file_path

        # 再按 ID 查找（直接匹配文件名）
        direct = self._persist_dir / f"{name_or_id}.json"
        if direct.exists():
            return direct

        # 模糊匹配：在所有 JSON 文件中搜索 name 字段
        for json_file in self._persist_dir.glob("*.json"):
            if json_file.name == self.INDEX_FILE:
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("name") == name_or_id:
                    return json_file
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def _rebuild_index(self) -> None:
        """扫描目录，重建 name → id 索引。"""
        self._index.clear()
        index_path = self._persist_dir / self.INDEX_FILE
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                # 验证索引中的文件是否还存在
                stale = [
                    name for name, cid in self._index.items()
                    if not (self._persist_dir / f"{cid}.json").exists()
                ]
                for name in stale:
                    del self._index[name]
                if stale:
                    self._write_index()
                return
            except (json.JSONDecodeError, KeyError):
                self._index.clear()

        # 没有索引文件，从 JSON 文件中重建
        for json_file in self._persist_dir.glob("*.json"):
            if json_file.name == self.INDEX_FILE:
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("name", "")
                conv_id = data.get("id", json_file.stem)
                if name:
                    self._index[name] = conv_id
            except (json.JSONDecodeError, KeyError):
                continue
        self._write_index()

    def _write_index(self) -> None:
        """将索引写入 _index.json。"""
        index_path = self._persist_dir / self.INDEX_FILE
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def _read_created_at(self, conv_id: str) -> str | None:
        """读取某个对话的创建时间。"""
        file_path = self._persist_dir / f"{conv_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("created_at")
            except (json.JSONDecodeError, KeyError):
                pass
        return None
