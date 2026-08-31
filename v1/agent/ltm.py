"""
ltm.py — 长期记忆管理（Long-Term Memory）
======
提供持久化的长期记忆存储，支持双通道检索（关键词 + 语义向量）。

核心能力：
  - 记忆片段存储（内容 + 关键词 + 嵌入向量）
  - 双通道检索：关键词匹配 + 语义相似度
  - LLM 自动提取：从对话中提取值得长期记住的信息
  - JSON 持久化：存储到 LTM_STORE_DIR/ltm.json

使用方式：
  ltm = LongTermMemory()
  ltm.add("用户偏好简短的回答", keywords=["偏好", "简短"])
  results = ltm.retrieve("请简短回答", top_k=3)
  ltm.extract_from_exchange(user_msg, ai_msg, llm)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config import (
    LTM_STORE_DIR,
    LTM_RETRIEVAL_K,
    LTM_SIMILARITY_THRESHOLD,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


# ============================================================
#  MemorySnippet — 长期记忆片段
# ============================================================

@dataclass
class MemorySnippet:
    """一条长期记忆片段。"""
    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    source_turn: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "embedding": self.embedding,
            "source_turn": self.source_turn,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemorySnippet:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            content=data["content"],
            keywords=data.get("keywords", []),
            embedding=data.get("embedding"),
            source_turn=data.get("source_turn", 0),
            timestamp=data.get("timestamp", ""),
        )


# ============================================================
#  LongTermMemory — 长期记忆存储与检索
# ============================================================

class LongTermMemory:
    """
    长期记忆管理器。

    存储：JSON 文件持久化到 LTM_STORE_DIR/ltm.json
    检索：关键词匹配（精确）+ 语义相似度（概念）混合打分

    混合分数 = 0.4 × keyword_score + 0.6 × semantic_score
    """

    KEYWORD_WEIGHT = 0.4
    SEMANTIC_WEIGHT = 0.6

    def __init__(self, store_dir: Path | None = None):
        self._store_dir = Path(store_dir) if store_dir else LTM_STORE_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._snippets: list[MemorySnippet] = []
        self._keyword_index: dict[str, set[int]] = {}  # token → {snippet_index, ...}
        self._embedder = None  # lazy-loaded
        self._load_json()

    # ----- 公共 API -----

    def add(
        self,
        content: str,
        keywords: list[str] | None = None,
        auto_embed: bool = True,
    ) -> str:
        """
        添加一条长期记忆。

        参数:
            content:     记忆内容（一句话）
            keywords:    关键词列表（None 则自动从 content 提取）
            auto_embed:  是否自动生成嵌入向量

        返回:
            str: 新记忆的 ID
        """
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")

        if keywords is None:
            keywords = self._extract_keywords(content)

        snippet = MemorySnippet(
            id=uuid.uuid4().hex[:12],
            content=content,
            keywords=keywords,
            source_turn=0,
        )

        # 自动生成嵌入向量
        if auto_embed:
            try:
                snippet.embedding = self._embed_text(content)
            except Exception:
                pass  # 嵌入失败不阻塞添加

        idx = len(self._snippets)
        self._snippets.append(snippet)

        # 更新关键词索引
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = set()
            self._keyword_index[kw_lower].add(idx)
        # 也从内容中提取 token 建立索引
        for token in self._tokenize(content):
            token_lower = token.lower()
            if token_lower not in self._keyword_index:
                self._keyword_index[token_lower] = set()
            self._keyword_index[token_lower].add(idx)

        self._save_json()
        print(f"[LTM] [ADD] 记忆已存储: {content[:50]}... (id={snippet.id})")
        return snippet.id

    def retrieve(
        self,
        query: str,
        top_k: int = LTM_RETRIEVAL_K,
        threshold: float = LTM_SIMILARITY_THRESHOLD,
    ) -> list[MemorySnippet]:
        """
        双通道检索：关键词 + 语义混合。

        参数:
            query:     查询文本
            top_k:     返回数量
            threshold: 最低相关性分数

        返回:
            list[MemorySnippet]: 按混合分数降序排列
        """
        if not self._snippets:
            return []

        # 通道 1: 关键词匹配
        keyword_scores = self._keyword_search(query)

        # 通道 2: 语义相似度（如果有嵌入向量）
        semantic_scores = self._semantic_search(query)

        # 混合打分
        results = []
        for i, snippet in enumerate(self._snippets):
            k_score = keyword_scores.get(i, 0.0)
            s_score = semantic_scores.get(i, 0.0)

            # 如果有关键词命中但无语义分数，给关键词更高权重
            if k_score > 0 and s_score == 0:
                hybrid = k_score * 0.7
            elif k_score == 0 and s_score > 0:
                hybrid = s_score * 0.8
            else:
                hybrid = self.KEYWORD_WEIGHT * k_score + self.SEMANTIC_WEIGHT * s_score

            if hybrid >= threshold:
                results.append((hybrid, snippet))

        # 按分数降序
        results.sort(key=lambda x: x[0], reverse=True)

        top = [s for _, s in results[:top_k]]
        if top:
            scores_str = ", ".join(
                f"{s.id[:8]}({sc:.2f})"
                for sc, s in results[:top_k]
            )
            print(f"[LTM] [RETRIEVE] {len(top)} 条记忆 (scores: {scores_str})")
        return top

    def extract_from_exchange(
        self,
        user_msg: str,
        ai_msg: str,
        llm: ChatOpenAI,
        turn_number: int = 0,
    ) -> list[str]:
        """
        使用 LLM 从一轮对话中提取值得长期记忆的信息。

        参数:
            user_msg:    用户消息
            ai_msg:      AI 回答
            llm:         LLM 实例
            turn_number: 当前对话轮数

        返回:
            list[str]: 提取出的记忆内容（已自动添加到 LTM）
        """
        prompt = (
            "你是一个知识提取器。请从以下对话中提取值得长期记住的信息。\n"
            "包括：用户偏好、重要结论、关键事实、上下文约束。\n"
            "每条信息不超过一句话。如果没有值得记住的内容，返回空列表。\n"
            "请严格输出 JSON 数组，每个元素是一条信息字符串。\n\n"
            f"用户: {user_msg}\n"
            f"助手: {ai_msg}\n\n"
            '输出示例: ["用户偏好简短回答", "用户正在研究变化检测方向的论文"]'
        )

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            # 提取 JSON 数组
            text = response.content if hasattr(response, "content") else str(response)
            facts = self._parse_json_array(text)
        except Exception:
            return []

        # 将提取的事实添加到 LTM
        added = []
        for fact in facts:
            if fact and fact.strip():
                sid = self.add(
                    content=fact.strip(),
                    keywords=self._extract_keywords(fact),
                    auto_embed=True,
                )
                added.append(sid)

        if added:
            print(f"[LTM] [EXTRACT] 从对话中提取了 {len(added)} 条长期记忆")

        return added

    def list_all(self) -> list[MemorySnippet]:
        """列出所有长期记忆。"""
        return list(self._snippets)

    def delete(self, snippet_id: str) -> bool:
        """删除一条长期记忆。"""
        for i, s in enumerate(self._snippets):
            if s.id == snippet_id:
                self._snippets.pop(i)
                self._rebuild_keyword_index()
                self._save_json()
                print(f"[LTM] [DEL] 已删除记忆: {snippet_id}")
                return True
        print(f"[LTM] [ERR] 未找到记忆: {snippet_id}")
        return False

    def clear(self) -> None:
        """清空所有长期记忆。"""
        self._snippets.clear()
        self._keyword_index.clear()
        self._save_json()
        print("[LTM] [CLEAR] 所有长期记忆已清空")

    def count(self) -> int:
        return len(self._snippets)

    def to_dict(self) -> dict:
        return {
            "snippets": [s.to_dict() for s in self._snippets],
        }

    @classmethod
    def from_dict(cls, data: dict, store_dir: Path | None = None) -> LongTermMemory:
        ltm = cls(store_dir=store_dir)
        ltm._snippets = []
        for s_data in data.get("snippets", []):
            ltm._snippets.append(MemorySnippet.from_dict(s_data))
        ltm._rebuild_keyword_index()
        return ltm

    # ----- 内部方法 -----

    def _keyword_search(self, query: str) -> dict[int, float]:
        """
        关键词检索：基于 token 重叠打分。

        返回:
            dict[int, float]: snippet_index → keyword_score (0~1)
        """
        query_tokens = set(t.lower() for t in self._tokenize(query))
        if not query_tokens:
            return {}

        scores: dict[int, float] = {}
        matched_indices: set[int] = set()

        for token in query_tokens:
            for idx in self._keyword_index.get(token, set()):
                matched_indices.add(idx)

        if not matched_indices:
            return scores

        for idx in matched_indices:
            snippet = self._snippets[idx]
            # 计算 token 重叠率
            snippet_tokens = set(t.lower() for t in self._tokenize(snippet.content))
            snippet_keywords = set(k.lower() for k in snippet.keywords)

            # 与内容 token 的重叠
            content_overlap = query_tokens & snippet_tokens
            # 与关键词的重叠（加权更高）
            keyword_overlap = query_tokens & snippet_keywords

            # 关键词命中权重更高
            content_score = len(content_overlap) / max(len(query_tokens), 1)
            keyword_score = len(keyword_overlap) / max(len(query_tokens), 1)

            scores[idx] = min(1.0, content_score * 0.6 + keyword_score * 1.2)

        return scores

    def _semantic_search(self, query: str) -> dict[int, float]:
        """
        语义检索：基于嵌入向量的余弦相似度。

        返回:
            dict[int, float]: snippet_index → semantic_score (0~1)
        """
        # 检查哪些 snippets 有嵌入向量
        indexed = [
            (i, s) for i, s in enumerate(self._snippets)
            if s.embedding is not None
        ]
        if not indexed:
            return {}

        try:
            query_emb = self._embed_text(query)
            if query_emb is None:
                return {}
        except Exception:
            return {}

        scores = {}
        query_vec = np.array(query_emb)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return {}

        for i, snippet in indexed:
            snippet_vec = np.array(snippet.embedding)
            snippet_norm = np.linalg.norm(snippet_vec)
            if snippet_norm == 0:
                continue
            cosine = float(np.dot(query_vec, snippet_vec) / (query_norm * snippet_norm))
            scores[i] = max(0.0, cosine)

        return scores

    def _embed_text(self, text: str) -> list[float] | None:
        """使用嵌入模型将文本转为向量。"""
        self._ensure_embedder()
        if self._embedder is None:
            return None
        try:
            result = self._embedder.embed_query(text)
            return result
        except Exception:
            return None

    def _ensure_embedder(self):
        """延迟加载嵌入模型。"""
        if self._embedder is not None:
            return
        try:
            from agent.embedder import get_embeddings
            self._embedder = get_embeddings()
            print("[LTM] [LOAD] 嵌入模型已就绪")
        except Exception as e:
            print(f"[LTM] [WARN] 无法加载嵌入模型（语义检索不可用）: {e}")
            self._embedder = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        对文本进行分词（中英文混合）。

        中文：使用字符级 bigram + unigram
        英文/数字：使用空格分词 + 小写
        """
        tokens = []

        # 英文单词
        english_words = re.findall(r"[a-zA-Z0-9]+", text)
        tokens.extend(w.lower() for w in english_words if len(w) >= 2)

        # 中文：提取中文字符
        chinese_text = re.sub(r"[^一-鿿]", "", text)
        if chinese_text:
            # unigram
            tokens.extend(chinese_text)
            # bigram
            for i in range(len(chinese_text) - 1):
                tokens.append(chinese_text[i:i + 2])

        return tokens

    @staticmethod
    def _extract_keywords(content: str, max_kw: int = 5) -> list[str]:
        """从内容中自动提取关键词（基于 token 频率启发式）。"""
        tokens = LongTermMemory._tokenize(content)
        # 取最长的几个 token 作为关键词
        unique = list(dict.fromkeys(tokens))  # 去重保序
        # 优先长 token
        unique.sort(key=lambda t: len(t), reverse=True)
        return unique[:max_kw]

    def _rebuild_keyword_index(self):
        """从 snippets 重建关键词索引。"""
        self._keyword_index.clear()
        for i, snippet in enumerate(self._snippets):
            for kw in snippet.keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._keyword_index:
                    self._keyword_index[kw_lower] = set()
                self._keyword_index[kw_lower].add(i)
            for token in self._tokenize(snippet.content):
                token_lower = token.lower()
                if token_lower not in self._keyword_index:
                    self._keyword_index[token_lower] = set()
                self._keyword_index[token_lower].add(i)

    def _save_json(self):
        """将记忆持久化到 JSON 文件。"""
        file_path = self._store_dir / "ltm.json"
        # 保存时不写入嵌入向量（太大了），而是在加载时惰性计算
        data = {
            "snippets": [
                {
                    "id": s.id,
                    "content": s.content,
                    "keywords": s.keywords,
                    "source_turn": s.source_turn,
                    "timestamp": s.timestamp,
                }
                for s in self._snippets
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_json(self):
        """从 JSON 文件加载记忆。"""
        file_path = self._store_dir / "ltm.json"
        if not file_path.exists():
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._snippets = [
                MemorySnippet(
                    id=s.get("id", uuid.uuid4().hex[:12]),
                    content=s["content"],
                    keywords=s.get("keywords", []),
                    embedding=None,  # 不在 JSON 中存储，需要时惰性生成
                    source_turn=s.get("source_turn", 0),
                    timestamp=s.get("timestamp", ""),
                )
                for s in data.get("snippets", [])
            ]
            self._rebuild_keyword_index()
            print(f"[LTM] [LOAD] 已加载 {len(self._snippets)} 条长期记忆")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[LTM] [WARN] 加载长期记忆失败: {e}")
            self._snippets = []

    @staticmethod
    def _parse_json_array(text: str) -> list[str]:
        """从 LLM 响应中解析 JSON 数组。"""
        # 尝试直接解析
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
        # 尝试提取 [...] 部分
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(x) for x in result]
            except json.JSONDecodeError:
                pass
        return []
