"""
memory.py — 对话记忆管理
=========
提供可插拔的对话记忆抽象，支持两种策略：
  - BufferMemory: 保留所有历史消息（无上限）
  - WindowMemory: 只保留最近 N 轮对话（控制 token 消耗）

使用方式：
  memory = create_memory("window", window_size=5)
  memory.add_user_message("什么是 RAG?")
  memory.add_ai_message("RAG 是检索增强生成...")
  history_text = memory.get_history_text()  # 格式化的对话历史
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

T = TypeVar("T", bound="BaseMemory")


# ============================================================
#  Message — 单条消息
# ============================================================

@dataclass
class Message:
    """一条对话消息。"""
    role: str            # "human" | "ai"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", ""),
        )


# ============================================================
#  BaseMemory — 抽象基类
# ============================================================

class BaseMemory(ABC):
    """对话记忆的抽象基类。所有记忆策略都继承自此类。"""

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """添加一条用户消息。"""
        ...

    @abstractmethod
    def add_ai_message(self, content: str) -> None:
        """添加一条 AI 消息。"""
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """返回所有当前保留的消息。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆。"""
        ...

    def get_history_text(self) -> str:
        """
        将对话历史格式化为可注入 LLM prompt 的字符串。

        格式：
          [对话历史]
          Q: <用户问题>
          A: <AI 回答>
          Q: <用户问题>
          A: <AI 回答>

        如果没有历史消息，返回空字符串。
        """
        messages = self.get_messages()
        if not messages:
            return ""

        lines = ["[对话历史]"]
        for msg in messages:
            if msg.role == "human":
                lines.append(f"Q: {msg.content}")
            elif msg.role == "ai":
                lines.append(f"A: {msg.content}")
        return "\n".join(lines)

    def message_count(self) -> int:
        """返回当前保留的消息总数。"""
        return len(self.get_messages())

    def turn_count(self) -> int:
        """
        返回对话轮数（一轮 = 一个用户问题 + 一个 AI 回答）。

        未配对的单条用户消息不计为完整轮次。
        """
        human_count = sum(1 for m in self.get_messages() if m.role == "human")
        ai_count = sum(1 for m in self.get_messages() if m.role == "ai")
        return min(human_count, ai_count)

    def estimate_tokens(self) -> int:
        """
        估算当前记忆占用的 token 数。

        启发式：中文 ~1.5 tok/char, 英文 ~1.3 tok/word
        """
        text = self.get_history_text()
        if not text:
            return 0
        # 中文字符
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        # 英文单词
        import re
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return int(chinese_chars * 1.5 + english_words * 1.3 + (len(text) - chinese_chars - english_words * 4) * 0.5)

    def summarize(self, llm) -> str:
        """
        使用 LLM 生成对话的结构化摘要。

        摘要包含：已讨论文献列表、关键结论、待解决问题。
        返回生成的摘要文本，并内部缓存。
        """
        history = self.get_history_text()
        if not history:
            return ""

        prompt = (
            "你是一个对话摘要生成器。请对以下对话生成结构化摘要。\n"
            "摘要应包含：\n"
            "1. 已讨论的论文/文献列表\n"
            "2. 关键结论和要点\n"
            "3. 仍待解决的问题\n"
            "请用中文输出，简洁但完整。不超过 500 字。\n\n"
            f"对话内容：\n{history}"
        )

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            self._cached_summary = text.strip()
            print(f"[MEMORY] [SUMMARY] 已生成对话摘要 ({len(self._cached_summary)} 字符)")
            return self._cached_summary
        except Exception as e:
            print(f"[MEMORY] [WARN] 摘要生成失败: {e}")
            return ""

    def get_summary(self) -> str:
        """返回已缓存的对话摘要（如果没有则返回空字符串）。"""
        return getattr(self, "_cached_summary", "")

    @abstractmethod
    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）。"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseMemory":
        """从字典反序列化。"""
        ...

    @property
    @abstractmethod
    def memory_type(self) -> str:
        """返回记忆策略类型名称。"""
        ...


# ============================================================
#  BufferMemory — 保留所有消息
# ============================================================

class BufferMemory(BaseMemory):
    """
    缓冲区记忆：保留所有对话消息，不做任何截断。

    适合短对话或需要完整上下文的场景。
    注意：长对话可能导致 prompt 超出 LLM token 限制。
    """

    memory_type = "buffer"

    def __init__(self):
        self._messages: list[Message] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(role="human", content=content))

    def add_ai_message(self, content: str) -> None:
        self._messages.append(Message(role="ai", content=content))

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "memory_config": {},
            "messages": [m.to_dict() for m in self._messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BufferMemory:
        memory = cls()
        for msg_data in data.get("messages", []):
            memory._messages.append(Message.from_dict(msg_data))
        return memory


# ============================================================
#  WindowMemory — 滑动窗口（保留最近 N 轮）
# ============================================================

class WindowMemory(BaseMemory):
    """
    窗口记忆：只保留最近 N 轮对话（一轮 = 用户问题 + AI 回答）。

    当消息数超过 2 * max_turns 时，自动丢弃最早的完整轮次。
    这种策略可以控制 token 消耗，适合长对话。

    参数:
        max_turns: 保留的最大对话轮数（默认 5）
    """

    memory_type = "window"

    def __init__(self, max_turns: int = 5):
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self._max_turns = max_turns
        self._messages: list[Message] = []

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(role="human", content=content))

    def add_ai_message(self, content: str) -> None:
        self._messages.append(Message(role="ai", content=content))
        self._trim()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """
        截断超出窗口的消息。

        按轮次计算：找到最早的完整 human+ai 配对的轮次，
        如果轮数超过 max_turns，丢弃最早的轮次。

        注意：末尾可能存在未配对的 human 消息（用户刚提问但 AI 尚未回答），
        这种情况不计入完整轮次，也不会被截断。
        """
        while True:
            pairs = self._find_pairs()
            if len(pairs) <= self._max_turns:
                break
            # 丢弃第一对完整轮次
            first_pair = pairs[0]
            # first_pair 是 (human_index, ai_index)
            # 删除包含这两条消息及它们之间可能存在的消息
            del self._messages[first_pair[0]:first_pair[1] + 1]

    def _find_pairs(self) -> list[tuple[int, int]]:
        """
        找到所有完整的 human+ai 配对。

        返回:
            list[tuple[int, int]]: 每个元素是 (human_index, ai_index)

        算法：从前往后扫描，遇到 human 就找它后面第一个 ai 作为配对。
        如果 human 后面没有 ai（未完成回答），这个 human 不计入。
        """
        pairs = []
        i = 0
        while i < len(self._messages):
            if self._messages[i].role == "human":
                # 找该 human 之后第一个 ai
                for j in range(i + 1, len(self._messages)):
                    if self._messages[j].role == "ai":
                        pairs.append((i, j))
                        i = j + 1
                        break
                else:
                    # 没有找到配对的 ai，结束
                    break
            else:
                i += 1
        return pairs

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "memory_config": {"max_turns": self._max_turns},
            "messages": [m.to_dict() for m in self._messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> WindowMemory:
        max_turns = data.get("memory_config", {}).get("max_turns", 5)
        memory = cls(max_turns=max_turns)
        for msg_data in data.get("messages", []):
            memory._messages.append(Message.from_dict(msg_data))
        return memory


# ============================================================
#  HybridMemory — 短期 + 长期混合记忆
# ============================================================

class HybridMemory(BaseMemory):
    """
    混合记忆：短期记忆（滑动窗口）+ 长期记忆（持久化检索）。

    短期记忆：最近 N 轮对话，处理指代消解和上下文连贯。
    长期记忆：从对话中提取的关键事实，跨会话持久化，支持双通道检索。

    参数:
        stm_window_size: 短期记忆窗口大小（轮数，默认 5）
        ltm_store_dir:   长期记忆存储目录
    """

    memory_type = "hybrid"

    def __init__(
        self,
        stm_window_size: int = 5,
        ltm_store_dir: Path | None = None,
    ):
        from agent.ltm import LongTermMemory

        self._short_term = WindowMemory(max_turns=stm_window_size)
        self._long_term = LongTermMemory(store_dir=ltm_store_dir)
        self._cached_summary: str = ""
        self._state: dict[str, str] = {}

    # ----- STM 代理 -----

    def add_user_message(self, content: str) -> None:
        self._short_term.add_user_message(content)

    def add_ai_message(self, content: str) -> None:
        self._short_term.add_ai_message(content)

    def get_messages(self) -> list[Message]:
        return self._short_term.get_messages()

    def clear(self) -> None:
        self._short_term.clear()
        self._cached_summary = ""
        self._state.clear()

    @property
    def short_term(self) -> WindowMemory:
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        return self._long_term

    # ----- 状态管理 -----

    def update_state(self, key: str, value: str) -> None:
        """更新对话状态（如当前聚焦文献、分析进度等）。"""
        self._state[key] = value

    def get_state(self) -> dict[str, str]:
        """获取当前对话状态。"""
        return dict(self._state)

    def get_state_text(self) -> str:
        """将状态格式化为文本。"""
        if not self._state:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self._state.items())

    # ----- 摘要 -----

    def maybe_summarize(
        self,
        llm,
        max_turns: int = 8,
        max_tokens: int = 8000,
        token_warning_ratio: float = 0.6,
    ) -> bool:
        """
        自动触发摘要的条件：
          1. 对话轮数 > max_turns
          2. 估算 token 占用 > token_warning_ratio * max_tokens

        返回 True 表示执行了摘要。
        """
        turns = self.turn_count()
        tokens = self.estimate_tokens()
        threshold = int(max_tokens * token_warning_ratio)

        need_summarize = (turns > max_turns) or (tokens > threshold)

        if need_summarize and turns > 0:
            print(f"[MEMORY] [AUTO] 触发自动摘要 (turns={turns}, tokens~{tokens})")
            self._cached_summary = self.summarize(llm)
            return True

        return False

    # ----- 上下文组装 -----

    def get_history_text(self) -> str:
        """
        获取完整的记忆上下文文本。

        格式：
          [对话摘要]
          ...
          [近期对话]
          Q: ... / A: ...
        """
        parts = []

        # 摘要（如有）
        summary = self.get_summary()
        if summary:
            parts.append(f"[对话摘要]\n{summary}")

        # 短期对话
        stm_text = self._short_term.get_history_text()
        if stm_text:
            parts.append(stm_text)

        if not parts:
            return ""

        return "\n\n".join(parts)

    def get_context_for_query(self, query: str) -> dict:
        """
        为当前查询准备完整上下文（用于构建 RAG prompt）。

        返回:
            dict with keys: history, summary, long_term_memory, state
        """
        # 检索相关长期记忆
        ltm_facts = self._long_term.retrieve(query)
        ltm_text = ""
        if ltm_facts:
            ltm_lines = ["[相关长期记忆]"]
            for fact in ltm_facts:
                ltm_lines.append(f"- {fact.content}")
            ltm_text = "\n".join(ltm_lines)

        return {
            "history": self._short_term.get_history_text(),
            "summary": self.get_summary(),
            "long_term_memory": ltm_text,
            "state": self.get_state_text(),
        }

    # ----- 序列化 -----

    def to_dict(self) -> dict:
        stm_data = self._short_term.to_dict()
        return {
            "memory_type": self.memory_type,
            "memory_config": {"stm_window_size": self._short_term.max_turns},
            "short_term": {
                "messages": stm_data.get("messages", []),
            },
            "summary": self._cached_summary,
            "state": dict(self._state),
        }

    @classmethod
    def from_dict(cls, data: dict) -> HybridMemory:
        stm_size = data.get("memory_config", {}).get("stm_window_size", 5)
        memory = cls(stm_window_size=stm_size)
        # 恢复短期记忆
        for msg_data in data.get("short_term", {}).get("messages", []):
            memory._short_term._messages.append(Message.from_dict(msg_data))
        # 恢复摘要和状态
        memory._cached_summary = data.get("summary", "")
        memory._state = dict(data.get("state", {}))
        return memory


# ============================================================
#  create_memory — 工厂函数
# ============================================================

def create_memory(
    memory_type: str = "hybrid",
    window_size: int = 5,
    ltm_store_dir: Path | None = None,
) -> BaseMemory:
    """
    创建记忆实例的工厂函数。

    参数:
        memory_type: "buffer" — 保留所有消息
                     "window" — 只保留最近 N 轮
                     "hybrid" — 短期窗口 + 长期持久化记忆（默认推荐）
        window_size: 窗口大小（仅 window / hybrid 类型生效）
        ltm_store_dir: 长期记忆存储目录（仅 hybrid 类型生效）

    返回:
        BaseMemory: 记忆实例

    异常:
        ValueError: memory_type 不在支持列表中
    """
    memory_type = memory_type.strip().lower()
    if memory_type == "buffer":
        return BufferMemory()
    elif memory_type == "window":
        return WindowMemory(max_turns=window_size)
    elif memory_type == "hybrid":
        return HybridMemory(
            stm_window_size=window_size,
            ltm_store_dir=ltm_store_dir,
        )
    else:
        raise ValueError(
            f"不支持的记忆类型: '{memory_type}'。"
            f"可选值: 'buffer', 'window', 'hybrid'"
        )
