"""
intent_router.py — 用户意图统一路由
===============
在问题进入 RAG 管道或 Agent 之前，先做意图分类，决定：
  - 走知识库检索（RAG pipeline / Agent + search_literature）
  - 走文件管理（Agent + MCP 工具）
  - 直接拒绝（超出领域）
  - 需要澄清（模糊问题）
  - 简单回复（闲聊/问候）

设计：
  双层分类 = 快速规则（Tier 1, < 5ms）+ LLM 精判（Tier 2, ~500ms）
  规则命中直接返回，未命中才走 LLM，兼顾速度与准确。

Intent 类型:
  knowledge_retrieval  — 文献检索/对比/综述/事实问答
  file_management      — 论文文件整理/移动/归类/搜索
  out_of_domain        — 超出系统能力范围
  general_chat         — 问候/感谢/闲聊
  clarification_needed — 问题模糊需要追问

使用方式:
  from agent.intent_router import route_intent

  intent = route_intent("对比 DenseNet 和 Change-Agent 的方法", memory)
  # intent.intent_type == "knowledge_retrieval"
  # intent.suggested_pipeline == "agent"
  # intent.suggested_tools == ["search_literature", "compare_papers"]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    DASHSCOPE_BASE_URL,
    CLARIFY_ENABLED,
)

if TYPE_CHECKING:
    from agent.memory import BaseMemory

logger = logging.getLogger(__name__)


# ================================================================
#  Intent — 路由决策结果
# ================================================================

@dataclass
class Intent:
    """意图路由的完整决策结果。"""
    intent_type: str = "knowledge_retrieval"
    # 可选值:
    #   knowledge_retrieval  — 检索论文文献（本地知识库 / 在线检索）
    #   file_management      — 文件操作（整理/移动/归类）
    #   out_of_domain        — 超出领域范围
    #   general_chat         — 问候/闲聊
    #   clarification_needed — 需要追问澄清

    confidence: float = 1.0
    # 0.0 ~ 1.0，规则命中为 1.0，LLM 为模型输出值

    suggested_pipeline: str = "rag"
    # "rag"     = 传统 RAG 管道（本地检索）
    # "agent"   = Agent 模式（本地检索 / 文件管理 / 在线检索）
    # "direct"  = 直接回复（不需要检索/工具）
    # "clarify" = 需要先澄清

    retrieval_scope: str = "local"
    # 检索范围（仅 intent_type=knowledge_retrieval 时有效）:
    #   "local"     — 只查本地已索引论文
    #   "online"    — 只做在线检索（arXiv 等）
    #   "hybrid"    — 本地 + 在线都查
    #   "ambiguous" — 不确定，需要追问用户澄清

    suggested_tools: list[str] = field(default_factory=list)
    # 建议 Agent 加载的工具子集（为空表示用默认工具集）
    # file_management 时: ["search_files", "organize_paper", ...]

    reasoning: str = ""
    # 分类理由（用于调试/日志）

    original_question: str = ""
    # 保留原始问题

    needs_clarification: bool = False
    # 是否需要澄清

    clarification_hint: str = ""
    # 澄清提示文本（供 UI 展示）

    active_skill: str = ""
    # Skills 系统激活的技能名称（空 = 无技能激活）
    # 由 route_intent() 末尾调用 skill_registry.match() 填充


# ================================================================
#  Tier 1: 快速规则匹配
# ================================================================

# —— 文件管理关键词 ——
_FILE_MANAGEMENT_PATTERNS = [
    # 中文 — 文件操作
    r"整理.*(论文|文献|文件|PDF)",
    r"(移动|挪|搬).*(论文|文件|PDF)",
    r"(归类|分类|整理).*(文件|论文|目录)",
    r"(创建|新建|建立).*(目录|文件夹|分类)",
    r"(列出|显示|查看).*(目录|文件夹|分类|文件列表)",
    r"把.*(论文|PDF|文件).*(放到|移到|归类到|整理到|归类|分类|整理|管理)",
    r"将.*(论文|PDF|文件).*(归类|分类|整理|管理|归档)",
    r"(论文|文献|PDF|文件).*(归类|分类|整理|归档|管理)",
    r"按.*(主题|类别|类型|内容|方法).*(归类|分类|整理|管理|归档)",
    r"(归档|归入).*",
    r"(文件|目录|文件夹|论文).*在哪",
    r"有哪些.*(分类|目录|文件夹)",
    r"(搜索|查找|找).*(文件|PDF|论文文件)",
    r"(重命名|改名|删除|清理).*(文件|论文|PDF)",
    # 英文
    r"(organize|move|classify|archive|rename|delete|clean).*(paper|file|pdf|directory)",
    r"(list|show|display).*(directory|folder|file|category)",
    r"(create|mkdir|new).*(directory|folder|category)",
    r"search for.*(file|pdf|paper)",
]

# —— 检索范围模糊关键词 ——
# 这些词暗示用户可能想在线检索，但也可能是本地检索，
# 关键词重叠度太高，不做硬分类，而是触发澄清追问。
_RETRIEVAL_AMBIGUITY_PATTERNS = [
    # 明确提到 arXiv
    r"arxiv",
    r"(检索|搜索|查找).*arxiv",
    # "搜索/找 论文" — 可能是本地也可能是线上
    r"(搜索|搜一下|搜一搜|帮我搜|帮我查|帮我找).*(论文|文献|paper|article)",
    r"(找一下|找找|查找|查一下).*(论文|文献|paper|article)",
    # "网上/在线 搜索/检索"
    r"(网上|在线|联网).*(搜|查|找|检索)",
    # "最新/最近/前沿" — 本地 KB 可能没有
    r"(最新|最近|前沿|近期).*(论文|文献|研究|进展|paper)",
    r"最近.*(发表|发布|出版).*(论文|文献|paper)",
    # "有什么新论文/文献"
    r"有什么.*(新|最新|最近).*(论文|文献|研究|paper)",
    # 英文
    r"(search|find|look.up).*(arxiv|online|web|latest|recent).*(paper|article)",
    r"(latest|recent|new).*(paper|article|research|publication)",
]

# —— 超出领域关键词 ——
_OUT_OF_DOMAIN_PATTERNS = [
    r"天气|天气预报|气温|下雨|晴天",
    r"股票|股价|基金|比特币|加密货币|A股|港股",
    r"新闻|头条|热搜|最新消息",
    r"游戏|攻略|王者荣耀|原神|LOL|英雄联盟|手游",
    r"电影|电视剧|综艺|明星|演员|追剧",
    r"菜谱|做饭|菜怎么做|美食|食谱|烹饪",
    r"旅游|酒店|机票|景点|攻略",
    r"笑话|段子|讲个故事|聊天|解闷",
    r"体育|足球|篮球|NBA|世界杯|奥运会",
    r"音乐|歌曲|歌手|专辑|MV",
    r"手机|电脑|买什么|推荐.*(手机|电脑|车|耳机)",
    r"翻译|帮我写.*(代码|作文|邮件|文案)",
    r"debug|帮我改|帮我调|写一个.*程序",
    r"(你|您).*(是谁|叫什么|能做什么|有什么功能)",
    r"(今天|明天|昨天).*(星期|日期|几号)",
]

# —— 学术关键词（反过滤：命中直接视为 knowledge_retrieval）——
_ACADEMIC_HINTS = [
    r"论文|方法|实验|结果|模型|数据|训练|测试|评估",
    r"网络|架构|损失|优化|参数|精度|召回|准确|F1|mAP",
    r"检测|分割|分类|识别|预测|生成|推理",
    r"文献|研究|学术|综述|对比|分析|总结|概述",
    r"算法|深度学习|机器学习|神经网络|transformer|attention",
    r"NLP|CV|computer.vision|llm|大模型|大语言模型",
    r"embedding|嵌入|向量|检索|RAG|检索增强",
    r"paper|method|experiment|result|model|dataset|benchmark|baseline",
    r"变化检测|语义分割|目标检测|超分辨|change.detection",
    r"arxiv|preprint|conference|journal|ICCV|CVPR|NeurIPS|ICML|ECCV",
    r"这篇|这[个篇种]|该方法|该模型|该算法",
    r"什么是|解释.*(一下|这个)|介绍一下",
    r"有哪些.*(方法|模型|论文|技术|方案)",
    r"区别|差异|优缺点|比较|对比|vs|vs\.|相比",
]

# —— 闲聊/问候关键词 ——
_CHITCHAT_PATTERNS = [
    r"^(你好|hi|hello|hey|早|晚上好|下午好)\b",
    r"^(谢谢|thanks|thank.you|3Q|多谢|辛苦了)\b",
    r"^(在吗|在不在|你好呀|您好)\b",
    r"^(好的|ok|okay|嗯|知道了|明白了|懂了)\b",
    r"^(再见|bye|拜拜|回头见|下次)\b",
    r"^(你能做什么|你有什么功能|你可以做什么|介绍一下你自己)\b",
]

# —— 模糊/需要澄清的模式 ——
_CLARIFICATION_PATTERNS = [
    # 指代模糊 + 无历史 + 问题长度 <= 15 字（防止宽泛模式吃掉完整问题）
    # 使用字符集匹配而非 \S*$，避免中文无空格时全量吞入
    r"^(这[个篇种]|该[方法论文实验])[\w一-鿿]{0,12}$",
    r"^(它|她|他)[的们]?.*(怎么样|如何|是什么)",
    r"^(前面|上面|之前|刚才)[说的]?[\w一-鿿]{0,10}$",
    # 过于宽泛 — 严格限制为无实质内容的短句
    r"^(帮我|帮我看看|分析一下|讲一下|说一下)$",
    r"^(有什么|有哪些)$",
]


def _match_any_pattern(text: str, patterns: list[str]) -> bool:
    """检查文本是否匹配任一正则模式。"""
    text_lower = text.strip().lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def _tier1_classify(question: str, has_history: bool) -> Intent | None:
    """
    快速规则分类（Tier 1）。

    命中规则 → 返回 Intent；未命中 → 返回 None，交给 Tier 2。
    """
    q = question.strip()
    if not q:
        return Intent(
            intent_type="clarification_needed",
            confidence=1.0,
            suggested_pipeline="clarify",
            reasoning="空输入",
            original_question=question,
            needs_clarification=True,
        )

    # 1. 闲聊/问候（最高优先级，避免被后续规则误判）
    if _match_any_pattern(q, _CHITCHAT_PATTERNS):
        return Intent(
            intent_type="general_chat",
            confidence=1.0,
            suggested_pipeline="direct",
            reasoning="Tier1: 命中闲聊/问候模式",
            original_question=question,
        )

    # 2. 学术关键词 → 知识检索
    if _match_any_pattern(q, _ACADEMIC_HINTS):
        # 检查是否也包含文件管理意图（如 "把这些论文整理到变化检测目录"）
        if _match_any_pattern(q, _FILE_MANAGEMENT_PATTERNS):
            return Intent(
                intent_type="file_management",
                confidence=1.0,
                suggested_pipeline="agent",
                suggested_tools=["search_files", "organize_paper", "list_directory"],
                reasoning="Tier1: 命中学术+文件管理混合模式",
                original_question=question,
            )

        # 检查检索范围是否模糊 — 同时命中"找论文"/"最新"/"arXiv"等词
        if _match_any_pattern(q, _RETRIEVAL_AMBIGUITY_PATTERNS):
            return Intent(
                intent_type="knowledge_retrieval",
                confidence=0.8,
                suggested_pipeline="clarify",
                retrieval_scope="ambiguous",
                reasoning="Tier1: 学术关键词 + 检索范围模糊 → 需澄清",
                original_question=question,
                needs_clarification=True,
                clarification_hint=(
                    "你想从哪里检索？\n"
                    "- 📂 **本地知识库**：检索已上传的 PDF 论文\n"
                    "- 🌐 **在线搜索**：从 arXiv 等平台搜索最新文献\n"
                    "- 🔀 **两者都搜**：先查本地，本地不够再在线搜索"
                ),
            )

        return Intent(
            intent_type="knowledge_retrieval",
            confidence=1.0,
            suggested_pipeline="agent",
            retrieval_scope="local",
            reasoning="Tier1: 命中学术关键词（默认本地检索）",
            original_question=question,
        )

    # 3. 文件管理
    if _match_any_pattern(q, _FILE_MANAGEMENT_PATTERNS):
        return Intent(
            intent_type="file_management",
            confidence=1.0,
            suggested_pipeline="agent",
            suggested_tools=["search_files", "organize_paper", "list_directory"],
            reasoning="Tier1: 命中文件管理模式",
            original_question=question,
        )

    # 4. 超出领域
    if _match_any_pattern(q, _OUT_OF_DOMAIN_PATTERNS):
        return Intent(
            intent_type="out_of_domain",
            confidence=1.0,
            suggested_pipeline="direct",
            reasoning="Tier1: 命中非领域关键词",
            original_question=question,
        )

    # 5. 模糊/需要澄清（无历史时）
    if not has_history and _match_any_pattern(q, _CLARIFICATION_PATTERNS):
        return Intent(
            intent_type="clarification_needed",
            confidence=0.9,
            suggested_pipeline="clarify",
            reasoning="Tier1: 命中模糊模式且无历史上下文",
            original_question=question,
            needs_clarification=True,
        )

    # 6. 极短问题（< 6 字且无学术词且无历史）→ 可能需要澄清
    # 先检查是否命中学术关键词，命中则不应判为需澄清
    if len(q) < 6 and not has_history:
        if not _match_any_pattern(q, _ACADEMIC_HINTS):
            return Intent(
                intent_type="clarification_needed",
                confidence=0.7,
                suggested_pipeline="clarify",
                reasoning="Tier1: 极短问题且无历史且无学术词",
                original_question=question,
                needs_clarification=True,
            )
        # 有学术关键词 → 继续下面逻辑（可能会返回 knowledge_retrieval 通过后续处理）

    # 未命中 → Tier 2
    return None


# ================================================================
#  Tier 2: LLM 精判
# ================================================================

INTENT_CLASSIFY_PROMPT = """\
你是一个科研文献助手系统的意图分类器。请分析用户输入，判断其意图类型。

**上下文：**
- 这是一个科研文献检索和管理系统
- 用户可以从两个来源获取论文信息：
  a) **本地知识库**: 用户已上传/已索引的 PDF 论文
  b) **在线平台**: arXiv 等外部学术搜索引擎
- 用户还可以：整理论文文件
- 系统无法：查天气、写代码、翻译、闲聊、回答与学术无关的问题

**意图类型定义：**

1. `knowledge_retrieval` — 检索论文文献信息（包括本地和在线）
   例："Change-Agent 用了什么方法？"、"对比 DenseNet 和 ResNet"
   例："帮我搜索变化检测的最新论文"、"arXiv 上有哪些 transformer 论文"

2. `file_management` — 需要对论文 PDF 文件进行**文件操作**
   例："把这篇论文放到变化检测目录"、"列出所有分类"

3. `out_of_domain` — 与学术论文无关的问题
   例："今天天气怎么样"、"帮我写一段代码"、"推荐一款手机"

4. `general_chat` — 简单的问候、感谢、告别
   例："你好"、"谢谢"、"再见"

5. `clarification_needed` — 问题过于模糊，无法确定用户意图
   例："这个怎么样"（无上下文）、"帮我分析"（未指定分析什么）

**retrieval_scope 字段（仅 knowledge_retrieval 时需要）：**
- `"local"` — 用户的意思是查本地已索引论文（有指向具体论文内容的问题）
- `"online"` — 用户明确想从外部平台搜索论文（提到 arXiv、网上检索、在线搜索、联网搜索等）
- `"hybrid"` — 可能两者都需要
- `"ambiguous"` — 不确定，需要追问用户。典型特征：
  - "搜索XX论文" → 不确定是本地还是线上
  - "最新/最近 的研究/论文" → 本地库可能没有最新论文
  - 问题含"找论文"但本地库也可能是空的

**规则：**
- "这篇论文的方法"、"对比X和Y"、"实验中..." → knowledge_retrieval, scope=local
- "arXiv上"/"网上检索"/"在线搜" → knowledge_retrieval, scope=online
- "搜索/找 XX论文"、"最新论文" → knowledge_retrieval, scope=ambiguous（需要追问）
- 文件操作关键词 → file_management
- 问候 → general_chat
- 无关问题 → out_of_domain

请严格输出 JSON：
{{
  "intent_type": "knowledge_retrieval|file_management|out_of_domain|general_chat|clarification_needed",
  "retrieval_scope": "local|online|hybrid|ambiguous",
  "confidence": 0.0-1.0,
  "reasoning": "简短说明分类理由",
  "needs_clarification": true/false,
  "clarification_hint": "需要追问时的提示文字"
}}

对话历史：{history}

用户输入：{question}

JSON 输出："""


def _create_intent_llm():
    """创建意图分类用的轻量 LLM 实例（temperature=0 确保确定性）。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


def _tier2_classify(question: str, memory: BaseMemory | None = None) -> Intent:
    """
    LLM 精判（Tier 2）。

    仅在 Tier 1 规则无法确定时调用。
    """
    # 获取对话历史
    history_text = ""
    if memory and memory.message_count() > 0:
        try:
            history_text = memory.get_history_text()
            # 截断，防止 prompt 过长
            if len(history_text) > 1500:
                history_text = history_text[-1500:]
        except Exception:
            pass

    prompt = INTENT_CLASSIFY_PROMPT.format(
        history=history_text if history_text else "（无历史，这是首轮对话）",
        question=question,
    )

    try:
        llm = _create_intent_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        data = _parse_json(text)
        intent_type = data.get("intent_type", "knowledge_retrieval")
        confidence = float(data.get("confidence", 0.8))
        reasoning = data.get("reasoning", "Tier2: LLM 分类")
        suggested_tools = data.get("suggested_tools", [])

        # 校验 intent_type
        valid_types = {
            "knowledge_retrieval", "file_management",
            "out_of_domain", "general_chat", "clarification_needed",
        }
        if intent_type not in valid_types:
            intent_type = "knowledge_retrieval"
            reasoning += " (修正: 无效类型 → knowledge_retrieval)"

        # 解析检索范围
        retrieval_scope = data.get("retrieval_scope", "local")
        valid_scopes = {"local", "online", "hybrid", "ambiguous"}
        if retrieval_scope not in valid_scopes:
            retrieval_scope = "local"

        # 是否需要澄清（来自 LLM 判断）
        needs_clarify = (
            intent_type == "clarification_needed"
            or data.get("needs_clarification", False)
            or retrieval_scope == "ambiguous"
        )
        clarification_hint = data.get("clarification_hint", "")
        if retrieval_scope == "ambiguous" and not clarification_hint:
            clarification_hint = (
                "你想从哪里检索？\n"
                "- 📂 **本地知识库**：检索已上传的 PDF 论文\n"
                "- 🌐 **在线搜索**：从 arXiv 等平台搜索最新文献\n"
                "- 🔀 **两者都搜**：先查本地，本地不够再在线搜索"
            )

        # 映射到 pipeline
        pipeline_map = {
            "knowledge_retrieval": "agent",
            "file_management": "agent",
            "out_of_domain": "direct",
            "general_chat": "direct",
            "clarification_needed": "clarify",
        }

        intent = Intent(
            intent_type=intent_type,
            confidence=min(max(confidence, 0.0), 1.0),
            suggested_pipeline=pipeline_map.get(intent_type, "rag"),
            retrieval_scope=retrieval_scope,
            suggested_tools=suggested_tools,
            reasoning=reasoning,
            original_question=question,
            needs_clarification=needs_clarify,
            clarification_hint=clarification_hint,
        )

        logger.info(
            f"[INTENT] Tier2 → {intent_type} (conf={confidence:.2f}) "
            f"pipeline={intent.suggested_pipeline} | {reasoning}"
        )
        return intent

    except Exception as e:
        logger.warning(f"[INTENT] Tier2 LLM 调用失败: {e}，回退默认")
        # 降级：默认走知识检索
        return Intent(
            intent_type="knowledge_retrieval",
            confidence=0.5,
            suggested_pipeline="agent",
            reasoning=f"Tier2: LLM 失败({e})，回退默认",
            original_question=question,
        )


def _parse_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON 对象。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 {...} 部分（包括嵌套）
    match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
    if not match:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ================================================================
#  route_intent — 主入口
# ================================================================

def _apply_skill_match(question: str, intent: Intent) -> None:
    """在意图确定后，检查 Skills 系统是否有匹配的技能。

    如果 SKILLS_ENABLED 且 skill_registry 发现匹配，则设置 intent.active_skill。
    """
    try:
        from config import SKILLS_ENABLED
        if not SKILLS_ENABLED:
            return

        from skills import skill_registry
        matched = skill_registry.match(question, intent.intent_type)
        if matched is not None:
            intent.active_skill = matched.name
            logger.info(
                f"[SKILL] 激活技能: {matched.name} "
                f"(priority={matched.priority}) ← \"{question[:50]}...\""
            )
    except Exception as e:
        # Skills 系统故障不应影响核心路由
        logger.warning(f"[SKILL] 技能匹配失败（已跳过）: {e}")


def route_intent(
    question: str,
    memory: BaseMemory | None = None,
    force_llm: bool = False,
) -> Intent:
    """
    用户意图统一路由（主入口）。

    双层分类：
      Tier 1 — 快速规则匹配（< 5ms），覆盖 ~85% 的场景
      Tier 2 — LLM 精判（~500ms），处理规则无法覆盖的边界情况

    参数:
        question:  用户原始问题
        memory:    对话记忆（用于判断是否有历史上下文）
        force_llm: 强制走 Tier 2 LLM 分类（调试用）

    返回:
        Intent: 包含意图类型、推荐管线、建议工具等

    示例:
        >>> intent = route_intent("对比 Change-Agent 和 DenseNet 的性能")
        >>> intent.intent_type
        'knowledge_retrieval'
        >>> intent.suggested_pipeline
        'agent'

        >>> intent = route_intent("帮我把这些论文按主题整理一下")
        >>> intent.intent_type
        'file_management'
    """
    has_history = memory is not None and memory.message_count() > 0

    # Tier 1: 快速规则
    if not force_llm:
        result = _tier1_classify(question, has_history)
        if result is not None:
            logger.debug(
                f"[INTENT] Tier1 → {result.intent_type} "
                f"(conf={result.confidence:.2f}) | {result.reasoning}"
            )
            # Skills 匹配（在意图确定后）
            _apply_skill_match(question, result)
            return result

    # Tier 2: LLM 精判
    logger.info(f"[INTENT] Tier1 未命中，进入 Tier2 LLM 分类: \"{question[:60]}...\"")
    result = _tier2_classify(question, memory)
    # Skills 匹配（在意图确定后）
    _apply_skill_match(question, result)
    return result


# ================================================================
#  便捷函数
# ================================================================

def is_knowledge_query(question: str, memory: BaseMemory | None = None) -> bool:
    """快速判断是否是知识检索类问题。"""
    intent = route_intent(question, memory)
    return intent.intent_type == "knowledge_retrieval"


def is_file_operation(question: str, memory: BaseMemory | None = None) -> bool:
    """快速判断是否是文件操作类问题。"""
    intent = route_intent(question, memory)
    return intent.intent_type == "file_management"


def get_suggested_tools_for_intent(intent_type: str, retrieval_scope: str = "local") -> list[str]:
    """
    根据意图类型和检索范围返回建议的工具子集。

    用于 Agent 初始化时选择性加载工具，减少 token 消耗。

    参数:
        intent_type:     意图类型
        retrieval_scope: 检索范围 ("local" | "online" | "hybrid")
    """
    if intent_type == "file_management":
        return [
            "list_directory", "create_directory", "move_file",
            "search_files", "get_file_info", "organize_paper",
            "list_paper_categories",
        ]
    elif intent_type == "knowledge_retrieval":
        base_tools = [
            "search_literature", "get_paper_detail", "compare_papers",
            "search_long_term_memory", "get_conversation_context",
            "rewrite_query", "add_to_memory", "get_system_status",
        ]
        if retrieval_scope in ("online", "hybrid"):
            # 在线检索需要 arXiv MCP 工具（名称对应 mcp_simple_arxiv）
            base_tools.extend(["search_papers", "get_paper_data", "get_full_paper_text"])
        return base_tools
    else:
        return []  # 全部工具
