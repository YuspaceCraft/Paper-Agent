"""
config.py — 中央配置模块
========
加载环境变量，定义所有常量，是整个项目的"设置中心"。

为什么需要这个模块？
  把 API Key、模型名、路径等配置集中管理，其他模块只需 import 就能用。
  如果以后想换模型或调参数，只需改这一个文件。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载 .env 文件中的环境变量
#    load_dotenv() 会从项目根目录的 .env 文件中读取配置
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. 嵌入模型提供方选择
#    "local"     = 本地开源模型（免费、离线、隐私）
#    "dashscope" = 阿里云 DashScope API（需要联网 + API Key）
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

# 3. 本地嵌入模型配置（仅 EMBEDDING_PROVIDER="local" 时生效）
#    BAAI/bge-small-zh-v1.5: 512维, ~100MB, 中文优化, CPU 极快（推荐）
#    BAAI/bge-large-zh-v1.5: 1024维, ~1.3GB, 中文优化, 更准但更慢
#    intfloat/multilingual-e5-small: 384维, ~200MB, 多语言
#    sentence-transformers/all-MiniLM-L6-v2: 384维, ~80MB, 英文为主
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-0.6B"
).strip()

# Reranker 配置（二阶段检索：粗筛 → 精排）
# RERANK_MODEL 可以和 EMBEDDING_MODEL 相同（共享实例，省显存）
# 也可以指定专用 reranker 模型
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "Qwen/Qwen3-Embedding-0.6B"
).strip()
# 粗筛阶段返回的候选数（给 reranker 吃的）
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "20"))
# Reranker 精排后最终返回的数量
RERANK_FINAL_K = int(os.getenv("RERANK_FINAL_K", "5"))

# HuggingFace 镜像（国内下载加速）
# hf-mirror.com 是国内最常用的 HF 镜像站
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip()

# 嵌入设备选择
#   "auto" = 自动检测: CUDA > MPS(Apple) > CPU
#   "cuda" = 强制使用 NVIDIA GPU
#   "cpu"  = 强制使用 CPU
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda").strip().lower()

# 嵌入批次大小（越大吞吐越高，但吃更多显存/内存）
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# 4. DashScope 配置（仅 EMBEDDING_PROVIDER="dashscope" 时生效）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not DASHSCOPE_API_KEY:
    DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = DASHSCOPE_API_KEY  # generator.py 用

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()

# DashScope 嵌入模型名（仅 EMBEDDING_PROVIDER="dashscope" 时生效）
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3").strip()

# 如果选了 dashscope 但没有 API Key，报错
if EMBEDDING_PROVIDER == "dashscope" and not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "[ERR] EMBEDDING_PROVIDER=dashscope 但未找到 API Key！\n"
        "  方式1: 在 .env 中设置 DASHSCOPE_API_KEY=你的key\n"
        "  方式2: 改用本地模型: 在 .env 中设置 EMBEDDING_PROVIDER=local"
    )

# 5. LLM 模型配置（聊天生成用，始终走 DashScope OpenAI 兼容 API）
#    qwen-plus:   性价比最高（推荐）
#    qwen-max:    效果最好但更贵
#    qwen-turbo:  最快最省钱
#    注意：本地 LLM 后续可用 Ollama 等替代，当前保持 DashScope
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus").strip()

# 6. 文档切分参数
#    chunk_size: 每个文本块的最大字符数
#    chunk_overlap: 相邻块之间的重叠字符数（避免语义在边界被切断）
#    NOTE: chunk_size 建议 ≤ 400 字符，确保 cross-encoder reranker
#          能在 512 token 窗口内完整评估每个文档块
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 7. 路径配置
#    Path(__file__).resolve().parent 是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
# DEFAULT_DOC = "knowledge.txt"

# 8. 检索配置
#    top_k: 每次检索返回最相关的 k 个文档片段
TOP_K = 5

# 10. 记忆管理配置（多轮对话）
#     "buffer" = 保留所有消息（无上限，token 消耗会持续增长）
#     "window" = 只保留最近 N 轮对话（控制 token 消耗）
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "hybrid").strip().lower()
if MEMORY_TYPE not in ("buffer", "window", "hybrid"):
    raise ValueError(f"MEMORY_TYPE 必须是 'buffer'、'window' 或 'hybrid'，当前值: {MEMORY_TYPE}")
# window 策略下保留的对话轮数（一轮 = 用户问题 + AI 回答）
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
# 对话持久化目录
MEMORY_PERSIST_DIR = PROJECT_ROOT / os.getenv("MEMORY_PERSIST_DIR", "conversations")

# 长期记忆配置
LTM_AUTO_EXTRACT = os.getenv("LTM_AUTO_EXTRACT", "true").strip().lower() in ("true", "1", "yes")
LTM_RETRIEVAL_K = int(os.getenv("LTM_RETRIEVAL_K", "3"))
LTM_SIMILARITY_THRESHOLD = float(os.getenv("LTM_SIMILARITY_THRESHOLD", "0.5"))
LTM_STORE_DIR = PROJECT_ROOT / os.getenv("LTM_STORE_DIR", "long_term_memory")
# 上下文管理
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
AUTO_SUMMARIZE_TURNS = int(os.getenv("AUTO_SUMMARIZE_TURNS", "8"))
TOKEN_WARNING_RATIO = float(os.getenv("TOKEN_WARNING_RATIO", "0.6"))
# 动态 Top-K（根据查询类型调整检索数量）
FACT_QUERY_TOP_K = int(os.getenv("FACT_QUERY_TOP_K", "3"))
REVIEW_QUERY_TOP_K = int(os.getenv("REVIEW_QUERY_TOP_K", "8"))

# 9. PDF 论文解析配置
#    论文专用的切分参数（比通用文本更大，保持章节语义完整性）
PAPER_CHUNK_SIZE = int(os.getenv("PAPER_CHUNK_SIZE", "800"))
PAPER_CHUNK_OVERLAP = int(os.getenv("PAPER_CHUNK_OVERLAP", "100"))

# PDF 提取后端优先级（按顺序尝试）
#    可选: pymupdf, pdfplumber, pypdf
#    逗号分隔, 例如: "pymupdf,pdfplumber,pypdf"
PDF_EXTRACTION_BACKENDS = os.getenv(
    "PDF_EXTRACTION_BACKENDS",
    "pymupdf,pdfplumber,pypdf"
).strip().split(",")

# 是否将参考文献纳入向量索引
#    参考文献通常噪声大，语义检索价值低，默认关闭
INCLUDE_REFERENCES = os.getenv(
    "INCLUDE_REFERENCES", "false"
).strip().lower() in ("true", "1", "yes")

# 11. Agent 模式配置
#     AGENT_MODE: 是否启用 Agent 模式（true=Agent 自主决策, false=传统 RAG 管道）
AGENT_MODE = os.getenv("AGENT_MODE", "true").strip().lower() in ("true", "1", "yes")
#     AGENT_TYPE: Agent 类型
#       "react"       — ReAct 循环（Thought → Action → Observation，适合大多数查询）
#       "plan_execute"— Plan-Execute 循环（先制定计划再逐步执行，适合复杂多步查询）
#       "reflective"  — ReAct + 反思修正循环（适合需要高质量答案的场景）
AGENT_TYPE = os.getenv("AGENT_TYPE", "react").strip().lower()
if AGENT_TYPE not in ("react", "plan_execute", "reflective"):
    raise ValueError(f"AGENT_TYPE 必须是 'react'、'plan_execute' 或 'reflective'，当前值: {AGENT_TYPE}")
#     AGENT_MAX_ITERATIONS: Agent 最大迭代步数（防止无限循环）
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
#     AGENT_VERBOSE: 是否输出 Agent 中间步骤日志
AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "true").strip().lower() in ("true", "1", "yes")
# 14. LangSmith 可观测性配置
#     LANGSMITH_TRACING_V2: 是否启用 LangSmith 追踪（true 时自动捕获所有 LLM/Tool/Retriever 调用）
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "false").strip().lower() in ("true", "1", "yes")
#     LANGSMITH_API_KEY: LangSmith API Key（在 https://smith.langchain.com 获取）
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
#     LANGSMITH_PROJECT: 项目名，在 LangSmith 面板中分组显示
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "paper-agent").strip()
#     LANGSMITH_ENDPOINT: LangSmith API 地址（默认 SaaS，自托管时改为自己的地址）
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()

# 15. Debug 日志配置
#     DEBUG_LOG_ENABLED: 是否启用结构化 JSON Lines 调试日志
DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG_ENABLED", "true").strip().lower() in ("true", "1", "yes")
#     DEBUG_LOG_DIR: 调试日志输出目录
DEBUG_LOG_DIR = os.getenv("DEBUG_LOG_DIR", str(PROJECT_ROOT / "logs"))
#     DEBUG_LOG_MAX_FILES: 最多保留的日志文件数量（超出后自动清理旧文件）
DEBUG_LOG_MAX_FILES = int(os.getenv("DEBUG_LOG_MAX_FILES", "50"))
#     AGENT_LLM_TEMPERATURE: Agent 决策用的 LLM temperature（0=确定，>0=更多样）
AGENT_LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0"))
# 反思模块配置
REFLECTION_MAX_ROUNDS = int(os.getenv("REFLECTION_MAX_ROUNDS", "2"))
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.3"))
# 澄清模块配置
CLARIFY_ENABLED = os.getenv("CLARIFY_ENABLED", "true").strip().lower() in ("true", "1", "yes")
# Plan-Execute 复杂度阈值：问题长度超过此值或包含复杂关键词时使用 plan_execute
PLAN_EXECUTE_COMPLEXITY_THRESHOLD = int(os.getenv("PLAN_EXECUTE_COMPLEXITY_THRESHOLD", "30"))

# 12. MCP 声明式配置文件路径
MCP_CONFIG_PATH = PROJECT_ROOT / "mcp_config.yaml"

# 13. 在线检索配置
ONLINE_SEARCH_ENABLED = os.getenv("ONLINE_SEARCH_ENABLED", "true").strip().lower() in ("true", "1", "yes")
# 在线检索 Provider（"arxiv" = arXiv MCP Server）
ONLINE_SEARCH_PROVIDER = os.getenv("ONLINE_SEARCH_PROVIDER", "arxiv").strip().lower()
# Hybrid 模式下在线检索返回的最大结果数
HYBRID_ONLINE_TOP_K = int(os.getenv("HYBRID_ONLINE_TOP_K", "5"))

# 14. Skills 系统配置
#     SKILLS_DIR: 技能目录路径
SKILLS_DIR = PROJECT_ROOT / os.getenv("SKILLS_DIR", "skills")
#     SKILLS_ENABLED: 是否启用技能系统
SKILLS_ENABLED = os.getenv("SKILLS_ENABLED", "true").strip().lower() in ("true", "1", "yes")
#     SKILLS_MAX_INSTRUCTIONS_CHARS: 技能指令最大字符数（防止 token 膨胀）
SKILLS_MAX_INSTRUCTIONS_CHARS = int(os.getenv("SKILLS_MAX_INSTRUCTIONS_CHARS", "2000"))


# ================================================================
# LangSmith 初始化（在 import config 时自动执行一次）
# ================================================================

LANGSMITH_TRACING_SAMPLING_RATE = os.getenv(
    "LANGSMITH_TRACING_SAMPLING_RATE", "1.0"
).strip()

def _init_langsmith():
    """初始化 LangSmith 追踪（Agent v2.0）。

    追踪层级（从外到内）：
      1. agent_run (run_type="chain")  — @traceable on UnifiedAgentLoop.run()
         ├── 意图路由 (tags: intent:xxx, scope:xxx)
         └── create_agent → LLM/Tool 调用 (LangChain 自动追踪)

    LangSmith 库自动读取 LANGCHAIN_* 环境变量，但显式调用
    确保追踪钩子在所有 LangChain 组件初始化之前注册。
    """
    if not LANGSMITH_TRACING_V2:
        return

    if not LANGSMITH_API_KEY:
        print("[CONFIG] [WARN] LANGSMITH_TRACING_V2=true 但 LANGSMITH_API_KEY 为空，跳过 LangSmith 初始化")
        return

    try:
        # 显式设置环境变量（langsmith 库内部读取）
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT

        # 采样率（可通过环境变量控制，默认 1.0 = 全量追踪）
        _sample_rate = float(LANGSMITH_TRACING_SAMPLING_RATE)
        if _sample_rate < 1.0:
            os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = str(_sample_rate)

        # 触发 langsmith 的追踪注册（必须在任何 LangChain 组件之前调用）
        import langsmith  # noqa: F401

        _sample_info = f", 采样率={_sample_rate}" if _sample_rate < 1.0 else ""
        print(f"[CONFIG] [OK] LangSmith 追踪已启用 → 项目: {LANGSMITH_PROJECT}, Agent v2.0{_sample_info}")
    except ImportError:
        print("[CONFIG] [WARN] langsmith 包未安装，跳过追踪。安装: pip install langsmith")
    except Exception as e:
        print(f"[CONFIG] [WARN] LangSmith 初始化失败: {e}")


_init_langsmith()
