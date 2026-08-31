"""
app.py — 科研文献助手 Web 界面 (Streamlit)
=====
纯 UI 层：渲染界面、管理 session state、调用 agent 层。

不包含业务逻辑 — 所有查询/上传逻辑在 agent/handler.py 中。

运行方式：
  streamlit run web/app.py
"""

# ================================================================
# Windows 修复：必须在 sys.path 被污染前预热导入所有重型第三方库
# ================================================================
import os as _os
_os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st       # noqa: E402
import langchain_text_splitters  # noqa: E402
import sentence_transformers    # noqa: E402
import pyarrow                  # noqa: E402
import pandas                   # noqa: E402
import sklearn                  # noqa: E402

# ================================================================
# 全局崩溃日志
# ================================================================
import sys
import logging
import traceback as _traceback
from pathlib import Path
from datetime import datetime
import io as _io

_CRASH_LOG_PATH = Path(_os.environ.get("TMP", _os.environ.get("TEMP", str(Path.home())))) / "literature_assistant_crash.log"

def _global_exception_handler(exc_type, exc_value, exc_tb):
    buf = _io.StringIO()
    _traceback.print_exception(exc_type, exc_value, exc_tb, file=buf)
    _crash_msg = f"[UNHANDLED] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{buf.getvalue()}"
    try:
        with open(_CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{_crash_msg}{'='*60}\n")
            f.flush()
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _global_exception_handler

if not hasattr(logging.getLogger(), "_crash_handler_installed"):
    _file_handler = logging.FileHandler(
        str(_CRASH_LOG_PATH), mode="a", encoding="utf-8", delay=True,
    )
    _file_handler.setLevel(logging.WARNING)
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log_root = logging.getLogger()
    _log_root.addHandler(_file_handler)
    _log_root._crash_handler_installed = True

logging.getLogger("streamlit").propagate = True

# 项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    MEMORY_WINDOW_SIZE, AGENT_MODE, AGENT_TYPE,
    LLM_MODEL, LOCAL_EMBEDDING_MODEL, RERANK_MODEL,
    EMBEDDING_DEVICE, EMBEDDING_PROVIDER,
    AGENT_MAX_ITERATIONS, REFLECTION_MAX_ROUNDS,
)

# ================================================================
# Streamlit 配置
# ================================================================
st.set_page_config(
    page_title="科研文献助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stChatMessage { padding: 0.5rem 1rem; }
    .upload-section {
        border: 2px dashed #4a90d9; border-radius: 10px;
        padding: 1rem; margin: 1rem 0;
        background-color: rgba(74, 144, 217, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# 初始化 session state
# ================================================================

def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        from agent.memory import create_memory
        st.session_state.memory = create_memory(
            memory_type="hybrid", window_size=MEMORY_WINDOW_SIZE,
        )

    if "store_version" not in st.session_state:
        st.session_state.store_version = 0

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "conversation_name" not in st.session_state:
        st.session_state.conversation_name = "当前对话"

    if "toast" not in st.session_state:
        st.session_state.toast = None

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = AGENT_MODE
        st.session_state.agent_type = AGENT_TYPE

    # 澄清状态：从 agent 返回的待处理澄清
    if "pending_clarify" not in st.session_state:
        st.session_state.pending_clarify = None  # LoopResult or None


_init_session_state()


# ================================================================
# @st.cache_resource — 缓存重型资源
# ================================================================

@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    from agent.embedder import get_embeddings
    return get_embeddings()

@st.cache_resource(show_spinner=False)
def get_cached_llm():
    from agent.generator import create_llm
    return create_llm()

@st.cache_resource(show_spinner=False)
def get_cached_reranker():
    from agent.retriever import Reranker
    return Reranker(model_name=RERANK_MODEL)

@st.cache_resource(show_spinner=False)
def get_cached_vector_store(_version: int):
    from agent.store import load_vector_store, store_exists
    if not store_exists():
        return None
    return load_vector_store(get_cached_embeddings())


# ================================================================
# 侧边栏
# ================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📚 科研文献助手")
        mode_label = "🤖 Agent" if st.session_state.agent_mode else "📡 RAG"
        st.caption(f"{mode_label} 模式 — 智能文献分析系统")
        st.divider()

        render_agent_controls()
        st.divider()
        render_system_status()
        st.divider()
        render_skills_section()
        st.divider()
        render_upload_section()
        st.divider()
        render_conversation_controls()
        st.divider()
        render_model_info()

        if st.session_state.toast:
            st.toast(st.session_state.toast)
            st.session_state.toast = None


def render_agent_controls():
    st.markdown("### 🤖 Agent 设置")
    agent_mode = st.toggle(
        "启用 Agent 模式",
        value=st.session_state.agent_mode,
        help="开启后 AI 自主决策调用工具进行多步检索和分析。",
    )
    if agent_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = agent_mode

    if agent_mode:
        agent_type = st.selectbox(
            "Agent 类型",
            options=["react", "plan_execute", "reflective"],
            index=["react", "plan_execute", "reflective"].index(st.session_state.agent_type)
            if st.session_state.agent_type in ["react", "plan_execute", "reflective"] else 0,
            format_func=lambda x: {
                "react": "ReAct (快速推理)",
                "plan_execute": "Plan-Execute (复杂任务)",
                "reflective": "Reflective (反思修正)",
            }.get(x, x),
        )
        if agent_type != st.session_state.agent_type:
            st.session_state.agent_type = agent_type

        with st.expander("⚙️ 高级设置"):
            st.caption(f"最大迭代次数: {AGENT_MAX_ITERATIONS}")
            st.caption(f"反思轮数: {REFLECTION_MAX_ROUNDS}")
    else:
        st.caption("Agent 模式已关闭，使用传统 RAG 管道")


def render_system_status():
    vs = get_cached_vector_store(st.session_state.store_version)
    chunk_count = 0
    if vs is not None:
        try:
            chunk_count = vs._collection.count()
        except Exception:
            chunk_count = "?"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("向量块", chunk_count)
    with col2:
        st.metric("已上传", len(st.session_state.uploaded_files))
    with col3:
        st.metric("对话轮数", st.session_state.memory.turn_count())

    if vs is not None:
        st.success("🟢 向量数据库就绪")
    else:
        st.warning("🟡 向量数据库为空 — 请上传 PDF")


def render_skills_section():
    """渲染 Skills 系统区域。"""
    st.markdown("### 🎯 技能 (Skills)")

    try:
        from skills import skill_registry
        skills_list = skill_registry.list_all()
    except Exception:
        st.caption("技能系统未启用")
        return

    if not skills_list:
        st.caption("暂无已安装技能。")
        st.caption(f"将技能放入 `skills/` 目录即可自动加载。")
        return

    # 显示技能列表
    for skill in skills_list:
        status_icon = "✅" if skill.enabled else "⛔"
        with st.expander(f"{status_icon} {skill.name}"):
            st.caption(skill.description)
            st.caption(f"版本: {skill.version} | 优先级: {skill.priority}")
            if skill.triggers:
                trigger_patterns = [
                    f"[{t.type}] {t.pattern[:50]}" for t in skill.triggers
                ]
                st.caption(f"触发: {'; '.join(trigger_patterns[:3])}")
            if skill.tool_names:
                st.caption(f"工具: {', '.join(skill.tool_names)}")

    # 刷新按钮
    if st.button("🔄 刷新技能", key="refresh_skills"):
        try:
            from skills import skill_registry
            count = skill_registry.refresh()
            st.toast(f"技能扫描完成，发现 {count} 个技能")
        except Exception:
            st.warning("刷新失败")


def render_upload_section():
    st.markdown("### 📤 上传文献")
    uploaded_files = st.file_uploader(
        "拖拽 PDF 文件到此处",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        new_files = [uf for uf in uploaded_files
                     if uf.name not in st.session_state.uploaded_files]
        if new_files:
            from agent.handler import handle_upload
            for uf in new_files:
                with st.spinner(f"正在处理 {uf.name}..."):
                    result = handle_upload(
                        uf.getbuffer(), uf.name,
                        embeddings=get_cached_embeddings(),
                        vector_store=get_cached_vector_store(st.session_state.store_version),
                    )
                if result.success:
                    st.success(result.message)
                    st.session_state.uploaded_files.append(uf.name)
                else:
                    st.error(result.message)
            st.session_state.store_version += 1
            get_cached_vector_store.clear()
            st.rerun()

    if st.session_state.uploaded_files:
        st.markdown("**已索引的文件:**")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"📄 `{fname}`")

        if st.button("🗑️ 清空数据库", use_container_width=True, type="secondary"):
            st.session_state.confirm_clear = True

        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ 此操作将删除所有向量数据！")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("确认清空", use_container_width=True):
                    import shutil
                    from config import CHROMA_PERSIST_DIR
                    shutil.rmtree(str(CHROMA_PERSIST_DIR), ignore_errors=True)
                    st.session_state.uploaded_files = []
                    st.session_state.store_version += 1
                    get_cached_vector_store.clear()
                    st.session_state.confirm_clear = False
                    st.rerun()
            with c2:
                if st.button("取消", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()


def render_conversation_controls():
    st.markdown("### 💬 对话管理")

    if st.button("🆕 新建对话", use_container_width=True):
        if st.session_state.memory.message_count() > 0:
            _auto_save_conversation()
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.conversation_name = "当前对话"
        st.rerun()

    c1, c2 = st.columns([3, 1])
    with c1:
        conv_name = st.text_input(
            "对话名称", value=st.session_state.conversation_name,
            key="conv_name_input", label_visibility="collapsed",
            placeholder="对话名称...",
        )
        st.session_state.conversation_name = conv_name
    with c2:
        if st.button("💾", help="保存对话", use_container_width=True):
            if st.session_state.memory.message_count() > 0:
                _save_conversation(conv_name)
            else:
                st.warning("对话为空，无法保存")

    st.markdown("**已保存的对话:**")
    from agent.conversation import ConversationStore
    from config import MEMORY_PERSIST_DIR
    store = ConversationStore(MEMORY_PERSIST_DIR)
    records = store.list_all()
    if not records:
        st.caption("(暂无已保存的对话)")
    else:
        for r in records[:10]:
            c1, c2, c3 = st.columns([5, 1, 1])
            with c1:
                if st.button(f"{r.name} ({r.message_count}条)", key=f"load_{r.id}",
                             use_container_width=True):
                    _load_conversation(store, r.id)
                    st.rerun()
            with c3:
                if st.button("🗑️", key=f"del_{r.id}", help=f"删除: {r.name}"):
                    store.delete(r.id)
                    st.rerun()


def _save_conversation(name: str):
    from agent.conversation import ConversationStore
    from config import MEMORY_PERSIST_DIR
    store = ConversationStore(MEMORY_PERSIST_DIR)
    store.save(name, st.session_state.memory)
    st.session_state.toast = f"✅ 已保存: {name}"


def _auto_save_conversation():
    _save_conversation(f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def _load_conversation(store, conv_id: str):
    loaded = store.load(conv_id)
    if loaded:
        st.session_state.memory.clear()
        for msg in loaded.get_messages():
            if msg.role == "human":
                st.session_state.memory.add_user_message(msg.content)
                st.session_state.messages.append({"role": "user", "content": msg.content})
            elif msg.role == "ai":
                st.session_state.memory.add_ai_message(msg.content)
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
        st.session_state.toast = f"✅ 已加载对话 (共 {loaded.turn_count()} 轮)"


def render_model_info():
    st.markdown("### ⚙️ 模型配置")
    st.markdown(f"""
| 组件 | 配置 |
|------|------|
| 嵌入模型 | `{LOCAL_EMBEDDING_MODEL}` |
| 精排模型 | `{RERANK_MODEL}` |
| 生成模型 | `{LLM_MODEL}` |
| 设备 | `{EMBEDDING_DEVICE}` |
| 模式 | `{EMBEDDING_PROVIDER}` |
| Agent | {'ON' if AGENT_MODE else 'OFF'} ({AGENT_TYPE}) |
""")

    try:
        from mcphub import registry
        status = registry.get_status()
        st.markdown("### 🔧 MCP 工具")
        with st.container():
            src_label = "📄 YAML" if status["config_source"] == "mcp_config.yaml" else "📝 .env"
            st.caption(f"配置来源: {src_label}  |  工具总数: {status['total_tools']}")
            for srv_name, srv_info in status["servers"].items():
                if srv_info.get("enabled"):
                    if srv_info["status"] == "ok":
                        st.caption(f"  ✅ `{srv_name}` — {srv_info['tool_count']} 个工具")
                    elif "error" in srv_info.get("status", ""):
                        err = srv_info.get("error", srv_info["status"])
                        st.caption(f"  ⚠️ `{srv_name}` — {err[:60]}")
                else:
                    st.caption(f"  ⏸️ `{srv_name}` — 已禁用")
            if st.button("🔄 重载 MCP", use_container_width=True):
                registry.refresh()
                st.rerun()
    except Exception as e:
        st.markdown("### 🔧 MCP 工具")
        st.caption(f"⚠️ MCP 初始化失败: {e}")


# ================================================================
# 主区域
# ================================================================

def _format_doc_for_display(doc, index: int) -> str:
    meta = doc.metadata
    filename = meta.get("filename", "未知")
    section = meta.get("section_name", "")
    paper_title = meta.get("paper_title", "")
    source_parts = []
    if meta.get("chunk_type") == "paper":
        if section:
            source_parts.append(f"章节: {section}")
        if paper_title:
            source_parts.append(f"论文: {paper_title[:60]}")
    source_parts.append(f"文件: {filename}")
    source = " | ".join(source_parts)
    content = doc.page_content
    if len(content) > 500:
        content = content[:500] + "..."
    return f"**块 {index}** — {source}\n\n```\n{content}\n```"


def render_chat_history():
    """渲染聊天历史。每条 assistant 消息包含可折叠的操作步骤（树形结构）+ 检索上下文。"""
    import re as _re

    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            # Assistant 消息：操作步骤（可折叠）+ 最终回复
            if role == "assistant":
                # 操作步骤（树形结构，可收缩）
                steps = msg.get("steps", [])
                if steps:
                    with st.expander("🔍 查看 Agent 操作流程", expanded=False):
                        _current_phase = ""
                        _current_step = 0
                        for phase_id, message in steps:
                            # 阶段切换 → 渲染阶段标题
                            if phase_id != _current_phase:
                                _current_phase = phase_id
                                _current_step = 0
                                emoji = {"planning": "🔍", "executing": "⚙️",
                                         "reflecting": "🔬", "responding": "✅",
                                         "waiting": "❓"}.get(phase_id, "•")
                                phase_label = {"planning": "任务规划", "executing": "任务执行",
                                              "reflecting": "反思修正", "responding": "完成回复",
                                              "waiting": "等待输入"}.get(phase_id, phase_id)
                                st.markdown(f"**{emoji} {phase_label}**")

                            # 检测步骤编号 [N/M]
                            step_match = _re.search(r'\[(\d+)/(\d+)\]', message)
                            msg_clean = message.strip()

                            if step_match and phase_id == "executing":
                                idx = int(step_match.group(1))
                                total = int(step_match.group(2))
                                _current_step = idx
                                connector = "└─" if idx == total else "├─"
                                st.markdown(f"&nbsp;&nbsp;**{connector} 步骤{idx}/{total}** "
                                           f"{message[message.find(']')+1:].strip()[:100]}",
                                           unsafe_allow_html=True)
                            elif _current_step > 0 and phase_id == "executing":
                                # 工具结果：缩进在步骤下方
                                is_result = any(msg_clean.startswith(p)
                                              for p in ["✅", "❌", "⚠️", "ℹ️", "📁", "📄", "📂", "🔍"])
                                indent = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                                if is_result:
                                    st.markdown(f"{indent}{msg_clean[:150]}", unsafe_allow_html=True)
                                else:
                                    st.caption(f"{indent}{msg_clean[:150]}", unsafe_allow_html=True)
                            else:
                                # 非执行阶段的消息
                                st.caption(f"   {msg_clean[:150]}")
                # 最终回复
                st.markdown(msg["content"])
                # Agent 执行树（可折叠）
                if msg.get("execution_tree"):
                    _render_execution_tree(msg["execution_tree"])
                # 调试日志路径
                if msg.get("debug_log_path"):
                    st.caption(f"📋 调试日志: `{msg['debug_log_path']}`")
                # 检索上下文（可折叠）
                if msg.get("retrieved_docs"):
                    with st.expander("📎 查看检索上下文", expanded=False):
                        docs = msg["retrieved_docs"]
                        st.caption(f"共检索到 {len(docs)} 个相关文档块")
                        for j, doc in enumerate(docs, 1):
                            st.markdown(_format_doc_for_display(doc, j))
                        sources = set()
                        for d in docs:
                            fn = d.metadata.get("filename", "?")
                            sec = d.metadata.get("section_name", "")
                            if sec:
                                sources.add(f"{fn} ({sec})")
                            else:
                                sources.add(fn)
                        st.caption(f"来源: {', '.join(sorted(sources))}")
            else:
                # 用户消息：直接渲染
                st.markdown(msg["content"])


def render_main():
    st.markdown("### 💬 科研文献助手")

    vs = get_cached_vector_store(st.session_state.store_version)
    if vs is None and not st.session_state.pending_query:
        st.info("👋 欢迎使用科研文献助手！请先在左侧上传 PDF 文献，然后开始提问。")
        st.markdown("""
        **功能亮点:**
        - 📄 上传 PDF 论文自动解析
        - 🔍 二阶段检索 + Agent 自主决策
        - 🧠 混合记忆 (短期窗口 + 长期持久化)
        - 📊 结构化回答 (文献引用标注)
        """)
        render_chat_history()
        return

    # ---- 1. 先渲染已有聊天历史（保证用户消息立即可见）----
    render_chat_history()

    # ---- 2. 用户输入 ----
    chat_placeholder = "🤖 输入你的问题..." if st.session_state.agent_mode else "请输入你的问题..."
    if prompt := st.chat_input(chat_placeholder, key="chat_input"):
        # 立刻持久化用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_query = prompt
        st.rerun()

    # ---- 3. 处理待定查询（在已渲染的历史下方新增 assistant 回复）----
    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None

        # 收集 agent 步骤
        agent_steps: list[tuple[str, str]] = []

        with st.chat_message("assistant"):
            # Agent 操作流程 — 实时展示（处理中展开，完成后存入历史变为折叠）
            with st.status("🤖 Agent 正在处理...", expanded=True) as status_container:
                from agent.handler import handle_query

                current_phase = [""]
                current_step = [{"index": 0, "total": 0}]  # 当前子步骤追踪
                phase_messages: dict[str, list[str]] = {
                    "planning": [], "executing": [], "reflecting": [],
                    "responding": [], "waiting": [],
                }

                def _on_progress(phase_id: str, message: str):
                    """进度回调：按阶段+子步骤分层，实时渲染到 status 容器。"""
                    import re as _re2
                    agent_steps.append((phase_id, message))
                    phase_messages.setdefault(phase_id, []).append(message)

                    msg_stripped = message.strip()

                    # ---- 1. 阶段切换 → 先重置步骤计数器，再渲染阶段标题 ----
                    if phase_id != current_phase[0]:
                        current_phase[0] = phase_id
                        current_step[0] = {"index": 0, "total": 0}
                        phase_emoji = {"planning": "🔍", "executing": "⚙️",
                                       "reflecting": "🔬", "responding": "✅",
                                       "waiting": "❓"}.get(phase_id, "•")
                        phase_label = {"planning": "任务规划", "executing": "任务执行",
                                       "reflecting": "反思修正", "responding": "完成回复",
                                       "waiting": "等待输入"}.get(phase_id, phase_id)
                        status_container.markdown(
                            f"---\n{phase_emoji} **{phase_label}**"
                        )

                    # ---- 2. 检测步骤编号 [N/M]（在阶段切换之后，确保计数器不被重置）----
                    step_match = _re2.search(r'\[(\d+)/(\d+)\]', message)
                    is_step_header = bool(step_match and phase_id == "executing")
                    is_tool_result = (
                        phase_id == "executing"
                        and not is_step_header
                        and any(msg_stripped.startswith(p)
                               for p in ["✅", "❌", "⚠️", "ℹ️", "📁", "📄", "📂",
                                         "🔍", "🗑️", "✏️", "📋", "[OK]", "[ERR]"])
                    )

                    if is_step_header:
                        current_step[0] = {"index": int(step_match.group(1)),
                                          "total": int(step_match.group(2))}

                    # ---- 3. 按消息类型分层渲染 ----
                    if is_step_header:
                        # 子步骤标题：带树形连接符
                        idx = current_step[0]["index"]
                        total = current_step[0]["total"]
                        connector = "└─" if idx == total else "├─"
                        step_desc = message[message.find(']')+1:].strip()[:120] if ']' in message else message
                        status_container.markdown(
                            f"**{connector} 步骤{idx}/{total}** {step_desc}",
                            unsafe_allow_html=True,
                        )

                    elif is_tool_result:
                        # 工具执行结果：缩进在步骤下方
                        indent = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        if msg_stripped.startswith("❌") or msg_stripped.startswith("[ERR]"):
                            status_container.markdown(f"{indent}🔴 {msg_stripped[:200]}", unsafe_allow_html=True)
                        elif msg_stripped.startswith("⚠️") or "警告" in msg_stripped:
                            status_container.markdown(f"{indent}🟡 {msg_stripped[:200]}", unsafe_allow_html=True)
                        elif msg_stripped.startswith("✅") or msg_stripped.startswith("[OK]"):
                            status_container.markdown(f"{indent}🟢 {msg_stripped[:200]}", unsafe_allow_html=True)
                        else:
                            status_container.markdown(f"{indent}└ {msg_stripped[:200]}", unsafe_allow_html=True)

                    else:
                        # 非执行阶段消息（planning / reflecting / responding）
                        if msg_stripped.startswith("❌") or msg_stripped.startswith("[ERR]"):
                            status_container.error(f"{msg_stripped[:200]}")
                        elif msg_stripped.startswith("⚠️") or "警告" in msg_stripped:
                            status_container.warning(f"{msg_stripped[:200]}")
                        elif msg_stripped.startswith("✅") or msg_stripped.startswith("[OK]"):
                            status_container.success(f"{msg_stripped[:200]}")
                        else:
                            status_container.write(f"   {msg_stripped[:200]}")

                result = handle_query(
                    prompt,
                    st.session_state.memory,
                    agent_type=st.session_state.agent_type,
                    embeddings=get_cached_embeddings(),
                    reranker=get_cached_reranker(),
                    vector_store=vs,
                    progress_callback=_on_progress,
                )

                status_container.update(label="✅ 完成", state="complete", expanded=False)

            # ---- 渲染执行树（status 容器外） ----
            execution_tree = _build_execution_tree(agent_steps)
            if execution_tree:
                _render_execution_tree(execution_tree)
            if getattr(result, "debug_log_path", ""):
                st.caption(f"📋 调试日志: `{result.debug_log_path}`")

            # ---- 按 result.status 渲染回复 ----
            if result.status == "clarify_scope":
                st.markdown(f"💡 **{result.clarification_message}**")
                scope_options = [
                    ("📂 本地知识库", "local"),
                    ("🌐 在线搜索(arXiv)", "online"),
                    ("🔀 两者都搜", "hybrid"),
                ]
                cols = st.columns(len(scope_options))
                for i, ((label, scope_val), col) in enumerate(zip(scope_options, cols)):
                    with col:
                        if st.button(label, key=f"scope_{i}"):
                            with st.spinner("正在重新查询..."):
                                # 从澄清结果中提取已确认的计划，注入用户选择的 scope
                                _resume = result.pending_plan
                                if _resume:
                                    _resume["retrieval_scope"] = scope_val
                                result2 = handle_query(
                                    prompt, st.session_state.memory,
                                    agent_type=st.session_state.agent_type,
                                    retrieval_scope=scope_val,
                                    embeddings=get_cached_embeddings(),
                                    reranker=get_cached_reranker(),
                                    vector_store=vs,
                                    resume_plan=_resume,
                                )
                            _finalize_answer(prompt, result2, agent_steps)
                            st.rerun()
                # 不持久化澄清状态，等待用户选择
                return
            else:
                # 正常回复：持久化到 messages
                _finalize_answer(prompt, result, agent_steps)

        st.rerun()


def _finalize_answer(prompt: str, result, agent_steps: list | None = None):
    """将 agent 结果持久化到 messages 和 memory。"""
    execution_tree = _build_execution_tree(agent_steps or [])
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "steps": agent_steps or [],          # agent 操作步骤
        "execution_tree": execution_tree,    # 结构化执行树
        "retrieved_docs": result.retrieved_docs,
        "debug_log_path": getattr(result, "debug_log_path", ""),
    })
    st.session_state.memory.add_user_message(prompt)
    st.session_state.memory.add_ai_message(result.answer)


def _build_execution_tree(agent_steps: list) -> list[dict]:
    """将扁平的 agent_steps 列表转换为结构化执行树。"""
    if not agent_steps:
        return []

    phases_order = ["planning", "executing", "reflecting", "responding", "waiting"]
    phase_labels = {
        "planning": "任务规划", "executing": "任务执行",
        "reflecting": "反思修正", "responding": "完成回复",
        "waiting": "等待输入",
    }
    phase_emojis = {
        "planning": "🔍", "executing": "⚙️",
        "reflecting": "🔬", "responding": "✅",
        "waiting": "❓",
    }

    # 按阶段分组
    phase_groups: dict[str, list] = {}
    for step in agent_steps:
        if isinstance(step, tuple):
            phase_id, message = step
            phase_groups.setdefault(phase_id, []).append(message)
        elif isinstance(step, dict):
            phase_id = step.get("phase", "unknown")
            phase_groups.setdefault(phase_id, []).append(step)

    tree = []
    for phase_id in phases_order:
        if phase_id not in phase_groups:
            continue
        messages = phase_groups[phase_id]

        phase_node = {
            "phase": phase_id,
            "phase_label": phase_labels.get(phase_id, phase_id),
            "emoji": phase_emojis.get(phase_id, "•"),
            "items": [],
            "steps": [],
        }

        if phase_id == "executing":
            # 解析执行步骤：检测 [N/M] 模式和工具结果
            current_step = None
            for msg in messages:
                if isinstance(msg, dict):
                    phase_node["steps"].append(msg)
                    continue
                import re as _re3
                step_match = _re3.search(r'\[(\d+)/(\d+)\]', msg)
                if step_match:
                    if current_step:
                        phase_node["steps"].append(current_step)
                    idx = int(step_match.group(1))
                    total = int(step_match.group(2))
                    desc = msg[msg.find(']')+1:].strip()[:120] if ']' in msg else msg
                    current_step = {
                        "index": idx,
                        "total": total,
                        "description": desc,
                        "tool_results": [],
                    }
                elif current_step is not None:
                    # 工具结果行
                    msg_stripped = msg.strip()
                    is_result = any(msg_stripped.startswith(p)
                                  for p in ["✅", "❌", "⚠️", "ℹ️", "📁", "📄", "📂", "🔍"])
                    if is_result or msg_stripped:
                        current_step["tool_results"].append({
                            "message": msg_stripped[:300],
                            "is_error": msg_stripped.startswith("❌") or msg_stripped.startswith("[ERR]"),
                            "is_warning": msg_stripped.startswith("⚠️") or "警告" in msg_stripped,
                            "is_success": msg_stripped.startswith("✅"),
                        })
                else:
                    phase_node["items"].append({"type": "info", "content": str(msg)[:200]})
            if current_step:
                phase_node["steps"].append(current_step)

            # 如果解析不到步骤，回退为 info items
            if not phase_node["steps"]:
                for msg in messages:
                    content = msg if isinstance(msg, str) else msg.get("message", str(msg))
                    phase_node["items"].append({"type": "info", "content": str(content)[:200]})
        else:
            # 非执行阶段：归类为反思分数、问题、信息
            for msg in messages:
                content = msg if isinstance(msg, str) else msg.get("message", str(msg))
                content_str = str(content)[:300]

                # 检测反思评分行
                if "忠实度" in content_str and "完整性" in content_str:
                    phase_node["items"].append({"type": "score", "content": content_str})
                elif content_str.startswith("⚠️ 问题:") or content_str.startswith("⚠️"):
                    phase_node["items"].append({"type": "issue", "content": content_str})
                elif "修正" in content_str:
                    phase_node["items"].append({"type": "correction", "content": content_str})
                else:
                    phase_node["items"].append({"type": "info", "content": content_str})

        tree.append(phase_node)

    return tree


def _render_execution_tree(tree: list[dict]):
    """渲染可展开的执行树。"""
    if not tree:
        return

    with st.expander("🔍 查看Agent执行详情", expanded=False):
        for phase_node in tree:
            emoji = phase_node["emoji"]
            label = phase_node["phase_label"]

            # 计算阶段内的条目数量
            step_count = len(phase_node.get("steps", []))
            item_count = len(phase_node.get("items", []))
            total_count = step_count + item_count
            if total_count == 0:
                continue

            with st.expander(f"{emoji} {label} ({total_count} 项)", expanded=False):
                # 渲染执行步骤
                for step in phase_node.get("steps", []):
                    idx = step.get("index", 0)
                    total = step.get("total", 0)
                    desc = step.get("description", "")
                    step_label = f"步骤{idx}/{total}: {desc[:100]}"

                    # 统计工具结果中的状态
                    results = step.get("tool_results", [])
                    err_count = sum(1 for r in results if r.get("is_error"))
                    ok_count = sum(1 for r in results if r.get("is_success"))
                    status_icon = "✅" if err_count == 0 and ok_count > 0 else ("❌" if err_count > 0 else "•")

                    with st.expander(f"{status_icon} {step_label}", expanded=False):
                        for r in results:
                            msg = r.get("message", "")
                            if r.get("is_error"):
                                st.error(msg[:300])
                            elif r.get("is_warning"):
                                st.warning(msg[:300])
                            elif r.get("is_success"):
                                st.success(msg[:300])
                            else:
                                st.caption(msg[:300])

                # 渲染阶段内的信息条目
                for item in phase_node.get("items", []):
                    content = item.get("content", "")
                    item_type = item.get("type", "info")

                    if item_type == "score":
                        # 尝试提取分数并渲染进度条
                        import re as _re4
                        f_match = _re4.search(r'忠实度[：:]\s*([\d.]+)', content)
                        c_match = _re4.search(r'完整性[：:]\s*([\d.]+)', content)
                        a_match = _re4.search(r'准确性[：:]\s*([\d.]+)', content)
                        if f_match or c_match or a_match:
                            cols = st.columns(3)
                            if f_match:
                                v = float(f_match.group(1))
                                if v > 1.0:
                                    v = v / 100.0  # 百分比→小数
                                cols[0].metric("忠实度", f"{v:.0%}")
                                cols[0].progress(min(max(v, 0.0), 1.0))
                            if c_match:
                                v = float(c_match.group(1))
                                if v > 1.0:
                                    v = v / 100.0
                                cols[1].metric("完整性", f"{v:.0%}")
                                cols[1].progress(min(max(v, 0.0), 1.0))
                            if a_match:
                                v = float(a_match.group(1))
                                if v > 1.0:
                                    v = v / 100.0
                                cols[2].metric("准确性", f"{v:.0%}")
                                cols[2].progress(min(max(v, 0.0), 1.0))
                        else:
                            st.caption(content[:300])
                    elif item_type == "issue":
                        st.warning(content[:300])
                    elif item_type == "correction":
                        st.info(content[:300])
                    else:
                        st.caption(content[:300])


# ================================================================
# 入口
# ================================================================

def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
