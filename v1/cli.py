"""
cli.py — 命令行接口
======
整个项目的入口。用 python -m web.cli <命令> 启动。

可用命令：
  ingest       — 导入文档到向量数据库
  interactive  — 交互式问答模式（可连续提问）
  status       — 查看系统状态
"""

import argparse
import sys

from agent import ingest, agent_query
from agent.store import store_exists
from agent.memory import BaseMemory, HybridMemory, create_memory
from agent.conversation import ConversationStore
from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    CHROMA_PERSIST_DIR,
    MEMORY_TYPE,
    MEMORY_WINDOW_SIZE,
    MEMORY_PERSIST_DIR,
    LTM_AUTO_EXTRACT,
    LTM_RETRIEVAL_K,
    AGENT_MODE,
    AGENT_TYPE,
    AGENT_MAX_ITERATIONS,
    AGENT_VERBOSE,
    CLARIFY_ENABLED,
    REFLECTION_MAX_ROUNDS,
)


def cmd_ingest(args):
    """处理 'ingest' 子命令"""
    ingest(file_path=args.file, force=args.rebuild)


def cmd_interactive(args):
    """处理 'interactive' 子命令：多轮对话（带记忆管理 + Agent 模式）"""
    # ----- 初始化记忆和持久化 -----
    memory = create_memory(MEMORY_TYPE, MEMORY_WINDOW_SIZE)
    store = ConversationStore(MEMORY_PERSIST_DIR)

    # Agent 模式状态（用 dict 包装以便在 _handle_command 中修改）
    agent_state = {
        "enabled": AGENT_MODE,
        "agent_type": AGENT_TYPE,
    }

    memory_label = f"{MEMORY_TYPE}"
    if MEMORY_TYPE == "window":
        memory_label += f" (最近 {MEMORY_WINDOW_SIZE} 轮)"
    elif MEMORY_TYPE == "hybrid":
        memory_label += f" (STM:{MEMORY_WINDOW_SIZE}轮 + LTM:{LTM_RETRIEVAL_K}条)"

    mode_label = "Agent" if agent_state["enabled"] else "RAG 管道"
    print("=" * 60)
    print(f"  [AI] 科研文献助手 — {mode_label} 模式")
    print(f"  模型: {LLM_MODEL} | 嵌入: {EMBEDDING_MODEL}")
    print(f"  记忆: {memory_label} | LTM自动提取: {'ON' if LTM_AUTO_EXTRACT else 'OFF'}")
    if agent_state["enabled"]:
        print(f"  Agent: {agent_state['agent_type']} | 最大迭代: {AGENT_MAX_ITERATIONS} | "
              f"澄清: {'ON' if CLARIFY_ENABLED else 'OFF'} | "
              f"反思轮数: {REFLECTION_MAX_ROUNDS}")
    print("=" * 60)
    print("  命令:")
    print("    /quit      — 退出（自动保存对话）")
    print("    /new       — 开始新对话")
    print("    /history   — 查看对话历史")
    print("    /save [名称] — 保存对话")
    print("    /load <名称> — 载入对话")
    print("    /list      — 列出已保存的对话")
    print("    /ltm       — 查看长期记忆")
    print("    /ltm-add <内容> — 手动添加长期记忆")
    print("    /state     — 查看当前分析状态")
    print("    /summarize — 强制生成对话摘要")
    print("    /status    — 查看系统状态")
    print("    /agent     — 切换 Agent 模式 ON/OFF")
    print("    /agent-type <react|plan|reflective> — 切换 Agent 类型")
    print("    /clarify   — 手动触发澄清检查")
    print("    /help      — 显示此帮助")
    print("=" * 60)

    if not store_exists():
        print("\n[WARN] 尚未导入文档！请先运行: python -m web.cli ingest\n")

    # ----- 主循环 -----
    while True:
        try:
            user_input = input("\n[Q] 你的问题: ").strip()

            if not user_input:
                continue

            # ---- 命令分发 ----
            if user_input.startswith("/"):
                _handle_command(user_input, memory, store, agent_state)
                continue

            # ---- 澄清检查（Agent 模式下）----
            # ---- 调用 Agent（统一使用 Agent loop）----
            print()
            result = agent_query(
                user_input, memory,
                agent_type=agent_state["agent_type"],
            )

            # 处理 Agent 返回的结构化结果
            if result.status in ("clarify_scope",):
                print(f"\n[CLARIFY] {result.clarification_message}")
                clarification = input("\n[Q] 你的选择 (本地/在线/两者): ").strip()
                if clarification:
                    m = clarification.strip().lower()
                    if any(kw in m for kw in ["在线", "网上", "arxiv", "online"]):
                        scope = "online"
                    elif any(kw in m for kw in ["两者", "都", "混合", "hybrid", "both"]):
                        scope = "hybrid"
                    else:
                        scope = "local"
                    # 携带 pending_plan 恢复，跳过重复的 PLANNING
                    _resume = result.pending_plan
                    if _resume:
                        _resume["retrieval_scope"] = scope
                    result = agent_query(
                        user_input, memory,
                        agent_type=agent_state["agent_type"],
                        retrieval_scope=scope,
                        resume_plan=_resume,
                    )
                    answer = result.answer
                else:
                    answer = result.answer if result.answer else "已取消。"
            else:
                answer = result.answer

            # 更新记忆
            memory.add_user_message(user_input)
            memory.add_ai_message(answer)

            print("\n" + "=" * 60)
            print(answer)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n")
            _auto_save_on_exit(memory, store)
            break


# ============================================================
#  命令处理器
# ============================================================

def _handle_command(raw: str, memory: BaseMemory, store: ConversationStore, agent_state: dict):
    """解析并执行 / 命令。"""
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit":
        _auto_save_on_exit(memory, store)
        print("[BYE] 再见！")
        sys.exit(0)

    elif cmd == "/new" or cmd == "/clear":
        if memory.message_count() > 0:
            choice = input("  当前对话有未保存内容。是否先保存？(y/n/s=保存): ").strip().lower()
            if choice == "s":
                name = input("  保存名称 (回车跳过): ").strip()
                if name:
                    store.save(name, memory)
            elif choice == "n":
                pass
            else:
                # 默认不保存
                pass
        memory.clear()
        print("[OK] 已开始新对话。")

    elif cmd == "/history":
        if memory.message_count() == 0:
            print("[INFO] 对话历史为空。")
        else:
            print("\n" + "=" * 60)
            print(memory.get_history_text())
            print("=" * 60)
            print(f"[INFO] 共 {memory.turn_count()} 轮对话, {memory.message_count()} 条消息")

    elif cmd == "/save":
        name = arg
        if not name:
            name = input("  保存名称: ").strip()
        if not name:
            print("[ERR] 保存名称不能为空。")
        elif memory.message_count() == 0:
            print("[ERR] 对话为空，无法保存。")
        else:
            store.save(name, memory)

    elif cmd == "/load":
        name = arg
        if not name:
            name = input("  加载名称: ").strip()
        if not name:
            print("[ERR] 请指定要加载的对话名称。")
            return
        # 如果有未保存内容，提示
        if memory.message_count() > 0:
            choice = input("  当前对话将被替换，是否继续？(y/n): ").strip().lower()
            if choice != "y":
                print("[CANCEL] 已取消。")
                return
        loaded = store.load(name)
        if loaded is not None:
            # 用加载的消息替换当前 memory
            memory.clear()
            for msg in loaded.get_messages():
                if msg.role == "human":
                    memory.add_user_message(msg.content)
                elif msg.role == "ai":
                    memory.add_ai_message(msg.content)
            print(f"[OK] 已加载对话，共 {memory.turn_count()} 轮。")

    elif cmd == "/list":
        records = store.list_all()
        if not records:
            print("[INFO] 没有已保存的对话。")
        else:
            print("\n" + "=" * 60)
            print(f"  {'名称':<20} {'ID':<14} {'消息':<6} {'类型':<8} {'更新时间'}")
            print("  " + "-" * 56)
            for r in records:
                # 截断过长的字段
                name = r.name[:19] if len(r.name) > 19 else r.name
                updated = r.updated_at[:19] if r.updated_at else ""
                print(f"  {name:<20} {r.id:<14} {r.message_count:<6} "
                      f"{r.memory_type:<8} {updated}")
            print("=" * 60)
            print(f"[INFO] 共 {len(records)} 个已保存的对话")

    elif cmd == "/ltm":
        if not isinstance(memory, HybridMemory):
            print("[ERR] 当前记忆模式不支持长期记忆。请将 MEMORY_TYPE 设为 hybrid。")
        else:
            snippets = memory.long_term.list_all()
            if not snippets:
                print("[INFO] 长期记忆为空。")
            else:
                print("\n" + "=" * 60)
                print(f"  长期记忆 (共 {len(snippets)} 条)")
                print("  " + "-" * 56)
                for s in snippets:
                    print(f"  [{s.id[:8]}] {s.content}")
                    if s.keywords:
                        print(f"         关键词: {', '.join(s.keywords[:5])}")
                print("=" * 60)

    elif cmd == "/ltm-add":
        content = arg
        if not content:
            content = input("  记忆内容: ").strip()
        if not content:
            print("[ERR] 内容不能为空。")
        elif not isinstance(memory, HybridMemory):
            print("[ERR] 当前记忆模式不支持长期记忆。")
        else:
            sid = memory.long_term.add(content)
            print(f"[OK] 已添加长期记忆 (id={sid})")

    elif cmd == "/ltm-del":
        sid = arg
        if not sid:
            sid = input("  记忆ID: ").strip()
        if not sid:
            print("[ERR] 请指定要删除的记忆ID。")
        elif not isinstance(memory, HybridMemory):
            print("[ERR] 当前记忆模式不支持长期记忆。")
        else:
            memory.long_term.delete(sid)

    elif cmd == "/ltm-clear":
        if not isinstance(memory, HybridMemory):
            print("[ERR] 当前记忆模式不支持长期记忆。")
        else:
            choice = input(f"  确认清空所有长期记忆（共 {memory.long_term.count()} 条）？(y/n): ").strip().lower()
            if choice == "y":
                memory.long_term.clear()

    elif cmd == "/state":
        if not isinstance(memory, HybridMemory):
            print("[INFO] 当前记忆模式无状态追踪。")
        else:
            state = memory.get_state()
            if not state:
                print("[INFO] 当前无分析状态。")
            else:
                print("\n" + "=" * 60)
                print("  当前分析状态")
                print("  " + "-" * 56)
                for k, v in state.items():
                    print(f"  {k}: {v}")
                print("=" * 60)

    elif cmd == "/summarize":
        if memory.message_count() == 0:
            print("[INFO] 对话为空，无法生成摘要。")
        else:
            print("[INFO] 正在生成对话摘要...")
            from agent.generator import create_llm
            llm = create_llm()
            memory.summarize(llm)
            summary = memory.get_summary()
            if summary:
                print("\n" + "=" * 60)
                print(summary)
                print("=" * 60)
                print("[OK] 摘要已生成并缓存。")

    elif cmd == "/agent":
        agent_state["enabled"] = not agent_state["enabled"]
        status = "ON" if agent_state["enabled"] else "OFF"
        print(f"[OK] Agent 模式已切换为: {status}")
        if agent_state["enabled"]:
            print(f"     当前 Agent 类型: {agent_state['agent_type']}")

    elif cmd == "/agent-type":
        valid_types = {"react": "react", "plan": "plan_execute", "reflective": "reflective",
                       "plan_execute": "plan_execute"}
        new_type = arg.strip().lower() if arg else ""
        if not new_type:
            print(f"[INFO] 当前 Agent 类型: {agent_state['agent_type']}")
            print(f"  可用类型: react, plan_execute (简写 plan), reflective")
        elif new_type not in valid_types:
            print(f"[ERR] 无效的 Agent 类型: {new_type}")
            print(f"  可用类型: react, plan_execute, reflective")
        else:
            agent_state["agent_type"] = valid_types[new_type]
            print(f"[OK] Agent 类型已切换为: {agent_state['agent_type']}")

    elif cmd == "/clarify":
        if not arg:
            print("[INFO] 使用 /clarify <问题> 来测试澄清功能。")
        else:
            from agent.clarifier import clarify_if_needed
            cr = clarify_if_needed(arg, memory)
            print(f"\n[CLARIFY] 问题: {arg}")
            print(f"  需要澄清: {'是' if cr.needs_clarification else '否'}")
            if cr.needs_clarification:
                print(f"  类型: {cr.vagueness_type}")
                print(f"  澄清追问: {cr.clarification_message}")
                if cr.options:
                    for i, opt in enumerate(cr.options, 1):
                        print(f"    选项{i}: {opt}")

    elif cmd == "/status":
        cmd_status(None)

    elif cmd == "/help":
        print("\n  可用命令:")
        print("    /quit      — 退出（自动保存对话）")
        print("    /new       — 开始新对话")
        print("    /clear     — 同 /new")
        print("    /history   — 查看当前对话历史")
        print("    /save [名称] — 保存当前对话")
        print("    /load <名称> — 载入已保存的对话")
        print("    /list      — 列出所有已保存的对话")
        print("    /ltm       — 查看长期记忆列表")
        print("    /ltm-add <内容> — 手动添加长期记忆")
        print("    /ltm-del <id> — 删除长期记忆")
        print("    /ltm-clear — 清空长期记忆")
        print("    /state     — 查看当前分析状态")
        print("    /summarize — 强制生成对话摘要")
        print("    /status    — 查看系统状态")
        print("    /agent     — 切换 Agent 模式 ON/OFF")
        print("    /agent-type <react|plan|reflective> — 切换 Agent 类型")
        print("    /clarify   — 手动触发澄清检查")
        print("    /help      — 显示此帮助")

    else:
        print(f"[ERR] 未知命令: {cmd}。输入 /help 查看可用命令。")


def _auto_save_on_exit(memory: BaseMemory, store: ConversationStore):
    """退出时自动保存对话（如果有内容）。"""
    if memory.message_count() == 0:
        return
    # 自动保存为 "auto-<时间戳>"
    from datetime import datetime
    name = f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    store.save(name, memory)


def cmd_status(args):
    """处理 'status' 子命令：显示系统状态"""
    exists = store_exists()

    print("\n" + "=" * 60)
    print("  [STATUS] 系统状态")
    print("=" * 60)
    print(f"  向量数据库: {'[OK] 已就绪' if exists else '[ERR] 未创建（运行 ingest 导入文档）'}")
    print(f"  存储位置:   {CHROMA_PERSIST_DIR}")

    if exists:
        try:
            from agent.embedder import get_embeddings
            from agent.store import load_vector_store
            store = load_vector_store(get_embeddings())
            count = store._collection.count()
            print(f"  文档块数量: {count}")
        except Exception:
            pass

    print(f"  LLM 模型:   {LLM_MODEL}")
    print(f"  嵌入模型:   {EMBEDDING_MODEL}")
    print(f"  切分大小:   {CHUNK_SIZE} 字符 (重叠 {CHUNK_OVERLAP})")
    print(f"  检索数量:   Top-{TOP_K}")
    print("=" * 60)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        prog="python -m web.cli",
        description="LangChain + RAG 最小 Demo —— 基于文档的智能问答系统",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ----- ingest 子命令 -----
    parser_ingest = subparsers.add_parser("ingest", help="导入文档到向量数据库")
    parser_ingest.add_argument(
        "--file",
        default=None,
        help=f"文档路径 (默认: data/)",
    )
    parser_ingest.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建向量数据库（即使已存在）",
    )

    # ----- interactive 子命令 -----
    subparsers.add_parser("interactive", help="进入交互式问答模式")

    # ----- status 子命令 -----
    subparsers.add_parser("status", help="查看系统状态")

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n[TIP] 快速开始:")
        print("  1. python -m web.cli ingest          — 导入文档")
        print("  2. python -m web.cli interactive     — 交互模式")
        print("  3. python -m web.cli status          — 查看状态")
        sys.exit(0)

    # 分发到对应的处理函数
    handlers = {
        "ingest": cmd_ingest,
        "interactive": cmd_interactive,
        "status": cmd_status,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
