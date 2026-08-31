"""
paper_review/tools.py — 论文审阅技能专用工具

提供两个工具：
  extract_section          — 从已索引论文中提取特定章节内容
  generate_review_checklist — 生成论文审阅检查清单

工具返回结构化字符串，便于 Agent 解析。
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def extract_section(paper_title: str, section_name: str) -> str:
    """
    从已索引的论文中提取特定章节的内容。

    适用场景:
      - 用户想查看论文的"方法论"部分
      - 用户想确认论文的实验设置
      - 审阅时需要精读某个特定章节

    :param paper_title:  论文标题（支持模糊匹配）
    :param section_name: 章节名称，如 "method" / "experiment" / "conclusion" / "abstract"
    :return: 提取的章节文本（截断至 2000 字符）
    """
    # 通过 _ToolContext 访问向量库
    from agent.tools import _ctx

    if _ctx.vector_store is None:
        return "[ERR] 向量数据库未初始化，请先上传 PDF 论文"

    # 构造章节查询：论文标题 + 章节名
    section_keywords = {
        "abstract": "abstract 摘要",
        "introduction": "introduction 引言 介绍 背景",
        "method": "method methodology 方法 方法论 提出 模型 架构",
        "experiment": "experiment 实验 结果 评估 evaluation",
        "conclusion": "conclusion 结论 总结",
        "related": "related work 相关工作 文献综述",
    }

    # 获取章节关键词映射
    section_query = section_keywords.get(
        section_name.lower().strip(),
        section_name  # 未知章节直接用原名
    )

    # 组合查询
    query = f"{paper_title} {section_query}"

    try:
        results = _ctx.vector_store.similarity_search(query, k=5)
    except Exception as e:
        return f"[ERR] 检索失败: {e}"

    if not results:
        return f"[INFO] 未找到匹配论文 '{paper_title}' 的 '{section_name}' 章节内容"

    # 格式化输出
    lines = [f"📄 {paper_title} — {section_name.upper()}"]
    lines.append("=" * 50)

    total_text = ""
    for i, doc in enumerate(results):
        content = doc.page_content[:500]  # 每块最多 500 字符
        source = doc.metadata.get("source", "未知来源")
        lines.append(f"\n### 片段 {i+1} (来源: {source})")
        lines.append(content)
        total_text += content

        if len(total_text) > 2000:
            lines.append("\n... (内容过长，已截断)")
            break

    return "\n".join(lines)


@tool
def generate_review_checklist(paper_count: int = 1) -> str:
    """
    生成论文审阅检查清单，帮助系统化地审阅论文。

    :param paper_count: 需要审阅的论文数量
    :return: 格式化的审阅检查清单
    """
    checklist = f"""
# 📋 论文审阅检查清单 ({paper_count} 篇)

## 基本信息确认
- [ ] 论文标题和作者
- [ ] 发表年份和会议/期刊
- [ ] 论文所属领域和子方向
- [ ] 是否有开源代码 (GitHub 链接)

## 方法论审阅
- [ ] 核心方法描述是否清晰
- [ ] 与现有方法的区别和联系
- [ ] 理论基础是否充分
- [ ] 公式推导是否合理

## 实验审阅
- [ ] 数据集选择和规模
- [ ] 基线方法是否合理
- [ ] 评估指标是否全面
- [ ] 消融实验是否充分
- [ ] 结果是否有统计显著性

## 贡献评估
- [ ] 理论贡献
- [ ] 方法贡献
- [ ] 实验贡献
- [ ] 应用价值

## 不足分析
- [ ] 方法局限
- [ ] 实验覆盖不足
- [ ] 可复现性
- [ ] 写作质量

## 综合评分 (1-10)
| 维度 | 评分 |
|------|------|
| 创新性 | /10 |
| 技术深度 | /10 |
| 实验充分性 | /10 |
| 写作质量 | /10 |
| 实用价值 | /10 |
"""
    return checklist.strip()


# 导出列表（工具注册时使用）
__all__ = [
    "extract_section",
    "generate_review_checklist",
]
