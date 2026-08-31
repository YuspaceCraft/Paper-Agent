"""
dataset.py — 评测数据集加载器
=========
从 JSONL 文件加载评测样本，校验字段完整性。

JSONL 格式（每行一个 JSON 对象）：
  {
    "id": "q001",
    "question": "What is change captioning?",
    "relevant_sources": ["paper_a.pdf", "paper_b.pdf"],
    "reference_answer": "Change captioning is...",
    "metadata": {"category": "definition", "difficulty": "easy"}
  }

字段说明：
  id                — 必填，唯一标识
  question          — 必填，用户问题
  relevant_sources  — 检索评测用，文件名列表
  reference_answer  — 生成评测用，参考答案
  metadata          — 可选，用于分层统计
"""

import json
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """单个评测样本。

    必填字段: id, question
    可选字段: relevant_sources, reference_answer, metadata
    """
    id: str
    question: str
    relevant_sources: list[str] = field(default_factory=list)
    reference_answer: str = ""
    metadata: dict = field(default_factory=dict)

    def has_retrieval_gt(self) -> bool:
        """是否有检索阶段的 ground truth。"""
        return len(self.relevant_sources) > 0

    def has_generation_gt(self) -> bool:
        """是否有生成阶段的 ground truth（参考答案）。"""
        return bool(self.reference_answer.strip())


def _normalize_filename(name: str) -> str:
    """对文件名做 Unicode NFC 规范化 + 去空白，用于模糊匹配。"""
    return unicodedata.normalize("NFC", name).strip().lower()


def validate_sample(data: dict) -> list[str]:
    """校验单条样本，返回错误列表（空列表 = 合法）。

    规则：
      - id 必须是非空字符串
      - question 必须是非空字符串
      - relevant_sources（如果存在）必须是字符串列表
      - reference_answer（如果存在）必须是字符串
    """
    errors: list[str] = []

    # id
    sample_id = data.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append(f"缺少或无效的 'id' 字段: {sample_id!r}")

    # question
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append(f"缺少或无效的 'question' 字段: {question!r}")

    # relevant_sources（可选，但有就必须是 list[str]）
    if "relevant_sources" in data:
        srcs = data["relevant_sources"]
        if not isinstance(srcs, list) or not all(isinstance(s, str) for s in srcs):
            errors.append(f"'relevant_sources' 必须是字符串列表，实际: {type(srcs)}")

    # reference_answer（可选，但有就必须是 string）
    if "reference_answer" in data:
        ref = data["reference_answer"]
        if not isinstance(ref, str):
            errors.append(f"'reference_answer' 必须是字符串，实际: {type(ref)}")

    return errors


def load_dataset(path: str | Path) -> list[EvalSample]:
    """从 JSONL 文件加载评测数据集。

    参数:
        path: JSONL 文件路径

    返回:
        list[EvalSample]: 校验通过的样本列表（非法行跳过并警告）
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"[DATASET] [ERR] 文件不存在: {path}")

    samples: list[EvalSample] = []
    total_lines = 0
    skipped = 0

    # 尝试 UTF-8，失败则回退到 GBK（与 loader.py 保持一致）
    encodings = ["utf-8", "gbk", "utf-8-sig"]
    lines = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if lines is None:
        raise ValueError(f"[DATASET] [ERR] 无法解码文件: {path}")

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # 空行 / 注释

        total_lines += 1
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            print(f"[DATASET] [WARN] 第 {line_num} 行 JSON 解析失败: {e}")
            skipped += 1
            continue

        # 校验
        errors = validate_sample(data)
        if errors:
            print(f"[DATASET] [WARN] 第 {line_num} 行校验失败: {'; '.join(errors)}")
            skipped += 1
            continue

        # 构造 EvalSample
        samples.append(EvalSample(
            id=data["id"].strip(),
            question=data["question"].strip(),
            relevant_sources=[
                _normalize_filename(s) for s in data.get("relevant_sources", [])
            ],
            reference_answer=data.get("reference_answer", "").strip(),
            metadata=data.get("metadata", {}),
        ))

    print(f"[DATASET] [OK] 已加载 {len(samples)} 条样本"
          f"（共 {total_lines} 行，跳过 {skipped} 行）")
    return samples
