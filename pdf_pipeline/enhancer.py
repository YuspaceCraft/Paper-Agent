"""
enhancer.py — Stage 3 多模态语义增强
==========

对已提取的公式和图片分别调用 LLM/VLM 进行语义增强：

  - 公式: qwen3.6-max-preview → 自然语言解释
  - 图片: qwen3.7-plus (vision) → 详细图表描述

增强结果写入元素 metadata + 独立文件，供 Stage 4 注入 Markdown。

前置: bindings.json 已生成，formula_N.txt 和 picture_N.png 已保存。
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# ================================================================
# API client (lazy init)
# ================================================================

_client: OpenAI | None = None

# 统一的 DashScope endpoint，与 agent/nodes.py 保持一致
_DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# Legacy MaaS endpoint — 可通过 ENHANCER_BASE_URL 覆盖
_ENHANCER_BASE_URL = os.getenv("ENHANCER_BASE_URL", _DASHSCOPE_BASE_URL)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # 自动加载项目根目录 .env
        try:
            from dotenv import load_dotenv
            _env_path = Path(__file__).resolve().parent.parent / ".env"
            if _env_path.exists():
                load_dotenv(_env_path)
        except ImportError:
            pass

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未设置。请在环境变量或项目根目录 .env 中配置。"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=_ENHANCER_BASE_URL,
            # 与 agent/nodes.py 保持一致：单次 LLM 调用 2 分钟超时，
            # 避免上游 API 无响应时无限挂起（openai 默认 600s）。
            timeout=120.0,
            max_retries=1,
        )
    return _client


# ================================================================
# 输出质检
# ================================================================

# 描述文本的合理长度范围 (tokens 数近似为字符数/3.5)
_MIN_DESC_CHARS = 15    # 低于此视为空描述
_MAX_DESC_CHARS = 500   # 超过此视为跑偏 (prompt 要求 ≤90 tokens)

# 明显的失败模式 — 出现任意一个即视为增强失败
_VETO_PATTERNS: list[re.Pattern] = [
    re.compile(r'^I (cannot|am unable|do not have)', re.IGNORECASE),
    re.compile(r'^Sorry', re.IGNORECASE),
    re.compile(r'^(As an AI|I am an AI)', re.IGNORECASE),
    re.compile(r'^Here (is|are) the', re.IGNORECASE),  # "Here is the description:" boilerplate
    re.compile(r'^The (image|figure|formula|picture) (shows|depicts|is|contains)', re.IGNORECASE),
    re.compile(r'^\*\*.*\*\*$'),  # 单行 markdown bold 且无实质内容
    re.compile(r'^\[FORMULA_DESC:', re.IGNORECASE),  # 输出格式标记而非描述
    re.compile(r'.*description text here.*', re.IGNORECASE),  # prompt 模板泄露
]


def _validate_enhancement(desc: str, element_id: str) -> bool:
    """质检增强描述，返回 True 表示通过、False 表示应丢弃。

    检查：
    1. 非空且长度在合理范围内
    2. 不含明显的失败模式（拒绝、模板泄露、通用废话）
    """
    if not desc or not desc.strip():
        return False

    text = desc.strip()

    # 长度检查
    if len(text) < _MIN_DESC_CHARS:
        print(f"  [ENHANCE] {element_id}: discarded — too short ({len(text)} chars)")
        return False
    if len(text) > _MAX_DESC_CHARS:
        print(f"  [ENHANCE] {element_id}: discarded — too long ({len(text)} chars)")
        return False

    # 模式检查
    for pat in _VETO_PATTERNS:
        if pat.search(text):
            print(f"  [ENHANCE] {element_id}: discarded — veto pattern "
                  f"'{pat.pattern[:60]}...'")
            return False

    return True


# ================================================================
# 辅助：上下文提取
# ================================================================

def _extract_abstract(markdown: str) -> str:
    """从 Markdown 中提取摘要片段（前几个有效段落）。"""
    lines = markdown.split("\n")
    in_body = False
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##") or stripped.startswith("# "):
            in_body = True
            continue
        if in_body:
            if stripped:
                current.append(stripped)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
                if len(paragraphs) >= 3:
                    break
    if current:
        paragraphs.append(" ".join(current))

    for p in paragraphs:
        if len(p) > 100:
            return p[:1000]
    return paragraphs[0][:1000] if paragraphs else ""


def _find_ref_context(element_id: str, references: list[dict],
                      markdown: str = "") -> str:
    """收集正文中对某元素的所有引用上下文。ponytail: 直接取 context 字段即可。"""
    contexts = []
    for r in references:
        if r.get("target_element_id") == element_id:
            ctx = r.get("context", "")
            if ctx and ctx not in contexts:
                contexts.append(ctx)
    return " | ".join(contexts[:3])


# ================================================================
# 公式增强 (LLM) — 批处理模式
# ================================================================

_FORMULA_BATCH_PROMPT = """You are a RAG indexing engine for scientific literature. Generate concise, high-density natural language descriptions for the following formulas for vector retrieval.

Rules:
- MAX 80 tokens per description.
- Output ONLY in this exact format, one per formula (no markdown, no extra text):
  [FORMULA:element_id]: description text here
- Focus strictly on: input/output semantics + specific architectural role in THIS paper.
- If context is insufficient, describe only what is explicitly computable from the formula itself.
- No meta-commentary, no disclaimers.

Formulas to describe:
{formula_blocks}"""

_FORMULA_BATCH_PARSE = re.compile(
    r'\[FORMULA:(formula_\d+)\]:\s*(.+?)(?=\[FORMULA:formula_\d+\]:|\Z)',
    re.DOTALL,
)

# 单条公式 prompt（batch 解析失败时的 fallback）
_FORMULA_SINGLE_PROMPT = """You are a RAG indexing engine for scientific literature. Generate a concise, high-density natural language description of this formula for vector retrieval.

Rules:
- MAX 80 tokens. Output ONLY the description text inside [FORMULA_DESC: ...].
- NO section labels, no meta-commentary, no disclaimers.
- Focus strictly on: input/output semantics + specific architectural role in THIS paper.

Formula: {formula}
Bound Context: {context}"""

# 每批最多多少个公式
_FORMULA_BATCH_SIZE = 6


def _enhance_formulas_batch(
    batch: list[dict], references: list[dict], client: OpenAI, out_dir: Path | None,
) -> int:
    """一次 API 调用增强一批公式。返回成功数。"""
    blocks = []
    for elem in batch:
        ctx = _find_ref_context(elem["element_id"], references)
        blocks.append(
            f"[FORMULA:{elem['element_id']}]\n"
            f"Text: {elem['formula_text']}\n"
            f"Context: {ctx or 'N/A'}\n"
        )
    prompt = _FORMULA_BATCH_PROMPT.format(formula_blocks="\n".join(blocks))

    try:
        resp = client.chat.completions.create(
            model="qwen3.7-max-2026-06-08",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200 * len(batch),
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [ENHANCE] Batch FAILED: {exc}, falling back to individual")
        return sum(
            _enhance_single_formula(e, references, client, out_dir) > 0
            for e in batch
        )

    # 解析批量响应
    parsed: dict[str, str] = {}
    for m in _FORMULA_BATCH_PARSE.finditer(raw):
        parsed[m.group(1)] = m.group(2).strip()

    enhanced = 0
    for elem in batch:
        eid = elem["element_id"]
        if eid in parsed and parsed[eid]:
            desc = parsed[eid]
            if not _validate_enhancement(desc, eid):
                # 质检失败 → 尝试 solo fallback
                enhanced += _enhance_single_formula(elem, references, client, out_dir) > 0
                continue
            elem["formula_desc"] = desc
            elem["formula_desc_meta"] = _make_provenance(
                "qwen3.7-max-2026-06-08", prompt, elem.get("formula_text", ""),
            )
            if out_dir:
                (out_dir / f"{eid}_enhanced.txt").write_text(
                    desc, encoding="utf-8",
                )
            enhanced += 1
            preview = desc[:80].replace("\n", " ")
            try:
                print(f"  [ENHANCE] {eid}: {preview}...")
            except UnicodeEncodeError:
                print(f"  [ENHANCE] {eid}: "
                      f"{preview.encode('ascii', 'replace').decode()}...")
        else:
            print(f"  [ENHANCE] {eid}: not in batch response, retrying solo")
            enhanced += _enhance_single_formula(elem, references, client, out_dir) > 0

    return enhanced


def _enhance_single_formula(
    elem: dict, references: list[dict], client: OpenAI, out_dir: Path | None,
) -> int:
    """单条公式增强（批处理解析失败时的 fallback）。"""
    context = _find_ref_context(elem["element_id"], references)
    prompt = _FORMULA_SINGLE_PROMPT.format(formula=elem["formula_text"], context=context)
    try:
        resp = client.chat.completions.create(
            model="qwen3.7-max-2026-06-08",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        desc = resp.choices[0].message.content.strip()
        if not _validate_enhancement(desc, elem["element_id"]):
            return 0
        elem["formula_desc"] = desc
        elem["formula_desc_meta"] = _make_provenance(
            "qwen3.7-max-2026-06-08", prompt, elem.get("formula_text", ""),
        )
        if out_dir:
            (out_dir / f"{elem['element_id']}_enhanced.txt").write_text(
                desc, encoding="utf-8",
            )
        preview = desc[:80].replace("\n", " ")
        try:
            print(f"  [ENHANCE] {elem['element_id']}: {preview}...")
        except UnicodeEncodeError:
            print(f"  [ENHANCE] {elem['element_id']}: "
                  f"{preview.encode('ascii', 'replace').decode()}...")
        return 1
    except Exception as exc:
        print(f"  [ENHANCE] {elem['element_id']} FAILED: {exc}")
        return 0


def _make_provenance(model: str, prompt: str, input_text: str) -> dict:
    """Create provenance metadata for an LLM/VLM-generated description.

    ponytail: sibling meta field — does not change the shape of existing
    formula_desc / picture_desc strings, so downstream consumers are unaffected.
    """
    prompt_hash = hashlib.md5((prompt + input_text).encode()).hexdigest()[:12]
    return {
        "source": "llm_generated",
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_hash": prompt_hash,
        "verification": "none",  # none | auto_verified | human_reviewed
    }


def enhance_formulas(bindings: dict, markdown: str = "",
                     output_dir: str = "", batch_size: int = _FORMULA_BATCH_SIZE) -> int:
    """
    对所有公式批量调用 LLM 进行语义增强。

    Args:
        batch_size: 每批处理的公式数，默认 6。

    Returns: 成功增强的公式数。
    """
    client = _get_client()
    elements = bindings.get("elements", [])
    references = bindings.get("references", [])
    out = Path(output_dir) if output_dir else None

    formulas = [e for e in elements
                if e["type"] == "formula" and e.get("formula_text")]
    if not formulas:
        print("  [ENHANCE] No formulas to enhance")
        return 0

    print(f"  [ENHANCE] Enhancing {len(formulas)} formulas "
          f"in batches of {batch_size}...")

    enhanced = 0
    for i in range(0, len(formulas), batch_size):
        batch = formulas[i:i + batch_size]
        if len(batch) == 1:
            enhanced += _enhance_single_formula(batch[0], references, client, out)
        else:
            enhanced += _enhance_formulas_batch(batch, references, client, out)

    return enhanced


# ================================================================
# 图片增强 (VLM) — 并行请求
# ================================================================

_IMAGE_MAX_WORKERS = 4

_IMAGE_PROMPT = """You are a RAG engineer. Generate a concise, technical description of this figure for vector indexing.
Requirements:
- MAX 90 tokens.
- Structure: [Function] + [Mechanism] + [Paper Claim Support]
- Use only standard ML terms (no Turkish translations).
- Do NOT list components; focus on data flow and novelty.
- Output ONLY the description, no preamble.

Image caption: {caption}
Paper abstract excerpt: {abstract}
Referenced in text: {ref_context}"""


def _enhance_single_image(
    elem: dict, references: list[dict], abstract: str,
    client: OpenAI, out_dir: Path | None,
) -> tuple[str, int]:
    """增强单张图片。返回 (element_id, 1=ok 0=fail)。"""
    eid = elem["element_id"]
    img_path = Path(elem["image_path"])
    if not img_path.exists():
        print(f"  [ENHANCE] {eid}: image file missing")
        return (eid, 0)

    caption = elem.get("caption", "")
    ref_context = _find_ref_context(eid, references)

    img_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = _IMAGE_PROMPT.format(
        caption=caption,
        abstract=abstract[:500],
        ref_context=ref_context,
    )

    try:
        resp = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.3,
            max_tokens=200,
        )
        desc = resp.choices[0].message.content.strip()
        if not _validate_enhancement(desc, eid):
            return (eid, 0)
        elem["picture_desc"] = desc
        elem["picture_desc_meta"] = _make_provenance(
            "qwen3.7-plus", prompt, caption,
        )

        if out_dir:
            (out_dir / f"{eid}_description.txt").write_text(
                desc, encoding="utf-8",
            )

        preview = desc[:80].replace("\n", " ")
        try:
            print(f"  [ENHANCE] {eid}: {preview}...")
        except UnicodeEncodeError:
            print(f"  [ENHANCE] {eid}: "
                  f"{preview.encode('ascii', 'replace').decode()}...")
        return (eid, 1)
    except Exception as exc:
        print(f"  [ENHANCE] {eid} FAILED: {exc}")
        return (eid, 0)


def enhance_images(bindings: dict, markdown: str = "",
                   output_dir: str = "", max_workers: int = _IMAGE_MAX_WORKERS) -> int:
    """
    对所有关键图片并行调用 VLM 进行语义描述。

    Args:
        max_workers: 并行请求数，默认 4。

    Returns: 成功增强的图片数。
    """
    client = _get_client()
    elements = bindings.get("elements", [])
    references = bindings.get("references", [])
    out = Path(output_dir) if output_dir else None

    pictures = [e for e in elements
                if e["type"] == "picture" and e.get("image_path")]
    if not pictures:
        print("  [ENHANCE] No images to enhance")
        return 0

    abstract = _extract_abstract(markdown)

    print(f"  [ENHANCE] Enhancing {len(pictures)} images "
          f"with {max_workers} parallel workers...")

    enhanced = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _enhance_single_image, elem, references, abstract, client, out,
            ): elem
            for elem in pictures
        }
        for fut in as_completed(futures):
            try:
                _, ok = fut.result()
                enhanced += ok
            except Exception as exc:
                print(f"  [ENHANCE] Worker FAILED: {exc}")

    return enhanced


# ================================================================
# 端到端 API
# ================================================================

def enhance_all(bindings: dict, markdown: str = "",
                output_dir: str = "") -> dict:
    """Stage 3 完整流程：公式 + 图片增强。"""
    n_f = enhance_formulas(bindings, markdown, output_dir)
    n_i = enhance_images(bindings, markdown, output_dir)
    return {"formulas_enhanced": n_f, "images_enhanced": n_i}
