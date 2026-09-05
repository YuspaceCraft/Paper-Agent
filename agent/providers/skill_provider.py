"""
skill_provider.py — SKILL.md 开放标准实现。

Anthropic Agent Skills 开放标准 (2025.12)：
每个 skill 是一个目录，包含 SKILL.md（YAML frontmatter + Markdown body），
可选 resources/ 子目录存放参考文件。

渐进式三阶段：
1. list: 只返回 skill name + description（~50 tokens/skill）
2. load: 模型调用 load_skill(name) → 返回完整 SKILL.md body（<5k tokens）
3. read_resource: 按需加载资源文件

对 ToolProvider 接口，skills 暴露为两个伪工具：
- skill__list: 列出所有可用技能
- skill__load: 加载指定技能的完整指示
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ToolDef, ToolProvider


# ---- SKILL.md parser ----

def discover_skills(skills_dir: str | Path = "skills") -> dict[str, dict]:
    """公开的 skills 清单入口（配置中心 web/api/routers/config.py 用）。

    返回原始清单（含停用的）：{name: {path, description, body, resources}}。
    是否启用由调用方按 config_store.get_disabled_skills() 判定。
    """
    return _discover_skills(skills_dir)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 YAML frontmatter。返回 ({key: value}, body)。

    兼容 PyYAML 不可用时的简单 key: value 解析。
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 3:].strip()

    fm: dict[str, str] = {}
    try:
        import yaml
        parsed = yaml.safe_load(fm_text)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                fm[str(k)] = str(v) if v is not None else ""
    except Exception:
        # fallback: simple key: value 解析
        for line in fm_text.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")

    return fm, body


def _discover_skills(skills_dir: str | Path) -> dict[str, dict]:
    """扫描 skills/ 目录，返回 {name: {path, description, resources}}。

    匹配规则：skills/ 下的每个直接子目录，包含 SKILL.md 即视为一个 skill。
    深度限制 1 层。未包含 SKILL.md 的子目录忽略。
    """
    root = Path(skills_dir)
    if not root.is_dir():
        return {}

    skills: dict[str, dict] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue

        text = skill_md.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)

        name = fm.get("name", entry.name)
        description = fm.get("description", "")

        # collect resources
        resources_dir = entry / "resources"
        resources: list[str] = []
        if resources_dir.is_dir():
            resources = sorted([
                f.name for f in resources_dir.iterdir()
                if f.is_file()
            ])

        skills[name] = {
            "path": str(entry),
            "description": description,
            "body": body,
            "resources": resources,
        }

    return skills


# ---- provider ----

class SkillProvider(ToolProvider):
    """SKILL.md 标准技能提供器。

    暴露两个工具给 agent：
    - skill__list: 列出可用技能（name + description）
    - skill__load: 加载技能完整指示 → 注入 agent 上下文
    """

    name = "skill"

    def __init__(self, skills_dir: str = "skills"):
        self._skills_dir = skills_dir
        self._skills: dict[str, dict] | None = None
        self._loaded: dict[str, str] = {}  # session cache

    def _ensure_skills(self) -> dict[str, dict]:
        if self._skills is None:
            skills = _discover_skills(self._skills_dir)
            # 配置中心停用集合：停用的 skill 不出现在 skill__list（reload_tools 后重建生效）。
            try:
                from .. import config_store
                blocked = set(config_store.get_disabled_skills())
            except Exception:
                blocked = set()
            if blocked:
                skills = {n: s for n, s in skills.items() if n not in blocked}
            self._skills = skills
        return self._skills

    # ---- pseudo-tool definitions ----

    @staticmethod
    def _list_tooldef() -> ToolDef:
        return ToolDef(
            name="skill__list",
            description=(
                "List all available skills. Skills are reusable instruction sets "
                "that teach you HOW to perform specific research tasks. "
                "Call skill__load(name) to get full instructions for a skill."
            ),
            parameters={"type": "object", "properties": {}},
            source="skill",
            annotations={"readOnlyHint": True, "idempotentHint": True},
        )

    @staticmethod
    def _load_tooldef() -> ToolDef:
        return ToolDef(
            name="skill__load",
            description=(
                "Load full instructions for a specific skill. "
                "Call skill__list first to see available skills. "
                "The returned instructions should guide your subsequent tool usage."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name as returned by skill__list",
                    },
                },
                "required": ["name"],
            },
            source="skill",
            annotations={"readOnlyHint": True, "idempotentHint": True},
        )

    # ---- provider interface ----

    async def list_tools(self) -> list[ToolDef]:
        skills = self._ensure_skills()
        if not skills:
            return []
        return [self._list_tooldef(), self._load_tooldef()]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if name == "skill__list":
            return self._do_list()
        elif name == "skill__load":
            return self._do_load(arguments.get("name", ""))
        else:
            raise KeyError(f"Skill tool '{name}' not found (use skill__list or skill__load)")

    # ---- operations ----

    def _do_list(self) -> str:
        skills = self._ensure_skills()
        if not skills:
            return "(no skills available)"

        lines = ["Available skills:"]
        for sname, info in skills.items():
            desc = info.get("description", "") or ""
            lines.append(f"- **{sname}**: {desc}")
        lines.append("\nUse skill__load(\"<name>\") to get full instructions.")
        return "\n".join(lines)

    def _do_load(self, name: str) -> str:
        skills = self._ensure_skills()
        if name not in skills:
            available = ", ".join(skills.keys()) if skills else "(none)"
            return f"Skill '{name}' not found. Available: {available}"

        # Cache for session reuse (idempotent)
        if name not in self._loaded:
            info = skills[name]
            body = info.get("body", "")
            resources = info.get("resources", [])
            if resources:
                body += "\n\n## Resources\n" + "\n".join(
                    f"- {r}" for r in resources
                )
            self._loaded[name] = body

        return self._loaded[name]
