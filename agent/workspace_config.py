"""
workspace_config.py — 可配置项目路径（统一来源，agent/domains 与 web/api 共用）。

三个可配置根：
- project_path    ：文献问答 + 写作的项目路径（用户自选本地目录）。
                    未显式设置(None) → 项目根 = 代码根，写作目录 = web/workspace/docs
                                      （旧行为，数据不迁移）。
                    显式设置后       → 项目根 = 所选路径，写作目录 = {project_path}/writing。
- experiments_path：实验根，独立于文献问答。默认 web/workspace/experiments（兼容现有数据）。
- study_root      ：研究知识库根，跟随实验根的 parent/studies（默认 web/workspace/studies）。

配置持久化在 web/workspace/settings.json：{"project_path": str|null, "experiments_path": str|null}。
null = 未显式设置，走默认回退。路径安全边界 resolve_workspace_path（generic_provider）的
root 由此模块提供；本模块不涉及路径越界校验（那是 resolve 的职责）。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# 代码根（= 仓库根），未设置 project_path 时的项目根回退。
_CODE_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = _CODE_ROOT / "web" / "workspace" / "settings.json"

# 默认根（未显式设置时的回退）。
_DEFAULT_DOCS_DIR = _CODE_ROOT / "web" / "workspace" / "docs"
_DEFAULT_EXPERIMENTS_DIR = _CODE_ROOT / "web" / "workspace" / "experiments"
_DEFAULT_STUDIES_DIR = _CODE_ROOT / "web" / "workspace" / "studies"

# 写作保存子目录名（位于 project_path 内）。
_WRITING_SUBDIR = "writing"

# 测试接缝：getter 先查 override（key = getter 名），再查磁盘配置。
_OVERRIDES: dict[str, Path] = {}


# ---- 设置读写 ----

def load_settings() -> dict:
    """读 settings.json；缺失/损坏 → {}（全部走默认回退）。"""
    try:
        raw = SETTINGS_PATH.read_text(encoding="utf-8")
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except (OSError, ValueError):
        return {}


def save_settings(cfg: dict) -> None:
    """原子写 settings.json（临时文件 + os.replace，防写到一半崩溃留下半截）。"""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(SETTINGS_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SETTINGS_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _abs_or_none(value: str | None) -> str | None:
    """规范化绝对路径串；None/空 → None。非绝对路径抛 ValueError（调用方校验）。"""
    if value is None or not str(value).strip():
        return None
    p = Path(os.path.expanduser(str(value).strip()))
    if not p.is_absolute():
        raise ValueError(f"path must be absolute: {value}")
    return str(p.resolve())


def set_paths(project_path: str | None = None, experiments_path: str | None = None) -> dict:
    """更新并持久化配置。路径必须为绝对路径；不存在时尝试创建目录。

    返回更新后的配置 dict。校验失败抛 ValueError / OSError（web 层转 400）。
    """
    cfg = load_settings()
    if project_path is not None:
        pp = _abs_or_none(project_path)
        if pp:
            Path(pp).mkdir(parents=True, exist_ok=True)
        cfg["project_path"] = pp  # None（传入空串）→ 显式清除，恢复默认
    if experiments_path is not None:
        ep = _abs_or_none(experiments_path)
        if ep:
            Path(ep).mkdir(parents=True, exist_ok=True)
        cfg["experiments_path"] = ep
    save_settings(cfg)
    return cfg


# ---- 根解析 getter（测试 override 优先）----

def _resolve(key: str, default: Path) -> Path:
    return _OVERRIDES.get(key, default)


def get_project_path() -> str | None:
    """已显式设置的 project_path（绝对串），无则 None。"""
    cfg = _OVERRIDES.get("_raw_project_path")
    if cfg is not None:
        return str(cfg)
    return load_settings().get("project_path")


def get_project_root() -> Path:
    """文献问答/通用工具的项目根：project_path（显式设置）或代码根。"""
    return _resolve("project_root", Path(get_project_path() or _CODE_ROOT))


def get_experiments_path() -> Path:
    """实验根：experiments_path（显式设置）或 web/workspace/experiments。"""
    cfg = load_settings().get("experiments_path")
    return _resolve("experiments_path", Path(cfg or _DEFAULT_EXPERIMENTS_DIR))


def get_study_root() -> Path:
    """研究知识库根：默认实验根同级的 studies/（跟随实验根移动）。"""
    return _resolve("study_root", get_experiments_path().parent / "studies")


def get_docs_dir() -> Path:
    """写作文档根：project_path 显式设置 → {project_path}/writing；否则旧 web/workspace/docs。"""
    return _resolve("writing_dir", Path(get_project_path() + "/" + _WRITING_SUBDIR) if get_project_path() else _DEFAULT_DOCS_DIR)


def get_writing_dir() -> Path:
    """写作文档根别名（creation 域用，语义同 get_docs_dir）。"""
    return get_docs_dir()


# ---- 测试接缝 ----

def set_override(key: str, value: Path | str) -> None:
    _OVERRIDES[key] = Path(value)


def clear_overrides() -> None:
    _OVERRIDES.clear()