"""
config.py — YAML 配置加载
=========
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class APIEmbeddingConfig:
    api_base: str = "https://llm-xjt5kj6lz9uh275u.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "text-embedding-v4"
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30


@dataclass
class LocalEmbeddingConfig:
    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"              # cpu | cuda | cuda:0
    batch_size: int = 64
    normalize: bool = True


@dataclass
class EmbeddingConfig:
    backend: str = "api"             # api | local
    api: APIEmbeddingConfig = field(default_factory=APIEmbeddingConfig)
    local: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)


@dataclass
class ContextAssemblyConfig:
    retrieval_max_tokens: int = 512
    generation_max_tokens: int = 2048
    neighbor_window: int = 1            # prev/next chunk 各取几个
    include_keywords: bool = True       # 检索文本保留 [KEYWORDS: ...] 前缀


@dataclass
class ChromaStoreConfig:
    persist_dir: str = "./indexer/data/chroma"
    collection_name: str = "rag_chunks"


@dataclass
class QdrantStoreConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "rag_chunks"
    vector_size: int = 1024


@dataclass
class VectorStoreConfig:
    backend: str = "chroma"             # chroma | qdrant
    chroma: ChromaStoreConfig = field(default_factory=ChromaStoreConfig)
    qdrant: QdrantStoreConfig = field(default_factory=QdrantStoreConfig)


@dataclass
class DedupConfig:
    hash_algorithm: str = "sha256"


@dataclass
class PIIConfig:
    enabled: bool = True
    detection_method: str = "regex"     # regex | presidio
    action: str = "flag"                # flag | drop | redact


@dataclass
class HyDEConfig:
    enabled: bool = False
    model: str = "qwen3.6-max-preview"
    questions_per_chunk: int = 3


@dataclass
class EvalConfig:
    export_path: str = "./indexer/data/eval_manifest.jsonl"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"                # json | text


@dataclass
class IndexerConfig:
    schema_version: str = "1.0"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    context_assembly: ContextAssemblyConfig = field(default_factory=ContextAssemblyConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dict_to_dataclass(d: dict, dc: type):
    """ponytail: recursive dict → dataclass, no pydantic dep."""
    if d is None:
        return dc()
    # get_type_hints resolves string annotations caused by __future__.annotations
    import typing
    field_types = typing.get_type_hints(dc)
    kwargs = {}
    for k, v in d.items():
        if k in field_types:
            ft = field_types[k]
            if hasattr(ft, "__dataclass_fields__") and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(v, ft)
            else:
                kwargs[k] = v
    return dc(**kwargs)


DEFAULT_CONFIG_PATH = "./indexer/config.yaml"


def load_config(path: str = "") -> IndexerConfig:
    """从 YAML 文件加载配置，缺失值使用默认值。

    若 path 为空，默认读取 ./indexer/config.yaml。
    """
    config = IndexerConfig()

    if not path:
        path = DEFAULT_CONFIG_PATH

    cfg_path = Path(path)
    if cfg_path.exists():
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML config. pip install pyyaml")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if "embedding" in raw:
            emb_raw = raw["embedding"]
            config.embedding = EmbeddingConfig(
                backend=emb_raw.get("backend", "api"),
                api=_dict_to_dataclass(emb_raw.get("api", {}), APIEmbeddingConfig),
                local=_dict_to_dataclass(emb_raw.get("local", {}), LocalEmbeddingConfig),
            )
        if "context_assembly" in raw:
            config.context_assembly = _dict_to_dataclass(raw["context_assembly"], ContextAssemblyConfig)
        if "vector_store" in raw:
            config.vector_store = _dict_to_dataclass(raw["vector_store"], VectorStoreConfig)
        if "dedup" in raw:
            config.dedup = _dict_to_dataclass(raw["dedup"], DedupConfig)
        if "pii" in raw:
            config.pii = _dict_to_dataclass(raw["pii"], PIIConfig)
        if "hyde" in raw:
            config.hyde = _dict_to_dataclass(raw["hyde"], HyDEConfig)
        if "eval" in raw:
            config.eval = _dict_to_dataclass(raw["eval"], EvalConfig)
        if "logging" in raw:
            config.logging = _dict_to_dataclass(raw["logging"], LoggingConfig)
        if "schema_version" in raw:
            config.schema_version = raw["schema_version"]

    return config
