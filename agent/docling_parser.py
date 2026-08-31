"""向后兼容重导出。新代码请直接使用 pdf_pipeline.parser。

注意: 从 agent.* 路径导入 parser 可能触发 segfault
(agent/__init__.py → PyTorch/CUDA → docling onnxruntime 冲突)。
推荐直接使用: from pdf_pipeline.parser import parse_pdf_docling
"""
from pdf_pipeline.parser import *  # noqa: F401, F403
