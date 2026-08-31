"""
retrieval — 共享检索层
=====================

检索服务层的基础组件，供 retrieval_orchestrator（离线评估）和 web/api（在线服务）共用。

组件:
  - SparseRetriever: TF-IDF 稀疏检索
  - DenseRetriever:  向量稠密检索
  - rrf_fuse / weighted_fuse: 混合检索融合
  - RetrievalService: 一站式检索服务（从 optimal config 加载）
"""
from .sparse import SparseRetriever
from .fusion import rrf_fuse, weighted_fuse
from .service import RetrievalService, DenseRetriever
