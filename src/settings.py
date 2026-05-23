from dataclasses import dataclass


@dataclass
class Settings:
    # 文档目录
    docs_path: str = "docs"

    # 向量库存储路径
    storage_path: str = "storage/chroma"

    # Chroma collection 名称
    collection_name: str = "rag_practice_collection"

    # Embedding 模型
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str = "mps"

    # Rerank 模型
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 5
    rerank_use_fp16: bool = True

    # 向量召回数量
    retrieval_top_k: int = 50

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 支持的文档格式
    supported_exts: list = None

    def __post_init__(self):
        if self.supported_exts is None:
            self.supported_exts = [".pdf", ".txt"]


# 全局单例，各模块直接 import 使用
settings = Settings()
