from src.settings import settings
from src.logger import get_logger

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import QueryBundle
import chromadb

logger = get_logger(__name__)


class Retriever:
    def __init__(self):
        try:
            self._embed_model = HuggingFaceEmbedding(
                model_name=settings.embed_model_name,
                device=settings.embed_device
            )
            self._chroma_client = chromadb.PersistentClient(path=settings.storage_path)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=settings.collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self._embed_model
            )
            self._reranker = FlagEmbeddingReranker(
                model=settings.rerank_model_name,
                top_n=settings.rerank_top_n,
                use_fp16=settings.rerank_use_fp16
            )
            logger.info("Retriever 初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    def retrieve(self, query: str) -> list:
        try:
            # 向量召回
            retriever = self._index.as_retriever(similarity_top_k=settings.retrieval_top_k)
            nodes = retriever.retrieve(query)
            logger.info(f"向量召回 {len(nodes)} 个 chunks")

            # rerank
            query_bundle = QueryBundle(query_str=query)
            reranked_nodes = self._reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
            logger.info(f"rerank 后保留 {len(reranked_nodes)} 个 chunks")

            return reranked_nodes

        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
