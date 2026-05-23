import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from src.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)


class Ingest:
    def __init__(self):
        self._directory_path = settings.docs_path
        self._storage_path = settings.storage_path

        # 确保 Chroma 存储目录存在
        os.makedirs(self._storage_path, exist_ok=True)

        try:
            self._embed_model = HuggingFaceEmbedding(
                model_name=settings.embed_model_name,
                device=settings.embed_device
            )
            self._chroma_client = chromadb.PersistentClient(path=self._storage_path)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=settings.collection_name
            )
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    def ingest(self) -> VectorStoreIndex:
        try:
            # 加载文档
            logger.info(f"加载文档：{self._directory_path}")
            reader = SimpleDirectoryReader(
                self._directory_path,
                required_exts=settings.supported_exts
            )
            documents = reader.load_data()
            logger.info(f"加载完成，共 {len(documents)} 个文档")

            # chunking
            splitter = SentenceSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents(documents)
            logger.info(f"切分完成，共 {len(nodes)} 个 chunks")

            # 清空旧数据，避免重复入库
            existing = self._chroma_collection.count()
            if existing > 0:
                logger.warning(f"检测到已有 {existing} 条数据，清空后重新入库")
                self._chroma_client.delete_collection(settings.collection_name)
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=settings.collection_name
                )

            # embedding + 写入 Chroma
            logger.info("开始 embedding，首次运行需下载模型，请稍候...")
            vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self._embed_model
            )
            logger.info(f"入库完成，共写入 {len(nodes)} 个 chunks")

            return index

        except Exception as e:
            logger.error(f"入库失败: {e}")
            raise

        finally:
            if hasattr(self, "_chroma_client"):
                self._chroma_client.close()
                logger.info("Chroma 客户端连接已关闭")
