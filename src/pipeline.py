from src.retriever import Retriever
from src.generator import Generator
from src.logger import get_logger

logger = get_logger(__name__)

class Pipeline:
    def __init__(self):
        try:
            self._retriever = Retriever()
            self._generator = Generator()
            logger.info("Pipeline 初始化完成")
        except Exception as e:
            logger.error(f"Pipeline 初始化失败: {e}")
            raise

    def run(self, query: str) -> str:
        try:
            logger.info(f"接收到用户查询: {query}")
            
            reranked_nodes = self._retriever.retrieve(query)
            
            for i, node in enumerate(reranked_nodes):
                logger.debug(f"chunk {i+1} | score: {node.score:.4f} | {node.get_content()[:50]}...")

            answer = self._generator.generate(reranked_nodes, query)
            
            logger.info("生成完成")
            return answer

        except Exception as e:
            logger.error(f"Pipeline 运行失败: {e}")
            raise