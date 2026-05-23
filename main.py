from src.ingest import Ingest
from src.pipeline import Pipeline
from src.logger import get_logger
from src.settings import settings

import chromadb

logger = get_logger(__name__)

if __name__ == "__main__":
    try: 
        chroma_client = chromadb.PersistentClient(path=settings.storage_path)
        collection = chroma_client.get_or_create_collection(name=settings.collection_name)
        if collection.count() == 0:
            logger.info("检测到没有已入库的数据，开始执行 ingest")
            ingest = Ingest()
            ingest.ingest()
        else:
            logger.info(f"检测到已有 {collection.count()} 条数据，跳过 ingest 步骤")

        chroma_client.close()  # 关闭 Chroma 客户端连接

        pipeline = Pipeline()
        while True:
            query = input("\n请输入你的问题（输入 'exit' 或 'quit' 退出）：").strip()
            if query.lower() in ["exit", "quit"]:
                logger.info("用户选择退出程序")
                print("再见！")
                break

            if not query:
                continue

            answer = pipeline.run(query)
            print(f"\n回答：{answer}")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        raise