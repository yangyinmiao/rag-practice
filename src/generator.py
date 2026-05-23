
from src.settings import settings
from src.logger import get_logger
from openai import OpenAI
from dotenv import load_dotenv
import os

logger = get_logger(__name__)

class Generator:

    def __init__(self):

        try:
            load_dotenv()
            self._client = OpenAI(
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )
            self._model = os.getenv("LLM_MODEL", "deepseek-chat")
            self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    def generate(self, reranked_nodes: list, query: str) -> str:
        try:
            context = "\n\n".join([node.get_content() for node in reranked_nodes])
            prompt = f"""你是一个问答助手，请根据以下参考资料回答用户的问题。
    如果参考资料中没有相关信息，请直接说不知道，不要编造答案。

    参考资料：
    {context}

    用户问题：{query}
    """
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise