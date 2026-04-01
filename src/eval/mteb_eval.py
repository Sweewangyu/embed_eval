import os
import requests
import numpy as np
from typing import List, Union
from dotenv import load_dotenv
import mteb

load_dotenv()

# 与 recall.py 相同的代理设置
NO_PROXY_IPS = os.getenv("NO_PROXY_IPS", "10.246.99.82,localhost,127.0.0.1")
os.environ["no_proxy"] = NO_PROXY_IPS
os.environ["NO_PROXY"] = NO_PROXY_IPS

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://10.246.99.82:11027/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "dummy")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/models/Qwen3-Embedding-0.6B")
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "60"))

# MTEB 评估专属配置
MTEB_BATCH_SIZE = int(os.getenv("MTEB_BATCH_SIZE", "32"))
# 默认跑 C-MTEB (中文公开数据集) 中的两个经典任务作为示例：T2Retrieval (检索), Ocnli (分类/STS)
MTEB_TASKS = os.getenv("MTEB_TASKS", "T2Retrieval,Ocnli").split(",")


class APIEmbeddingModel:
    """
    包装类：将远程 Embedding API 封装成 MTEB 所需的格式
    MTEB 要求模型提供一个 encode 方法： encode(sentences, **kwargs) -> np.ndarray
    """
    def __init__(self):
        self.api_url = EMBEDDING_BASE_URL
        self.api_key = EMBEDDING_API_KEY
        self.model_name = EMBEDDING_MODEL
        self.timeout = EMBEDDING_TIMEOUT
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def encode(self, sentences: Union[str, List[str]], batch_size: int = MTEB_BATCH_SIZE, **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
            
        all_embeddings = []
        # 分批处理，避免一次请求文本过多导致报错或超时
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            payload = {
                "input": batch,
                "model": self.model_name,
            }
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if "data" not in data or not data["data"]:
                    raise ValueError(f"API 返回异常或无数据: {data}")
                    
                # 按照 index 排序，确保返回的 embeddings 顺序与输入的 sentences 一致
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                print(f"获取 Embedding 失败, Batch: {i} 到 {i+len(batch)}\n错误信息: {e}")
                if "response" in locals() and response is not None:
                    print(f"接口返回详情: {response.text}")
                # 向上抛出异常，MTEB 评测中遇到错误通常应中止
                raise e
                
        return np.array(all_embeddings)


def main():
    print(f"正在初始化 API 嵌入模型: {EMBEDDING_MODEL}")
    model = APIEmbeddingModel()
    
    tasks = [task.strip() for task in MTEB_TASKS if task.strip()]
    print(f"准备在以下 MTEB 任务上进行评估: {tasks}")
    
    # 实例化 MTEB 评测引擎
    evaluation = mteb.MTEB(tasks=tasks)
    
    # 运行评测，结果将输出到 mteb_results 文件夹
    output_folder = "mteb_results"
    print(f"开始评估，请耐心等待，结果将保存在 '{output_folder}' 目录下...")
    
    results = evaluation.run(model, output_folder=output_folder)
    print("\n评估完成！请查看 mteb_results 目录下的结果 JSON 文件。")


if __name__ == "__main__":
    main()
