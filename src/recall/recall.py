import os
import json
import requests
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv()

NO_PROXY_IPS = os.getenv("NO_PROXY_IPS", "10.246.99.82,localhost,127.0.0.1")
os.environ["no_proxy"] = NO_PROXY_IPS
os.environ["NO_PROXY"] = NO_PROXY_IPS

MILVUS_HOST = os.getenv("MILVUS_HOST", "10.246.99.82")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DB_NAME = os.getenv("DB_NAME", "default")  # 数据库名
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sdpolice")  # 集合（表）名
VECTOR_FIELD_NAME = os.getenv("VECTOR_FIELD_NAME", "vector")  # 向量字段名

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://10.246.99.82:11027/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "dummy")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/models/Qwen3-Embedding-0.6B")

INPUT_JSON_PATH = os.getenv("INPUT_JSON_PATH", "qa_rag.json")  # 包含 query 和 goldenchunk 的输入文件
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", "output.json")  # 输出文件

TOP_K = int(os.getenv("TOP_K", "5"))  # 召回的 chunk 数量

SEARCH_METRIC_TYPE = os.getenv("SEARCH_METRIC_TYPE", "COSINE")
SEARCH_NPROBE = int(os.getenv("SEARCH_NPROBE", "10"))
SEARCH_PARAMS = {
    "metric_type": SEARCH_METRIC_TYPE,
    "params": {"nprobe": SEARCH_NPROBE},
}

EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "10"))

OUTPUT_FIELDS = [
    "id",
    "chunk_text",
    "original_file_name",
]


def get_embedding(query_text):
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": query_text,
        "model": EMBEDDING_MODEL,
    }

    try:
        response = requests.post(
            EMBEDDING_BASE_URL, headers=headers, json=payload, timeout=EMBEDDING_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        vector = data["data"][0]["embedding"]
        return vector
    except Exception as e:
        print(f"获取 Embedding 失败, Query: '{query_text}'\n错误信息: {e}")
        if "response" in locals() and response is not None:
            print(f"接口返回详情: {response.text}")
        return None


def main():
    print(f"正在连接 Milvus {MILVUS_HOST}:{MILVUS_PORT} (已配置绕过代理)...")
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(name=COLLECTION_NAME, db_name=DB_NAME)
        collection.load()
        print(f"Milvus 数据库 '{DB_NAME}' 中的表 '{COLLECTION_NAME}' 连接并加载成功！")
    except Exception as e:
        print(f"Milvus 连接或加载失败: {e}")
        return

    try:
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"读取输入文件 {INPUT_JSON_PATH} 失败: {e}")
        return

    results = []

    for index, item in enumerate(input_data):
        query = item.get("question", "")
        golden_chunks = item.get("goldenchunks", [])

        if not query:
            continue

        print(f"[{index + 1}/{len(input_data)}] 正在处理 Question: {query}")

        query_vector = get_embedding(query)
        if not query_vector:
            print("  -> ⚠️ 跳过此条目，因向量获取失败。")
            continue

        try:
            search_result = collection.search(
                data=[query_vector],
                anns_field=VECTOR_FIELD_NAME,
                param=SEARCH_PARAMS,
                limit=TOP_K,
                expr=None,
                output_fields=OUTPUT_FIELDS,
            )
        except Exception as e:
            print(f"  -> ⚠️ Milvus 检索失败: {e}")
            continue

        retrieved_chunks = []
        for hits in search_result:
            for hit in hits:
                chunk_data = {
                    "distance": hit.distance,
                }
                for field in OUTPUT_FIELDS:
                    chunk_data[field] = hit.entity.get(field)

                retrieved_chunks.append(chunk_data)

        results.append(
            {
                "id": item.get("id"),
                "question": query,
                "answer": item.get("answer", ""),
                "goldenchunks": golden_chunks,
                "chunks": retrieved_chunks,
            }
        )

    try:
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(
            f"\n处理完成！共成功处理 {len(results)} 条数据，已保存至 {OUTPUT_JSON_PATH}"
        )
    except Exception as e:
        print(f"写入输出文件失败: {e}")


if __name__ == "__main__":
    main()
