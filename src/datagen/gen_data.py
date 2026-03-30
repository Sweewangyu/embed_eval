import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 从环境变量获取配置
INPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", "output.json")
DATAGEN_OUTPUT_PATH = os.getenv("DATAGEN_OUTPUT_PATH", "train_data.jsonl")
SCORES_OUTPUT_PATH = os.getenv("SCORES_OUTPUT_PATH", "scores_data.jsonl")

RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "http://10.246.99.82:11026/v1/rerank")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "dummy")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "dummy")

# 难负样本的分数区间
HARD_NEGATIVE_MIN = float(os.getenv("HARD_NEGATIVE_MIN", "0.15"))
HARD_NEGATIVE_MAX = float(os.getenv("HARD_NEGATIVE_MAX", "0.35"))

def get_rerank_scores(query, texts):
    """
    调用 Reranker 模型对 query 和候选 chunks 进行打分
    """
    if not texts:
        return []
    
    headers = {
        "Content-Type": "application/json"
    }
    if RERANKER_API_KEY and RERANKER_API_KEY != "dummy":
        headers["Authorization"] = f"Bearer {RERANKER_API_KEY}"
        
    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "texts": texts
    }
    
    try:
        response = requests.post(RERANKER_BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        scores = [0.0] * len(texts)
        
        def extract_score(item):
            # 兼容多种常见的 API 返回字段
            return item.get("relevance_score", item.get("score", 0.0))
            
        if isinstance(data, dict):
            if "results" in data:
                for item in data["results"]:
                    scores[item["index"]] = extract_score(item)
            elif "data" in data:
                for item in data["data"]:
                    scores[item["index"]] = extract_score(item)
            else:
                print(f"未能解析的返回格式: {data}")
        elif isinstance(data, list):
            for item in data:
                scores[item["index"]] = extract_score(item)
        else:
            print(f"未能解析的返回格式: {data}")
            
        return scores
    except Exception as e:
        print(f"调用 Reranker 失败, Query: '{query[:20]}...' 错误: {e}")
        return []

def normalize_scores(scores, max_score=None):
    """
    使用 Min-Max 对相对分数进行归一化到 [0, 1] 区间
    如果提供了 max_score，则以此作为上限
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max_score if max_score is not None else max(scores)
    
    if max_s <= min_s:
        return [0.0 for _ in scores]
        
    return [(s - min_s) / (max_s - min_s) for s in scores]

def main():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"输入文件 {INPUT_JSON_PATH} 不存在。请先运行 eval.py")
        return
        
    print(f"加载检索结果: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    train_data = []
    scores_data = []
    
    for item in tqdm(data, desc="生成难样本训练数据"):
        query = item.get("question", "")
        golden_chunks = item.get("goldenchunks", [])
        chunks = item.get("chunks", [])
        
        if not query or not golden_chunks or not chunks:
            continue
            
        texts = [c.get("chunk_text", "") for c in chunks]
        
        # 计算 golden chunks 的打分，并取最大值作为上限
        golden_scores = get_rerank_scores(query, golden_chunks)
        if not golden_scores:
            continue
        golden_max_score = max(golden_scores)
        
        # 获取打分并归一化
        scores = get_rerank_scores(query, texts)
        if not scores or len(scores) != len(texts):
            continue
            
        norm_scores = normalize_scores(scores, max_score=golden_max_score)
        
        # 记录每个 chunk 的得分情况
        chunk_scores = []
        for text, score, n_score in zip(texts, scores, norm_scores):
            chunk_scores.append({
                "chunk_text": text,
                "raw_score": score,
                "normalized_score": n_score
            })
            
        scores_record = {
            "query": query,
            "golden_chunks": golden_chunks,
            "golden_max_score": golden_max_score,
            "chunks": chunk_scores
        }
        scores_data.append(scores_record)
        
        hard_negatives = []
        for text, n_score in zip(texts, norm_scores):
            # 将相对归一化后分数在配置区间内的 chunk 作为难负样本
            if HARD_NEGATIVE_MIN <= n_score <= HARD_NEGATIVE_MAX:
                # 简单过滤，避免把 golden chunk 本身当作负样本
                is_golden = False
                for g in golden_chunks:
                    if g in text or text in g:
                        is_golden = True
                        break
                if not is_golden:
                    hard_negatives.append(text)
                    
        # 如果没有找到难样本，则跳过
        if not hard_negatives:
            continue
            
        # 按照指定的结构构造数据
        positive_messages = [[{"role": "user", "content": g}] for g in golden_chunks]
        negative_messages = [[{"role": "user", "content": hn}] for hn in hard_negatives]
        
        record = {
            "messages": [{"role": "user", "content": query}],
            "positive_messages": positive_messages,
            "negative_messages": negative_messages
        }
        train_data.append(record)
        
    # 保存结果到 jsonl 文件
    with open(DATAGEN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    with open(SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in scores_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"\n数据生成完成！共提取 {len(train_data)} 条包含难负样本的数据。")
    print(f"训练数据已保存至: {DATAGEN_OUTPUT_PATH}")
    print(f"所有打分记录已保存至: {SCORES_OUTPUT_PATH}")

if __name__ == "__main__":
    main()