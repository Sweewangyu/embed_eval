#!/usr/bin/env python3
"""
基于覆盖度的Embedding召回率评估脚本
从JSON文件读取queries和golden_chunks，通过Milvus检索并计算召回率
"""

import os
import json
import re
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
import numpy as np
from tqdm import tqdm


def load_test_data(json_path: str) -> Tuple[List[str], List[str]]:
    """从JSON文件加载测试数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data.get('queries', [])
    golden_chunks = data.get('golden_chunks', [])

    if len(queries) != len(golden_chunks):
        raise ValueError(f"queries数量({len(queries)})与golden_chunks数量({len(golden_chunks)})不一致")

    print(f"加载了 {len(queries)} 条测试数据")
    return queries, golden_chunks


def connect_milvus(config: Dict = None) -> None:
    """连接Milvus"""
    if config is None:
        load_dotenv()
        config = {
            'host': os.getenv('MILVUS_HOST', 'localhost'),
            'port': os.getenv('MILVUS_PORT', '19530'),
            'user': os.getenv('MILVUS_USER', ''),
            'password': os.getenv('MILVUS_PASSWORD', '')
        }

    connections.connect(
        alias='default',
        host=config['host'],
        port=config['port'],
        user=config.get('user', ''),
        password=config.get('password', '')
    )
    print(f"已连接Milvus: {config['host']}:{config['port']}")


def get_embedding(model, text: str) -> List[float]:
    """获取文本的embedding"""
    # 这里需要替换为你的embedding模型调用
    # 示例：使用sentence-transformers
    # from sentence_transformers import SentenceTransformer
    # return model.encode(text).tolist()

    # 或者使用API调用
    # response = client.embeddings.create(input=text, model="your-model")
    # return response.data[0].embedding

    raise NotImplementedError("请实现get_embedding函数，使用你的embedding模型")


def search_milvus(collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """在Milvus中搜索相关chunk"""
    collection = Collection(collection_name)
    collection.load()

    # 执行搜索
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",  # 请根据你的collection字段调整
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]  # 请根据你的collection字段调整
    )

    # 提取结果
    retrieved_chunks = []
    for hit in results[0]:
        retrieved_chunks.append({
            'text': hit.entity.get('text', ''),
            'score': hit.score
        })

    return retrieved_chunks


def split_sentences(text: str) -> List[str]:
    """将文本分句"""
    # 中文分句
    text = text.strip()
    sentences = re.split(r'[。！？\n；;]', text)
    return [s.strip() for s in sentences if s.strip()]


def coverage_score(golden_chunk: str, retrieved_chunks: List[Dict], threshold: float = 0.7) -> Dict:
    """
    计算覆盖度（召回率）

    Args:
        golden_chunk: golden标准答案文本
        retrieved_chunks: 检索到的chunks列表，每个包含'text'字段
        threshold: 判断句子被覆盖的相似度阈值

    Returns:
        包含详细覆盖度指标的字典
    """
    if not golden_chunk:
        return {'coverage': 0, 'total_sentences': 0, 'covered_sentences': 0, 'covered_sentences_list': []}

    # 分句
    golden_sentences = split_sentences(golden_chunk)
    total_sentences = len(golden_sentences)

    if total_sentences == 0:
        return {'coverage': 0, 'total_sentences': 0, 'covered_sentences': 0, 'covered_sentences_list': []}

    covered_sentences = []
    covered_count = 0

    for sentence in golden_sentences:
        is_covered = False
        for chunk in retrieved_chunks:
            retrieved_text = chunk.get('text', '')

            # 方法1: 精确包含（最严格）
            if sentence in retrieved_text:
                is_covered = True
                break

            # 方法2: 模糊匹配（可选）
            # 可以添加相似度计算，比如使用余弦相似度
            # if calculate_similarity(sentence, retrieved_text) > threshold:
            #     is_covered = True
            #     break

        if is_covered:
            covered_count += 1
            covered_sentences.append(sentence)

    coverage = covered_count / total_sentences

    return {
        'coverage': coverage,
        'total_sentences': total_sentences,
        'covered_sentences': covered_count,
        'covered_sentences_list': covered_sentences,
        'uncovered_sentences': [s for s in golden_sentences if s not in covered_sentences]
    }


def calculate_recall_at_k(queries: List[str], golden_chunks: List[str],
                          collection_name: str, model, top_k: int = 10,
                          threshold: float = 0.7) -> List[Dict]:
    """
    计算Recall@K

    Args:
        queries: 查询列表
        golden_chunks: 对应的golden chunks列表
        collection_name: Milvus collection名称
        model: embedding模型
        top_k: 召回的数量
        threshold: 覆盖度阈值

    Returns:
        每个query的详细评估结果列表
    """
    results = []

    for query, golden_chunk in tqdm(zip(queries, golden_chunks),
                                    total=len(queries),
                                    desc="评估中"):
        try:
            # 获取query的embedding
            query_embedding = get_embedding(model, query)

            # 从Milvus检索
            retrieved = search_milvus(collection_name, query_embedding, top_k)

            # 计算覆盖度
            coverage_metrics = coverage_score(golden_chunk, retrieved, threshold)

            results.append({
                'query': query,
                'golden_chunk': golden_chunk,
                'retrieved_count': len(retrieved),
                'coverage': coverage_metrics['coverage'],
                'total_sentences': coverage_metrics['total_sentences'],
                'covered_sentences': coverage_metrics['covered_sentences'],
                'covered_sentences_list': coverage_metrics['covered_sentences_list'],
                'uncovered_sentences': coverage_metrics['uncovered_sentences'],
                'hit_at_k': 1 if coverage_metrics['coverage'] >= threshold else 0
            })

        except Exception as e:
            print(f"处理query失败: {query[:50]}... 错误: {e}")
            results.append({
                'query': query,
                'golden_chunk': golden_chunk,
                'retrieved_count': 0,
                'coverage': 0,
                'error': str(e)
            })

    return results


def print_evaluation_summary(results: List[Dict], top_k: int):
    """打印评估摘要"""
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("没有有效的评估结果")
        return

    # 计算统计指标
    avg_coverage = np.mean([r['coverage'] for r in valid_results])
    hit_at_k = np.mean([r['hit_at_k'] for r in valid_results])

    print("\n" + "="*60)
    print("评估结果摘要")
    print("="*60)
    print(f"评估样本数: {len(valid_results)}")
    print(f"Top-K: {top_k}")
    print(f"平均覆盖度 (Recall@K): {avg_coverage:.3f}")
    print(f"Hit@K (覆盖率>{args.threshold}): {hit_at_k:.3f}")
    print(f"平均句子数: {np.mean([r['total_sentences'] for r in valid_results]):.1f}")
    print(f"平均覆盖句子数: {np.mean([r['covered_sentences'] for r in valid_results]):.1f}")

    # 打印分位数
    coverages = [r['coverage'] for r in valid_results]
    print(f"\n覆盖度分布:")
    print(f"  中位数: {np.median(coverages):.3f}")
    print(f"  25%分位: {np.percentile(coverages, 25):.3f}")
    print(f"  75%分位: {np.percentile(coverages, 75):.3f}")
    print(f"  最小值: {np.min(coverages):.3f}")
    print(f"  最大值: {np.max(coverages):.3f}")

    # 打印覆盖率分布
    coverage_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"\n覆盖率分布:")
    for low, high in coverage_ranges:
        count = sum(1 for c in coverages if low <= c < high)
        percentage = count / len(coverages) * 100
        print(f"  {low}-{high}: {count} ({percentage:.1f}%)")
    full_coverage = sum(1 for c in coverages if c >= 0.95)
    print(f"  0.95-1.0: {full_coverage} ({full_coverage/len(coverages)*100:.1f}%)")


def save_results(results: List[Dict], output_path: str):
    """保存评估结果到JSON文件"""
    # 准备输出数据（简化某些字段）
    output_results = []
    for r in results:
        output_r = r.copy()
        # 截断过长的文本以避免输出文件过大
        if 'golden_chunk' in output_r:
            output_r['golden_chunk'] = output_r['golden_chunk'][:200] + '...' if len(output_r['golden_chunk']) > 200 else output_r['golden_chunk']
        if 'query' in output_r:
            output_r['query'] = output_r['query'][:100] + '...' if len(output_r['query']) > 100 else output_r['query']
        output_results.append(output_r)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='基于覆盖度的Embedding召回率评估')
    parser.add_argument('--data', type=str, required=True, help='测试数据JSON文件路径')
    parser.add_argument('--collection', type=str, required=True, help='Milvus collection名称')
    parser.add_argument('--top-k', type=int, default=10, help='召回的chunk数量')
    parser.add_argument('--threshold', type=float, default=0.7, help='覆盖度阈值')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='结果输出路径')

    global args
    args = parser.parse_args()

    # 加载测试数据
    print(f"加载测试数据: {args.data}")
    queries, golden_chunks = load_test_data(args.data)

    # 连接Milvus
    connect_milvus()

    # 检查collection是否存在
    if not utility.has_collection(args.collection):
        print(f"错误: Collection '{args.collection}' 不存在")
        return

    # 初始化embedding模型（需要用户自己实现）
    model = None  # 在这里初始化你的embedding模型

    # 运行评估
    print(f"\n开始评估 (Top-K={args.top_k}, 阈值={args.threshold})...")
    results = calculate_recall_at_k(
        queries=queries,
        golden_chunks=golden_chunks,
        collection_name=args.collection,
        model=model,
        top_k=args.top_k,
        threshold=args.threshold
    )

    # 打印摘要
    print_evaluation_summary(results, args.top_k)

    # 保存结果
    save_results(results, args.output)


if __name__ == '__main__':
    main()
