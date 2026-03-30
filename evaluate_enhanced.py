#!/usr/bin/env python3
"""
增强版Embedding召回率评估脚本
支持多种匹配策略（词级别Jaccard相似度、N-gram重叠、句子覆盖度）
多种输出格式（CSV、JSON、HTML可视化报告）

注意：此脚本仅做评估，需要从output.json读取已检索的chunks
"""

import os
import json
import re
import argparse
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from tqdm import tqdm
import jieba


def load_evaluation_data(json_path: str) -> List[Dict]:
    """
    从JSON文件加载评估数据

    期望格式：
    [
      {
        "query": "问题1",
        "golden_chunk": "标准答案...",
        "retrieved_chunks": [
          {"text": "chunk1", "score": 0.95},
          {"text": "chunk2", "score": 0.90}
        ]
      },
      ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 验证格式
    if not isinstance(data, list):
        raise ValueError("output.json应该是一个列表格式")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第{i}项不是字典格式")

        if 'query' not in item:
            raise ValueError(f"第{i}项缺少'query'字段")
        if 'golden_chunk' not in item:
            raise ValueError(f"第{i}项缺少'golden_chunk'字段")
        if 'retrieved_chunks' not in item:
            raise ValueError(f"第{i}项缺少'retrieved_chunks'字段")

    print(f"加载了 {len(data)} 条评估数据")
    return data


def has_chinese(text: str) -> bool:
    """检测文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def tokenize(text: str) -> List[str]:
    """分词（支持中文和英文）"""
    text = text.strip()
    if not text:
        return []

    # 移除标点符号（保留中文字符）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

    if has_chinese(text):
        # 中文使用jieba分词
        words = list(jieba.cut(text))
    else:
        # 英文按空格分词
        words = text.split()

    # 过滤空字符串
    return [w for w in words if w]


def jaccard_similarity(text1: str, text2: str) -> float:
    """计算词级别的Jaccard相似度"""
    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def get_ngrams(text: str, n: int) -> Set[str]:
    """获取字符n-gram集合"""
    text = re.sub(r'\s', '', text.lower())  # 移除空格并转小写
    if len(text) < n:
        return set()

    return {text[i:i+n] for i in range(len(text) - n + 1)}


def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """计算字符n-gram重叠度"""
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0


def split_sentences(text: str) -> List[str]:
    """将文本分句"""
    text = text.strip()
    # 支持中英文标点
    sentences = re.split(r'[。！？\n；;.!?]', text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_coverage(golden_chunk: str, retrieved_chunks: List[Dict],
                     threshold: float = 0.7) -> Dict:
    """计算句子级别的覆盖度"""
    golden_sentences = split_sentences(golden_chunk)
    total_sentences = len(golden_sentences)

    if total_sentences == 0:
        return {
            'coverage': 0.0,
            'total_sentences': 0,
            'covered_sentences': 0,
            'covered_list': [],
            'uncovered_list': []
        }

    covered_sentences = []
    uncovered_sentences = []

    for sentence in golden_sentences:
        is_covered = False
        for chunk in retrieved_chunks:
            retrieved_text = chunk.get('text', '')
            # 精确包含
            if sentence in retrieved_text:
                is_covered = True
                break
            # 模糊匹配（Jaccard相似度）
            if jaccard_similarity(sentence, retrieved_text) >= threshold:
                is_covered = True
                break

        if is_covered:
            covered_sentences.append(sentence)
        else:
            uncovered_sentences.append(sentence)

    coverage = len(covered_sentences) / total_sentences

    return {
        'coverage': coverage,
        'total_sentences': total_sentences,
        'covered_sentences': len(covered_sentences),
        'covered_list': covered_sentences,
        'uncovered_list': uncovered_sentences
    }


def calculate_similarity(text1: str, text2: str, strategy: str,
                       ngram_size: int = 3) -> float:
    """根据策略计算两个文本的相似度"""
    if strategy == 'jaccard':
        return jaccard_similarity(text1, text2)
    elif strategy == 'ngram':
        return ngram_overlap(text1, text2, ngram_size)
    elif strategy == 'sentence':
        # 句子级别的相似度：计算句子重叠比例
        sentences1 = split_sentences(text1)
        sentences2 = split_sentences(text2)

        if not sentences1 and not sentences2:
            return 1.0
        if not sentences1 or not sentences2:
            return 0.0

        covered = 0
        for s1 in sentences1:
            for s2 in sentences2:
                if jaccard_similarity(s1, s2) >= 0.7:
                    covered += 1
                    break

        return covered / len(sentences1)
    elif strategy == 'combined':
        # 组合策略：加权平均
        jaccard = jaccard_similarity(text1, text2)
        ngram = ngram_overlap(text1, text2, ngram_size)
        return (jaccard * 0.6 + ngram * 0.4)
    else:
        raise ValueError(f"不支持的匹配策略: {strategy}")


def calculate_coverage(
    golden_chunk: str,
    retrieved_chunks: List[Dict],
    strategy: str = 'jaccard',
    threshold: float = 0.3,
    ngram_size: int = 3,
    top_n_matches: int = 5
) -> Dict:
    """
    计算覆盖度（支持多种匹配策略）

    Args:
        golden_chunk: golden标准答案文本
        retrieved_chunks: 检索到的chunks列表
        strategy: 匹配策略 (jaccard, ngram, sentence, combined)
        threshold: 判断被覆盖的相似度阈值
        ngram_size: n-gram大小
        top_n_matches: 返回top N个最佳匹配

    Returns:
        包含详细覆盖度指标的字典
    """
    if not golden_chunk:
        return {
            'coverage_ratio': 0.0,
            'sentence_coverage': {},
            'word_coverage': {'total_words': 0, 'covered_words': 0, 'coverage': 0.0},
            'best_matches': []
        }

    # 方法1: 整体相似度匹配（计算golden与每个retrieved chunk的相似度）
    similarities = []
    for chunk in retrieved_chunks:
        sim = calculate_similarity(golden_chunk, chunk['text'], strategy, ngram_size)
        similarities.append({
            'chunk_text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
            'similarity': sim,
            'score': chunk.get('score', 0.0)
        })

    # 按相似度排序
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    # 计算覆盖度（最高的相似度作为整体覆盖度）
    max_similarity = similarities[0]['similarity'] if similarities else 0.0
    coverage_ratio = 1.0 if max_similarity >= threshold else max_similarity / threshold

    # 方法2: 句子级别覆盖度
    sentence_result = sentence_coverage(golden_chunk, retrieved_chunks, threshold)

    # 方法3: 词级别覆盖度
    golden_words = set(tokenize(golden_chunk))
    retrieved_texts = ' '.join([c['text'] for c in retrieved_chunks])
    retrieved_words = set(tokenize(retrieved_texts))

    total_words = len(golden_words)
    if total_words > 0:
        covered_words = len(golden_words & retrieved_words)
        word_coverage_ratio = covered_words / total_words
    else:
        covered_words = 0
        word_coverage_ratio = 0.0

    # 最佳匹配
    best_matches = similarities[:top_n_matches]

    return {
        'coverage_ratio': coverage_ratio,
        'sentence_coverage': sentence_result,
        'word_coverage': {
            'total_words': total_words,
            'covered_words': covered_words,
            'coverage': word_coverage_ratio
        },
        'best_matches': best_matches,
        'strategy': strategy
    }


def evaluate_data(
    data: List[Dict],
    strategy: str = 'jaccard',
    threshold: float = 0.3,
    ngram_size: int = 3
) -> List[Dict]:
    """
    对已有数据进行评估

    Args:
        data: 加载的评估数据列表
        strategy: 匹配策略 (jaccard, ngram, sentence, combined)
        threshold: 覆盖度阈值
        ngram_size: n-gram大小

    Returns:
        每个query的详细评估结果列表
    """
    results = []

    for idx, item in enumerate(
        tqdm(data, total=len(data), desc="评估中")
    ):
        try:
            query = item['query']
            golden_chunk = item['golden_chunk']
            retrieved = item['retrieved_chunks']

            # 计算覆盖度
            coverage_metrics = calculate_coverage(
                golden_chunk,
                retrieved,
                strategy=strategy,
                threshold=threshold,
                ngram_size=ngram_size
            )

            top_k = len(retrieved)

            # 计算Hit@K (覆盖度超过阈值)
            hit_at_k = 1 if coverage_metrics['coverage_ratio'] >= threshold else 0

            # 计算位置加权召回（前面的结果更重要）
            position_weights = [1.0 / (i + 1) for i in range(len(retrieved))]
            similarities = [m['similarity'] for m in coverage_metrics['best_matches']]
            weighted_recall = sum(w * s for w, s in zip(position_weights, similarities))

            # 计算Precision@K
            precision = hit_at_k / top_k if top_k > 0 else 0.0

            # 计算Recall@不同K值
            recall_at_1 = 1.0 if coverage_metrics['coverage_ratio'] >= threshold else 0.0
            recall_at_3 = recall_at_1 if len(retrieved) >= 3 else coverage_metrics['coverage_ratio']
            recall_at_5 = recall_at_1 if len(retrieved) >= 5 else coverage_metrics['coverage_ratio']
            recall_at_10 = coverage_metrics['coverage_ratio']

            result = {
                'query_id': idx,
                'query': query,
                'golden_chunk': golden_chunk,
                'retrieved_count': len(retrieved),
                'strategy': strategy,
                'coverage_ratio': coverage_metrics['coverage_ratio'],
                'sentence_coverage': coverage_metrics['sentence_coverage']['coverage'],
                'word_coverage': coverage_metrics['word_coverage']['coverage'],
                'hit_at_k': hit_at_k,
                'precision_at_k': precision,
                'weighted_recall': weighted_recall,
                'recall_at_1': recall_at_1,
                'recall_at_3': recall_at_3,
                'recall_at_5': recall_at_5,
                'recall_at_10': recall_at_10,
                'best_match_score': coverage_metrics['best_matches'][0]['similarity'] if coverage_metrics['best_matches'] else 0.0,
                'best_match_text': coverage_metrics['best_matches'][0]['chunk_text'] if coverage_metrics['best_matches'] else '',
                'best_matches': coverage_metrics['best_matches']
            }

            results.append(result)

        except Exception as e:
            print(f"处理query失败: {item.get('query', '')[:50]}... 错误: {e}")
            results.append({
                'query_id': idx,
                'query': item.get('query', ''),
                'golden_chunk': item.get('golden_chunk', ''),
                'retrieved_count': 0,
                'strategy': strategy,
                'error': str(e)
            })

    return results


def calculate_metrics(results: List[Dict], top_k: int) -> Dict:
    """
    计算所有评估指标

    Returns:
        包含所有统计指标的字典
    """
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        return {'error': '没有有效的评估结果'}

    # 覆盖度统计
    coverages = [r['coverage_ratio'] for r in valid_results]
    avg_coverage = np.mean(coverages)
    median_coverage = np.median(coverages)
    std_coverage = np.std(coverages)

    # Recall@K统计
    hit_at_k = [r['hit_at_k'] for r in valid_results]
    avg_hit_at_k = np.mean(hit_at_k)

    # 不同K值的Recall
    recall_1 = np.mean([r.get('recall_at_1', 0) for r in valid_results])
    recall_3 = np.mean([r.get('recall_at_3', 0) for r in valid_results])
    recall_5 = np.mean([r.get('recall_at_5', 0) for r in valid_results])
    recall_10 = np.mean([r.get('recall_at_10', 0) for r in valid_results])

    # Precision统计
    precisions = [r['precision_at_k'] for r in valid_results]
    avg_precision = np.mean(precisions)

    # F1统计
    f1_scores = []
    for r in valid_results:
        p = r['precision_at_k']
        rec = r['coverage_ratio']
        if p + rec > 0:
            f1 = 2 * p * rec / (p + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    avg_f1 = np.mean(f1_scores)

    # mAP (Mean Average Precision)
    average_precisions = []
    for r in valid_results:
        # 简化的AP计算：考虑位置
        ap = 0.0
        for k in range(1, min(top_k, len(r.get('best_matches', []))) + 1):
            if r['best_matches'][k-1]['similarity'] >= 0.3:  # 使用阈值判断相关
                ap += 1.0 / k
        average_precisions.append(ap / top_k if top_k > 0 else 0.0)
    map_score = np.mean(average_precisions)

    # 分位数统计
    percentiles = {
        'p25': np.percentile(coverages, 25),
        'p50': np.percentile(coverages, 50),
        'p75': np.percentile(coverages, 75),
        'p90': np.percentile(coverages, 90),
        'p95': np.percentile(coverages, 95),
        'min': np.min(coverages),
        'max': np.max(coverages)
    }

    # 覆盖度区间分布
    coverage_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0)]
    distribution = {}
    for low, high in coverage_ranges:
        count = sum(1 for c in coverages if low <= c < high)
        distribution[f'{low}-{high}'] = {
            'count': count,
            'percentage': count / len(coverages) * 100 if coverages else 0.0
        }

    return {
        'sample_count': len(valid_results),
        'top_k': top_k,
        'coverage': {
            'mean': avg_coverage,
            'median': median_coverage,
            'std': std_coverage,
            'percentiles': percentiles,
            'distribution': distribution
        },
        'recall': {
            'hit_at_k': avg_hit_at_k,
            'recall_at_1': recall_1,
            'recall_at_3': recall_3,
            'recall_at_5': recall_5,
            'recall_at_10': recall_10
        },
        'precision': {
            'mean': avg_precision
        },
        'f1': {
            'mean': avg_f1
        },
        'map': map_score,
        'strategy': valid_results[0].get('strategy', 'unknown') if valid_results else 'unknown'
    }


def main():
    parser = argparse.ArgumentParser(description='增强版Embedding召回率评估（仅评估模式）')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径（output.json，包含已检索的chunks）')
    parser.add_argument('--strategy', type=str, default='jaccard',
                       choices=['jaccard', 'ngram', 'sentence', 'combined'],
                       help='匹配策略 (jaccard=词级别Jaccard, ngram=字符n-gram, sentence=句子级别, combined=组合)')
    parser.add_argument('--threshold', type=float, default=0.3, help='覆盖度阈值')
    parser.add_argument('--ngram-size', type=int, default=3, help='n-gram大小')
    parser.add_argument('--output-dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--formats', type=str, default='csv,json,html',
                       help='输出格式，逗号分隔 (csv, json, html)')

    args = parser.parse_args()

    # 解析输出格式
    output_formats = [f.strip().lower() for f in args.formats.split(',')]
    valid_formats = {'csv', 'json', 'html'}
    output_formats = [f for f in output_formats if f in valid_formats]

    if not output_formats:
        print("警告: 没有有效的输出格式，默认使用json")
        output_formats = ['json']

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载评估数据
    print(f"加载评估数据: {args.input}")
    data = load_evaluation_data(args.input)

    # 计算平均Top-K
    avg_top_k = int(np.mean([len(item['retrieved_chunks']) for item in data]))

    # 运行评估
    print(f"\n开始评估 (平均Top-K={avg_top_k}, 策略={args.strategy}, 阈值={args.threshold})...")
    results = evaluate_data(
        data=data,
        strategy=args.strategy,
        threshold=args.threshold,
        ngram_size=args.ngram_size
    )

    # 计算指标
    metrics = calculate_metrics(results, avg_top_k)

    # 打印摘要
    print("\n" + "="*60)
    print("评估结果摘要")
    print("="*60)
    print(f"策略: {args.strategy}")
    print(f"评估样本数: {metrics['sample_count']}")
    print(f"平均Top-K: {avg_top_k}")
    print(f"\n覆盖度指标:")
    print(f"  平均覆盖度: {metrics['coverage']['mean']:.3f}")
    print(f"  中位数: {metrics['coverage']['median']:.3f}")
    print(f"  标准差: {metrics['coverage']['std']:.3f}")
    print(f"\n召回指标:")
    print(f"  Hit@{avg_top_k}: {metrics['recall']['hit_at_k']:.3f}")
    print(f"  Recall@1: {metrics['recall']['recall_at_1']:.3f}")
    print(f"  Recall@3: {metrics['recall']['recall_at_3']:.3f}")
    print(f"  Recall@5: {metrics['recall']['recall_at_5']:.3f}")
    print(f"  Recall@10: {metrics['recall']['recall_at_10']:.3f}")
    print(f"\n其他指标:")
    print(f"  平均Precision@{avg_top_k}: {metrics['precision']['mean']:.3f}")
    print(f"  平均F1: {metrics['f1']['mean']:.3f}")
    print(f"  mAP: {metrics['map']:.3f}")

    print(f"\n覆盖度分位数:")
    p = metrics['coverage']['percentiles']
    print(f"  25%分位: {p['p25']:.3f}")
    print(f"  50%分位: {p['p50']:.3f}")
    print(f"  75%分位: {p['p75']:.3f}")
    print(f"  90%分位: {p['p90']:.3f}")
    print(f"  95%分位: {p['p95']:.3f}")
    print(f"  最小值: {p['min']:.3f}")
    print(f"  最大值: {p['max']:.3f}")

    print(f"\n覆盖度分布:")
    for range_name, stats in metrics['coverage']['distribution'].items():
        print(f"  {range_name}: {stats['count']} ({stats['percentage']:.1f}%)")

    # 保存结果
    from output_formatters import (
        save_csv_summary,
        save_json_detailed,
        generate_html_report
    )

    strategy_suffix = f"_{args.strategy}"

    if 'csv' in output_formats:
        csv_path = os.path.join(args.output_dir, f'evaluation_summary{strategy_suffix}.csv')
        save_csv_summary(results, metrics, csv_path)
        print(f"\nCSV报告已保存到: {csv_path}")

    if 'json' in output_formats:
        json_path = os.path.join(args.output_dir, f'evaluation_detailed{strategy_suffix}.json')
        save_json_detailed(results, json_path)
        print(f"JSON报告已保存到: {json_path}")

    if 'html' in output_formats:
        html_path = os.path.join(args.output_dir, f'evaluation_report{strategy_suffix}.html')
        generate_html_report(results, metrics, html_path)
        print(f"HTML可视化报告已保存到: {html_path}")


if __name__ == '__main__':
    main()
