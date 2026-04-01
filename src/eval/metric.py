#!/usr/bin/env python3
"""
Embedding召回率评估脚本
使用Jaccard策略，输出覆盖度指标和召回指标
"""

import os
import json
import re
import argparse
from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm
import jieba


def load_evaluation_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    normalized = []
    for item in data:
        normalized.append({
            'query': item['question'],
            'golden_chunk': '\n\n'.join(item['goldenchunks']),
            'retrieved_chunks': [
                {
                    'text': chunk['chunk_text'],
                    'score': chunk['distance'],
                }
                for chunk in item['chunks']
            ],
        })

    print(f"加载了 {len(normalized)} 条评估数据")
    return normalized


def has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    if has_chinese(text):
        words = list(jieba.cut(text))
    else:
        words = text.split()
    return [w for w in words if w]


def jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'[。！？\n；;.!?]', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def sentence_coverage(golden_chunk: str, retrieved_chunks: List[Dict],
                      threshold: float = 0.7) -> float:
    golden_sentences = split_sentences(golden_chunk)
    if not golden_sentences:
        return 0.0

    covered = 0
    for sentence in golden_sentences:
        for chunk in retrieved_chunks:
            retrieved_text = chunk.get('text', '')
            if sentence in retrieved_text or jaccard_similarity(sentence, retrieved_text) >= threshold:
                covered += 1
                break

    return covered / len(golden_sentences)


def calculate_coverage(
    golden_chunk: str,
    retrieved_chunks: List[Dict],
    threshold: float = 0.3,
) -> Dict:
    if not golden_chunk:
        return {
            'coverage_ratio': 0.0,
            'sentence_coverage': 0.0,
            'word_coverage': 0.0,
            'best_match_similarity': 0.0,
        }

    similarities = [
        jaccard_similarity(golden_chunk, chunk['text'])
        for chunk in retrieved_chunks
    ]

    max_similarity = max(similarities) if similarities else 0.0
    coverage_ratio = 1.0 if max_similarity >= threshold else max_similarity / threshold

    sent_cov = sentence_coverage(golden_chunk, retrieved_chunks, threshold)

    golden_words = set(tokenize(golden_chunk))
    retrieved_words = set(tokenize(' '.join(c['text'] for c in retrieved_chunks)))
    word_cov = len(golden_words & retrieved_words) / len(golden_words) if golden_words else 0.0

    return {
        'coverage_ratio': coverage_ratio,
        'sentence_coverage': sent_cov,
        'word_coverage': word_cov,
        'best_match_similarity': max_similarity,
    }



def _hit_at_k(golden: str, chunks: List[Dict], k: int, threshold: float) -> float:
    topk = chunks[:k]
    cov = calculate_coverage(golden, topk, threshold=threshold)
    return 1.0 if cov['best_match_similarity'] >= threshold else 0.0


def evaluate_data(data: List[Dict], threshold: float = 0.3) -> List[Dict]:
    results = []

    for idx, item in enumerate(tqdm(data, desc="评估中")):
        try:
            retrieved_chunks = item['retrieved_chunks']

            # 调试：打印前5条的详细信息
            # if idx < 5:
            #     print(f"\n[DEBUG idx={idx}] {item['query'][:30]}")
            #     print(f"  总chunk数: {len(retrieved_chunks)}")
            #     for i, c in enumerate(retrieved_chunks):
            #         sim = jaccard_similarity(item['golden_chunk'], c['text'])
            #         print(f"  chunk[{i}] sim={sim:.3f} | {c['text'][:50]}")

            full_coverage = calculate_coverage(
                item['golden_chunk'],
                retrieved_chunks,
                threshold=threshold,
            )

            recall_at_1  = _hit_at_k(item['golden_chunk'], retrieved_chunks, 1,  threshold)
            recall_at_3  = _hit_at_k(item['golden_chunk'], retrieved_chunks, 3,  threshold)
            recall_at_5  = _hit_at_k(item['golden_chunk'], retrieved_chunks, 5,  threshold)
            recall_at_10 = _hit_at_k(item['golden_chunk'], retrieved_chunks, 10, threshold)

            if idx < 5:
                print(f"  full best_sim={full_coverage['best_match_similarity']:.3f}")
                print(f"  recall@1={recall_at_1} @3={recall_at_3} @5={recall_at_5} @10={recall_at_10}")

            results.append({
                'query_id': idx,
                'query': item['query'],
                'retrieved_count': len(retrieved_chunks),
                'coverage_ratio': full_coverage['coverage_ratio'],
                'sentence_coverage': full_coverage['sentence_coverage'],
                'word_coverage': full_coverage['word_coverage'],
                'best_match_similarity': full_coverage['best_match_similarity'],
                'recall_at_1':  recall_at_1,
                'recall_at_3':  recall_at_3,
                'recall_at_5':  recall_at_5,
                'recall_at_10': recall_at_10,
            })

        except Exception as e:
            print(f"处理query失败: {item.get('query', '')[:50]}... 错误: {e}")
            results.append({
                'query_id': idx,
                'query': item.get('query', ''),
                'error': str(e),
            })

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return {'error': '没有有效的评估结果'}

    coverages = [r['coverage_ratio'] for r in valid]

    return {
        'sample_count': len(valid),
        'coverage': {
            'mean': float(np.mean(coverages)),
            'median': float(np.median(coverages)),
            'std': float(np.std(coverages)),
        },
        'recall': {
            'recall_at_1': float(np.mean([r['recall_at_1'] for r in valid])),
            'recall_at_3': float(np.mean([r['recall_at_3'] for r in valid])),
            'recall_at_5': float(np.mean([r['recall_at_5'] for r in valid])),
            'recall_at_10': float(np.mean([r['recall_at_10'] for r in valid])),
        },
    }


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description='Embedding召回率评估（Jaccard策略）')
    parser.add_argument('--input', type=str, 
                        default=os.getenv('OUTPUT_JSON_PATH', 'output.json'),
                        help='输入文件路径（output.json）')
    parser.add_argument('--threshold', type=float, 
                        default=float(os.getenv('EVAL_THRESHOLD', '0.3')), 
                        help='覆盖度阈值')
    parser.add_argument('--output-dir', type=str, 
                        default=os.getenv('EVAL_OUTPUT_DIR', 'results'), 
                        help='结果输出目录')
    parser.add_argument('--formats', type=str, 
                        default=os.getenv('EVAL_FORMATS', 'csv,json,html'),
                        help='输出格式，逗号分隔 (csv, json, html)')
    args = parser.parse_args()
    output_formats = [f.strip().lower() for f in args.formats.split(',')]
    valid_formats = {'csv', 'json', 'html'}
    output_formats = [f for f in output_formats if f in valid_formats]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载评估数据: {args.input}")
    data = load_evaluation_data(args.input)
    avg_top_k = int(np.mean([len(item['retrieved_chunks']) for item in data]))

    print(f"\n开始评估 (平均Top-K={avg_top_k}, 策略=jaccard, 阈值={args.threshold})...")
    results = evaluate_data(data, threshold=args.threshold)
    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"评估样本数: {metrics['sample_count']}")
    print(f"平均Top-K:  {avg_top_k}")

    print(f"\n覆盖度指标:")
    print(f"  平均覆盖度: {metrics['coverage']['mean']:.3f}")
    print(f"  中位数:      {metrics['coverage']['median']:.3f}")
    print(f"  标准差:      {metrics['coverage']['std']:.3f}")

    print(f"\n召回指标:")
    r = metrics['recall']
    print(f"  Recall@1:   {r['recall_at_1']:.3f}")
    print(f"  Recall@3:   {r['recall_at_3']:.3f}")
    print(f"  Recall@5:   {r['recall_at_5']:.3f}")
    print(f"  Recall@10:  {r['recall_at_10']:.3f}")

    out_path = os.path.join(args.output_dir, 'evaluation_jaccard.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}")
    valid = [r for r in results if 'error' not in r]
    print("\n=== 覆盖度分布 ===")
    for r in valid[:10]:
        print(f"query: {r['query'][:30]}")
        print(f"  best_match_similarity: {r['best_match_similarity']:.3f}")
        print(f"  word_coverage:         {r['word_coverage']:.3f}")  
        print(f"  sentence_coverage:     {r['sentence_coverage']:.3f}")
        print(f"  coverage_ratio:        {r['coverage_ratio']:.3f}")
    # 注意：下面的代码依赖外部模块 formatters，请确保该文件存在
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.formatters import (
        save_csv_summary,
        save_json_detailed,
        generate_html_report
    )

    strategy_suffix = "_jaccard"

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