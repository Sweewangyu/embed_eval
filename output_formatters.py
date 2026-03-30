#!/usr/bin/env python3
"""
输出格式化工具
支持CSV、JSON、HTML可视化报告
"""

import json
import os
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime


def save_csv_summary(results: List[Dict], metrics: Dict, output_path: str):
    """
    保存CSV统计表格

    Args:
        results: 评估结果列表
        metrics: 指标统计
        output_path: 输出文件路径
    """
    # 准备数据
    data = []

    for r in results:
        if 'error' in r:
            continue

        data.append({
            'Query ID': r['query_id'],
            'Query': r['query'][:100] + '...' if len(r['query']) > 100 else r['query'],
            'Coverage Ratio': f"{r['coverage_ratio']:.4f}",
            'Sentence Coverage': f"{r['sentence_coverage']:.4f}",
            'Word Coverage': f"{r['word_coverage']:.4f}",
            'Hit@K': r['hit_at_k'],
            'Precision@K': f"{r['precision_at_k']:.4f}",
            'Recall@1': f"{r.get('recall_at_1', 0):.4f}",
            'Recall@3': f"{r.get('recall_at_3', 0):.4f}",
            'Recall@5': f"{r.get('recall_at_5', 0):.4f}",
            'Recall@10': f"{r.get('recall_at_10', 0):.4f}",
            'Best Match Score': f"{r['best_match_score']:.4f}",
            'Strategy': r.get('strategy', 'unknown')
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 同时保存摘要统计
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_data = {
        'Metric': [
            'Sample Count',
            'Strategy',
            'Mean Coverage',
            'Median Coverage',
            'Std Coverage',
            'Hit@K',
            'Recall@1',
            'Recall@3',
            'Recall@5',
            'Recall@10',
            'Mean Precision@K',
            'Mean F1',
            'mAP',
            'Min Coverage',
            '25% Percentile',
            '75% Percentile',
            '90% Percentile',
            '95% Percentile',
            'Max Coverage'
        ],
        'Value': [
            metrics['sample_count'],
            metrics.get('strategy', 'unknown'),
            f"{metrics['coverage']['mean']:.4f}",
            f"{metrics['coverage']['median']:.4f}",
            f"{metrics['coverage']['std']:.4f}",
            f"{metrics['recall']['hit_at_k']:.4f}",
            f"{metrics['recall']['recall_at_1']:.4f}",
            f"{metrics['recall']['recall_at_3']:.4f}",
            f"{metrics['recall']['recall_at_5']:.4f}",
            f"{metrics['recall']['recall_at_10']:.4f}",
            f"{metrics['precision']['mean']:.4f}",
            f"{metrics['f1']['mean']:.4f}",
            f"{metrics['map']:.4f}",
            f"{metrics['coverage']['percentiles']['min']:.4f}",
            f"{metrics['coverage']['percentiles']['p25']:.4f}",
            f"{metrics['coverage']['percentiles']['p75']:.4f}",
            f"{metrics['coverage']['percentiles']['p90']:.4f}",
            f"{metrics['coverage']['percentiles']['p95']:.4f}",
            f"{metrics['coverage']['percentiles']['max']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')


def save_json_detailed(results: List[Dict], output_path: str):
    """
    保存JSON详细结果

    Args:
        results: 评估结果列表
        output_path: 输出文件路径
    """
    # 准备输出数据（截断过长的文本）
    output_results = []

    for r in results:
        output_r = r.copy()

        # 截断过长的文本
        if 'golden_chunk' in output_r:
            chunk = output_r['golden_chunk']
            output_r['golden_chunk'] = chunk[:500] + '...' if len(chunk) > 500 else chunk

        if 'query' in output_r:
            query = output_r['query']
            output_r['query'] = query[:200] + '...' if len(query) > 200 else query

        if 'best_match_text' in output_r:
            text = output_r['best_match_text']
            output_r['best_match_text'] = text[:500] + '...' if len(text) > 500 else text

        # 格式化best_matches
        if 'best_matches' in output_r:
            output_r['best_matches'] = output_r['best_matches'][:5]  # 只保留前5个

        output_results.append(output_r)

    # 保存JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)


def generate_html_report(results: List[Dict], metrics: Dict, output_path: str):
    """
    生成HTML可视化报告

    Args:
        results: 评估结果列表
        metrics: 指标统计
        output_path: 输出文件路径
    """
    # 准备图表数据
    coverage_data = [r['coverage_ratio'] for r in results if 'error' not in r]

    # 生成HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding评估报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        .chart-card h3 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 15px;
        }}
        .distribution-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }}
        .distribution-table th,
        .distribution-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .distribution-table th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        .distribution-table tr:hover {{
            background: #f8f9fa;
        }}
        .distribution-table td:last-child {{
            font-weight: bold;
            color: #667eea;
        }}
        .query-list {{
            margin-top: 20px;
        }}
        .query-item {{
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .query-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .query-id {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}
        .query-text {{
            font-style: italic;
            color: #666;
        }}
        .query-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-mini {{
            text-align: center;
        }}
        .metric-mini-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #764ba2;
        }}
        .metric-mini-label {{
            font-size: 0.8em;
            color: #999;
        }}
        .coverage-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .coverage-high {{
            background: #28a745;
        }}
        .coverage-medium {{
            background: #ffc107;
            color: #333;
        }}
        .coverage-low {{
            background: #dc3545;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
        .best-match {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 6px;
            margin-top: 10px;
        }}
        .best-match-title {{
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 5px;
        }}
        .best-match-text {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Embedding 评估报告</h1>
            <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>

        <div class="content">
            <div class="section">
                <h2>📊 核心指标</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['sample_count']}</div>
                        <div class="metric-label">评估样本数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['coverage']['mean']:.3f}</div>
                        <div class="metric-label">平均覆盖度</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']['hit_at_k']:.3f}</div>
                        <div class="metric-label">Hit@{metrics['top_k']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['f1']['mean']:.3f}</div>
                        <div class="metric-label">平均 F1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['map']:.3f}</div>
                        <div class="metric-label">mAP</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['strategy']}</div>
                        <div class="metric-label">匹配策略</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📈 召回指标</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']['recall_at_1']:.3f}</div>
                        <div class="metric-label">Recall@1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']['recall_at_3']:.3f}</div>
                        <div class="metric-label">Recall@3</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']['recall_at_5']:.3f}</div>
                        <div class="metric-label">Recall@5</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']['recall_at_10']:.3f}</div>
                        <div class="metric-label">Recall@10</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📉 覆盖度分布</h2>
                <div class="charts-container">
                    <div class="chart-card">
                        <h3>覆盖度直方图</h3>
                        <canvas id="coverageHistogram"></canvas>
                    </div>
                    <div class="chart-card">
                        <h3>分位数分析</h3>
                        <canvas id="percentileChart"></canvas>
                    </div>
                </div>
                <table class="distribution-table">
                    <thead>
                        <tr>
                            <th>覆盖度区间</th>
                            <th>样本数量</th>
                            <th>占比</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # 添加覆盖度分布数据
    for range_name, stats in metrics['coverage']['distribution'].items():
        html_content += f"""
                        <tr>
                            <td>{range_name}</td>
                            <td>{stats['count']}</td>
                            <td>{stats['percentage']:.1f}%</td>
                        </tr>
"""

    html_content += f"""
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h2>🔍 详细评估结果</h2>
                <div class="query-list">
"""

    # 添加每个查询的详细信息
    for r in results:
        if 'error' in r:
            continue

        # 根据覆盖度设置徽章颜色
        coverage = r['coverage_ratio']
        if coverage >= 0.7:
            badge_class = 'coverage-high'
        elif coverage >= 0.4:
            badge_class = 'coverage-medium'
        else:
            badge_class = 'coverage-low'

        html_content += f"""
                    <div class="query-item">
                        <div class="query-header">
                            <div>
                                <span class="query-id">Query #{r['query_id']}</span>
                                <span class="coverage-badge {badge_class}" style="margin-left: 10px;">
                                    覆盖度: {coverage:.3f}
                                </span>
                            </div>
                        </div>
                        <div class="query-text">"{r['query']}"</div>
                        <div class="query-metrics">
                            <div class="metric-mini">
                                <div class="metric-mini-value">{r['precision_at_k']:.3f}</div>
                                <div class="metric-mini-label">Precision@{metrics['top_k']}</div>
                            </div>
                            <div class="metric-mini">
                                <div class="metric-mini-value">{r['sentence_coverage']:.3f}</div>
                                <div class="metric-mini-label">句子覆盖度</div>
                            </div>
                            <div class="metric-mini">
                                <div class="metric-mini-value">{r['word_coverage']:.3f}</div>
                                <div class="metric-mini-label">词覆盖度</div>
                            </div>
                            <div class="metric-mini">
                                <div class="metric-mini-value">{r['best_match_score']:.3f}</div>
                                <div class="metric-mini-label">最佳匹配分数</div>
                            </div>
                        </div>
"""

        if r.get('best_match_text'):
            html_content += f"""
                        <div class="best-match">
                            <div class="best-match-title">🎯 最佳匹配片段</div>
                            <div class="best-match-text">{r['best_match_text']}</div>
                        </div>
"""

        html_content += """
                    </div>
"""

    html_content += f"""
                </div>
            </div>
        </div>

        <footer>
            <p>评估报告自动生成 | Embedding评估工具 v1.0</p>
        </footer>
    </div>

    <script>
        // 覆盖度直方图
        const coverageCtx = document.getElementById('coverageHistogram').getContext('2d');
        new Chart(coverageCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r for r in metrics['coverage']['distribution'].keys()])},
                datasets: [{{
                    label: '样本数量',
                    data: {json.dumps([stats['count'] for stats in metrics['coverage']['distribution'].values()])},
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.7)',
                        'rgba(253, 126, 20, 0.7)',
                        'rgba(255, 193, 7, 0.7)',
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(23, 162, 184, 0.7)',
                        'rgba(102, 126, 234, 0.7)'
                    ],
                    borderColor: [
                        'rgb(220, 53, 69)',
                        'rgb(253, 126, 20)',
                        'rgb(255, 193, 7)',
                        'rgb(40, 167, 69)',
                        'rgb(23, 162, 184)',
                        'rgb(102, 126, 234)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // 分位数图表
        const percentileCtx = document.getElementById('percentileChart').getContext('2d');
        new Chart(percentileCtx, {{
            type: 'bar',
            data: {{
                labels: ['最小值', '25%', '50% (中位数)', '75%', '90%', '95%', '最大值'],
                datasets: [{{
                    label: '覆盖度',
                    data: [
                        {metrics['coverage']['percentiles']['min']},
                        {metrics['coverage']['percentiles']['p25']},
                        {metrics['coverage']['percentiles']['p50']},
                        {metrics['coverage']['percentiles']['p75']},
                        {metrics['coverage']['percentiles']['p90']},
                        {metrics['coverage']['percentiles']['p95']},
                        {metrics['coverage']['percentiles']['max']}
                    ],
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: 'rgb(102, 126, 234)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
