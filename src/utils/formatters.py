#!/usr/bin/env python3
"""
输出格式化工具
支持 CSV、JSON、HTML 可视化报告
"""

import json
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from html import escape


def _truncate_text(text: Any, max_len: int) -> str:
    """安全截断文本。"""
    text = "" if text is None else str(text)
    return text[:max_len] + "..." if len(text) > max_len else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_metrics_defaults(metrics: Dict) -> Dict:
    """补齐 metrics 缺失字段，避免格式化阶段报错。"""
    metrics = metrics or {}

    coverage = metrics.get("coverage", {}) or {}
    recall = metrics.get("recall", {}) or {}

    return {
        "sample_count": metrics.get("sample_count", 0),
        "top_k": metrics.get("top_k", 0),
        "coverage": {
            "mean": _safe_float(coverage.get("mean", 0.0)),
            "median": _safe_float(coverage.get("median", 0.0)),
            "std": _safe_float(coverage.get("std", 0.0)),
        },
        "recall": {
            "hit_at_k": _safe_float(recall.get("hit_at_k", 0.0)),
            "recall_at_1": _safe_float(recall.get("recall_at_1", 0.0)),
            "recall_at_3": _safe_float(recall.get("recall_at_3", 0.0)),
            "recall_at_5": _safe_float(recall.get("recall_at_5", 0.0)),
            "recall_at_10": _safe_float(recall.get("recall_at_10", 0.0)),
        },
    }


def save_csv_summary(results: List[Dict], metrics: Dict, output_path: str):
    """保存逐条结果 CSV 和汇总 CSV。"""
    safe_metrics = _get_metrics_defaults(metrics)

    data = []
    for r in results:
        if "error" in r:
            continue

        query = _truncate_text(r.get("query", ""), 100)

        data.append({
            "Query ID": r.get("query_id", ""),
            "Query": query,
            "Coverage Ratio": f"{_safe_float(r.get('coverage_ratio', 0.0)):.4f}",
            "Sentence Coverage": f"{_safe_float(r.get('sentence_coverage', 0.0)):.4f}",
            "Word Coverage": f"{_safe_float(r.get('word_coverage', 0.0)):.4f}",
            "Hit@K": r.get("hit_at_k", 0),
            "Recall@1": f"{_safe_float(r.get('recall_at_1', 0.0)):.4f}",
            "Recall@3": f"{_safe_float(r.get('recall_at_3', 0.0)):.4f}",
            "Recall@5": f"{_safe_float(r.get('recall_at_5', 0.0)):.4f}",
            "Recall@10": f"{_safe_float(r.get('recall_at_10', 0.0)):.4f}",
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_path = output_path.replace(".csv", "_summary.csv")
    summary_data = {
        "Metric": [
            "Sample Count",
            "Mean Coverage",
            "Median Coverage",
            "Std Coverage",
            "Hit@K",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
        ],
        "Value": [
            safe_metrics["sample_count"],
            f"{safe_metrics['coverage']['mean']:.4f}",
            f"{safe_metrics['coverage']['median']:.4f}",
            f"{safe_metrics['coverage']['std']:.4f}",
            f"{safe_metrics['recall']['hit_at_k']:.4f}",
            f"{safe_metrics['recall']['recall_at_1']:.4f}",
            f"{safe_metrics['recall']['recall_at_3']:.4f}",
            f"{safe_metrics['recall']['recall_at_5']:.4f}",
            f"{safe_metrics['recall']['recall_at_10']:.4f}",
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False, encoding="utf-8-sig")


def save_json_detailed(results: List[Dict], output_path: str):
    """保存详细 JSON。"""
    output_results = []
    for r in results:
        output_r = r.copy()

        if "golden_chunk" in output_r:
            output_r["golden_chunk"] = _truncate_text(output_r.get("golden_chunk", ""), 500)

        if "query" in output_r:
            output_r["query"] = _truncate_text(output_r.get("query", ""), 200)

        output_results.append(output_r)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)


def generate_html_report(results: List[Dict], metrics: Dict, output_path: str):
    """生成 HTML 可视化报告。"""
    safe_metrics = _get_metrics_defaults(metrics)

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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        .chart-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            margin-bottom: 30px;
        }}
        .chart-card h3 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 15px;
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
            word-break: break-word;
        }}
        .query-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
        .coverage-high {{ background: #28a745; }}
        .coverage-medium {{ background: #ffc107; color: #333; }}
        .coverage-low {{ background: #dc3545; }}
        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
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
                <h2>📊 覆盖度指标</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['sample_count']}</div>
                        <div class="metric-label">评估样本数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['coverage']['mean']:.3f}</div>
                        <div class="metric-label">平均覆盖度</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['coverage']['median']:.3f}</div>
                        <div class="metric-label">中位数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['coverage']['std']:.3f}</div>
                        <div class="metric-label">标准差</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📈 召回指标</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['recall']['hit_at_k']:.3f}</div>
                        <div class="metric-label">Hit@{safe_metrics['top_k']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['recall']['recall_at_1']:.3f}</div>
                        <div class="metric-label">Recall@1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['recall']['recall_at_3']:.3f}</div>
                        <div class="metric-label">Recall@3</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['recall']['recall_at_5']:.3f}</div>
                        <div class="metric-label">Recall@5</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{safe_metrics['recall']['recall_at_10']:.3f}</div>
                        <div class="metric-label">Recall@10</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📉 覆盖度分布</h2>
                <div class="chart-card">
                    <h3>各样本覆盖度</h3>
                    <canvas id="coverageChart"></canvas>
                </div>
            </div>

            <div class="section">
                <h2>🔍 详细评估结果</h2>
                <div class="query-list">
"""

    for r in results:
        if "error" in r:
            continue

        coverage = _safe_float(r.get("coverage_ratio", 0.0))
        if coverage >= 0.7:
            badge_class = "coverage-high"
        elif coverage >= 0.4:
            badge_class = "coverage-medium"
        else:
            badge_class = "coverage-low"

        query_text = escape(str(r.get("query", "")))

        html_content += f"""
                    <div class="query-item">
                        <div class="query-header">
                            <div>
                                <span class="query-id">Query #{r.get('query_id', '')}</span>
                                <span class="coverage-badge {badge_class}" style="margin-left: 10px;">
                                    覆盖度: {coverage:.3f}
                                </span>
                            </div>
                        </div>
                        <div class="query-text">"{query_text}"</div>
                        <div class="query-metrics">
                            <div class="metric-mini">
                                <div class="metric-mini-value">{_safe_float(r.get('sentence_coverage', 0.0)):.3f}</div>
                                <div class="metric-mini-label">句子覆盖度</div>
                            </div>
                            <div class="metric-mini">
                                <div class="metric-mini-value">{_safe_float(r.get('word_coverage', 0.0)):.3f}</div>
                                <div class="metric-mini-label">词覆盖度</div>
                            </div>
                            <div class="metric-mini">
                                <div class="metric-mini-value">{r.get('hit_at_k', 0)}</div>
                                <div class="metric-mini-label">Hit@K</div>
                            </div>
                        </div>
                    </div>
"""

    coverage_values = [_safe_float(r.get("coverage_ratio", 0.0)) for r in results if "error" not in r]
    query_ids = [f"Q{r.get('query_id', '')}" for r in results if "error" not in r]

    bar_colors = [
        "rgba(40,167,69,0.7)" if v >= 0.7
        else "rgba(255,193,7,0.7)" if v >= 0.4
        else "rgba(220,53,69,0.7)"
        for v in coverage_values
    ]

    html_content += f"""
                </div>
            </div>
        </div>

        <footer>
            <p>评估报告自动生成 | Embedding评估工具 v1.0</p>
        </footer>
    </div>

    <script>
        const coverageCtx = document.getElementById('coverageChart').getContext('2d');
        new Chart(coverageCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(query_ids, ensure_ascii=False)},
                datasets: [{{
                    label: '覆盖度',
                    data: {json.dumps(coverage_values, ensure_ascii=False)},
                    backgroundColor: {json.dumps(bar_colors, ensure_ascii=False)},
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, max: 1.0 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)