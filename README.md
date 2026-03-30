# Embed Eval & Data Generation

本项目是一个用于评测 Embedding 召回效果，并通过 Reranker 模型挖掘难负样本以构建训练数据的工具链。

## 核心流程与功能

本项目包含以下三个主要阶段，形成了一套完整的 RAG 检索评测与训练数据构造流程：

1. **向量召回评测数据生成 (`src/recall/recall.py`)**
   - 请求 Embedding 模型接口，对 `qa_rag.json` 中的问题提取特征向量。
   - 连接 Milvus 向量数据库进行 ANN 相似度检索，召回 Top-K 相关文本块 (chunks)。
   - 结果保存至 `output.json`，其中包含原始问题、标准答案 (golden chunks) 以及召回的候选文本块。

2. **召回效果评测 (`src/eval/metric.py`)**
   - 读取召回结果 `output.json`。
   - 使用基于 Jaccard 相似度的覆盖度计算策略，对检索回来的文本块与标准答案进行比对评估。
   - 计算包含命中率 (Hit@K)、召回率 (Recall@K) 等核心指标。
   - 输出详细的评估报告，支持导出为 JSON、CSV 和 HTML 格式（存放于 `results/` 目录下）。

3. **难负样本训练数据构造 (`src/datagen/gen_data.py`)**
   - 读取召回结果 `output.json`。
   - 调用 Reranker 模型对每个候选 Chunk 重新打分。
   - 以 Golden Chunk 的得分作为上限进行分数的 Min-Max 归一化。
   - 提取归一化分数落在指定区间（如 0.15 - 0.35，可通过 `.env` 配置）的候选 Chunk 作为难负样本。
   - 最终输出为 JSONL 格式的微调训练数据 (`train_data.jsonl`)，可直接用于 Reranker 或 Embedding 模型的对比学习训练。
   - 同时输出一份详细打分记录文件 (`scores_data.jsonl`)，记录每个 Chunk 的原始得分和归一化得分，方便人工抽查分析。

## 快速开始

### 1. 环境准备

建议使用 Python 3.8+，首先安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件创建 `.env` 文件，并填写相关配置：

```bash
cp .env.example .env
```

主要配置项说明：
- **Milvus 配置**: `MILVUS_HOST`, `MILVUS_PORT`, `DB_NAME`, `COLLECTION_NAME` 等。
- **Embedding 接口**: `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` 等。
- **Reranker 接口**: `RERANKER_BASE_URL`, `RERANKER_API_KEY`, `RERANKER_MODEL` 等。
- **难负样本区间**: `HARD_NEGATIVE_MIN` (默认 0.15), `HARD_NEGATIVE_MAX` (默认 0.35)。
- **输出文件配置**: `DATAGEN_OUTPUT_PATH` (默认 `train_data.jsonl`), `SCORES_OUTPUT_PATH` (默认 `scores_data.jsonl`)。

### 3. 运行流程

**Step 1: 执行向量召回**
```bash
python src/recall/recall.py
```
> 读取输入文件 `qa_rag.json`，在 Milvus 数据库中检索后输出至 `output.json`。

**Step 2: 评估召回效果 (可选)**
```bash
python src/eval/metric.py --threshold 0.3
```
> 生成评估指标并在 `results/` 下生成 CSV/JSON/HTML 报告。

**Step 3: 构造难负样本训练数据**
```bash
python src/datagen/gen_data.py
```
> 利用 Reranker 给候选文本块打分，挖掘出难负样本并输出至 `train_data.jsonl`。

## 文件结构说明

```
embed_eval/
├── src/
│   ├── recall/
│   │   └── recall.py          # 连接 Milvus 执行向量检索
│   ├── eval/
│   │   └── metric.py          # 基于 Jaccard 策略的召回率评测脚本
│   ├── datagen/
│   │   └── gen_data.py        # Reranker 打分与难负样本挖掘脚本
│   └── utils/
│       └── formatters.py      # 提供评测结果格式化导出 (CSV/HTML/JSON) 功能
├── requirements.txt           # 项目依赖
└── .env.example               # 环境变量模板配置
```