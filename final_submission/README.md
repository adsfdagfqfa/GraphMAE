# Final Project Submission Package

本目录整理了本项目提交所需的报告、核心代码、运行脚本、环境依赖和实验日志。

## 目录结构

```text
final_submission/
├── README.md
├── code/
│   ├── graphmae/
│   ├── main_graph.py
│   ├── main_transductive.py
│   ├── configs.yml
│   ├── requirements.txt
│   ├── requirements_autodl.txt
│   └── scripts/
├── logs/
│   ├── baseline_random/
│   ├── pagerank_preserve_fast_r020_power05/
│   └── pagerank_preserve_large_r020_power05/
└── report/
    └── graphmae_pagerank_mask_report.md
```

## 核心代码

- `code/graphmae/models/edcoder.py`  
  GraphMAE 原始随机 mask baseline。

- `code/graphmae/models/edcoder_pagerank_preserve_mask.py`  
  本项目主要方法 PageRank preserve mask。该方法保留高 PageRank 节点作为上下文，更倾向遮盖低 PageRank 节点。

- `code/graphmae/models/edcoder_pagerank_mask.py`  
  PageRank preserve 的内部辅助文件，提供 PageRank 计算和 curriculum 调度；它不作为本文单独提交的方法。

- `code/graphmae/models/__init__.py`  
  只保留 `random` baseline 和 `pagerank_preserve_curriculum` 两种 mask 入口。

- `code/graphmae/utils.py`  
  包含 `mask_strategy`、`pagerank_mask_ratio`、`pagerank_preserve_power` 等命令行参数。

- `code/main_transductive.py`  
  节点分类实验入口。

- `code/main_graph.py`  
  图分类实验入口。

## 运行脚本

- `code/scripts/run_medium_3090.sh`  
  随机 mask baseline 的节点分类和图分类实验。

- `code/scripts/run_pagerank_preserve_curriculum_fast.sh`  
  PageRank preserve 在 fast 数据集上的实验。

- `code/scripts/run_pagerank_preserve_curriculum_large.sh`  
  PageRank preserve 在 NCI1、COLLAB、REDDIT-BINARY 上的实验。

## 实验日志

- `logs/baseline_random/`  
  原始 GraphMAE 随机 mask baseline。来源：`logs/medium_3090_20260620_183824`。

- `logs/pagerank_preserve_fast_r020_power05/`  
  PageRank preserve 主实验配置：`pagerank_mask_ratio=0.20`，`pagerank_preserve_power=0.5`。包含 Citeseer、Cora、Pubmed、MUTAG、IMDB-BINARY、IMDB-MULTI、PROTEINS。

- `logs/pagerank_preserve_large_r020_power05/`  
  PageRank preserve 大图实验配置：`pagerank_mask_ratio=0.20`，`pagerank_preserve_power=0.5`。包含 NCI1、COLLAB、REDDIT-BINARY。

每个日志目录中的 `summary.log` 是主要结果汇总；单个 `.log` 文件保留了对应数据集的完整训练和评估输出。

## 环境依赖

优先使用：

```bash
pip install -r code/requirements_autodl.txt
```

`requirements_autodl.txt` 针对 AutoDL / CUDA 环境整理，包含 DGL、NumPy、SciPy、scikit-learn、OGB 等依赖。

## 报告

报告 Markdown 文件位于：

```text
report/graphmae_pagerank_mask_report.md
```
