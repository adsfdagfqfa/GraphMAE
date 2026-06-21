# 基于 PageRank Preserve 的 GraphMAE Mask 策略改进

## 1. 研究动机

GraphMAE 通过随机遮盖节点属性，并利用剩余图结构和节点特征重构被遮盖节点来学习图表示。原始随机 mask 默认所有节点同等重要，但真实图中节点的结构作用不同：高中心性节点往往连接关键区域，也更可能提供有效上下文。因此，本项目尝试利用 PageRank 改进 GraphMAE 的 mask 策略。

本文主要评估 PageRank preserve，即“基于 PageRank 的重要节点保留策略”。它保留高 PageRank 节点作为可见上下文，更倾向遮盖低 PageRank 节点。实验中保持 encoder、decoder、重构损失和下游评估方式不变，只改变预训练阶段的 mask 节点选择。

## 2. 方法设计

原始 GraphMAE 的预训练目标保持不变。给定图 G 和节点特征 X，模型选择被遮盖节点 M，扰动其输入特征，再用 encoder 编码，并只在 M 上重构原始特征。本文只改“如何选择 M”。

原始 GraphMAE 均匀随机选择 mask 节点，不考虑 degree、PageRank 等结构信息。本文将它作为 baseline。PageRank preserve 引入结构中心性，但不改变 mask rate、模型结构和重构目标。

PageRank preserve 先计算每个节点的 PageRank 分数，并做 min-max 归一化。节点 i 的 mask 权重定义为：

`w_i = (1 - normalized_pagerank_i) ^ preserve_power`

因此 PageRank 越高，被 mask 的概率越低；PageRank 越低，越可能成为重构目标。主实验使用 `pagerank_mask_ratio=0.20`、`pagerank_mask_steps=5`、PageRank 迭代 20 次、阻尼系数 0.85、`pagerank_preserve_power=0.5`。其中 `pagerank_mask_ratio` 控制 preserve 采样占 mask 预算的比例，剩余部分仍随机采样；`pagerank_mask_steps` 用于 curriculum，使训练初期接近随机遮盖，后期逐步增强 preserve 影响。

直觉上，GraphMAE 重构 masked nodes 依赖可见节点提供上下文；保留中心节点可能让模型获得更稳定的结构信息。实现上，原始随机 mask 在 `graphmae/models/edcoder.py` 中；PageRank preserve 在 `graphmae/models/edcoder_pagerank_preserve_mask.py` 中，复用 PageRank curriculum 的 PageRank 计算和训练调度，只重写采样权重。模型通过 `mask_strategy=pagerank_preserve_curriculum` 选择该策略。

## 3. 实验设置

实验覆盖 GraphMAE 的节点分类和图分类任务。

**节点分类。** 使用 transductive setting，包括 Citeseer、Cora 和 Pubmed。模型先无监督预训练，再冻结 encoder，用节点 embedding 训练线性分类器。指标为多随机种子下的测试准确率，用于检验 node-level embedding 质量。

**图分类。** 使用 MUTAG、IMDB-BINARY、IMDB-MULTI、PROTEINS、NCI1、COLLAB 和 REDDIT-BINARY。预训练后先得到节点 embedding，再 pooling 成 graph embedding，并用 10-fold SVM 评估。指标为 micro-F1/accuracy，用于检验 graph-level embedding 质量。

为保证公平，本文不按数据集事后挑参，而固定 `pagerank_mask_ratio=0.20`，并采用 fast sweep 中整体最稳定的 `preserve_power=0.5`。

## 4. 实验结果

### 4.1 节点分类结果

表 1 展示节点分类结果。PageRank preserve 在三个数据集上都略高于随机 baseline，说明保留高 PageRank 节点对 node-level 表示有一定帮助，但提升幅度整体较小。

**表 1. 节点分类测试准确率。**

| Dataset | Random | PageRank preserve |
|---|---:|---:|
| Citeseer | 0.7327 +/- 0.0021 | **0.7332 +/- 0.0021** |
| Cora | 0.8423 +/- 0.0036 | **0.8440 +/- 0.0034** |
| Pubmed | 0.8110 +/- 0.0065 | **0.8117 +/- 0.0090** |

### 4.2 图分类结果

表 2 展示图分类结果。PageRank preserve 在 IMDB-BINARY、IMDB-MULTI、PROTEINS、COLLAB 和 REDDIT-BINARY 上超过 baseline，说明该策略也能改善部分 graph-level embedding。MUTAG 略低于 baseline，NCI1 下降较明显，说明 PageRank 中心性并不总能匹配任务所需结构。

**表 2. 图分类 micro-F1/accuracy。**

| Dataset | Random | PageRank preserve |
|---|---:|---:|
| MUTAG | **0.8767 +/- 0.0170** | 0.8715 +/- 0.0167 |
| IMDB-BINARY | 0.7520 +/- 0.0054 | **0.7570 +/- 0.0020** |
| IMDB-MULTI | 0.5153 +/- 0.0031 | **0.5157 +/- 0.0063** |
| PROTEINS | 0.7513 +/- 0.0054 | **0.7533 +/- 0.0047** |
| NCI1 | **0.7039 +/- 0.0028** | 0.6851 +/- 0.0054 |
| COLLAB | 0.8040 +/- 0.0021 | **0.8062 +/- 0.0019** |
| REDDIT-BINARY | 0.8783 +/- 0.0016 | **0.8790 +/- 0.0025** |

## 5. 分析与总结

实验结果说明，保留结构重要节点作为上下文是一个有意义的 GraphMAE mask 改进方向。与随机 mask 相比，PageRank preserve 在所有节点分类数据集上都有小幅提升，在多数图分类数据集上也有正向效果。这支持本文的核心直觉：中心节点不一定应被优先遮盖，它们作为可见上下文可能更有价值。

但该方法也有局限。NCI1 上的下降表明 PageRank 只是通用结构中心性指标，在分子图中未必对应有用的化学子结构。MUTAG 的小幅下降也说明，小型图中过度依赖中心性可能引入噪声。因此，PageRank preserve 更适合作为轻量级结构先验，而不是通用最优策略。

总体而言，PageRank preserve 是对 GraphMAE 的轻量级改进：不改模型结构和损失函数，只改 mask 节点选择。后续可以结合 degree、PageRank 和领域结构指标，或根据图类型自适应决定重要节点应被遮盖还是保留。
