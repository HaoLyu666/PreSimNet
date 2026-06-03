# Top-k 专家聚合路由分析报告

## 1. 分析目的

审稿人认为 Top-k expert aggregation 的物理含义不够充分。四个专家对应四种异质交互类型，概念上互斥；但模型使用 `k = 2` 的 Top-k 聚合，可能混合来自 ACC 与 IDM 两类物理模型的候选加速度。因此，本报告重点回答两个问题：

1. 如果采用硬路由，只选 Top-1 专家，交互类型识别错误会导致多少专家选择错误？
2. Top-2 是否能覆盖这些硬路由错选样本，并且在这些样本上的 RMSE 表现如何？
3. 若 Top-2 出现 ACC/IDM 跨物理模型选择，第二专家的实际 softmax 权重是否足够大，会不会形成实质性的物理模型混合？

## 2. 实验设置

- 主模型 checkpoint：`checkponint/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2/epoch21_e.tar`
- 测试集：`../data/test_data.npy`
- 测试样本数：`2,246,939`
- 模型文件：`model/model_MoE_gru_new.py`
- 分析脚本：`analyze_routing_topk.py`
- 输出目录：`fig/vis/routing_analysis/`

交互类型与物理专家 family 的对应关系如下：

| Type id | 类型 | 物理 family |
|---:|---|---|
| 0 | AV-HV | ACC |
| 1 | AV-AV | ACC |
| 2 | HV-HV | IDM |
| 3 | HV-AV | IDM |

本报告比较四种路由方式：

| 路由方式 | 定义 |
|---|---|
| `soft_all` | 当前代码实际实现：4 个专家加速度按 softmax 概率全量加权。 |
| `hard_top1` | 硬路由：只选 softmax 最大的专家。 |
| `top2` | 论文中描述的 Top-2：保留概率最大的两个专家，并重新归一化加权。 |
| `family_top2` | 物理约束替代设计：先选择 ACC 或 IDM family，再只在同一 family 内两个专家之间加权。 |

RMSE 计算遵循 `evaluate_new.py` 的评估口径：每个 batch 先累积 `sqrt(sum squared error)` 和 `sqrt(count)`，最后二者相除。

## 3. 核心结论

1. 硬路由本身的专家错选比例较低。  
   全测试集上，`hard_top1` 选错专家的样本数为 `15,175 / 2,246,939 = 0.675%`。

2. Top-2 几乎消除了由交互类型识别错误导致的专家遗漏。  
   真实专家不在 Top-2 中的样本仅有 `208 / 2,246,939 = 0.0093%`。在 `15,175` 个 hard Top-1 错选样本中，Top-2 能覆盖真实专家的样本为 `14,967`，覆盖率为 `98.63%`。

3. 在 hard Top-1 错选样本上，Top-2 比 hard Top-1 更稳健。  
   对 `15,175` 个 hard Top-1 错选样本，hard Top-1 的平均 gap RMSE 和 velocity RMSE 分别为 `0.3863 m` 与 `0.2435 m/s`；Top-2 分别为 `0.3825 m` 与 `0.2405 m/s`，接近当前 soft-all 聚合结果。

4. Top-2 的确可能选到跨 ACC/IDM family 的第二专家，但第二专家权重通常极小。  
   按 Top-2 pair 身份统计，跨 family pair 占 `45.25%`。但在这些跨 family pair 中，第二专家 raw softmax 权重均值仅为 `0.00535`，中位数几乎为 `0`。因此最终加速度几乎由 Top-1 专家主导，Top-2 更像是对类型识别不确定性的低权重补偿，而不是 ACC 与 IDM 的均衡混合。

5. 若审稿人坚持物理互斥性，可提供 `family_top2` 作为替代路由设计。  
   该设计从机制上禁止 ACC/IDM 混合，性能与 Top-2 接近。在 hard Top-1 错选样本上，`family_top2` 的 velocity RMSE Avg 为 `0.2389 m/s`，略优于 Top-2 的 `0.2405 m/s`；但 gap RMSE Avg 为 `0.3849 m`，略差于 Top-2 的 `0.3825 m`。

## 4. 图 1：硬路由与 Top-2 专家选择错误分析

![Routing selection analysis](fig/vis/routing_analysis/routing_selection_analysis.png)

该图包含三部分：

- 各类型下 hard Top-1 的专家错选率与 Top-2 miss rate；
- hard Top-1 错选样本中，Top-2 能覆盖回来的样本数；
- Top-2 选择中 ACC/IDM 跨 family pair 的比例。

图 1 对应数据如下：

| 类型 | 总样本数 | Hard 错选数 | Hard 错选率 | Top-2 miss 数 | Top-2 miss 率 | Top-2 覆盖数 | hard 错选中的覆盖率 | 跨 family Top-2 数 | 跨 family 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV-AV | 404,724 | 2,003 | 0.495% | 13 | 0.0032% | 1,990 | 99.35% | 27,868 | 6.886% |
| AV-HV | 552,061 | 6,537 | 1.184% | 40 | 0.0072% | 6,497 | 99.39% | 391,160 | 70.854% |
| HV-AV | 543,312 | 1,818 | 0.335% | 45 | 0.0083% | 1,773 | 97.52% | 22,407 | 4.124% |
| HV-HV | 746,842 | 4,817 | 0.645% | 110 | 0.0147% | 4,707 | 97.72% | 575,342 | 77.037% |
| 总体 | 2,246,939 | 15,175 | 0.675% | 208 | 0.0093% | 14,967 | 98.63% | 1,016,777 | 45.252% |

## 5. 图 2：Softmax 路由权重分布

![Selected weight distribution](fig/vis/routing_analysis/selected_weight_distribution.png)

该图包含三部分：

- 全样本下 Top-1/Top-2 的 raw softmax 权重与 Top-2 归一化权重；
- hard Top-1 错选样本下的权重分布；
- 按真实类型统计的平均 softmax 概率矩阵。

图 2 对应主要数据如下：

| 分组 | 指标 | n | 均值 | 标准差 | P05 | P25 | 中位数 | P75 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | top1_raw | 2,246,939 | 0.992419 | 0.046193 | 0.991456 | 0.999988 | 1.000000 | 1.000000 | 1.000000 |
| all | top2_raw | 2,246,939 | 0.007516 | 0.045886 | 0.000000 | 0.000000 | 0.000000 | 0.000012 | 0.008365 |
| all | top1_renorm | 2,246,939 | 0.992470 | 0.045974 | 0.991632 | 0.999988 | 1.000000 | 1.000000 | 1.000000 |
| all | top2_renorm | 2,246,939 | 0.007530 | 0.045974 | 0.000000 | 0.000000 | 0.000000 | 0.000012 | 0.008368 |
| all | true_expert_prob | 2,246,939 | 0.989289 | 0.072246 | 0.991234 | 0.999988 | 1.000000 | 1.000000 | 1.000000 |
| top1_wrong | top1_raw | 15,175 | 0.729607 | 0.157738 | 0.515215 | 0.588781 | 0.708315 | 0.866963 | 0.991693 |
| top1_wrong | top2_raw | 15,175 | 0.267659 | 0.156742 | 0.008094 | 0.129760 | 0.287611 | 0.408310 | 0.482631 |
| top1_wrong | top1_renorm | 15,175 | 0.731499 | 0.157092 | 0.516903 | 0.590179 | 0.710823 | 0.869629 | 0.991906 |
| top1_wrong | top2_renorm | 15,175 | 0.268501 | 0.157092 | 0.008094 | 0.130371 | 0.289177 | 0.409821 | 0.483097 |
| top1_wrong | true_expert_prob | 15,175 | 0.266146 | 0.157562 | 0.007312 | 0.126460 | 0.285955 | 0.407696 | 0.482596 |
| top2_miss | top1_raw | 208 | 0.771226 | 0.187298 | 0.448129 | 0.610209 | 0.807247 | 0.942367 | 0.998455 |
| top2_miss | top2_raw | 208 | 0.168423 | 0.142613 | 0.001093 | 0.041150 | 0.141992 | 0.278640 | 0.433853 |
| top2_miss | true_expert_prob | 208 | 0.057979 | 0.072020 | 0.000052 | 0.004105 | 0.026442 | 0.093614 | 0.221759 |

## 6. Hard Top-1 错选样本上的 RMSE

下表只统计 hard Top-1 选错专家的样本，即 `top1_wrong`，样本数为 `15,175`。这是最直接回应审稿意见的实验，因为它专门考察交互类型识别错误导致专家选择错误时，不同路由方式的影响。

| 路由方式 | n | Gap 0.5s | Gap 1.0s | Gap 1.5s | Gap 2.0s | Gap Avg | Vel 0.5s | Vel 1.0s | Vel 1.5s | Vel 2.0s | Vel Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_all | 15,175 | 0.2348 | 0.3811 | 0.5088 | 0.6516 | 0.3824 | 0.1644 | 0.2273 | 0.3071 | 0.4202 | 0.2405 |
| hard_top1 | 15,175 | 0.2348 | 0.3837 | 0.5154 | 0.6624 | 0.3863 | 0.1678 | 0.2291 | 0.3107 | 0.4267 | 0.2435 |
| top2 | 15,175 | 0.2348 | 0.3811 | 0.5088 | 0.6518 | 0.3825 | 0.1645 | 0.2273 | 0.3070 | 0.4201 | 0.2405 |
| family_top2 | 15,175 | 0.2346 | 0.3829 | 0.5132 | 0.6586 | 0.3849 | 0.1656 | 0.2238 | 0.3039 | 0.4199 | 0.2389 |

结论：

- hard Top-1 在 hard 错选样本上误差更大；
- Top-2 与 soft-all 基本一致，说明第二候选专家对错分类样本有补偿作用；
- family-constrained Top-2 可以作为更强物理约束的替代方案。

## 7. 全样本 RMSE 对比

| 路由方式 | n | Gap 0.5s | Gap 1.0s | Gap 1.5s | Gap 2.0s | Gap Avg | Vel 0.5s | Vel 1.0s | Vel 1.5s | Vel 2.0s | Vel Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_all | 2,246,939 | 0.2296 | 0.3616 | 0.4784 | 0.6155 | 0.3636 | 0.2701 | 0.3068 | 0.3555 | 0.4433 | 0.3089 |
| hard_top1 | 2,246,939 | 0.2300 | 0.3601 | 0.4715 | 0.6031 | 0.3598 | 0.2703 | 0.3212 | 0.3751 | 0.4616 | 0.3203 |
| top2 | 2,246,939 | 0.2296 | 0.3616 | 0.4784 | 0.6155 | 0.3636 | 0.2701 | 0.3068 | 0.3555 | 0.4433 | 0.3089 |
| family_top2 | 2,246,939 | 0.2300 | 0.3602 | 0.4715 | 0.6032 | 0.3598 | 0.2703 | 0.3211 | 0.3750 | 0.4614 | 0.3202 |

结论：

- `soft_all` 与 `top2` 几乎完全一致，因为 softmax 分布高度尖锐；
- `hard_top1` 与 `family_top2` 的 gap RMSE 略低，但 velocity RMSE 更高；
- 对审稿意见而言，更有解释力的是上一节的 `top1_wrong` 子集。

## 8. 跨 ACC/IDM family 的 Top-2 权重分析

虽然 Top-2 pair 身份上经常跨 ACC/IDM family，但第二专家权重通常很小，因此实际加速度不是两个异质物理模型的均衡混合。

| 类型 | 总样本数 | 跨 family 数 | 跨 family 比例 | 跨 family 时 Top-1 raw 均值 | 跨 family 时 Top-2 raw 均值 | 跨 family 时 Top-2 raw 中位数 | 跨 family 时 Top-2 renorm 均值 | 跨 family 时 Top-2 renorm 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AV-AV | 404,724 | 27,868 | 6.886% | 0.996431 | 0.003359 | 0.000003 | 0.003391 | 0.000003 |
| AV-HV | 552,061 | 391,160 | 70.854% | 0.997002 | 0.002901 | 0.000001 | 0.002918 | 0.000001 |
| HV-AV | 543,312 | 22,407 | 4.124% | 0.980510 | 0.018786 | 0.000014 | 0.018956 | 0.000014 |
| HV-HV | 746,842 | 575,342 | 77.037% | 0.993350 | 0.006591 | 0.000000 | 0.006605 | 0.000000 |
| 总体 | 2,246,939 | 1,016,777 | 45.252% | 0.994556 | 0.005352 | 0.000000 | 0.005371 | 0.000000 |

这个结果可以支撑如下表述：

> Although the Top-2 set may occasionally include experts from different physical families, the routing distribution is highly sparse. For cross-family pairs, the second expert receives only 0.00535 softmax probability on average, with a median close to zero. Therefore, the aggregated acceleration is dominated by the most probable physically matched expert, while the second expert only acts as a small uncertainty-compensation term.

## 9. Hard Top-1 混淆矩阵

行是真实类型，列是 hard Top-1 预测类型。

| 真实类型 | Pred AV-AV | Pred AV-HV | Pred HV-AV | Pred HV-HV |
|---|---:|---:|---:|---:|
| AV-AV | 402,721 | 1,934 | 12 | 57 |
| AV-HV | 5,793 | 545,524 | 364 | 380 |
| HV-AV | 3 | 716 | 541,494 | 1,099 |
| HV-HV | 129 | 1,392 | 3,296 | 742,025 |

## 10. Top-2 专家组合计数

| Top-1 | Top-2 | 样本数 | 是否跨 family |
|---|---|---:|---:|
| AV-AV | AV-HV | 380,720 | 0 |
| AV-AV | HV-AV | 4,306 | 1 |
| AV-AV | HV-HV | 23,620 | 1 |
| AV-HV | AV-AV | 157,033 | 0 |
| AV-HV | HV-AV | 52,713 | 1 |
| AV-HV | HV-HV | 339,820 | 1 |
| HV-AV | AV-AV | 997 | 1 |
| HV-AV | AV-HV | 21,071 | 1 |
| HV-AV | HV-HV | 523,098 | 0 |
| HV-HV | AV-AV | 25,350 | 1 |
| HV-HV | AV-HV | 548,900 | 1 |
| HV-HV | HV-AV | 169,311 | 0 |

注：`torch.topk(k=2)` 不会返回重复专家，所以 Top-1 与 Top-2 相同的组合计数为 0，此处省略。

## 11. 用于回复审稿人的建议逻辑

可以按下面逻辑写回复：

1. 承认审稿人的担忧是合理的：四类交互专家具有明确物理语义，尤其 ACC 与 IDM family 不应被理解为任意均衡混合。

2. 说明 Top-2 的实际作用不是均衡混合，而是处理类型识别不确定性。实验证明，hard Top-1 的错选率为 `0.675%`，而 Top-2 miss rate 只有 `0.0093%`，能覆盖 `98.63%` 的 hard-routing error。

3. 说明 softmax 权重高度稀疏。全样本 Top-1 平均权重为 `0.9924`，Top-2 平均权重仅为 `0.0075`。即使 Top-2 跨 ACC/IDM family，第二专家平均权重也只有 `0.00535`，所以不会形成实质性的异质物理模型混合。

4. 报告 hard 错选样本 RMSE：Top-2 相比 hard Top-1 降低了 gap 与 velocity 平均 RMSE：
   - Gap Avg：`0.3863 m` 到 `0.3825 m`
   - Velocity Avg：`0.2435 m/s` 到 `0.2405 m/s`

5. 若需要更强物理约束，可补充 `family_top2` 路由作为替代设计：先选择 ACC/IDM family，再在同一 family 内聚合，从机制上避免 ACC-IDM 混合。

## 12. 生成文件清单

| 文件 | 说明 |
|---|---|
| `analyze_routing_topk.py` | 独立分析脚本。 |
| `fig/vis/routing_analysis/routing_selection_analysis.png` | 硬路由与 Top-2 选择错误分析图。 |
| `fig/vis/routing_analysis/routing_selection_analysis.pdf` | 选择错误分析图 PDF。 |
| `fig/vis/routing_analysis/selected_weight_distribution.png` | softmax 权重分布图。 |
| `fig/vis/routing_analysis/selected_weight_distribution.pdf` | 权重分布图 PDF。 |
| `fig/vis/routing_analysis/routing_selection_summary.csv` | 图 1 数据：错选率、Top-2 覆盖率、跨 family 比例。 |
| `fig/vis/routing_analysis/routing_rmse_eval_style.csv` | 各路由方式与各样本子集的 RMSE。 |
| `fig/vis/routing_analysis/weight_distribution_summary.csv` | 权重分布统计。 |
| `fig/vis/routing_analysis/cross_family_weight_summary.csv` | 跨 family Top-2 pair 的权重统计。 |
| `fig/vis/routing_analysis/hard_top1_confusion_matrix.csv` | hard Top-1 混淆矩阵。 |
| `fig/vis/routing_analysis/top2_pair_counts.csv` | Top-2 专家组合计数。 |
| `fig/vis/routing_analysis/routing_weight_data.npz` | 保存的原始路由概率与派生数组。 |
