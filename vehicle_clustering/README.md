# vehicle_clustering/ — 车辆级聚类分析模块

## 概述

本目录包含电动汽车多层级聚类分析中**车辆级别**的所有脚本。整体流程基于片段级聚类结果（来自 `analysis_complete_vehicles/`），将数据从片段 → 行程 → 车辆逐步聚合，最终完成车辆级聚类及后续特征分析。

---

## 完整运行流程

```
analysis_complete_vehicles/step7_clustering.py   （片段级聚类，生成聚类标签）
        │
        ▼
integrate_clustering_complete.py                 （数据集成：片段 → 行程 → 车辆）
        │
        ▼
step8_vehicle_clustering.py                      （车辆级 GMM 聚类，K=4）
        │
        ▼
step9_feature_dimension_analysis.py              （特征维度分析，深入理解聚类结果）
        │
        ▼
step8_vehicle_clustering_analysis_figures.py      （论文级可视化图表生成）
        │
        ▼
coupling_analysis/                               （耦合分析 / 下游建模）
```

### 运行命令

```bash
# 1. 数据集成（片段 → 行程 → 车辆特征聚合）
python vehicle_clustering/integrate_clustering_complete.py

# 2. 车辆级聚类（GMM K=4）
python vehicle_clustering/step8_vehicle_clustering.py

# 3. 特征维度分析（分析聚类结果的特征重要性、区分度等）
python vehicle_clustering/step9_feature_dimension_analysis.py

# 4. 论文图表生成（可选）
python vehicle_clustering/step8_vehicle_clustering_analysis_figures.py
```

---

## 各文件详细说明

### 核心流程文件

| 文件 | 说明 |
|------|------|
| `integrate_clustering_complete.py` | **数据集成脚本**。将 `analysis_complete_vehicles/` 的片段聚类结果整合，为每个片段分配 `trip_id`，然后逐步聚合到行程级和车辆级。输出 `vehicles_aggregated_features.csv` 供后续聚类使用。 |
| `step8_vehicle_clustering.py` | **车辆级聚类**。使用 GMM（高斯混合模型），固定 K=4，对车辆进行聚类。输入为 `vehicles_aggregated_features.csv`，输出聚类标签及评价指标。 |
| `step9_feature_dimension_analysis.py` | **特征维度分析**（原名 `Step 9: Feature Dimension Analysis for Vehicle Clusters.py`）。在 Step 8 聚类完成后运行，用于深入分析各车辆簇的特征特性，包括：特征重要性排名（方差 + 互信息）、聚类特征画像、特征区分度分析（簇间/簇内方差比）、优化的聚类标签生成、雷达图和热力图可视化。 |
| `step8_vehicle_clustering_analysis_figures.py` | **论文级图表生成**。在 Step 8 聚类完成后，专门为论文中 3.2.2、3.2.3、4.3.4 等章节生成专业可视化图表。 |

### 辅助/实验文件

| 文件 | 说明 |
|------|------|
| `diagnose_and_integrate_clustering.py` | **诊断脚本**。检查片段聚类结果的完整性，验证数据集成是否正确。在数据出现异常时使用。 |
| `step9_vehicle_clustering.py` | 实验版本：改进的车辆聚类可视化 + 优化参数（非主流程）。 |
| `step9_vehicle_clustering_v3.py` | 实验版本：K=3 聚类 + 片段信息整合（非主流程）。 |
| `step11_detailed_comparison_charts.py` | 详细的聚类算法比较图表（K-means / GMM / DBSCAN / 层次聚类 / 谱聚类对比）。 |

---

## Step 9 特征维度分析 — 详细说明

`step9_feature_dimension_analysis.py` 是 Step 8 聚类的**后处理分析步骤**，主要回答以下问题：

1. **哪些特征最重要？** — 通过方差分析和互信息（Mutual Information）排名特征重要性
2. **哪些特征最能区分不同车辆簇？** — 计算簇间方差 / 总方差（Discrimination Ratio）
3. **每个车辆簇的特征画像是什么？** — 为每个簇生成均值/标准差特征描述
4. **聚类标签是否合理？** — 基于特征分析自动生成优化的聚类标签和描述

### 输出文件

| 文件 | 内容 |
|------|------|
| `feature_importance.csv` | 特征重要性综合排名 |
| `cluster_profiles.csv` | 各簇的特征画像（均值 + 标准差） |
| `feature_discrimination.csv` | 特征区分度分析 |
| `optimized_cluster_labels.json` | 优化后的聚类标签及描述 |
| `feature_analysis_report.json` | 结构化分析报告 |
| `feature_analysis_report.txt` | 文本版完整分析报告 |
| `*.png / *.pdf` | 特征重要性、热力图、雷达图、相关性矩阵等可视化 |

### 为什么需要运行 Step 9？

Step 8 只完成了聚类本身（分配标签），但**不会告诉你聚类好不好、各簇有什么区别**。Step 9 通过特征分析帮助你：

- 验证聚类质量：如果区分度低，说明聚类可能需要调整
- 理解聚类含义：每个簇的驾驶行为特征是什么
- 为后续分析（coupling analysis）提供依据

---

## 为什么之前只运行了 integrate_clustering_complete 和 step8？

`integrate_clustering_complete.py` 和 `step8_vehicle_clustering.py` 是**最小必要流程**：

1. `integrate_clustering_complete.py` — 数据准备（没有这步，Step 8 没有输入数据）
2. `step8_vehicle_clustering.py` — 核心聚类（生成车辆聚类标签）

Step 9（`step9_feature_dimension_analysis.py`）是聚类完成后的**可选分析步骤**，用于深入理解聚类结果。当你发现聚类结果不理想时，运行 Step 9 可以帮助你诊断问题，例如特征是否有区分度、某些特征是否冗余等。

**建议**：如果对聚类结果有疑问，请在 Step 8 之后运行 Step 9 进行特征分析。
