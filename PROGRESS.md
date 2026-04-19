# 📊 多层级电动车分析项目 — 当前进度与下一步指引

> **更新时间**: 2026-04-14  
> **当前状态**: ⚠️ 所有脚本已就绪，**尚未执行任何步骤**

---

## 🔍 当前状态总结

仓库中包含完整的三阶段分析管线脚本，但 **所有中间数据和结果文件均不存在**（已被 `.gitignore` 排除）。这意味着：

- ❌ 没有找到原始预处理数据 (`data_*_processed.csv`)
- ❌ 没有任何阶段的输出结果 (`results/` 目录为空)
- ❌ 没有训练好的模型文件 (`.pth`, `.pt`)
- ❌ 没有特征数据集 (`.h5`, `.npz`)

---

## 📋 你现在应该执行的步骤

### ✅ 第零步：准备原始数据（必须先完成）

你需要在 **项目根目录** 放置预处理好的 CSV 数据文件：

```
./data_20250701_processed.csv
./data_20250702_processed.csv
...（31天的数据）
```

**必需列**: `vehicle_id`, `time`, `soc`, `v`, `i`, `power`, `spd`, `lat`, `lon`, `is_charging`

> ⚠️ 如果你已经在服务器上有这些数据，请将它们放到项目根目录，或创建软链接。

---

### 然后按以下顺序执行：

## 阶段一：片段聚类 (`analysis_complete_vehicles/`)

| 顺序 | 脚本 | 说明 | 输入 | 输出 |
|:---:|------|------|------|------|
| **1** | `step1_check_coverage.py` | 检查车辆31天覆盖率 | `data_*_processed.csv` | `vehicle_coverage_31days.csv` |
| **2** | `step2_filter_complete_vehicles.py` | 筛选≥28天的完整车辆 | 上一步输出 + 原始数据 | 完整车辆数据 |
| **3** | `step3_extract_discharge_segments.py` | 提取放电片段（SOC下降≥3%） | 筛选后数据 | `discharge_segments.csv` |
| **4** | `step4_build_dual_channel_dataset_fixed.py` | 构建双通道时序数据集 | 放电片段 | `dual_channel_dataset.h5` |
| **5** | `step5_dual_gru_model.py` + `step6_train_final_tensorboard.py` | 训练双通道GRU模型，提取潜在向量 | `.h5` 数据集 | `latent_vectors.npz` |
| **6** | `step7_clustering.py` | 潜在空间K-means/GMM聚类 | `latent_vectors.npz` | `clustering_v3_results.npz` |

## 阶段二：车辆聚类 (`vehicle_clustering/`)

| 顺序 | 脚本 | 说明 | 输入 | 输出 |
|:---:|------|------|------|------|
| **7** | `integrate_clustering_complete.py` | 整合片段→行程→车辆特征 | 阶段一结果 + 行程数据 | `segments_integrated_complete.csv` |
| **8** | `step8_vehicle_clustering.py` | GMM车辆级聚类(K=4) | 聚合特征 | `vehicle_clustering_gmm_k4.csv` |

## 阶段三：耦合分析 (`coupling_analysis/`)

| 顺序 | 脚本 | 说明 | 输入 | 输出 |
|:---:|------|------|------|------|
| **9** | `step12_extract_trips.py` | 提取充电间行程 | 片段+充电事件+车辆 | `inter_charge_trips.csv` |
| **10** | `step_A_build_variables.py` | 构建驾驶模式变量体系 | 行程数据 | 驾驶模式聚类 |
| **11** | `step_3_3_coupling_analysis.py` | XGBoost+SHAP耦合分析 | 所有前序结果 | 耦合分析报告 |

---

## 🔗 数据依赖关系图

```
data_*_processed.csv (原始数据 — 你需要准备这个！)
    │
    ├── step1 → vehicle_coverage_31days.csv
    ├── step2 → 完整车辆筛选
    ├── step3 → discharge_segments.csv
    │
    └── step4 → dual_channel_dataset.h5
            │
            └── step5+6 → latent_vectors.npz
                    │
                    └── step7 → clustering_v3_results.npz ⭐
                            │
                            ├── integrate → segments_integrated_complete.csv
                            │       │
                            │       └── step8 → vehicle_clustering_gmm_k4.csv ⭐
                            │
                            └── step12 → inter_charge_trips.csv
                                    │
                                    ├── step_A → 驾驶模式变量
                                    │
                                    └── step_3.3 → 耦合分析最终结果 🎯
```

---

## ⚡ 快速执行命令

准备好数据后，依次运行：

```bash
# 阶段一：片段聚类
python analysis_complete_vehicles/step1_check_coverage.py
python analysis_complete_vehicles/step2_filter_complete_vehicles.py
python analysis_complete_vehicles/step3_extract_discharge_segments.py
python analysis_complete_vehicles/step4_build_dual_channel_dataset_fixed.py
python analysis_complete_vehicles/step6_train_final_tensorboard.py
python analysis_complete_vehicles/step7_clustering.py

# 阶段二：车辆聚类
python vehicle_clustering/integrate_clustering_complete.py
python vehicle_clustering/step8_vehicle_clustering.py

# 阶段三：耦合分析
python coupling_analysis/step12_extract_trips.py
python coupling_analysis/step_A_build_variables.py
python coupling_analysis/step_3_3_coupling_analysis.py
```

---

## 📌 备注

- 所有结果文件 (`results/`, `.csv`, `.h5`, `.npz`, `.pth`) 已通过 `.gitignore` 排除，不会被提交到 Git
- 每一步的脚本都包含详细的中文注释和进度提示
- 建议在有 GPU 的环境（如 AutoDL）上运行 step5/step6 的模型训练
- 可视化相关脚本 (`step7c_*`, `step8_*`, `step9_*` 等) 可在完成对应阶段后选择性运行
