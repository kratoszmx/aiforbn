# HANDOFF.md

## 项目
- 名称：AI for BN PoC
- 路径：`$HOME/projects/ai_for_bn`
- 目标：构建一个可运行、可展示、方法上尽量诚实的 AI for BN 最小 PoC
- 当前默认运行环境：zsh 的 `quant` 环境

## 当前状态
- 主任务仍是 `band_gap` 预测与 BN 主题下的候选排序演示。
- 训练数据仍来自 JARVIS / 2DMatPedia 的 `twod_matpd`。
- 默认评估协议仍是 **按 `formula` 分组切分**，`train/val/test` 的 formula overlap 为 0。
- 当前特征路径已升级为 **两条 composition-only 路径并存**：
  - `basic_formula_composition`：手写基础控制组；
  - `matminer_composition`：`pymatgen Composition` + matminer 的 19 维精选组成描述符。
- 当前选择流程不再只是选模型，而是：
  - 在验证集上对一个很小的 `{feature_set} x {model_type}` 组合空间做选择；
  - 候选组合默认为：
    - `basic_formula_composition + hist_gradient_boosting`
    - `basic_formula_composition + linear_regression`
    - `matminer_composition + hist_gradient_boosting`
    - `matminer_composition + linear_regression`
  - 选中组合后用 `train + val` 重新训练；
  - 在测试集上输出完整 benchmark，并保留 `dummy_mean` 作为 feature-agnostic baseline。
- 当前候选空间仍明确标注为 `toy_iii_v_demo_grid`，输出文件仍是 `demo_candidate_ranking.csv`。
- 候选排序现在会显式写出：
  - `ranking_basis = composition_only_predicted_band_gap`
  - `ranking_feature_set`
  - `ranking_model_type`
  - `ranking_note`

## 本轮已完成
- 保留了 `basic_formula_composition` 作为明确控制组，没有把 richer features 偷偷替换成默认唯一特征。
- 新增 `matminer_composition` 路径，当前实现为：
  - `Stoichiometry` 的若干 norm 特征；
  - 选定的 `Magpie` 元素属性统计；
  - 总计 19 个 composition-only 特征。
- 新增 `build_feature_tables()` 与 `select_feature_model_combo()`，把验证集选择从“只选模型”升级为“联合选择特征表示与模型”。
- 对 richer feature path 增加了显式失败处理：
  - 若某个 feature set 不能完整 featurize 数据，会在元数据里记录并从选择中跳过；
  - 候选排序若无法完整 featurize 候选公式，会直接报错而不是静默丢行。
- 扩展了 `benchmark_results.csv`，现在测试集 benchmark 会显式记录：
  - `feature_set`
  - `feature_family`
  - `n_features`
  - `model_type`
  - `benchmark_role`
  - `selected_by_validation`
- 扩展了 `experiment_summary.json`，现在会记录：
  - `features.selected_feature_set`
  - `features.feature_set_results`
  - `feature_model_selection`
  - `screening.ranking_feature_set`
  - `screening.ranking_model_type`
- 更新了 focused tests，覆盖：
  - matminer featurization 成功/失败行为；
  - 联合 feature/model 选择；
  - benchmark 与 ranking 新元数据；
  - `main.py` 新编排。

## 最近一次已验证运行
- 验证命令：`/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python main.py`
- 数据规模：6351 rows，4381 unique formulas。
- BN 主题切片：12 rows，10 unique BN formulas。
- grouped split：
  - train = 5101 rows / 3505 formulas
  - val = 618 rows / 438 formulas
  - test = 632 rows / 438 formulas
- 验证集选中的组合：
  - `matminer_composition + hist_gradient_boosting`
- 最新验证集 MAE：
  - `basic_formula_composition + hist_gradient_boosting`: `0.9579`
  - `basic_formula_composition + linear_regression`: `0.9976`
  - `matminer_composition + hist_gradient_boosting`: `0.6006`
  - `matminer_composition + linear_regression`: `0.8303`
- 最新测试集指标：
  - `MAE = 0.5822`
  - `RMSE = 0.8611`
  - `R² = 0.5939`
- 最新测试集 benchmark：
  - `matminer_composition + hist_gradient_boosting`: `MAE = 0.5822`
  - `basic_formula_composition + hist_gradient_boosting`: `MAE = 0.9219`
  - `matminer_composition + linear_regression`: `MAE = 0.8555`
  - `basic_formula_composition + linear_regression`: `MAE = 1.0056`
  - `dummy_mean`: `MAE = 1.1118`

## 当前目录重点
- `main.py`：线性主入口；保持 notebook-friendly。
- `configs/default.py`：默认 split / feature / model / screening 配置。
- `src/pipeline/features.py`：特征构造、联合选模、benchmark、demo candidate ranking。
- `src/pipeline/reporting.py`：artifact 输出与实验摘要。
- `artifacts/metrics.json`：当前被选组合的测试集指标与元数据。
- `artifacts/benchmark_results.csv`：测试集 feature/model benchmark。
- `artifacts/experiment_summary.json`：数据、特征、联合选择与 ranking 摘要。
- `artifacts/demo_candidate_ranking.csv`：带 composition-only ranking 元数据的候选排序结果。
- `给见微的说明.md`：面向见微的项目状态说明。
- `项目汇报.md`：面向导师/评审的正式汇报稿。

## 仍然明确的限制
- 当前虽然比上一轮强，但**仍然只是 composition-only baseline**，不是 structure-aware 模型。
- 当前 richer path 仍然只看化学式推导出的组成描述符，没有结构、相稳定性、形成能、声子稳定性等信息。
- BN slice 仍然很小，当前训练主体仍是全体 2D 数据，不是 BN-only 学习。
- 候选空间仍然只是 25 个 Group 13/15 组合的 toy demo grid，没有稳定性、结构、合成可行性约束。
- 还没有 uncertainty-aware ranking，也没有 inference API 解耦。

## 交接时重点关注
- 后续若继续强化方法学，优先级现在应转向：
  - 从 composition-only 继续往 structure-aware 表示推进
  - 提升候选空间约束的可信度
  - 加入不确定性并进一步解耦推理入口
- 保持 `main.py` 线性，不要把当前简单流程抽象成重启动器。
- 若继续扩展特征路径，优先保持：
  - 命名清楚；
  - 候选空间小而透明；
  - 失败显式暴露；
  - docs 与 artifact 同步更新。
