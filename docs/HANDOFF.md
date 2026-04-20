# HANDOFF.md

## 项目
- 名称：AI for BN PoC
- 路径：`$HOME/projects/ai_for_bn`
- 默认环境：用户 zsh 下的 `quant`
- 当前优先级：**先把状态记录清楚，不继续推进新的长建模波次**

## 一句话结论
- **最后一个可直接回退的稳定主线**仍然是此前已完整验证并已保存的主线波次。
- **当前 live working tree 是实验性 dirty tree**，主要在试探新的 composition-only 现代模型方向，但这些实验**还没有证据强到可以并入默认主线**。
- 目前最稳妥的结论仍然是：
  - overall evaluation best 依旧不是 candidate-compatible screening best；
  - 当前最可信的 candidate-compatible neural control 仍然是 **`matminer_composition + torch_mlp_ensemble`**；
  - attention / Roost-like / kNN 这些新试探都**没有**在短 BN-slice pilot 上形成足够强的新证据。

## 最后一个稳定主线（可回退认定）
- 当前应把此前已经完整跑通 `pytest -q` 和 `python main.py` 的主线看作稳定基线。
- 该稳定基线已经包含：
  - grouped-by-formula robustness
  - BN formula holdout
  - BN family holdout
  - BN vs non-BN stratified error
  - candidate-compatible BN honesty table
  - ranking explainability / uncertainty / abstention
  - BN-centered alternative ranking
  - structure-generation handoff / first-pass execution artifacts
- 这条稳定主线里的核心方法学定位仍然是：
  - **overall evaluation** 可以使用 lightweight structure-aware 路径
  - **formula-only screening** 必须使用 candidate-compatible 路径
  - 不可把二者混成一个“AI 发现 BN 新材料”的强 claim

## 当前 live working tree 状态
当前 dirty tree 主要包含以下实验性改动：
- `configs/default.py`
- `src/pipeline/features.py`
- `src/pipeline/torch_models.py`
- `tests/test_config_default.py`
- `tests/test_features_pipeline.py`
- `tasks/literature_mining/MODEL_UPGRADE_RESEARCH_2026-04-20.md`
- `artifacts/pilot/`

另外，后续又做了四项轻量仓库整理：
- Streamlit artifact viewer 已从 `apps/streamlit_app.py` 移到 `src/streamlit_app.py`
- `src/core/io_utils.py` 的 `ensure_runtime_dirs(...)` 已去掉对 `apps/`、`tests/`、`notebooks/` 这类非运行时目录的自动创建逻辑，因此旧的 notebook/notebooks 自动生成来源已经移除
- `src/core/io_utils.py` 已从“兼容旧 shim”进一步收敛到当前 `myutils` 的真实目录式布局，直接对齐 `file_utils/`、`ai_utils/`、`net_utils/` 等子目录的导入方式
- 顺手把项目里重复出现的 JSON 读写 / JSON-safe 转换逻辑收拢到 `myutils/file_utils/json_io.py`，并在 `ai_for_bn` 的 `data` / `reporting` / `streamlit_app` / `structure_execution` 中开始复用

另有以下 **非本轮应编辑对象** 也在 working tree 中呈现 dirty 状态：
- `skill.txt`
- `skills.txt`
- `skills_ai.txt`

注意：
- 这些 skill 文件本轮**只读取，不应编辑**。
- 若之后要 commit，必须先检查并排除这些不该一起提交的改动。

## 当前默认主线与实验分界
### 默认主线仍保持不变
默认 `model.candidate_types` 仍应视为：
- `linear_regression`
- `hist_gradient_boosting`
- `torch_mlp`
- `torch_mlp_ensemble`

这意味着当前主线默认 sweep **没有**把下面这些模型并入：
- `torch_fractional_attention`
- `torch_sparse_fractional_attention`
- `torch_roost_like`

### 实验模型的定位
当前代码里已经有以下实验模型实现，但它们都应视为 **pilot-only / experimental**：
- `torch_fractional_attention`
- `torch_sparse_fractional_attention`
- `torch_roost_like`

它们当前都只允许和：
- `fractional_composition_vector`
搭配。

不要把这些实验模型误写成已经进入默认主线。

## 本轮新增但尚未主线化的实验结论
### 1) Dense fractional attention pilot
相关 artifacts：
- `artifacts/pilot/fractional_attention_pilot_summary.json`
- `artifacts/pilot/fractional_attention_pilot_benchmark_results.csv`
- `artifacts/pilot/fractional_attention_pilot_bn_slice_results.csv`

结论：
- 在短 BN-slice pilot 上没有打赢更强的现有 candidate-compatible control。
- 不值得主线化。

### 2) Sparse fractional attention pilot
相关 artifacts：
- `artifacts/pilot/sparse_fractional_attention_pilot_summary.json`
- `artifacts/pilot/sparse_fractional_attention_pilot_benchmark_results.csv`
- `artifacts/pilot/sparse_fractional_attention_pilot_bn_slice_results.csv`

结论：
- 在小 pilot 上出现了 validation selection 与 BN-slice evidence 不一致的问题。
- 不只是“本机算力不够”，而是模型本身没有形成稳定正信号。
- 继续在这条 attention 变体线上投入不划算。

### 3) Roost-like 短 pilot
相关 artifacts：
- `artifacts/pilot/roost_like_pilot_summary.json`
- `artifacts/pilot/roost_like_pilot_benchmark_results.csv`
- `artifacts/pilot/roost_like_pilot_bn_slice_results.csv`

本次小 pilot（`341 rows / 240 formulas / 10 BN formulas`）的关键信息：
- test benchmark：
  - `matminer_composition + hist_gradient_boosting`: `MAE = 0.5717`
  - `fractional_composition_vector + torch_mlp_ensemble`: `MAE = 0.8246`
  - `fractional_composition_vector + torch_roost_like`: `MAE = 0.8405`
- BN-slice：
  - `dummy_mean`: `MAE = 1.3439`
  - `matminer_composition + hist_gradient_boosting`: `MAE = 1.6158`
  - `fractional_composition_vector + torch_mlp_ensemble`: `MAE = 1.4772`
  - `fractional_composition_vector + torch_roost_like`: `MAE = 1.3784`

解读：
- `torch_roost_like` 比同批 fractional neural controls 更接近真正的 BN-slice 目标。
- 但它**仍然没有打赢 dummy**。
- 因此它最多算“有一点方向感”，**还不能主线化**。

### 4) Roost-like 配置小扫
相关 artifact：
- `artifacts/pilot/roost_like_config_sweep_summary.json`

关键结果：
- `roost_like_small`: `MAE = 1.3784`，未过 dummy
- `roost_like_medium`: `MAE = 1.3984`，未过 dummy
- `roost_like_wider`: `MAE = 2.0710`，明显更差

结论：
- 更宽/更重的局部配置并没有把 BN-slice 拉起来。
- 当前还没有“已经证明需要更重算力才会成功”的证据。

### 5) 零改代码 kNN 小 pilot
相关 artifacts：
- `artifacts/pilot/knn_bn_slice_pilot_summary.json`
- `artifacts/pilot/knn_bn_slice_pilot_results.csv`

最佳结果：
- `fractional_composition_vector + k=7 + distance`
- `BN-slice MAE = 1.8808`

结论：
- 比 Roost-like 更差。
- “局部传统基线”不是这轮的解。

### 6) TabPFN 可行性检查
当前环境中已完成：
- `quant` 环境已安装 `tabpfn==7.1.1`

但当前 blocker 是：
- `TabPFNLicenseError`
- 本地权重下载需要先接受 license，并设置 `TABPFN_TOKEN`
- 这是 **license / auth blocker**，不是算力 blocker

因此当前状态应写成：
- **TabPFN 已完成安装，但尚未完成真正 pilot**
- 缺的不是 GPU，而是 `TABPFN_TOKEN`

## 当前最可信的项目结论
截至目前，最可信的项目结论仍然是：
1. 主线方法学修补已经基本到位，项目不再是“只会报一个漂亮 test MAE 的 PoC”。
2. BN-centered 诊断已经比早期清楚很多，但 BN 子域仍然明显更难。
3. 当前最可信的 candidate-compatible neural baseline 仍然是：
   - `matminer_composition + torch_mlp_ensemble`
4. 当前实验 dirty tree 里的新模型方向：
   - dense attention：负面
   - sparse attention：负面
   - Roost-like：略有希望，但仍未过 dummy
   - kNN：负面
   - TabPFN：还没开始真正评估，卡在 license token
5. 因此当前**不应**把实验 dirty tree 叙述成“已经找到比现有主线更强的新路线”。

## 当前验证状态
本轮为了整理文档并确认 live tree 状态，已经做过两层验证：

1. 极窄实验验证：
- 先清缓存：`clear_project_cache('.')`（底层已对齐到最新 `myutils/file_utils/filesystem.delete_cache(...)`）
- 再跑：
  - `tests/test_config_default.py::test_default_config_has_expected_poc_defaults`
  - `tests/test_features_pipeline.py::test_select_feature_model_combo_marks_attention_models_as_feature_specific`
  - `tests/test_features_pipeline.py::test_fractional_composition_feature_table_and_torch_models_fit_predict`
- 结果：`3 passed, 1 warning in 4.26s`

2. 当前 live experimental tree 的完整 `pytest -q`：
- 同样先清缓存后运行
- 结果：`32 passed, 6 warnings in 20.10s`

重要说明：
- 本轮曾尝试重新跑一次 `python main.py` 来满足旧的提交前检查习惯
- 该进程一直在真实占用 CPU，不是假挂，但在完成前被中断
- 随后用户已明确要求：**这一轮只做文档/状态整理，不再继续跑程序；若要跑也只做快速 dry run**
- 因此当前 live experimental tree 的最新状态应描述为：
  - **完整 `pytest -q` 已通过**
  - **`python main.py` 本轮未完成，不应表述成已重新验证通过**
- 所以这个 dirty tree 仍应看作“文档已同步、测试已过、但主流程未重新完整验证”的实验树，而不是新的 fully re-verified stable checkpoint

## 当前最重要的记录文件
### 应继续保留并视为主状态文件
- `HANDOFF.md`：中文交接与当前状态摘要
- `PY_FILES_SUMMARY.md`：AI-facing Python surface 摘要
- `tasks/literature_mining/MODEL_UPGRADE_RESEARCH_2026-04-20.md`：AI-facing 建模方向技术备忘

### 当前实验 artifacts
- `artifacts/pilot/fractional_attention_pilot_*`
- `artifacts/pilot/sparse_fractional_attention_pilot_*`
- `artifacts/pilot/roost_like_pilot_*`
- `artifacts/pilot/roost_like_small_bn_slice_results.csv`
- `artifacts/pilot/roost_like_medium_bn_slice_results.csv`
- `artifacts/pilot/roost_like_wider_bn_slice_results.csv`
- `artifacts/pilot/roost_like_config_sweep_summary.json`
- `artifacts/pilot/knn_bn_slice_pilot_summary.json`
- `artifacts/pilot/knn_bn_slice_pilot_results.csv`

## 恢复工作时的直接起点
当前用户要求是：
- **先把状态记录清楚**
- **目前有其他更重要的工作，先不要继续在这里推进长建模波次**

因此恢复时的默认动作应是：
1. 先读：
   - `skill.txt`
   - `skills.txt`
   - `skills_ai.txt`
   - `HANDOFF.md`
2. 先确认当前工作目标是不是继续建模，还是只做文档/状态维护。
3. 如果恢复建模：
   - 不要直接把 experimental models 并入主线
   - 先决定是否要继续 TabPFN 路线
   - 若继续 TabPFN，本地还需要 `TABPFN_TOKEN`
4. 若未来要形成 checkpoint：
   - 先排除不该提交的 skill 文件改动
   - 先清缓存
   - 再跑 `pytest -q`
   - 再跑 `python main.py`
   - 全通过后再 `git add / commit`

## 当前不应丢失的判断
- 不要因为本机不是 CUDA 机器就自动退缩换方向。
- 但也不要在没有正向证据时，仅因为“模型更重”就要求 GPU。
- 当前实验结论还不足以说明“只要上 GPU 就能赢”。
- 目前真正的下一个 blocker 不是 GPU，而是：
  - **TabPFN license/token**
- 在用户没有明确要求恢复建模前，当前最合理动作就是：
  - **把状态文件维护清楚，保留现场，停止继续扩散实验面。**
