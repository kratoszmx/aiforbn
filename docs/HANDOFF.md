# HANDOFF.md

## 项目
- 名称：AI for BN PoC
- 路径：`$HOME/projects/ai_for_bn`
- 默认环境：用户 zsh 下的 `quant`
- 当前优先级：**结合 `docs/老師回覆.txt`，补齐更适合导师阅读的证据型摘要 artifact 与更冷静的项目文档叙事，再决定下一步单模块 coding**

## 一句话结论
- **最后一个可直接回退的稳定主线**仍然是此前已完整验证并已保存的主线波次。
- 当前更适合对外汇报的项目定位应是：
  - general grouped-split predictor 已经明显优于 dummy；
  - BN-specific diagnostics 已被单独拆开并更诚实地暴露出来；
  - 当前 candidate ranking 应被视为 **low-confidence formula-level prioritization for follow-up**，而不是 discovery claim。
- 目前最稳妥的技术判断仍然是：
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
当前 live tree 已完成一轮 **去掉 wrapper 的 src 顶层模块化重排**，目标是不改行为，但把主要模块都上提到 `src` 顶层，避免继续藏在 `pipeline/` 下面。

当前代码组织已经变成：
- `src/config.py`
  - 实际默认配置文件，`main.py` 和测试直接使用它
- `conftest.py`
  - pytest bootstrap 放在仓库根，而不是 `src` 顶层
- `src/runtime/`
  - 独立 runtime 模块，保留 `io_utils.py` 和 `schema.py`
- `src/materials/`
  - 业务主模块，吸收了原先分散的 dataset / features / reporting / structure-execution 逻辑
  - 关键文件包括 `data.py`、`candidate_space.py`、`feature_building.py`、`modeling.py`、`selection.py`、`benchmarking.py`、`screening.py`、`common.py`、`ranking_tables.py`、`structure_artifacts.py`、`structure_helpers.py`、`structure_execution.py`、`summary.py`、`artifacts.py`、`plots.py`
- `src/torch_models/`
  - 独立 PyTorch 模型模块，保留 `base.py`、`attention.py`、`sparse_attention.py`、`roost_like.py`、`ensemble.py`
- `src/ui/`
  - 独立 UI 模块，保留 `streamlit_app.py`

每个正式模块目录现在都已有模板文件：
- `AGENTS.md`
- `PY_FILES_SUMMARY.md`
- `utils.py`

测试布局也已随模块调整为：
- `src/runtime/tests/`
- `src/materials/tests/`
- `src/ui/tests/`
- `src/tests/`

根目录旧 `tests/` 已移除，`src/pipeline/`、`src/core/`、`src/dataset/`、`src/features/`、`src/reporting/`、`src/structure_execution/` 这些旧顶层也都已退出 live 结构。

另外，本轮还完成了几项关键整理：
- `main.py` 的导入链已改成直接指向新的真实模块，不再经过 façade
- `main.py` 现在新增 `--dry-run` 快速烟测入口，可在不跑完整主流程的情况下验证配置、候选空间、特征表构建、以及模型导入/实例化是否仍然通畅
- 各模块（含 `template` 与 `tests`）现在都已有自己的 `PY_FILES_SUMMARY.md`，用于记录该目录对外暴露的可调用函数/类；模块内部实现细节则继续放在各自的 `AGENTS.md`
- `src/config.py` 保持为真实配置文件，不再保留兼容层
- 顶层模块目录已去掉 `__init__.py`、package-relative imports 和依赖 `__all__` 的包式导出，回到“repo root + src 路径直接使用”的非包态模式
- `core` 这个顶层名字已移除，原通用运行时职责收敛到 `src/runtime/`
- `reporting` 和 `structure_execution` 不再作为假独立 sibling module 存在，而是并回 `materials`，避免只在目录层面独立、实际仍强依赖主业务流
- Streamlit UI 仍位于 `src/ui/streamlit_app.py`，并已去掉对 `runtime` 的不必要模块依赖，直接复用 `myutils` 的 JSON 读取能力
- `src/runtime/io_utils.py` 的 `ensure_runtime_dirs(...)` 已去掉对 `apps/`、`tests/`、`notebooks/` 这类非运行时目录的自动创建逻辑，因此旧的 notebook/notebooks 自动生成来源已经移除
- `src/runtime/io_utils.py` 已对齐当前 `myutils` 的目录式布局，直接使用 `file_utils/`、`ai_utils/`、`net_utils/` 等子目录导入
- 项目里重复出现的 JSON 读写 / JSON-safe 转换逻辑继续尽量复用 `myutils/file_utils/json_io.py`

另有以下 **非本轮应编辑对象** 也在 working tree 中呈现 dirty 状态：
- `skill.txt`
- `skills/*.txt`

注意：
- 这些 skill 文件应继续视为**只读取、不编辑**的约束对象。
- 若之后工作树再次出现它们的改动，commit 前必须显式排除。

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
4. 本轮新增的 reporting wave 没有引入新的 benchmark logic，而是把现有证据压缩成更容易给导师直接阅读的摘要产物：
   - `artifacts/bn_model_role_comparison.csv`
   - `artifacts/demo_candidate_rank_stability_summary.csv`
   - `artifacts/demo_candidate_structure_followup_report.csv`
   - 并同步接入 `artifacts/experiment_summary.json`
5. 因此当前**更安全的汇报口径**应是：
   - BN-themed formula-level screening PoC with honest diagnostics
   - not BN-centered discovery

## 当前验证状态
本轮围绕“模块拆分后兼容性”已经做过这些验证：

1. 语法级验证：
- `python3 -m compileall src main.py`
- 结果：通过

2. 定向回归验证：
- `pytest -q src/materials/tests/test_features_pipeline.py -k "feature_pipeline_can_train_evaluate_benchmark_and_rank_demo_candidates or screen_candidates_can_apply_bn_local_band_gap_alignment_penalty"`
- 结果：通过

3. 入口与配置兼容验证：
- `pytest -q src/tests/test_config.py src/tests/test_main.py`
- 结果：`2 passed`

4. 模块拆分后的完整 src 测试：
- `PYTHONPATH=src pytest -q src`
- 结果：`36 passed, 6 warnings in 5.67s`

5. 快速 dry-run 烟测：
- `python3 main.py --dry-run`
- 结果：通过，已确认配置加载、候选空间生成、tiny in-memory feature-table 构建、以及配置中模型的导入/实例化均可完成

6. reporting-artifact 波次的短验证：
- `/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python3 main.py --dry-run`
- `/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python3 -m pytest -q src/materials/tests/test_reporting.py`
- 结果：`2 passed`

7. `main.py` 烟测：
- 在用户 zsh / `quant` 环境里实际启动过 `python3 main.py`
- 修复 `src/config.py` 缺失后，程序已不再在入口阶段立即因 import/config 路径报错
- 该运行随后进入持续计算阶段，未在本轮等待到完整结束

因此当前最准确的表述应是：
- **模块拆分后的 `pytest -q src` 已通过**
- **`main.py` 已确认能启动并进入主流程，但本轮未等待到完整跑完**
- 这更接近“结构重构已验证、主流程做过短烟测”的状态，而不是“完整重算后的新稳定 checkpoint”

## 当前最重要的记录文件
### 应继续保留并视为主状态文件
- `HANDOFF.md`：中文交接与当前状态摘要
- `PY_FILES_SUMMARY.md`：AI-facing Python surface 摘要
- `tasks/literature_mining/MODEL_UPGRADE_RESEARCH_2026-04-20.md`：AI-facing 建模方向技术备忘

### 当前实验 / 汇报 artifacts
- `artifacts/pilot/fractional_attention_pilot_*`
- `artifacts/pilot/sparse_fractional_attention_pilot_*`
- `artifacts/pilot/roost_like_pilot_*`
- `artifacts/pilot/roost_like_small_bn_slice_results.csv`
- `artifacts/pilot/roost_like_medium_bn_slice_results.csv`
- `artifacts/pilot/roost_like_wider_bn_slice_results.csv`
- `artifacts/pilot/roost_like_config_sweep_summary.json`
- `artifacts/pilot/knn_bn_slice_pilot_summary.json`
- `artifacts/pilot/knn_bn_slice_pilot_results.csv`
- `artifacts/bn_model_role_comparison.csv`
- `artifacts/demo_candidate_rank_stability_summary.csv`
- `artifacts/demo_candidate_structure_followup_report.csv`

## 恢复工作时的直接起点
当前用户要求是：
- **结合 `docs/老師回覆.txt` 继续补齐老师认为应提升的点**
- **同步完善项目文档**
- **模块内 coding 继续走 Codex，但一次只准改一个模块，读完前置文件后直接开始改**

因此恢复时的默认动作应是：
1. 先读：
   - `skill.txt`
   - `skills/template.txt`
   - `skills/workflow.txt`
   - 其余 `skills/*.txt`
   - `HANDOFF.md`
   - `docs/老師回覆.txt`
2. 先确认这轮是：
   - 文档 / 汇报整理
   - 还是单模块 coding
3. 如果进入 coding：
   - 只选一个模块
   - 先写 `claw_memory/N.md`
   - 明确允许改哪些文件、禁止碰哪些文件
   - 让 Codex 读完对应前置文件后直接开始修改
   - OpenClaw 自己负责测试、审核、必要时接手修补
4. 若形成 checkpoint：
   - 先排除不该提交的文件
   - 先清缓存
   - 再跑短验证
   - 通过后再 `git add / commit / push`

## 当前不应丢失的判断
- 不要因为本机不是 CUDA 机器就自动退缩换方向。
- 但也不要在没有正向证据时，仅因为“模型更重”就要求 GPU。
- 当前实验结论还不足以说明“只要上 GPU 就能赢”。
- 目前真正的下一个 blocker 不是 GPU，而是：
  - **TabPFN license/token**
- 从结构规范角度看，当前代码已经进一步收敛到 4 个正式生产模块：`runtime`、`materials`、`torch_models`、`ui`。
- 模块模板要求目前已满足，每个正式模块都带有自己的 `AGENTS.md` 和 `utils.py`。
- 当前生产依赖关系也已明显收敛：`runtime -> []`、`torch_models -> []`、`ui -> []`、`materials -> [runtime, torch_models]`。
- 也就是说，之前那种 `reporting` / `structure_execution` 只是目录独立、实现上却从属于主业务链的问题，已经通过并回 `materials` 解决。
- 当前最合理动作不是盲目扩展实验面，而是：
  - **先把老师会追问的证据和口径补齐，再决定下一步单模块 coding 目标。**
