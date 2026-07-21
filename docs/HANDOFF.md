# HANDOFF.md

## 项目
- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`
- 名称：AI for BN PoC
- 路径：`/Users/zmx/Projects/projects/aiforbn`
- 默认执行环境：agent shell 下的 `quant`
- 当前优先级：**保持 relocated checkout 可执行，并以契约测试锁住 formula-only screening、BN diagnostics、artifact 安全边界与完整测试入口**

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

当前新增的 AI-native inspection 层：
- `docs/AGENT_MANIFEST.json`
  - 机器可读的项目契约，记录入口命令、模块边界、验证命令和安全边界
  - 现已记录 v18 research-plan alignment：源文件、实现锚点、非 claim 边界、以及 deliverable chain，并由 `--verify-agent-contract` 检查
- `python3 main.py --emit-agent-commands`
  - 输出 entrypoints、validation commands、validation profiles、project skills 和 retired guidance 文件清单
- `python3 main.py --emit-agent-state`
  - 输出 live JSON 项目状态
- `python3 main.py --verify-agent-contract`
  - 检查 AI-native 布局；只有缺少关键契约文件这类阻断错误才非零退出
- `skills/ai_native_workflow.txt`
  - 当前唯一 active plain-text project runtime guidance
- `.agents/skills/aiforbn-workflow/SKILL.md`
  - 当前 repo-scoped Codex workflow skill
- `.agents/skills/aiforbn-overleaf-proposal/SKILL.md`
  - research-plan / Overleaf 专用 repo-scoped Codex skill

当前 `quant` 环境已补齐 `requirements.txt` 中完整测试需要的关键依赖：
- `pyarrow`
- `torch`
- `streamlit`
- `jarvis-tools`

因此新的默认广覆盖验证命令可以使用：
- `python3 -m pytest -q src`

测试布局也已随模块调整为：
- `src/runtime/tests/`
- `src/materials/tests/`
- `src/torch_models/tests/`
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
- Streamlit UI 仍位于 `src/ui/streamlit_app.py`，并通过公开的 `runtime.io_utils.read_json_file` 复用 JSON 读取能力，避免重复维护 `myutils` 定位逻辑
- `src/runtime/io_utils.py` 的 `ensure_runtime_dirs(...)` 已去掉对 `apps/`、`tests/`、`notebooks/` 这类非运行时目录的自动创建逻辑，因此旧的 notebook/notebooks 自动生成来源已经移除
- `src/runtime/io_utils.py` 会从仓库位置向上寻找相邻的 `myutils/file_utils/`，也支持显式 `MYUTILS_ROOT`；不再依赖固定父目录深度
- 项目里重复出现的 JSON 读写 / JSON-safe 转换逻辑继续尽量复用 `myutils/file_utils/json_io.py`

根 `skill.txt` 和旧 `skills/*_skill.txt` shards 已不再作为入口文件；项目级 agent 规则收敛到：
- `AGENTS.md`
- `docs/AGENT_MANIFEST.json`
- `.agents/skills/aiforbn-workflow/SKILL.md`
- `.agents/skills/aiforbn-overleaf-proposal/SKILL.md`
- `skills/ai_native_workflow.txt`

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
   - `artifacts/bn_model_role_comparison.csv`（现已收敛为 compact 的 5 行 BN 角色对照表）
   - `artifacts/demo_candidate_rank_stability_summary.csv`
   - `artifacts/demo_candidate_structure_followup_report.csv`
   - 并同步接入 `artifacts/experiment_summary.json`
5. 因此当前**更安全的汇报口径**应是：
   - BN-themed formula-level screening PoC with honest diagnostics
   - not BN-centered discovery

## 当前验证状态
2026-07-19 接管轮已修复 relocated checkout 暴露的固定父目录导入故障，并完成测试完备性补强。当前验证证据：

2026-07-20 监督维护轮进一步完成：
- 把 `human_docs/` 的用户所有、默认只读边界提升为 manifest 字段和所有已声明 agent instruction surfaces 的稳定 marker，并加入削弱/缺失负向测试；没有修改任何 `human_docs/` 文件
- 移除零仓内调用、仅作 backward compatibility 的 `select_model_type(...)` façade
- 把 Streamlit 已过期的 `use_container_width` 参数迁移为 `width='stretch'`，新增真实 Streamlit `AppTest` render regression，并完成有时限的 headless server health check
- 严格复核后补齐 command index 的模块依赖 round-trip，并把 public-surface AST guard 接入 architecture focused profile；修复了原先标题解析不匹配导致的空集合假通过
- 对 runtime 目录、agent-state、JSON、dataset、artifact、plot 输出和 cache 清理加入 `human_docs/` 写入/删除阻断，同时把全部 manifest module public surfaces 纳入 policy marker 验证
- 退役仍可执行的非 BN toy candidate grid，保留且测试唯一的 bounded BN-centered candidate space；移除零引用的旧 rank-stability table builder
- grouped robustness 预测保留 DataFrame feature names，消除 sklearn feature-name warnings
- Round 3 进一步封死伪造 project root、直接/软链接 human-doc cache root、输出叶子软链接和硬链接别名；runtime/dataset/report/plot 会在任何目录创建、写入或删除前预检全部目录及具体输出叶子的根目录归属、类型和父链，结构配置、动态 CIF 及 stale-CIF 清理均保留/检查原始叶子身份，cache 清理安全跳过目录软链接
- Round 4 补齐大小写等价的 human-doc 路径识别、cache root 任意软链接组件与 discovery 逃逸阻断、JSON/agent-state 序列化先于目录创建，并把 Python config bytecode、Matplotlib/JARVIS 的间接 cache/archive 写入纳入同一 canonical guard；没有修改或重算 `human_docs/` 与 scientific artifacts
- Round 5 固定单次校验后的 JARVIS metadata snapshot，拒绝绝对路径、遍历、分隔符、空值和畸形 archive tag 后再把同一 URL/tag 与 canonical `store_dir` 交给依赖；空白 `MPLCONFIGDIR` 不再退化为当前目录，v18 alignment status 也纳入 contract verifier；没有联网下载、修改 `human_docs/` 或重算 scientific artifacts
- Round 9 为未来完整运行增加本地 source/config/dataset artifact provenance 完成标记，viewer 会显式把现有未重算快照标为 unverified，并跟随配置的 artifact root 与 summary 中的 execution 路径；同时修复重复 prediction source、disabled optional outputs、空 structure bridge 元数据、大小写 CIF stale cleanup 与 CSV failure-before-replace，仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 10 修复 completion marker 在“写完后抛错”时仍可能把部分 bundle 标为 current 的故障；provenance 现在要求完整 marker 字段与 schema-valid dataset manifest，viewer 对缺失 core bundle 或畸形 provenance/summary/manifest fail closed；同时移除无消费者的 `screening.enabled` 假开关并补齐根 Python summary 对 runtime/UI callable 的非空验证，仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 11 把 provenance 升级为实际成功发布文件的 v2 内容承诺：固定、可选、配置化、动态 CIF 与 parity plot 输出均在成功写入后登记并以相对路径及 SHA-256 固化，marker 严格最后发布；viewer 对缺失、篡改、畸形或未纳入本轮发布的已知输出 fail closed，同时继续忽略无关 extra/cache 文件；仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 12 验证重复运行会让 marker 只承诺本轮实际成功输出，并修复三处相邻 truth-contract 缺口：viewer 不再渲染 provenance 非 current 的已承诺表格，BN slice/family 数据不足时 summary 不再崩溃或误报空 prediction 文件，candidate generator 会把当前 chemical-plausibility 配置传入注释器；同时让 control-plane 测试不依赖 checkout 目录名，仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 13 补齐 BN slice/family prediction 的四状态、双向同目录切换与 provenance 交叉验证，并修复 viewer 对畸形/legacy marker 或 viewer 二次降级 bundle 仍渲染 committed-looking 内容的 fail-open：现在只有最终 assessment 为 current 且存在明确 v2 committed-path set 时才渲染任何 report table；仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 14 用 source-derived 30-section render inventory 和真实 AppTest 复核单一最终 render gate，并修复两条 dynamic execution 路径契约：custom execution 变为空时不再回退并误认 stale default 文件；summary 中 present 但无效、缺失、别名或未承诺的 JSON/CSV 路径会令最终 assessment fail closed。默认/custom/empty 同目录切换及三类 dynamic output 的 missing/byte-mismatch 均已回归覆盖；仍未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 15 修复 current-v2 summary 的 nested object-shape 与 file-identity 漏洞：`screening`/`structure_generation_bridge` 的错误 JSON 类型不再崩溃或静默通过，三个 dynamic execution 声明必须指向各自配置并 guard 后的同一文件，不能把已承诺的 BN slice 或另一 execution CSV 重新贴标签；absent/null/empty container、规范化路径与本机真实 same-file 大小写别名仍保留有效。全报告仍只经过单一最终 render gate；未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- Round 16 将同一 dynamic execution role 契约前移到 public writer 的纯预检：错误 nested shape、缺失/错后缀/跨角色/固定输出别名及 inactive declaration 会在创建目录、失效旧 marker 或写删任何 bundle 文件前拒绝；summary builder、writer 与 viewer 共用 runtime role registry，viewer 继续独立防御持久化后的畸形状态。32 个拒绝状态均验证无 prior root 与同目录 prior-valid bundle 的字节级原子性，9 个 fallback/规范化/same-file 控制仍可成功发布 current v2；未修改或重算 `human_docs/`、`data/` 或 scientific artifacts
- contract verifier 现精确锁定 validation-profile 命令序列、三个 active project-skill 记录与七个 retired-guidance 路径；public-surface 测试同时核对模块摘要及根摘要的 callable 参数顺序和 keyword-only 边界
- `--verify-agent-contract` 现会精确锁定六个 module 的 path/role/public-surface/agent-rules/local-utils/allowed-dependencies；公开 surface 测试逐个要求四个生产模块非空，并显式覆盖 import re-export

1. AI-native contract 与命令索引：
- `conda run -n quant python3 main.py --emit-agent-commands`
- `conda run -n quant python3 main.py --verify-agent-contract`
- 结果：两者通过，contract status 为 `ok`，无 warnings / errors

2. 快速主线烟测：
- `conda run -n quant python3 main.py --dry-run`
- 结果：通过；候选空间、4 组 feature sets、4 组默认模型与 dummy baseline 均可导入和实例化

3. 完整 src 测试：
- `conda run -n quant python3 -m pytest -q src`
- 结果：`453 passed, 1 warning`
- 剩余 warning 是 PyTorch nested-tensor prototype 提示，不是测试失败；原 sklearn feature-name warnings 已消除

4. UI 文字化验证：
- `conda run -n quant python3 -m pytest -q src/ui/tests/test_streamlit_app.py`
- 结果：`99 passed`，包含真实 Streamlit renderer、source-derived 全 section gate、dynamic custom/empty 路径切换、nested object-shape 与 shared file-identity/role matrix、无效或未承诺 summary declaration fail-closed、三类 dynamic output 的 missing/byte-mismatch 交叉矩阵、completion/provenance matrix、BN slice/family 四状态及非对称 provenance cross-product、content-mixed bundle mutation matrix 与非 current committed-output 抑制
- 有时限的 `streamlit run ... --server.headless=true` 启动后，`/_stcore/health` 与根页面均成功响应；验证后进程已终止

5. 语法与 diff 卫生：
- `conda run -n quant python3 -m compileall -q main.py src`
- `git diff --check`
- 结果：通过

本轮新增或强化的主要契约测试覆盖：
- relocated checkout 与 `MYUTILS_ROOT` override
- control-plane 命令在 `myutils` 不可用时仍保持纯 JSON、可独立检查 contract
- manifest command mapping、模块依赖边界、跨模块 private/wildcard imports、反向 public-surface 文档校验
- 完整 `main.py` BN-centered alternative branch 与 summary/artifact 参数传递
- formula-only screening 强边界、候选公式 featurization 原子失败、自定义 formula column
- BN diagnostic disabled / insufficient-data 状态与 BN-centered selection 复核
- processed-cache provenance 与 target-column identity、raw-record lookup、Pydantic schema、Torch regressor 快速契约
- JARVIS store/archive 与 Matplotlib import-time cache 的 canonical output guard，以及无效 JSON payload 的 pre-effect 原子失败
- validation profiles、project skills、retired guidance 与两层 Python surface callable signatures 的非空/精确契约
- BN stratified diagnostics 强制 formula grouping 并按唯一公式聚合，避免重复公式跨 fold 泄漏
- decision policy disabled 语义、结构工件路径 containment、core/pairwise/case/Unicode/hardlink alias 防护，以及空结果第二轮清除旧 JSON/CSV/CIF

因此当前最准确的表述是：
- **代码、contract、dry-run 与完整测试套件均已通过**
- **本轮没有重算完整 scientific artifacts；如需刷新 research/demo 产物，应单独运行 full pipeline 并审查生成物**

## 当前最重要的记录文件
### 应继续保留并视为主状态文件
- `HANDOFF.md`：中文交接与当前状态摘要
- `PY_FILES_SUMMARY.md`：AI-facing Python surface 摘要

### 只读的人类上下文
- `human_docs/` 全部由用户管理，默认只读，不属于 agent-owned 状态或 AI-facing contract。
- `human_docs/task_notes/literature_mining/MODEL_UPGRADE_RESEARCH_2026-04-20.md` 只能作为历史建模方向证据；采用其中建议前必须用当前代码、测试与研究边界重新验证。

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
默认恢复动作：
1. 先读：
   - `AGENTS.md`
   - `docs/AGENT_MANIFEST.json`
   - `skills/ai_native_workflow.txt`
   - `.agents/skills/aiforbn-workflow/SKILL.md`
   - `docs/HANDOFF.md`
2. 运行：
   - `python3 main.py --emit-agent-commands`
   - `python3 main.py --verify-agent-contract`
3. 先确认这轮是：
   - architecture / docs / skills / contract maintenance
   - single-module coding
   - scientific pipeline or artifact regeneration
   - research-plan / Overleaf delivery
4. 如果进入单模块 coding：
   - 只选一个模块
   - 读最近的模块 `AGENTS.md` 和 `PY_FILES_SUMMARY.md`
   - 明确允许改哪些文件、禁止碰哪些文件
   - 可对低风险局部实现使用 `spark_coder`，但主 Codex 必须审查 diff 和测试
5. 若形成 checkpoint：
   - 排除不该提交的文件
   - 清缓存
   - 按 `--emit-agent-commands` 给出的最小验证 profile 跑验证
   - 通过后再 `git add / commit / push`

只有当任务明确涉及老师回覆、导师汇报或 proposal 时，才额外读：
   - `human_docs/project_reports/老師回覆.txt`
   - `human_docs/project_reports/项目汇报.md`
   - `human_docs/project_reports/给见微的说明.md`

## 当前不应丢失的判断
- 不要因为本机不是 CUDA 机器就自动退缩换方向。
- 但也不要在没有正向证据时，仅因为“模型更重”就要求 GPU。
- 当前实验结论还不足以说明“只要上 GPU 就能赢”。
- 目前真正的下一个 blocker 不是 GPU，而是：
  - **TabPFN license/token**
- 从结构规范角度看，当前代码已经进一步收敛到 4 个正式生产模块：`runtime`、`materials`、`torch_models`、`ui`。
- 模块模板要求目前已满足，每个正式模块都带有自己的 `AGENTS.md` 和 `utils.py`。
- 当前生产依赖关系也已明显收敛：`runtime -> []`、`torch_models -> []`、`ui -> [runtime]`、`materials -> [runtime, torch_models]`。
- 也就是说，之前那种 `reporting` / `structure_execution` 只是目录独立、实现上却从属于主业务链的问题，已经通过并回 `materials` 解决。
- 当前工程动作应先通过 `--emit-agent-commands` 选择最小验证 profile，再进入单模块 coding 或 artifact regeneration。
- 当前科研动作仍不应盲目扩展实验面；只有任务明确涉及导师汇报时，才回到老师回覆与证据口径补齐。
