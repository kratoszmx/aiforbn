# ai_for_bn 主要 Python 文件说明

本文档概括当前项目中主要 Python 文件及其核心函数用途。

说明约定：
- 本项目不作为 Python 包维护。
- 默认从仓库根目录运行 `python main.py`。
- 主要通用函数优先复用 `../myutils`。
- 本文档侧重“当前有哪些主要函数、做什么用”，不展开实现细节。

---

## 1. `main.py`

**文件用途**：项目主入口，负责串起完整 PoC 流程。

### `main()`
- 功能：执行完整 BN PoC pipeline。
- 主流程包括：
  - 清理 Python 缓存目录
  - 读取配置
  - 准备运行目录
  - 加载/构建数据集
  - 提取 BN 子集
  - 生成候选材料
  - 构造特征
  - 划分数据集
  - 训练模型
  - 评估模型
  - 候选筛选
  - 保存指标与图表
- 输出：控制台打印当前运行摘要。

---

## 2. `src/core/io_utils.py`

**文件用途**：配置读取、运行目录初始化、缓存清理。

### `load_yaml(path)`
- 功能：读取 YAML 配置文件并解析为字典。

### `ensure_runtime_dirs(cfg)`
- 功能：根据配置准备运行所需目录。
- 当前会准备：
  - `data/raw`
  - `data/processed`
  - `artifacts`
  - 若干项目运行辅助目录
- 复用：底层目录创建复用了 `../myutils/file_utils.py` 的 `ensure_dirs`。

### `clear_project_cache(project_root_path='.')`
- 功能：清理项目下的 `__pycache__` 与 `.pytest_cache`。
- 复用：直接调用 `../myutils/file_utils.py` 的 `delete_cache`。

---

## 3. `src/core/schema.py`

**文件用途**：定义核心数据结构。

### `DatasetManifest`
- 功能：描述数据集来源、获取时间、版本提示等信息。

### `MaterialRecord`
- 功能：定义材料记录的基础结构。

---

## 4. `src/pipeline/data.py`

**文件用途**：数据获取、规范化、缓存保存。

### `load_or_build_dataset(cfg)`
- 功能：
  - 若本地已有处理后数据与 manifest，则直接读取
  - 否则通过 JARVIS-Tools 下载 2DMatPedia 数据并进行规范化
- 输出：`(DataFrame, manifest_dict)`

---

## 5. `src/pipeline/features.py`

**文件用途**：BN 过滤、候选生成、特征构造、数据切分、模型训练、评估与筛选。

### `extract_elements(formula)`
- 功能：从化学式中提取元素符号列表。

### `filter_bn(df, formula_col='formula')`
- 功能：筛出同时包含 `B` 和 `N` 的记录。

### `generate_bn_candidates()`
- 功能：生成一组简单的 BN 类候选材料组合。

### `build_feature_table(df, formula_col='formula')`
- 功能：从化学式构造基础数值特征表。

### `make_split_masks(df, cfg)`
- 功能：按配置生成 train/val/test 布尔掩码。

### `make_model(cfg)`
- 功能：根据配置构建 sklearn 回归模型。
- 当前支持：
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`

### `train_baseline_model(df, split_masks, cfg)`
- 功能：训练当前基线模型。
- 输出：`(model, feature_columns)`

### `evaluate_predictions(df, split_masks, model, feature_columns)`
- 功能：在测试集上评估模型并生成预测结果表。
- 输出：
  - 指标字典
  - 带预测值的 DataFrame

### `screen_candidates(candidate_df, model, feature_columns, cfg)`
- 功能：对候选材料进行打分并返回 Top-K 筛选结果。

---

## 6. `src/pipeline/reporting.py`

**文件用途**：保存结构化产物与基础图表。

### `save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg)`
- 功能：保存本次运行的关键产物，包括：
  - `metrics.json`
  - `predictions.csv`
  - `bn_slice.csv`
  - `screened_candidates.csv`
  - `manifest.json`

### `save_basic_plots(prediction_df, cfg)`
- 功能：生成并保存基础 parity plot。

---

## 7. `apps/streamlit_app.py`

**文件用途**：读取 `artifacts/` 下结果并做最小可展示 UI。

当前页面主要展示：
- 指标 JSON
- 预测结果样例
- 候选筛选结果样例

---

## 8. `tests/`

**文件用途**：最小测试集。

### `tests/conftest.py`
- 在测试开始前清理项目缓存。

### `tests/test_bn_filter.py`
- 验证元素提取与 BN 过滤逻辑。

### `tests/test_schema.py`
- 验证 schema 基本构造可用。
