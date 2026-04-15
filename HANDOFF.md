# HANDOFF.md

## 项目
- 名称：AI for BN PoC
- 路径：`$HOME/projects/ai_for_bn`
- 目标：构建一个可运行、可展示的 AI for BN 最小 PoC
- 当前范围：纯软件 MVP，不涉及实验设备或实验流程

## 当前状态
- 主任务已固定为：BN 相关材料的性质预测与候选筛选
- 主目标列：`band_gap`
- 数据源：2DMatPedia（通过 JARVIS-Tools 获取）
- 当前基线：sklearn 回归模型 + 基础组成特征
- 当前展示方式：Streamlit
- 当前默认运行环境：zsh 的 `quant` 环境

## 已完成工作
- 建立了从数据加载、BN 过滤、特征构造、训练、评估、筛选到产物输出的完整主流程
- 保留了 `main.py` 作为线性入口，便于快速阅读和迁移到 notebook
- 增加了 `PY_FILES_SUMMARY.md`，用于说明本项目主要 Python 文件与函数用途
- 增加了 `项目简报_初稿.md`，用于向非化学/非机器学习背景读者快速介绍项目
- 将项目结构整理为非包化但分层明确的形式：
  - `src/core/`：配置与 schema
  - `src/pipeline/`：数据、特征、训练、评估、输出
  - `apps/`：展示入口
- 已接入 `../myutils` 的通用能力，当前至少复用：
  - `file_utils.delete_cache`
  - `file_utils.ensure_dirs`

## 当前目录重点
- `main.py`：主入口
- `configs/default.py`：配置
- `src/core/io_utils.py`：配置读取、目录初始化、缓存清理
- `src/core/schema.py`：基础 schema
- `src/pipeline/data.py`：数据集加载与规范化
- `src/pipeline/features.py`：BN 过滤、特征、切分、训练、评估、筛选
- `src/pipeline/reporting.py`：结果保存与图表输出
- `apps/streamlit_app.py`：展示界面
- `PY_FILES_SUMMARY.md`：主要函数说明文档
- `项目简报_初稿.md`：面向汇报场景的中文说明稿

## 当前判断
- 目前实现适合作为演示型 PoC，不适合直接视为科研级强基线
- 当前特征与候选生成逻辑偏简化，优先服务于快速演示与后续扩展
- 后续若继续深化，应优先考虑：
  - 更合理的特征工程
  - 更严格的数据切分与验证
  - 更可信的候选生成约束

## 交接时重点关注
- 是否还能进一步复用 `../myutils` 的现成函数
- 是否有真正低耦合、跨项目可复用的逻辑可以沉淀到 `../myutils`
- `main.py` 是否仍保持线性、清晰、低认知负担
- 文档之间是否存在重复描述，如有应继续精简
