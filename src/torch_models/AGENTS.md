# Torch models

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

Owns sklearn-style regressors with no production dependency on other project modules.

- Public classes are in `base.py`, `ensemble.py`, `attention.py`, `sparse_attention.py` and `roost_like.py`; business code usually reaches them through `materials.modeling.make_model`.
- Keep fit/predict, input validation, reproducible seeds and device policy independently testable.
- Attention/Roost-like implementations are experimental and outside the default sweep; successful model construction does not establish training quality or GPU execution.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/torch_models/tests`.
