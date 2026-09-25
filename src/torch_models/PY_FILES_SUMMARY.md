# torch_models module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the documented public classes that external code may instantiate or import from `torch_models`.
Anything underscore-prefixed or omitted here should be treated as internal.

## base.py

- `TorchMLPRegressor`
  - Lightweight neural regressor over tabular composition-derived features.

## ensemble.py

- `TorchMLPEnsembleRegressor`
  - Multi-seed ensemble wrapper around `TorchMLPRegressor`.

## attention.py

- `TorchFractionalAttentionRegressor`
  - Dense attention regressor over fractional-composition vectors.

## sparse_attention.py

- `TorchSparseFractionalAttentionRegressor`
  - Sparse-token attention regressor over fractional-composition vectors.

## roost_like.py

- `TorchRoostLikeRegressor`
  - Roost-inspired present-element stoichiometry network.

## utils.py

- No public functions are currently exposed.

## tests/

- Run `conda run -n quant python -m pytest -q src/torch_models/tests` from the repository root; see [TESTING.md](../../TESTING.md) for prerequisites and the full change profile.
- `test_regressor_contracts.py` covers fast sklearn-style input, fit-state, attention-shape, device-policy, and ensemble-seed contracts.

## Notes

- External business logic usually reaches these classes through `materials.modeling.make_model(...)`.
- Underscore-prefixed helpers in `base.py` are internal to the `torch_models` module.
- The models accept numeric feature matrices through sklearn-style `fit` / `predict`; ensembles expose per-member predictions. Attention and Roost-like families require 118 fractional-composition features and remain outside the default sweep.
- MLP auto-device selection prefers CUDA, then MPS, then CPU. The attention family (including its sparse and Roost-like subclasses) uses CUDA or CPU in auto mode; explicit device choices remain caller-controlled. A test pass does not prove GPU execution or scientific accuracy.
