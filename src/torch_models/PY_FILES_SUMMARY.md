# torch_models module public surface

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

## Notes

- External business logic usually reaches these classes through `materials.modeling.make_model(...)`.
- Underscore-prefixed helpers in `base.py` are internal to the `torch_models` module.
