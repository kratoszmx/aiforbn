from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from materials.constants import *
from materials.feature_building import *
from materials.feature_building import _feature_columns, _feature_valid_mask

def make_model(cfg: dict, model_type: str | None = None):
    model_type = model_type or cfg['model']['type']
    if model_type == 'linear_regression':
        return LinearRegression(**cfg['model'].get('linear_regression', {}))
    if model_type == 'random_forest':
        return RandomForestRegressor(**cfg['model']['random_forest'])
    if model_type == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**cfg['model']['hist_gradient_boosting'])
    if model_type == 'torch_mlp':
        from torch_models.base import TorchMLPRegressor

        return TorchMLPRegressor(**cfg['model'].get('torch_mlp', {}))
    if model_type == 'torch_mlp_ensemble':
        from torch_models.ensemble import TorchMLPEnsembleRegressor

        return TorchMLPEnsembleRegressor(**cfg['model'].get('torch_mlp_ensemble', {}))
    if model_type == 'torch_fractional_attention':
        from torch_models.attention import TorchFractionalAttentionRegressor

        return TorchFractionalAttentionRegressor(**cfg['model'].get('torch_fractional_attention', {}))
    if model_type == 'torch_sparse_fractional_attention':
        from torch_models.sparse_attention import TorchSparseFractionalAttentionRegressor

        return TorchSparseFractionalAttentionRegressor(
            **cfg['model'].get('torch_sparse_fractional_attention', {})
        )
    if model_type == 'torch_roost_like':
        from torch_models.roost_like import TorchRoostLikeRegressor

        return TorchRoostLikeRegressor(**cfg['model'].get('torch_roost_like', {}))
    if model_type == 'dummy_mean':
        return DummyRegressor(**cfg['model'].get('dummy_mean', {'strategy': 'mean'}))
    raise ValueError(f'Unsupported model type: {model_type}')


def train_baseline_model(
    df: pd.DataFrame,
    split_masks,
    cfg: dict,
    model_type: str | None = None,
    include_validation: bool = False,
) -> tuple[object, list[str]]:
    feature_columns = _feature_columns(df)
    train_mask = pd.Series(split_masks['train'], index=df.index, dtype=bool)
    if include_validation:
        val_mask = pd.Series(split_masks['val'], index=df.index, dtype=bool)
        training_mask = train_mask | val_mask
    else:
        training_mask = train_mask
    train_df = df.loc[training_mask].copy()
    train_df = train_df[train_df['target'].notna()].copy()
    train_df = train_df.loc[_feature_valid_mask(train_df, feature_columns)].copy()
    if train_df.empty:
        raise ValueError('No training rows remain after filtering invalid feature rows')

    X = train_df[feature_columns]
    y = train_df['target']

    model = make_model(cfg, model_type=model_type)
    model.fit(X, y)
    return model, feature_columns


def _regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        'mae': float(mean_absolute_error(y_true_arr, y_pred_arr)),
        'rmse': float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        'r2': float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else None,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    split_masks,
    model,
    feature_columns: list[str],
    split_name: str = 'test',
):
    eval_mask = pd.Series(split_masks[split_name], index=df.index, dtype=bool)
    requested_eval_df = df.loc[eval_mask].copy()
    requested_eval_df = requested_eval_df[requested_eval_df['target'].notna()].copy()
    if requested_eval_df.empty:
        raise ValueError(f'No evaluation rows available for split: {split_name}')

    valid_mask = _feature_valid_mask(requested_eval_df, feature_columns)
    if int(valid_mask.sum()) != len(requested_eval_df):
        failed_formulas = requested_eval_df.loc[~valid_mask, 'formula'].astype(str).head(5).tolist()
        raise ValueError(
            f'Feature set cannot evaluate all {split_name} rows; '
            f'invalid formulas include: {failed_formulas}'
        )

    eval_df = requested_eval_df.loc[valid_mask].copy()
    X = eval_df[feature_columns]
    y = eval_df['target']
    pred = model.predict(X)

    metrics = _regression_metrics(y, pred)

    prediction_df = eval_df[['formula', 'target']].copy()
    prediction_df['prediction'] = pred
    prediction_df['abs_error'] = (prediction_df['target'] - prediction_df['prediction']).abs()
    return metrics, prediction_df

