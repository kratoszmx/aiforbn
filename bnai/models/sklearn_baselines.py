from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def make_model(cfg: dict):
    model_type = cfg['model']['type']
    if model_type == 'random_forest':
        return RandomForestRegressor(**cfg['model']['random_forest'])
    if model_type == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**cfg['model']['hist_gradient_boosting'])
    raise ValueError(f'Unsupported model type: {model_type}')
