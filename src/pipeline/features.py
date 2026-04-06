from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ATOMIC_NUMBERS = {
    'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'Al': 13, 'P': 15,
    'Ga': 31, 'As': 33, 'In': 49, 'Sb': 51, 'Tl': 81, 'Bi': 83,
}


def extract_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def filter_bn(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    mask = df[formula_col].astype(str).apply(lambda x: {'B', 'N'}.issubset(set(extract_elements(x))))
    out = df.loc[mask].copy()
    out['elements'] = out[formula_col].astype(str).apply(extract_elements)
    return out


def generate_bn_candidates() -> pd.DataFrame:
    group13 = ['B', 'Al', 'Ga', 'In', 'Tl']
    group15 = ['N', 'P', 'As', 'Sb', 'Bi']
    rows = []
    for left in group13:
        for right in group15:
            rows.append({'formula': f'{left}{right}', 'candidate_source': 'simple_bn_substitutions'})
    return pd.DataFrame(rows)


def _basic_features(formula: str) -> dict:
    elements = extract_elements(formula)
    z = [ATOMIC_NUMBERS.get(e, 0) for e in elements]
    return {
        'n_elements': len(elements),
        'sum_z': sum(z),
        'max_z': max(z) if z else 0,
        'min_z': min(z) if z else 0,
        'mean_z': sum(z) / len(z) if z else 0,
        'contains_B': int('B' in elements),
        'contains_N': int('N' in elements),
    }


def build_feature_table(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    base = df.copy()
    feature_rows = base[formula_col].astype(str).apply(_basic_features).apply(pd.Series)
    return pd.concat([base.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)


def make_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg['project']['random_seed'])
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * cfg['split']['train_ratio'])
    val_end = train_end + int(n * cfg['split']['val_ratio'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    masks = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    return masks


def _feature_columns(df: pd.DataFrame) -> list[str]:
    banned = {'record_id', 'source', 'formula', 'target', 'elements'}
    return [c for c in df.columns if c not in banned]


def make_model(cfg: dict):
    model_type = cfg['model']['type']
    if model_type == 'random_forest':
        return RandomForestRegressor(**cfg['model']['random_forest'])
    if model_type == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**cfg['model']['hist_gradient_boosting'])
    raise ValueError(f'Unsupported model type: {model_type}')


def train_baseline_model(df: pd.DataFrame, split_masks, cfg: dict) -> tuple[object, list[str]]:
    feature_columns = _feature_columns(df)
    train_df = df.loc[split_masks['train']].copy()
    train_df = train_df[train_df['target'].notna()].copy()
    X = train_df[feature_columns]
    y = train_df['target']

    model = make_model(cfg)
    model.fit(X, y)
    return model, feature_columns


def evaluate_predictions(df: pd.DataFrame, split_masks, model, feature_columns: list[str]):
    test_df = df.loc[split_masks['test']].copy()
    test_df = test_df[test_df['target'].notna()].copy()
    X = test_df[feature_columns]
    y = test_df['target']
    pred = model.predict(X)

    metrics = {
        'mae': float(mean_absolute_error(y, pred)),
        'rmse': float(mean_squared_error(y, pred) ** 0.5),
        'r2': float(r2_score(y, pred)),
    }

    prediction_df = test_df[['formula', 'target']].copy()
    prediction_df['prediction'] = pred
    prediction_df['abs_error'] = (prediction_df['target'] - prediction_df['prediction']).abs()
    return metrics, prediction_df


def screen_candidates(candidate_df: pd.DataFrame, model, feature_columns: list[str], cfg: dict) -> pd.DataFrame:
    feature_df = build_feature_table(candidate_df, formula_col='formula')
    pred = model.predict(feature_df[feature_columns])
    out = feature_df[['formula', 'candidate_source']].copy()
    out['prediction'] = pred
    return out.sort_values('prediction', ascending=False).head(cfg['screening']['top_k']).reset_index(drop=True)
