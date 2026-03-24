from __future__ import annotations

from bnai.models.base import ModelBundle
from bnai.models.sklearn_baselines import make_model


def _feature_columns(df):
    banned = {'record_id', 'source', 'formula', 'target', 'elements'}
    return [c for c in df.columns if c not in banned]


def train_baseline_model(df, split_masks, cfg):
    feature_columns = _feature_columns(df)
    train_df = df.loc[split_masks['train']].copy()
    train_df = train_df[train_df['target'].notna()].copy()
    X = train_df[feature_columns]
    y = train_df['target']

    model = make_model(cfg)
    model.fit(X, y)
    return ModelBundle(name=cfg['model']['type'], model=model, feature_columns=feature_columns)
