from __future__ import annotations

from bnai.preprocess.featurize import build_feature_table


def screen_candidates(candidate_df, model_bundle, cfg):
    feature_df = build_feature_table(candidate_df, formula_col='formula')
    pred = model_bundle.model.predict(feature_df[model_bundle.feature_columns])
    out = feature_df[['formula', 'candidate_source']].copy()
    out['prediction'] = pred
    out = out.sort_values('prediction', ascending=False).head(cfg['screening']['top_k']).reset_index(drop=True)
    return out
