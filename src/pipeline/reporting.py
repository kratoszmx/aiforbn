from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt


def build_experiment_summary(dataset_df, bn_df, candidate_df, split_masks, selection_summary, cfg):
    formula_col = cfg['data']['formula_column']
    target_col = cfg['data']['target_column']
    split_metadata = split_masks.get('metadata', {})
    selected_feature_set = selection_summary.get('selected_feature_set', cfg['features']['feature_set'])
    selected_model_type = selection_summary.get('selected_model_type', cfg['model']['type'])

    return {
        'dataset': {
            'dataset_name': cfg['data']['dataset'],
            'target_column': target_col,
            'rows': int(len(dataset_df)),
            'target_non_null_rows': int(dataset_df['target'].notna().sum()),
            'unique_formulas': int(dataset_df[formula_col].astype(str).nunique()),
            'bn_rows': int(len(bn_df)),
            'bn_unique_formulas': int(bn_df[formula_col].astype(str).nunique()) if not bn_df.empty else 0,
        },
        'features': {
            'feature_family': cfg['features'].get('feature_family', 'composition_only'),
            'configured_default_feature_set': cfg['features']['feature_set'],
            'candidate_feature_sets': selection_summary.get(
                'candidate_feature_sets',
                cfg['features'].get('candidate_sets', [cfg['features']['feature_set']]),
            ),
            'selected_feature_set': selected_feature_set,
            'selected_feature_count': selection_summary.get('selected_feature_count'),
            'feature_set_results': selection_summary.get('feature_set_results', []),
        },
        'split': split_metadata,
        'feature_model_selection': selection_summary,
        'benchmarking': {
            'benchmark_artifact': 'benchmark_results.csv',
            'candidate_model_types': selection_summary.get(
                'candidate_model_types',
                cfg['model'].get('candidate_types', [cfg['model']['type']]),
            ),
            'dummy_baselines': cfg['model'].get('benchmark_baselines', ['dummy_mean']),
        },
        'screening': {
            'candidate_space_name': cfg['screening']['candidate_space_name'],
            'candidate_space_kind': cfg['screening']['candidate_space_kind'],
            'candidate_space_note': cfg['screening']['candidate_space_note'],
            'candidate_rows': int(len(candidate_df)),
            'top_k': int(cfg['screening']['top_k']),
            'ranking_artifact': 'demo_candidate_ranking.csv',
            'ranking_basis': 'composition_only_predicted_band_gap',
            'ranking_note': 'Composition-only ranking from formula-derived features; not structure-aware.',
            'ranking_feature_set': selected_feature_set,
            'ranking_model_type': selected_model_type,
        },
    }


def save_metrics_and_predictions(
    metrics,
    prediction_df,
    bn_df,
    screened_df,
    benchmark_df,
    experiment_summary,
    manifest,
    cfg,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'demo_candidate_ranking.csv', index=False)
    benchmark_df.to_csv(artifact_dir / 'benchmark_results.csv', index=False)
    (artifact_dir / 'experiment_summary.json').write_text(json.dumps(experiment_summary, indent=2))
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    legacy_screen_path = artifact_dir / 'screened_candidates.csv'
    if legacy_screen_path.exists():
        legacy_screen_path.unlink()


def save_basic_plots(prediction_df, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(prediction_df['target'], prediction_df['prediction'], alpha=0.7)
    ax.set_xlabel('True target')
    ax.set_ylabel('Predicted target')
    ax.set_title('Parity plot')
    fig.tight_layout()
    fig.savefig(artifact_dir / 'parity_plot.png', dpi=160)
    plt.close(fig)
