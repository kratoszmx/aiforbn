from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.features import (
    DOMAIN_SUPPORT_RANKING_NOTE,
    NOVELTY_ANNOTATION_RANKING_NOTE,
    NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
    NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
    get_feature_family,
    get_screening_ranking_metadata,
    feature_set_supports_formula_only_screening,
)


def build_experiment_summary(dataset_df, bn_df, candidate_df, split_masks, selection_summary, cfg):
    formula_col = cfg['data']['formula_column']
    target_col = cfg['data']['target_column']
    split_metadata = split_masks.get('metadata', {})
    selected_feature_set = selection_summary.get('selected_feature_set', cfg['features']['feature_set'])
    selected_model_type = selection_summary.get('selected_model_type', cfg['model']['type'])
    screening_feature_set = selection_summary.get('screening_selected_feature_set', selected_feature_set)
    screening_model_type = selection_summary.get('screening_selected_model_type', selected_model_type)
    selected_feature_family = selection_summary.get(
        'selected_feature_family',
        get_feature_family(selected_feature_set),
    )
    screening_feature_family = selection_summary.get(
        'screening_selected_feature_family',
        get_feature_family(screening_feature_set),
    )
    screening_matches_overall = bool(
        selection_summary.get(
            'screening_selection_matches_overall',
            screening_feature_set == selected_feature_set and screening_model_type == selected_model_type,
        )
    )
    screening_selection_note = selection_summary.get(
        'screening_selection_note',
        'Formula-only screening reuses the overall selected combo.',
    )
    chemical_plausibility_cfg = cfg['screening'].get('chemical_plausibility', {})
    chemical_plausibility_enabled = bool(chemical_plausibility_cfg.get('enabled', True))
    ranking_config_metadata = get_screening_ranking_metadata(cfg)

    plausibility_pass_count = None
    plausibility_fail_count = None
    plausibility_failed_formulas = []
    if 'chemical_plausibility_pass' in candidate_df.columns:
        plausibility_pass_mask = candidate_df['chemical_plausibility_pass'].fillna(False).astype(bool)
        plausibility_pass_count = int(plausibility_pass_mask.sum())
        plausibility_fail_count = int((~plausibility_pass_mask).sum())
        plausibility_failed_formulas = (
            candidate_df.loc[~plausibility_pass_mask, formula_col].astype(str).tolist()
        )

    domain_support_reference_formula_count = None
    domain_support_penalized_rows = None
    domain_support_low_support_rows = None
    if 'domain_support_reference_formula_count' in candidate_df.columns and not candidate_df.empty:
        domain_support_reference_formula_count = int(
            candidate_df['domain_support_reference_formula_count'].fillna(0).iloc[0]
        )
    if 'domain_support_penalty' in candidate_df.columns:
        domain_support_penalties = candidate_df['domain_support_penalty'].fillna(0.0)
        domain_support_penalized_rows = int((domain_support_penalties > 0).sum())
    if 'domain_support_percentile' in candidate_df.columns:
        percentile_threshold = float(ranking_config_metadata['domain_support_penalize_below_percentile'])
        domain_support_low_support_rows = int(
            candidate_df['domain_support_percentile'].fillna(100.0).lt(percentile_threshold).sum()
        )

    ranking_metadata = get_screening_ranking_metadata(
        cfg,
        domain_support_penalty_applied=bool(domain_support_penalized_rows),
    )
    ranking_basis = ranking_metadata['ranking_basis']
    ranking_note = ranking_metadata['ranking_note']
    if bool(ranking_metadata['domain_support_enabled']):
        ranking_note = f'{ranking_note} {DOMAIN_SUPPORT_RANKING_NOTE}'
    if not screening_matches_overall:
        ranking_note = (
            f'{ranking_note} The best overall validation combo is structure-aware, so formula-only '
            'candidate screening falls back to the best candidate-compatible combo.'
        )
    if chemical_plausibility_enabled:
        ranking_note = (
            f'{ranking_note} Candidates are also annotated with a lightweight pymatgen oxidation-state '
            'plausibility screen, and passing formulas are prioritized ahead of formulas that fail '
            'basic charge-balance checks.'
        )
    novelty_annotation_enabled = bool(
        {
            'candidate_novelty_bucket',
            'candidate_is_formula_level_extrapolation',
        }.issubset(candidate_df.columns)
    )
    novelty_bucket_order = [
        NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
        NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
        NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    ]
    novelty_bucket_counts = None
    standard_top_k_novelty_bucket_counts = None
    formula_level_extrapolation_candidate_count = None
    formula_level_extrapolation_shortlist = []
    novelty_interpretation_note = None
    if novelty_annotation_enabled:
        novelty_bucket_counts = {
            bucket: int(
                candidate_df['candidate_novelty_bucket'].eq(bucket).sum()
            )
            for bucket in novelty_bucket_order
        }
        if 'screening_selected_for_top_k' in candidate_df.columns:
            top_k_candidate_df = candidate_df.loc[
                candidate_df['screening_selected_for_top_k'].fillna(False).astype(bool)
            ]
            standard_top_k_novelty_bucket_counts = {
                bucket: int(
                    top_k_candidate_df['candidate_novelty_bucket'].eq(bucket).sum()
                )
                for bucket in novelty_bucket_order
            }
        formula_level_extrapolation_candidate_count = int(
            candidate_df['candidate_is_formula_level_extrapolation'].fillna(False).astype(bool).sum()
        )
        novel_df = candidate_df.loc[
            candidate_df['candidate_novelty_bucket'].eq(NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION)
        ].copy()
        if 'ranking_rank' in novel_df.columns:
            novel_df = novel_df.sort_values('ranking_rank', ascending=True)
        for _, row in novel_df.head(5).iterrows():
            shortlist_row = {'formula': str(row[formula_col])}
            if 'ranking_rank' in row and pd.notna(row['ranking_rank']):
                shortlist_row['ranking_rank'] = int(row['ranking_rank'])
            if 'novel_formula_rank' in row and pd.notna(row['novel_formula_rank']):
                shortlist_row['novel_formula_rank'] = int(row['novel_formula_rank'])
            if 'ranking_score' in row and pd.notna(row['ranking_score']):
                shortlist_row['ranking_score'] = float(row['ranking_score'])
            if 'chemical_plausibility_pass' in row and pd.notna(row['chemical_plausibility_pass']):
                shortlist_row['chemical_plausibility_pass'] = bool(row['chemical_plausibility_pass'])
            if 'screening_selected_for_top_k' in row and pd.notna(row['screening_selected_for_top_k']):
                shortlist_row['screening_selected_for_top_k'] = bool(row['screening_selected_for_top_k'])
            if 'screening_selection_decision' in row and pd.notna(row['screening_selection_decision']):
                shortlist_row['screening_selection_decision'] = str(row['screening_selection_decision'])
            formula_level_extrapolation_shortlist.append(shortlist_row)
        novelty_interpretation_note = (
            'Standard top-k remains the default ranking output, but novelty should be interpreted '
            'separately: train+val rediscovery is in-domain replay, held-out-known formulas are '
            'known elsewhere in the dataset, and formula-level extrapolation only means unseen '
            'formula compositions inside this toy candidate space.'
        )
        ranking_note = f'{ranking_note} {NOVELTY_ANNOTATION_RANKING_NOTE}'

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
            'feature_space_family': cfg['features'].get('feature_family', 'mixed_formula_and_structure'),
            'configured_default_feature_set': cfg['features']['feature_set'],
            'candidate_feature_sets': selection_summary.get(
                'candidate_feature_sets',
                cfg['features'].get('candidate_sets', [cfg['features']['feature_set']]),
            ),
            'selected_feature_set': selected_feature_set,
            'selected_feature_family': selected_feature_family,
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
            'candidate_formulas_have_structures': False,
            'top_k': int(cfg['screening']['top_k']),
            'ranking_artifact': 'demo_candidate_ranking.csv',
            'ranking_basis': ranking_basis,
            'ranking_note': ranking_note,
            'ranking_feature_set': screening_feature_set,
            'ranking_feature_family': screening_feature_family,
            'ranking_model_type': screening_model_type,
            'domain_support_enabled': bool(ranking_metadata['domain_support_enabled']),
            'domain_support_method': ranking_metadata['domain_support_method'],
            'domain_support_distance_metric': ranking_metadata['domain_support_distance_metric'],
            'domain_support_reference_split': ranking_metadata['domain_support_reference_split'],
            'domain_support_reference_formula_count': domain_support_reference_formula_count,
            'domain_support_k_neighbors': int(ranking_metadata['domain_support_k_neighbors']),
            'domain_support_note': ranking_metadata['domain_support_note'],
            'domain_support_penalty_enabled': bool(ranking_metadata['domain_support_penalty_enabled']),
            'domain_support_penalty_active': bool(ranking_metadata['domain_support_penalty_active']),
            'domain_support_penalty_weight': float(ranking_metadata['domain_support_penalty_weight']),
            'domain_support_penalize_below_percentile': float(
                ranking_metadata['domain_support_penalize_below_percentile']
            ),
            'domain_support_penalized_rows': domain_support_penalized_rows,
            'domain_support_low_support_rows': domain_support_low_support_rows,
            'chemical_plausibility_enabled': chemical_plausibility_enabled,
            'chemical_plausibility_method': chemical_plausibility_cfg.get(
                'method',
                'pymatgen_common_oxidation_state_balance',
            ),
            'chemical_plausibility_selection_policy': chemical_plausibility_cfg.get(
                'selection_policy',
                'annotate_and_prioritize_passing_candidates',
            ),
            'chemical_plausibility_note': chemical_plausibility_cfg.get(
                'note',
                (
                    'Formula-level plausibility annotation using pymatgen oxidation-state guesses. '
                    'This is not a structure, thermodynamic stability, phonon stability, or '
                    'synthesis feasibility filter.'
                ),
            ),
            'chemical_plausibility_passed_rows': plausibility_pass_count,
            'chemical_plausibility_failed_rows': plausibility_fail_count,
            'chemical_plausibility_failed_formulas': plausibility_failed_formulas,
            'novelty_annotation_enabled': novelty_annotation_enabled,
            'novelty_bucket_counts': novelty_bucket_counts,
            'standard_top_k_novelty_bucket_counts': standard_top_k_novelty_bucket_counts,
            'formula_level_extrapolation_candidate_count': formula_level_extrapolation_candidate_count,
            'formula_level_extrapolation_shortlist': formula_level_extrapolation_shortlist,
            'novelty_interpretation_note': novelty_interpretation_note,
            'screening_selection_scope': selection_summary.get(
                'screening_selection_scope',
                'candidate_compatible_formula_only',
            ),
            'screening_candidate_feature_sets': selection_summary.get(
                'screening_candidate_feature_sets',
                [],
            ),
            'screening_selection_note': screening_selection_note,
            'ranking_matches_best_overall_evaluation': screening_matches_overall,
            'best_overall_evaluation_feature_set': selected_feature_set,
            'best_overall_evaluation_feature_family': selected_feature_family,
            'best_overall_evaluation_model_type': selected_model_type,
            'best_overall_evaluation_candidate_compatible': feature_set_supports_formula_only_screening(
                selected_feature_set
            ),
            'ranking_uncertainty_method': ranking_metadata['ranking_uncertainty_method'],
            'ranking_uncertainty_penalty': float(ranking_metadata['ranking_uncertainty_penalty']),
            'candidate_annotations': [
                'domain_support_reference_formula_count',
                'domain_support_k_neighbors',
                'domain_support_nearest_formula',
                'domain_support_nearest_distance',
                'domain_support_mean_k_distance',
                'domain_support_percentile',
                'domain_support_penalty',
                'chemical_plausibility_pass',
                'chemical_plausibility_guess_count',
                'chemical_plausibility_primary_oxidation_state_guess',
                'chemical_plausibility_note',
                'seen_in_dataset',
                'dataset_formula_row_count',
                'seen_in_train_plus_val',
                'train_plus_val_formula_row_count',
                'candidate_is_seen_in_dataset',
                'candidate_is_seen_in_train_plus_val',
                'candidate_is_formula_level_extrapolation',
                'candidate_novelty_bucket',
                'candidate_novelty_priority',
                'candidate_novelty_note',
                'novelty_rank_within_bucket',
                'novel_formula_rank',
                'screening_selected_for_top_k',
                'screening_selection_decision',
            ],
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
