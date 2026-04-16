from __future__ import annotations

from functools import lru_cache
import os
import re

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ATOMIC_NUMBERS = {
    'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'Al': 13, 'P': 15,
    'Ga': 31, 'As': 33, 'In': 49, 'Sb': 51, 'Tl': 81, 'Bi': 83,
}
DEFAULT_CANDIDATE_SPACE_NAME = 'toy_iii_v_demo_grid'
DEFAULT_CANDIDATE_SPACE_KIND = 'toy_demo'
DEFAULT_CANDIDATE_SPACE_NOTE = (
    'Formula-only Group 13/15 enumeration without stability, structure, or synthesis constraints.'
)
BASIC_FEATURE_SET = 'basic_formula_composition'
MATMINER_FEATURE_SET = 'matminer_composition'
FEATURE_FAMILY = 'composition_only'
DUMMY_FEATURE_SET = 'feature_agnostic_dummy'
RANKING_BASIS = 'composition_only_predicted_band_gap'
RANKING_NOTE = 'Composition-only ranking from formula-derived features; not structure-aware.'
FEATURE_SET_NOTES = {
    BASIC_FEATURE_SET: (
        'Hand-written control baseline from formula tokens and a small atomic-number lookup.'
    ),
    MATMINER_FEATURE_SET: (
        'Curated matminer composition descriptors from pymatgen Composition objects using '
        'Stoichiometry plus selected Magpie elemental-property statistics.'
    ),
    DUMMY_FEATURE_SET: 'Dummy regressor baseline that ignores composition features.',
}
BASIC_FEATURE_COLUMNS = (
    'n_elements',
    'sum_z',
    'max_z',
    'min_z',
    'mean_z',
    'contains_B',
    'contains_N',
)
MATMINER_SELECTED_RAW_LABELS = (
    '0-norm',
    '2-norm',
    '3-norm',
    'MagpieData mean Number',
    'MagpieData range Number',
    'MagpieData mean MendeleevNumber',
    'MagpieData range MendeleevNumber',
    'MagpieData mean AtomicWeight',
    'MagpieData range AtomicWeight',
    'MagpieData mean Row',
    'MagpieData range Row',
    'MagpieData mean Column',
    'MagpieData range Column',
    'MagpieData mean Electronegativity',
    'MagpieData range Electronegativity',
    'MagpieData mean NsValence',
    'MagpieData mean NpValence',
    'MagpieData mean NdValence',
    'MagpieData mean NfValence',
)


def extract_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def filter_bn(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    mask = df[formula_col].astype(str).apply(lambda x: {'B', 'N'}.issubset(set(extract_elements(x))))
    out = df.loc[mask].copy()
    out['elements'] = out[formula_col].astype(str).apply(extract_elements)
    return out


def generate_bn_candidates(cfg: dict | None = None) -> pd.DataFrame:
    screening_cfg = (cfg or {}).get('screening', {})
    candidate_space_name = screening_cfg.get('candidate_space_name', DEFAULT_CANDIDATE_SPACE_NAME)
    candidate_space_kind = screening_cfg.get('candidate_space_kind', DEFAULT_CANDIDATE_SPACE_KIND)
    candidate_space_note = screening_cfg.get('candidate_space_note', DEFAULT_CANDIDATE_SPACE_NOTE)

    group13 = ['B', 'Al', 'Ga', 'In', 'Tl']
    group15 = ['N', 'P', 'As', 'Sb', 'Bi']
    rows = []
    for left in group13:
        for right in group15:
            rows.append({
                'formula': f'{left}{right}',
                'candidate_space_name': candidate_space_name,
                'candidate_space_kind': candidate_space_kind,
                'candidate_generation_strategy': 'group13_group15_cartesian_product',
                'candidate_space_note': candidate_space_note,
            })
    return pd.DataFrame(rows)


def _ordered_values(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def get_candidate_feature_sets(cfg: dict) -> list[str]:
    features_cfg = cfg.get('features', {})
    default_feature_set = features_cfg.get('feature_set', BASIC_FEATURE_SET)
    candidate_sets = list(features_cfg.get('candidate_sets', [default_feature_set]))
    return _ordered_values([default_feature_set] + candidate_sets)


def get_candidate_model_types(cfg: dict) -> list[str]:
    model_cfg = cfg.get('model', {})
    default_model_type = model_cfg.get('type', 'hist_gradient_boosting')
    candidate_types = list(model_cfg.get('candidate_types', [default_model_type]))
    return _ordered_values([default_model_type] + candidate_types)


def _basic_features(formula: str) -> tuple[dict[str, float], str | None]:
    elements = extract_elements(formula)
    if not elements:
        return (
            {column: np.nan for column in BASIC_FEATURE_COLUMNS},
            'ValueError: no elements parsed from formula',
        )

    z = [ATOMIC_NUMBERS.get(e, 0) for e in elements]
    return ({
        'n_elements': len(elements),
        'sum_z': sum(z),
        'max_z': max(z),
        'min_z': min(z),
        'mean_z': sum(z) / len(z),
        'contains_B': int('B' in elements),
        'contains_N': int('N' in elements),
    }, None)


def _clean_feature_label(label: str, prefix: str) -> str:
    normalized = re.sub(r'[^0-9a-zA-Z]+', '_', label).strip('_').lower()
    return f'{prefix}_{normalized}' if normalized else prefix


@lru_cache(maxsize=1)
def _matminer_featurizer() -> MultipleFeaturizer:
    featurizer = MultipleFeaturizer([
        Stoichiometry(),
        ElementProperty.from_preset('magpie'),
    ])
    featurizer.set_n_jobs(1)
    return featurizer


@lru_cache(maxsize=1)
def _matminer_feature_spec() -> tuple[tuple[int, str], ...]:
    selected_labels = set(MATMINER_SELECTED_RAW_LABELS)
    labels: list[tuple[int, str]] = []
    seen: set[str] = set()
    for idx, raw_label in enumerate(_matminer_featurizer().feature_labels()):
        if raw_label not in selected_labels:
            continue
        clean_label = _clean_feature_label(raw_label, prefix='matminer')
        if clean_label in seen:
            raise ValueError(f'Duplicate cleaned matminer feature label: {clean_label}')
        seen.add(clean_label)
        labels.append((idx, clean_label))

    missing = sorted(selected_labels - {raw for raw in _matminer_featurizer().feature_labels() if raw in selected_labels})
    if missing:
        raise ValueError(f'Missing configured matminer labels: {missing}')
    return tuple(labels)


def _matminer_features(formula: str) -> tuple[dict[str, float], str | None]:
    feature_spec = _matminer_feature_spec()
    labels = [label for _, label in feature_spec]
    try:
        values = _matminer_featurizer().featurize(Composition(str(formula)))
        selected_values = [values[idx] for idx, _ in feature_spec]
        values_array = np.asarray(selected_values, dtype=float)
    except Exception as exc:  # pragma: no cover - exact exception type depends on formula failure mode
        return {label: np.nan for label in labels}, f'{type(exc).__name__}: {exc}'

    if not np.isfinite(values_array).all():
        return {label: np.nan for label in labels}, 'ValueError: non-finite matminer features'

    return dict(zip(labels, values_array.tolist())), None


def build_feature_table(
    df: pd.DataFrame,
    formula_col: str = 'formula',
    feature_set: str = BASIC_FEATURE_SET,
) -> pd.DataFrame:
    base = df.copy().reset_index(drop=True)
    formula_series = base[formula_col].astype(str)

    feature_dicts: list[dict[str, float]] = []
    errors: list[str | None] = []

    if feature_set == BASIC_FEATURE_SET:
        for formula in formula_series:
            feature_row, error = _basic_features(formula)
            feature_dicts.append(feature_row)
            errors.append(error)
    elif feature_set == MATMINER_FEATURE_SET:
        for formula in formula_series:
            feature_row, error = _matminer_features(formula)
            feature_dicts.append(feature_row)
            errors.append(error)
    else:
        raise ValueError(f'Unsupported feature set: {feature_set}')

    feature_rows = pd.DataFrame(feature_dicts)
    out = pd.concat([base, feature_rows], axis=1)
    out['feature_set'] = feature_set
    out['feature_generation_failed'] = pd.Series([error is not None for error in errors], dtype=bool)
    out['feature_generation_error'] = errors
    return out


def build_feature_tables(
    df: pd.DataFrame,
    cfg: dict,
    formula_col: str = 'formula',
) -> dict[str, pd.DataFrame]:
    return {
        feature_set: build_feature_table(df, formula_col=formula_col, feature_set=feature_set)
        for feature_set in get_candidate_feature_sets(cfg)
    }


def _ratio_counts(total: int, ratios: list[float]) -> list[int]:
    if total <= 0:
        return [0] * len(ratios)

    raw_ratios = np.asarray(ratios, dtype=float)
    if raw_ratios.sum() <= 0:
        raise ValueError('Split ratios must sum to a positive value')

    normalized = raw_ratios / raw_ratios.sum()
    raw_counts = normalized * total
    counts = np.floor(raw_counts).astype(int)

    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1

    positive_indices = [idx for idx, ratio in enumerate(raw_ratios) if ratio > 0]
    if total >= len(positive_indices):
        for idx in positive_indices:
            if counts[idx] > 0:
                continue
            donor_order = np.argsort(counts)[::-1]
            for donor_idx in donor_order:
                if donor_idx != idx and counts[donor_idx] > 1:
                    counts[donor_idx] -= 1
                    counts[idx] += 1
                    break

    return counts.astype(int).tolist()


def _build_split_metadata(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    group_col: str,
    method: str,
    cfg: dict,
) -> dict:
    group_sets = {
        name: set(df.loc[masks[name], group_col].astype(str).tolist())
        for name in ('train', 'val', 'test')
    }
    return {
        'method': method,
        'group_column': group_col,
        'ratios': {
            'train': float(cfg['split']['train_ratio']),
            'val': float(cfg['split']['val_ratio']),
            'test': float(cfg['split']['test_ratio']),
        },
        'row_counts': {
            name: int(np.asarray(masks[name]).sum())
            for name in ('train', 'val', 'test')
        },
        'group_counts': {
            name: int(len(group_sets[name]))
            for name in ('train', 'val', 'test')
        },
        'group_overlap_counts': {
            'train_val': int(len(group_sets['train'] & group_sets['val'])),
            'train_test': int(len(group_sets['train'] & group_sets['test'])),
            'val_test': int(len(group_sets['val'] & group_sets['test'])),
        },
    }


def _make_random_row_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
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
    masks['metadata'] = _build_split_metadata(
        df=df,
        masks=masks,
        group_col=cfg['split'].get('group_column', 'formula'),
        method='random_row',
        cfg=cfg,
    )
    return masks


def _make_grouped_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    group_col = cfg['split'].get('group_column', 'formula')
    if group_col not in df.columns:
        raise KeyError(f'Group split column not found: {group_col}')

    rng = np.random.default_rng(cfg['project']['random_seed'])
    group_series = df[group_col].astype(str)
    unique_groups = group_series.drop_duplicates().to_numpy()
    rng.shuffle(unique_groups)

    train_count, val_count, test_count = _ratio_counts(
        total=len(unique_groups),
        ratios=[
            cfg['split']['train_ratio'],
            cfg['split']['val_ratio'],
            cfg['split']['test_ratio'],
        ],
    )

    train_groups = set(unique_groups[:train_count])
    val_groups = set(unique_groups[train_count:train_count + val_count])
    test_groups = set(unique_groups[train_count + val_count:train_count + val_count + test_count])

    masks = {
        'train': group_series.isin(train_groups).to_numpy(),
        'val': group_series.isin(val_groups).to_numpy(),
        'test': group_series.isin(test_groups).to_numpy(),
    }
    masks['metadata'] = _build_split_metadata(
        df=df,
        masks=masks,
        group_col=group_col,
        method='group_by_formula',
        cfg=cfg,
    )
    return masks


def make_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    method = cfg['split'].get('method', 'group_by_formula')
    if method == 'group_by_formula':
        return _make_grouped_split_masks(df, cfg)
    if method == 'random_row':
        return _make_random_row_split_masks(df, cfg)
    raise ValueError(f'Unsupported split method: {method}')


def _feature_columns(df: pd.DataFrame) -> list[str]:
    banned = {
        'record_id',
        'source',
        'formula',
        'target',
        'elements',
        'feature_set',
        'feature_generation_failed',
        'feature_generation_error',
        'candidate_space_name',
        'candidate_space_kind',
        'candidate_generation_strategy',
        'candidate_space_note',
        'ranking_label',
        'ranking_basis',
        'ranking_note',
        'ranking_feature_set',
        'ranking_model_type',
    }
    return [c for c in df.columns if c not in banned]


def _feature_valid_mask(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    if 'feature_generation_failed' in df.columns:
        mask &= ~df['feature_generation_failed'].fillna(False).astype(bool)
    if feature_columns:
        mask &= df[feature_columns].notna().all(axis=1)
    return mask


def summarize_feature_table(feature_df: pd.DataFrame, feature_set: str | None = None) -> dict:
    inferred_feature_set = feature_set
    if inferred_feature_set is None and 'feature_set' in feature_df.columns and not feature_df.empty:
        inferred_feature_set = str(feature_df['feature_set'].iloc[0])

    feature_columns = _feature_columns(feature_df)
    failed_mask = feature_df.get(
        'feature_generation_failed',
        pd.Series(False, index=feature_df.index, dtype=bool),
    ).astype(bool)
    formula_examples = []
    if 'formula' in feature_df.columns:
        formula_examples = feature_df.loc[failed_mask, 'formula'].astype(str).head(5).tolist()
    error_examples = []
    if 'feature_generation_error' in feature_df.columns:
        error_examples = (
            feature_df.loc[failed_mask, 'feature_generation_error']
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )
    status = 'ok'
    if not feature_columns:
        status = 'no_features'
    elif failed_mask.any():
        status = 'featurization_incomplete'

    return {
        'feature_set': inferred_feature_set,
        'feature_family': FEATURE_FAMILY,
        'feature_note': FEATURE_SET_NOTES.get(inferred_feature_set or '', ''),
        'n_features': int(len(feature_columns)),
        'status': status,
        'selection_eligible': bool(status == 'ok'),
        'failed_formula_count': int(failed_mask.sum()),
        'failed_formula_examples': formula_examples,
        'failed_error_examples': error_examples,
    }


def make_model(cfg: dict, model_type: str | None = None):
    model_type = model_type or cfg['model']['type']
    if model_type == 'linear_regression':
        return LinearRegression(**cfg['model'].get('linear_regression', {}))
    if model_type == 'random_forest':
        return RandomForestRegressor(**cfg['model']['random_forest'])
    if model_type == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**cfg['model']['hist_gradient_boosting'])
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
    training_mask = split_masks['train'] | split_masks['val'] if include_validation else split_masks['train']
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
    requested_eval_df = df.loc[split_masks[split_name]].copy()
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


def _ordered_model_types(values: list[str]) -> list[str]:
    return _ordered_values(values)


def _metric_key(value):
    if value is None:
        return np.inf
    return value


def select_feature_model_combo(feature_tables: dict[str, pd.DataFrame], split_masks, cfg: dict) -> dict:
    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    selection_metric = cfg['model'].get('selection_metric', 'mae')
    use_validation_selection = bool(cfg['model'].get('use_validation_selection', True))
    default_feature_set = cfg['features'].get('feature_set', candidate_feature_sets[0])
    default_model_type = cfg['model'].get('type', candidate_model_types[0])

    feature_set_results = [
        summarize_feature_table(feature_tables[feature_set], feature_set=feature_set)
        for feature_set in candidate_feature_sets
    ]
    eligible_feature_sets = [
        item['feature_set']
        for item in feature_set_results
        if item['selection_eligible']
    ]

    if not eligible_feature_sets:
        raise ValueError('No candidate feature set could featurize the full dataset')

    if default_feature_set not in eligible_feature_sets:
        default_feature_set = eligible_feature_sets[0]

    summary = {
        'selection_space': 'feature_set_and_model_type',
        'selection_scope': FEATURE_FAMILY,
        'candidate_feature_sets': candidate_feature_sets,
        'candidate_model_types': candidate_model_types,
        'selection_metric': selection_metric,
        'used_validation_selection': False,
        'selected_feature_set': default_feature_set,
        'selected_model_type': default_model_type,
        'selected_feature_count': int(
            summarize_feature_table(feature_tables[default_feature_set], feature_set=default_feature_set)['n_features']
        ),
        'feature_set_results': feature_set_results,
        'validation_results': [],
    }

    if not use_validation_selection or len(candidate_feature_sets) * len(candidate_model_types) == 1:
        return summary

    if int(np.asarray(split_masks['val']).sum()) == 0:
        summary['selection_note'] = 'validation_split_empty'
        return summary

    best_metrics = None
    best_feature_set = default_feature_set
    best_model_type = default_model_type

    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        for model_type in candidate_model_types:
            result = {
                'feature_set': feature_set,
                'model_type': model_type,
                'n_features': feature_info['n_features'],
                'status': 'ok',
                'note': '',
                'mae': None,
                'rmse': None,
                'r2': None,
            }

            if not feature_info['selection_eligible']:
                result['status'] = 'skipped_featurization_failure'
                result['note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                summary['validation_results'].append(result)
                continue

            try:
                model, feature_columns = train_baseline_model(
                    df=feature_df,
                    split_masks=split_masks,
                    cfg=cfg,
                    model_type=model_type,
                    include_validation=False,
                )
                metrics, _ = evaluate_predictions(
                    df=feature_df,
                    split_masks=split_masks,
                    model=model,
                    feature_columns=feature_columns,
                    split_name='val',
                )
            except Exception as exc:
                result['status'] = 'evaluation_failed'
                result['note'] = f'{type(exc).__name__}: {exc}'
                summary['validation_results'].append(result)
                continue

            result.update(metrics)
            summary['validation_results'].append(result)

            if selection_metric == 'r2':
                is_better = (
                    best_metrics is None
                    or _metric_key(metrics['r2']) > _metric_key(best_metrics['r2'])
                )
            else:
                is_better = (
                    best_metrics is None
                    or _metric_key(metrics[selection_metric]) < _metric_key(best_metrics[selection_metric])
                )
            if is_better:
                best_metrics = metrics
                best_feature_set = feature_set
                best_model_type = model_type

    if best_metrics is None:
        raise ValueError('Validation selection failed for every candidate feature/model combination')

    summary['used_validation_selection'] = True
    summary['selected_feature_set'] = best_feature_set
    summary['selected_model_type'] = best_model_type
    summary['selected_feature_count'] = int(
        summarize_feature_table(feature_tables[best_feature_set], feature_set=best_feature_set)['n_features']
    )
    return summary


def select_model_type(feature_tables: dict[str, pd.DataFrame], split_masks, cfg: dict) -> dict:
    return select_feature_model_combo(feature_tables, split_masks, cfg)


def benchmark_regressors(
    feature_tables: dict[str, pd.DataFrame],
    split_masks,
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
) -> pd.DataFrame:
    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    baseline_types = _ordered_model_types(cfg['model'].get('benchmark_baselines', ['dummy_mean']))

    rows = []
    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        for model_type in candidate_model_types:
            row = {
                'feature_set': feature_set,
                'feature_family': FEATURE_FAMILY,
                'n_features': feature_info['n_features'],
                'model_type': model_type,
                'benchmark_role': 'candidate_model',
                'selected_by_validation': bool(
                    feature_set == selected_feature_set and model_type == selected_model_type
                ),
                'training_scope': 'train_plus_val',
                'evaluation_split': 'test',
                'benchmark_status': 'ok',
                'benchmark_note': '',
                'mae': None,
                'rmse': None,
                'r2': None,
            }

            if row['selected_by_validation']:
                row['benchmark_role'] = 'selected_model'

            if not feature_info['selection_eligible']:
                row['benchmark_status'] = 'skipped_featurization_failure'
                row['benchmark_note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                rows.append(row)
                continue

            model, feature_columns = train_baseline_model(
                df=feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=True,
            )
            metrics, _ = evaluate_predictions(
                df=feature_df,
                split_masks=split_masks,
                model=model,
                feature_columns=feature_columns,
                split_name='test',
            )
            row.update(metrics)
            rows.append(row)

    selected_feature_df = feature_tables[selected_feature_set]
    selected_feature_columns = _feature_columns(selected_feature_df)
    selected_feature_count = len(selected_feature_columns)
    for model_type in baseline_types:
        model, feature_columns = train_baseline_model(
            df=selected_feature_df,
            split_masks=split_masks,
            cfg=cfg,
            model_type=model_type,
            include_validation=True,
        )
        metrics, _ = evaluate_predictions(
            df=selected_feature_df,
            split_masks=split_masks,
            model=model,
            feature_columns=feature_columns,
            split_name='test',
        )
        rows.append({
            'feature_set': DUMMY_FEATURE_SET,
            'feature_family': 'feature_agnostic_baseline',
            'n_features': int(selected_feature_count),
            'model_type': model_type,
            'benchmark_role': 'dummy_baseline',
            'selected_by_validation': False,
            'training_scope': 'train_plus_val',
            'evaluation_split': 'test',
            'benchmark_status': 'ok',
            'benchmark_note': FEATURE_SET_NOTES[DUMMY_FEATURE_SET],
            **metrics,
        })

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def screen_candidates(
    candidate_df: pd.DataFrame,
    model,
    feature_columns: list[str],
    cfg: dict,
    feature_set: str,
    model_type: str,
) -> pd.DataFrame:
    feature_df = build_feature_table(candidate_df, formula_col='formula', feature_set=feature_set)
    feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
    if not feature_info['selection_eligible']:
        raise ValueError(
            'Candidate ranking aborted because the selected feature set could not featurize '
            f'candidate formulas: {feature_info["failed_formula_examples"]}'
        )

    pred = model.predict(feature_df[feature_columns])
    out = feature_df[[
        'formula',
        'candidate_space_name',
        'candidate_space_kind',
        'candidate_generation_strategy',
        'candidate_space_note',
    ]].copy()
    out['predicted_band_gap'] = pred
    out['ranking_label'] = cfg['screening'].get('ranking_label', 'demo_candidate_ranking')
    out['ranking_basis'] = RANKING_BASIS
    out['ranking_note'] = RANKING_NOTE
    out['ranking_feature_set'] = feature_set
    out['ranking_model_type'] = model_type
    return (
        out.sort_values('predicted_band_gap', ascending=False)
        .head(cfg['screening']['top_k'])
        .reset_index(drop=True)
    )
