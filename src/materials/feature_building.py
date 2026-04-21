from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition, Element

from materials.data import STRUCTURE_SUMMARY_COLUMNS

from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import _ordered_values

def get_candidate_feature_sets(cfg: dict) -> list[str]:
    features_cfg = cfg.get('features', {})
    default_feature_set = features_cfg.get('feature_set', BASIC_FEATURE_SET)
    candidate_sets = list(features_cfg.get('candidate_sets', [default_feature_set]))
    return _ordered_values([default_feature_set] + candidate_sets)


def get_candidate_screening_feature_sets(cfg: dict) -> list[str]:
    return [
        feature_set
        for feature_set in get_candidate_feature_sets(cfg)
        if feature_set_supports_formula_only_screening(feature_set)
    ]


def get_candidate_model_types(cfg: dict) -> list[str]:
    model_cfg = cfg.get('model', {})
    default_model_type = model_cfg.get('type', 'hist_gradient_boosting')
    candidate_types = list(model_cfg.get('candidate_types', [default_model_type]))
    return _ordered_values([default_model_type] + candidate_types)


def compatible_model_types_for_feature_set(cfg: dict, feature_set: str) -> list[str]:
    return [
        model_type
        for model_type in get_candidate_model_types(cfg)
        if model_type_supports_feature_set(model_type, feature_set)
    ]


def model_type_supports_feature_set(model_type: str, feature_set: str) -> bool:
    allowed_feature_sets = MODEL_FEATURE_SET_COMPATIBILITY.get(str(model_type))
    if allowed_feature_sets is None:
        return True
    return str(feature_set) in set(allowed_feature_sets)


def incompatible_model_feature_note(model_type: str, feature_set: str) -> str:
    if str(model_type) in {
        'torch_fractional_attention',
        'torch_sparse_fractional_attention',
        'torch_roost_like',
    }:
        return (
            f'Skipped because {model_type} is currently defined only for the '
            'fractional_composition_vector feature set.'
        )
    return (
        f'Skipped because model_type={model_type} is not configured for feature_set={feature_set}.'
    )


def _default_model_type_for_feature_set(cfg: dict, feature_set: str) -> str:
    compatible_model_types = compatible_model_types_for_feature_set(cfg, feature_set)
    if not compatible_model_types:
        raise ValueError(
            'No compatible candidate model type is available for '
            f'feature_set={feature_set}.'
        )
    configured_default_model_type = str(cfg['model'].get('type', compatible_model_types[0]))
    if configured_default_model_type in compatible_model_types:
        return configured_default_model_type
    return compatible_model_types[0]


def get_feature_family(feature_set: str) -> str:
    return FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[0]


def feature_set_supports_formula_only_screening(feature_set: str) -> bool:
    return bool(FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[1])


def get_feature_note(feature_set: str) -> str:
    return FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[2]


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


def _fractional_composition_features(formula: str) -> tuple[dict[str, float], str | None]:
    empty_row = {column: 0.0 for column in FRACTIONAL_COMPOSITION_COLUMNS}
    try:
        composition = Composition(str(formula))
        fractional = composition.fractional_composition.get_el_amt_dict()
    except Exception as exc:  # pragma: no cover - exact type depends on parsing failure mode
        return {column: np.nan for column in FRACTIONAL_COMPOSITION_COLUMNS}, f'{type(exc).__name__}: {exc}'

    feature_row = empty_row.copy()
    total_fraction = 0.0
    for symbol, value in fractional.items():
        column = f'frac_{str(symbol).lower()}'
        numeric_value = float(value)
        feature_row[column] = numeric_value
        total_fraction += numeric_value

    if not np.isfinite(total_fraction) or total_fraction <= 0:
        return {column: np.nan for column in FRACTIONAL_COMPOSITION_COLUMNS}, 'ValueError: invalid fractional composition'
    return feature_row, None


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


def _build_base_frame(df: pd.DataFrame, formula_col: str, feature_set: str) -> pd.DataFrame:
    if formula_col not in df.columns:
        raise KeyError(f'Formula column not found: {formula_col}')

    base = pd.DataFrame(index=df.index)
    for column in BASE_PASSTHROUGH_COLUMNS:
        if column == 'formula':
            base[column] = df[formula_col].astype(str)
        elif column in df.columns:
            base[column] = df[column]

    if feature_set == STRUCTURE_AWARE_FEATURE_SET:
        for column in STRUCTURE_AWARE_REQUIRED_COLUMNS:
            if column in df.columns:
                base[column] = df[column]
            else:
                base[column] = np.nan

    return base.reset_index(drop=True)


def _validate_structure_summary_features(row: pd.Series) -> str | None:
    missing_columns = [column for column in STRUCTURE_AWARE_REQUIRED_COLUMNS if column not in row.index]
    if missing_columns:
        return f'ValueError: missing structure summary columns: {missing_columns}'

    invalid_columns = []
    for column in STRUCTURE_AWARE_REQUIRED_COLUMNS:
        value = row[column]
        if pd.isna(value):
            invalid_columns.append(column)
            continue
        try:
            numeric_value = float(value)
        except Exception:
            invalid_columns.append(column)
            continue
        if not np.isfinite(numeric_value):
            invalid_columns.append(column)

    if invalid_columns:
        return f'ValueError: missing structure summary values: {invalid_columns}'
    return None


def _combine_feature_errors(*errors: str | None) -> str | None:
    messages = [error for error in errors if error]
    if not messages:
        return None
    return '; '.join(messages)


def build_feature_table(
    df: pd.DataFrame,
    formula_col: str = 'formula',
    feature_set: str = BASIC_FEATURE_SET,
) -> pd.DataFrame:
    base = _build_base_frame(df, formula_col=formula_col, feature_set=feature_set)
    formula_series = df[formula_col].astype(str).reset_index(drop=True)

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
    elif feature_set == FRACTIONAL_COMPOSITION_FEATURE_SET:
        for formula in formula_series:
            feature_row, error = _fractional_composition_features(formula)
            feature_dicts.append(feature_row)
            errors.append(error)
    elif feature_set == STRUCTURE_AWARE_FEATURE_SET:
        for formula, (_, row) in zip(formula_series, base.iterrows(), strict=False):
            matminer_row, matminer_error = _matminer_features(formula)
            structure_error = _validate_structure_summary_features(row)
            feature_dicts.append(matminer_row)
            errors.append(_combine_feature_errors(matminer_error, structure_error))
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
        *REFERENCE_PROPERTY_COLUMNS,
        'elements',
        'feature_set',
        'feature_generation_failed',
        'feature_generation_error',
        'candidate_space_name',
        'candidate_space_kind',
        'candidate_generation_strategy',
        'candidate_space_note',
        'candidate_family',
        'candidate_template',
        'candidate_family_note',
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
        'feature_family': get_feature_family(inferred_feature_set or ''),
        'feature_note': get_feature_note(inferred_feature_set or ''),
        'candidate_compatible': feature_set_supports_formula_only_screening(inferred_feature_set or ''),
        'n_features': int(len(feature_columns)),
        'status': status,
        'selection_eligible': bool(status == 'ok'),
        'failed_formula_count': int(failed_mask.sum()),
        'failed_formula_examples': formula_examples,
        'failed_error_examples': error_examples,
    }

